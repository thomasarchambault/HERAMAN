# Rôle : Exécuter la sélection ROI et créer les jeux de données spectraux utilisés par le pipeline.

# roi_select.py — sélection simple de ROIs → dataset 

from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path
import numpy as np
import cv2
from shared.image_io import read_image, write_image
import shutil
from datetime import datetime
from scipy.signal import savgol_filter

from shared.paths import portable_project_path
from shared.dataset_provenance import (
    cube_id_from_path,
    cube_ids_from_dataset,
    cube_paths_from_dataset,
    migrate_dataset_cube_ids,
)

def _write_library_json(path, library):
    """Serialize before replacing an existing library, so errors cannot empty it."""
    path = Path(path)
    payload = json.dumps(library, indent=2, ensure_ascii=False)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ----------------------- Utils  -----------------------
def _snv_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standard Normal Variate (SNV).

    args:
        X: (n_samples, n_bands) 
    returns:
        X_snv: (n_samples, n_bands) SNV-normalized"""
    
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def _central_trim_mask(X: np.ndarray, keep_frac: float = 0.95) -> np.ndarray:
    """ Keep central fraction of rows in X (L2 distance to mean). keep_frac in (0,1].

    args:
        X: (n_samples, n_bands)
    returns:
        boolean mask (n_samples,) where True = keep
    """
    if X.shape[0] == 0 or keep_frac >= 1.0:
        return np.ones((X.shape[0],), dtype=bool)
    m = X.mean(axis=0, keepdims=True)
    d = np.linalg.norm(X - m, axis=1)
    thr = np.quantile(d, keep_frac)
    return d <= thr


def _parse_wavelengths_from_folder(
    folder: Path,
    n_bands: int | None = None,
) -> np.ndarray | None:
    """Lit l'axe spectral depuis les fichiers corr_XXXnm proches du cube."""
    folder = Path(folder)
    if not folder.is_dir():
        return None

    allowed_ext = {".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    found: set[int] = set()

    try:
        paths = folder.iterdir()
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in allowed_ext:
                continue
            match = re.search(r"(\d{3,4})\s*nm", path.name.lower())
            if match:
                found.add(int(match.group(1)))
    except OSError:
        return None

    if not found:
        return None

    wavelengths = np.asarray(sorted(found), dtype=np.float32)
    if n_bands is not None and wavelengths.size != int(n_bands):
        return None
    return wavelengths


def _infer_wavelengths_near_cube(
    cube_path: str | Path,
    n_bands: int,
) -> np.ndarray | None:
    """Cherche les longueurs d'onde dans le dossier cube puis son parent."""
    if Path(cube_path).suffix.lower() in ('.hdr', '.raw', '.img', '.dat', '.bin'):
        from shared.envi import load_spectral_cube
        try:
            cube, axis = load_spectral_cube(cube_path)
            count = cube.shape[2]
            mapping = getattr(cube, '_mmap', None)
            if mapping is not None: mapping.close()
            return axis if n_bands is None or len(axis) == n_bands else None
        except (OSError, ValueError):
            return None

    path = Path(cube_path)
    from shared.corrected_layout import nearby_image_folders
    candidates = nearby_image_folders(path)

    for folder in candidates:
        wavelengths = _parse_wavelengths_from_folder(folder, n_bands=n_bands)
        if wavelengths is not None:
            return wavelengths

    # Compatibilité avec l'ancien axe complet seulement lorsque sa taille concorde.
    historical = np.arange(420, 721, 10, dtype=np.float32)
    if historical.size == int(n_bands):
        return historical
    return None


def _resolve_cube_wavelengths(
    cube_path: str | Path,
    n_bands: int,
    provided: list[float] | np.ndarray | None,
) -> np.ndarray:
    """Retourne un axe spectral validé pour le cube courant."""
    if provided is not None:
        wavelengths = np.asarray(provided, dtype=np.float32).reshape(-1)
        if wavelengths.size != int(n_bands):
            raise SystemExit(
                f"[STOP] {wavelengths.size} longueurs d'onde fournies pour "
                f"un cube de {n_bands} bandes."
            )
        return wavelengths

    wavelengths = _infer_wavelengths_near_cube(cube_path, n_bands=n_bands)
    if wavelengths is None:
        raise SystemExit(
            "[STOP] Impossible de déterminer les longueurs d'onde du cube. "
            "Conserver les fichiers corr_XXXnm près du cube ou fournir --wavelengths."
        )
    return wavelengths


def _read_dataset_wavelengths(dataset, n_bands: int) -> np.ndarray | None:
    """Lit l'axe spectral d'un dataset récent ou tente de migrer un ancien format."""
    wavelengths = None

    if "wavelengths" in dataset.files:
        wavelengths = np.asarray(dataset["wavelengths"], dtype=np.float32).reshape(-1)
    elif "meta" in dataset.files:
        try:
            meta = dataset["meta"].item()
            if isinstance(meta, dict) and "wavelengths" in meta:
                wavelengths = np.asarray(meta["wavelengths"], dtype=np.float32).reshape(-1)
        except Exception:
            wavelengths = None

    if wavelengths is not None and wavelengths.size == int(n_bands):
        return wavelengths

    if "cube_paths" in dataset.files:
        try:
            for cube_path in np.asarray(dataset["cube_paths"], dtype=object).tolist():
                wavelengths = _infer_wavelengths_near_cube(cube_path, n_bands=n_bands)
                if wavelengths is not None:
                    return wavelengths
        except Exception:
            pass

    historical = np.arange(420, 721, 10, dtype=np.float32)
    if historical.size == int(n_bands):
        return historical
    return None


# ----------------------- ROI display backgrounds -----------------------
def _normalize_to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image 2D/3D quelconque en BGR uint8 affichable.
    Sert uniquement à l'affichage ROI; ne modifie pas le cube ni les pixels extraits.
    """
    if img is None:
        raise ValueError("Image vide.")

    arr = np.asarray(img)

    if arr.ndim == 2:
        arr_f = arr.astype(np.float32)
        finite = np.isfinite(arr_f)
        if not np.any(finite):
            out = np.zeros(arr_f.shape, dtype=np.uint8)
        else:
            p1, p99 = np.nanpercentile(arr_f[finite], [1, 99])
            if p99 <= p1:
                mn, mx = float(np.nanmin(arr_f[finite])), float(np.nanmax(arr_f[finite]))
                p1, p99 = mn, mx
            if p99 > p1:
                out = np.clip((arr_f - p1) / (p99 - p1), 0, 1)
                out = (out * 255).astype(np.uint8)
            else:
                out = np.zeros(arr_f.shape, dtype=np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            finite = np.isfinite(arr_f)
            if np.any(finite):
                p1, p99 = np.nanpercentile(arr_f[finite], [1, 99])
                if p99 > p1:
                    arr = np.clip((arr_f - p1) / (p99 - p1), 0, 1)
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = np.zeros(arr_f.shape, dtype=np.uint8)
            else:
                arr = np.zeros(arr_f.shape, dtype=np.uint8)
        return arr.astype(np.uint8)

    raise ValueError(f"Format image non supporté pour affichage ROI: shape={arr.shape}")


def _find_wavelength_image_near_cube(cube_path: Path, rgb_image_path: str | None, wavelength: int) -> Path | None:
    """
    Cherche une image corrigée du type corr_490nm.png autour du cube/base_image.
    Si rien n'est trouvé, le code ROI utilisera directement la bande du cube.
    """
    allowed_ext = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

    search_roots: list[Path] = []
    from shared.corrected_layout import nearby_image_folders
    for root in nearby_image_folders(cube_path):
        if root and root.exists() and root.is_dir() and root not in search_roots:
            search_roots.append(root)

    if rgb_image_path:
        rp = Path(rgb_image_path)
        for root in [rp.parent, rp.parent.parent]:
            if root and root.exists() and root.is_dir() and root not in search_roots:
                search_roots.append(root)

    # Recherche directe prioritaire, puis récursive limitée.
    for root in search_roots:
        try:
            files = list(root.iterdir())
        except Exception:
            files = []
        for path in files:
            if not path.is_file() or path.suffix.lower() not in allowed_ext:
                continue
            name = path.name.lower()
            if str(wavelength) in name and "nm" in name:
                return path

    for root in search_roots:
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in allowed_ext:
                    continue
                rel = path.relative_to(root)
                if len(rel.parts) > 3:  # évite de scanner trop profondément
                    continue
                name = path.name.lower()
                if str(wavelength) in name and "nm" in name:
                    return path
        except Exception:
            continue

    return None


def _make_roi_display_images(
    cube: np.ndarray,
    cube_path: Path,
    rgb_image_path: str | None,
    img8_default: np.ndarray,
    wavelengths: list[float] | np.ndarray | None = None,
    roi_background_wavelengths: tuple[int, ...] = (490, 560, 700),
) -> dict[str, np.ndarray]:
    """
    Construit les images d'affichage disponibles pour la sélection ROI.
    Priorité:
      1) RGB synthétique / image fournie
      2) fichiers corr_XXXnm trouvés autour du cube
      3) bande correspondante directement extraite du cube
    """
    H, W, N = cube.shape
    display_images: dict[str, np.ndarray] = {"Synthetic RGB": _normalize_to_uint8_bgr(img8_default)}

    if wavelengths is None:
        # Convention actuelle du système: 420:10:720 pour 31 bandes.
        if N == 31:
            wl_arr = np.arange(420, 721, 10, dtype=float)
        else:
            wl_arr = np.arange(N, dtype=float)
    else:
        wl_arr = np.asarray(wavelengths, dtype=float)
        if wl_arr.size != N:
            print(f"[ROI][WARN] wavelengths a {wl_arr.size} valeurs mais le cube a {N} bandes. Fallback index bands.")
            wl_arr = np.arange(N, dtype=float)

    for wl in roi_background_wavelengths:
        label = f"{int(wl)} nm"
        img_bgr = None

        found = _find_wavelength_image_near_cube(cube_path, rgb_image_path, int(wl))
        if found is not None:
            raw = read_image(str(found), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                img_bgr = _normalize_to_uint8_bgr(raw)
                print(f"[ROI][BG] {label} chargé depuis fichier: {found}")

        if img_bgr is None:
            band_idx = int(np.argmin(np.abs(wl_arr - float(wl))))
            img_bgr = _normalize_to_uint8_bgr(cube[..., band_idx])
            nearest = wl_arr[band_idx]
            print(f"[ROI][BG] {label} généré depuis cube: bande {band_idx} ({nearest:g})")

        if img_bgr.shape[:2] != (H, W):
            print(f"[ROI][BG] {label}: resize {img_bgr.shape[1]}x{img_bgr.shape[0]} -> {W}x{H}")
            img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        display_images[label] = img_bgr

    print("[ROI][BG] Images disponibles:", " | ".join(display_images.keys()))
    return display_images



def _find_roi_mask_overlay(cube_path: Path, rgb_image_path: str | None, out_dir: Path | None = None) -> Path | None:
    """
    Cherche l'image de masque/warp à afficher par-dessus le background pendant la sélection ROI.
    Nom attendu en priorité: FINAL_warped_segments_only.png
    """
    candidates: list[Path] = []
    names = [
        "FINAL_warped_segments_only.png",
        "final_warped_segments_only.png",
        "warped_segments_only.png",
        "FINAL_warped_mask.png",
        "warped_mask.png",
    ]

    roots: list[Path] = []
    from shared.corrected_layout import nearby_image_folders
    for root in nearby_image_folders(cube_path):
        if root and root.exists() and root.is_dir() and root not in roots:
            roots.append(root)

    if rgb_image_path:
        rp = Path(rgb_image_path)
        for root in [rp.parent, rp.parent.parent]:
            if root and root.exists() and root.is_dir() and root not in roots:
                roots.append(root)

    if out_dir is not None:
        for root in [out_dir, out_dir.parent]:
            if root and root.exists() and root.is_dir() and root not in roots:
                roots.append(root)

    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path

    # Recherche récursive limitée si le fichier n'est pas directement à côté.
    for root in roots:
        try:
            for path in root.rglob("*.png"):
                rel = path.relative_to(root)
                if len(rel.parts) > 4:
                    continue
                low = path.name.lower()
                if "warped" in low and ("segments" in low or "mask" in low):
                    candidates.append(path)
        except Exception:
            pass

    if candidates:
        # priorité au nom le plus proche de FINAL_warped_segments_only
        candidates = sorted(candidates, key=lambda x: ("final_warped_segments_only" not in x.name.lower(), len(str(x))))
        return candidates[0]

    return None


def _load_roi_mask_overlay(cube_path: Path, rgb_image_path: str | None, out_dir: Path | None, target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Charge le masque/segment warpé pour l'overlay ROI.
    Retourne (overlay_bgr, alpha_mask_bool, source_path).
    """
    path = _find_roi_mask_overlay(cube_path, rgb_image_path, out_dir)
    if path is None:
        print("[ROI][MASK] Aucun overlay FINAL_warped_segments_only.png trouvé.")
        return None

    raw = read_image(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"[ROI][MASK] Impossible de lire: {path}")
        return None

    if raw.ndim == 2:
        # _normalize_to_uint8_bgr() already returns a three-channel BGR image.
        overlay_bgr = _normalize_to_uint8_bgr(raw)
        alpha = raw > 0
    else:
        if raw.shape[2] == 4:
            overlay_bgr = raw[:, :, :3].copy()
            alpha = raw[:, :, 3] > 0
        else:
            overlay_bgr = raw[:, :, :3].copy()
            # Fond noir = transparent. C'est le cas typique de FINAL_warped_segments_only.png.
            alpha = np.any(overlay_bgr > 8, axis=2)

        if overlay_bgr.dtype != np.uint8:
            overlay_bgr = _normalize_to_uint8_bgr(overlay_bgr)

    H, W = target_hw
    if overlay_bgr.shape[:2] != (H, W):
        print(f"[ROI][MASK] Resize overlay {overlay_bgr.shape[1]}x{overlay_bgr.shape[0]} -> {W}x{H}")
        overlay_bgr = cv2.resize(overlay_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    print(f"[ROI][MASK] Overlay chargé: {path}")
    return overlay_bgr.astype(np.uint8), alpha.astype(bool), str(path)

def select_rois_zoomable(img8, prompt="ROI selection (h=help)", max_size=None, display_images=None):
    """
    Multi-ROI avec zoom/pan et retours visuels:
      - Scroll wheel: zoom 
      - Right/Middle-drag: pan
      - Left-drag: draw ROI
      - u/z: undo   |  c: clear   |  h: help on/off   |  d: delete ROI under cursor
      - Enter: validate  |   Esc: cancel (no ROI)
    Return [(x,y,w,h), ...] in original coordinates.
    """
    import cv2, numpy as np
    import unicodedata

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    # — Window & image setup —
    H, W = img8.shape[:2]

    # Images de référence alternatives pour l'affichage seulement.
    # Les masques ROI restent toujours en coordonnées du cube original.
    if display_images is None:
        display_images = {"Reference": img8}
    else:
        clean_display_images = {}
        for name, im in display_images.items():
            if im is None:
                continue
            im = _normalize_to_uint8_bgr(im)
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            clean_display_images[str(name)] = im
        display_images = clean_display_images or {"Reference": img8}

    bg_names = list(display_images.keys())
    bg_index = 0
    img8_current = display_images[bg_names[bg_index]]

    if max_size is None:
        try:
            import tkinter as tk
            r = tk.Tk(); r.withdraw()
            sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
            max_w, max_h = int(sw*0.9), int(sh*0.9)
        except Exception:
            max_w, max_h = 1280, 800
    else:
        max_w, max_h = max_size

    win_w, win_h = max_w, max_h
    s0 = min(1.0, max_w / W, max_h / H)   
    s, s_min, s_max = s0, max(0.02, s0*0.1), 20.0

    def make_scaled():
        return cv2.resize(
            img8_current if 'img8_current' in locals() else img8, (max(1, int(W*s)), max(1, int(H*s))),
            interpolation=cv2.INTER_AREA if s <= 1.0 else cv2.INTER_LINEAR
        )

    scaled = make_scaled()
    vx = vy = 0  # pan offsets in scaled image coords

    # — États d’interaction —
    rects = []              # (x,y,w,h) in original coords 
    drawing = False
    x0 = y0 = 0
    cur = (0, 0)            # Original image coords under cursor
    panning_r = panning_m = False
    pan_start = (0, 0)
    vx0 = vy0 = 0
    show_help = True

    win = "ROIs (zoom/pan)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, win_w, win_h)

    def screen_to_img(mx, my):
        ix = int(round((mx + vx) / s))
        iy = int(round((my + vy) / s))
        return clamp(ix, 0, W-1), clamp(iy, 0, H-1)

    def clamp_view():
        nonlocal vx, vy
        sw, sh = scaled.shape[1], scaled.shape[0]
        vx = clamp(vx, 0, max(0, sw - win_w))
        vy = clamp(vy, 0, max(0, sh - win_h))

    def find_rect_at(ix, iy):
        for k in range(len(rects)-1, -1, -1):  # last drawn = topmost
            x, y, w, h = rects[k]
            if x <= ix <= x+w and y <= iy <= y+h:
                return k
        return -1

    def redraw():
        # Windowed content
        sw, sh = scaled.shape[1], scaled.shape[0]
        x2, y2 = min(vx+win_w, sw), min(vy+win_h, sh)
        view = scaled[vy:y2, vx:x2]
        vis = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR) if len(view.shape)==2 else view.copy()

        # Overlay existing ROIs + INDEX
        overlay = vis.copy()
        for i, (rx, ry, rw, rh) in enumerate(rects, 1):
            X1 = int(rx*s) - vx; Y1 = int(ry*s) - vy
            X2 = int((rx+rw)*s) - vx; Y2 = int((ry+rh)*s) - vy
            if X1 >= vis.shape[1] or Y1 >= vis.shape[0] or X2 < 0 or Y2 < 0: 
                continue
            cv2.rectangle(overlay, (X1, Y1), (X2, Y2), (0,255,255), -1)
            cv2.rectangle(overlay, (X1, Y1), (X2, Y2), (0,180,255), 2)
            cv2.putText(overlay, f"#{i}", (X1+6, Y1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"#{i}", (X1+6, Y1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # Current drawing rectangle
        if drawing:
            X1 = int(x0*s) - vx; Y1 = int(y0*s) - vy
            X2 = int(cur[0]*s) - vx; Y2 = int(cur[1]*s) - vy
            cv2.rectangle(vis, (X1, Y1), (X2, Y2), (0,255,0), 2)

        # Help & infos
        info1 = unicodedata.normalize("NFKD", str(prompt)).encode("ascii", "ignore").decode("ascii")
        info2 = "Scroll=zoom | Right/Middle-drag=pan | Left-drag=ROI | u/z=undo c=clear h=help d=delete | Enter=OK Esc=Cancel"
        cv2.putText(vis, info1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        if show_help:
            cv2.putText(vis, info2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 2, cv2.LINE_AA)
        cv2.putText(vis, f"#ROIs: {len(rects)}", (10, vis.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        return vis

    def on_mouse(event, x, y, flags, param):
        nonlocal s, scaled, vx, vy, drawing, x0, y0, cur
        nonlocal panning_r, panning_m, pan_start, vx0, vy0

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = screen_to_img(x, y)
            x0, y0 = ix, iy
            cur = (ix, iy)
            drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                ix, iy = screen_to_img(x, y); cur = (ix, iy)
            elif panning_r or panning_m:
                dx = x - pan_start[0]; dy = y - pan_start[1]
                vx = vx0 - dx; vy = vy0 - dy
                clamp_view()
            else:
                ix, iy = screen_to_img(x, y); cur = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                ix, iy = screen_to_img(x, y)
                rx, ry = min(x0, ix), min(y0, iy)
                rw, rh = abs(ix - x0), abs(iy - y0)
                if rw >= 3 and rh >= 3:
                    rects.append((rx, ry, rw, rh))
                drawing = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            panning_r = True; pan_start = (x, y); vx0, vy0 = vx, vy
        elif event == cv2.EVENT_RBUTTONUP:
            panning_r = False

        elif event == cv2.EVENT_MBUTTONDOWN:
            panning_m = True; pan_start = (x, y); vx0, vy0 = vx, vy
        elif event == cv2.EVENT_MBUTTONUP:
            panning_m = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)  # multiple de 120
            except Exception:
                delta = 120 if flags > 0 else -120
            ix = (x + vx) / s; iy = (y + vy) / s
            zoom = 1.25 ** (delta / 120.0)
            s_new = max(s_min, min(s_max, s * zoom))
            if abs(s_new - s) < 1e-6: return
            s = s_new; scaled = make_scaled()
            vx = int(ix * s - x); vy = int(iy * s - y)
            clamp_view()

    cv2.setMouseCallback(win, on_mouse)

    # Key loop
    while True:
        cv2.imshow(win, redraw())
        k = cv2.waitKey(15) & 0xFF
        if k in (13, 10):  # Enter
            vis = redraw()
            cv2.putText(vis, f"Saved {len(rects)} ROI(s)", (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,255,80), 2, cv2.LINE_AA)
            cv2.imshow(win, vis); cv2.waitKey(250)
            break
        elif k == 27:      # Esc
            rects = []; break
        elif k in (ord('u'), ord('U'), ord('z'), ord('Z')):
            if rects: rects.pop()
        elif k in (ord('c'), ord('C')):
            rects.clear()
        elif k in (ord('h'), ord('H')):
            show_help = not show_help
        elif k in (ord('d'), ord('D')):
            ix, iy = cur
            j = find_rect_at(ix, iy)
            if j >= 0: rects.pop(j)

    cv2.destroyWindow(win)

    # Final clamping
    out = []
    for (x, y, w, h) in rects:
        x = clamp(int(x), 0, W-1); y = clamp(int(y), 0, H-1)
        w = clamp(int(w), 1, W - x); h = clamp(int(h), 1, H - y)
        out.append((x, y, w, h))
    return out

def select_rois_zoomable_shapes(img8, prompt="ROI selection", max_size=None, display_images=None, mask_overlay=None,
                               fat_exclusion_mask=None, exclusion_masks=None):
    """
    Retourne une liste de masques bool (H,W) en coords originales.

    UI améliorée:
      - image affichée avec ratio H/W conservé
      - panneau latéral lisible
      - g: changer image de référence
      - m: afficher/masquer overlay FINAL_warped_segments_only
      - , / .: diminuer/augmenter opacité overlay
      - f: plein écran
      - r / p / b / w / e: rectangle / polygon / brush / wand / eraser
      - [ / ]: tool radius; - / +: wand colour tolerance
      - x: optional class subtraction; margin slider in native cube pixels
      - Esc / Cancel all: discard the complete ROI selection
    """
    import cv2, numpy as np
    import unicodedata
    from shared.roi_drawing_tools import wand_lab_image, wand_reference, wand_region, paint_stroke, erase_roi_masks, exclusion_zone

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    H, W = img8.shape[:2]
    exclusion_enabled = False
    exclusion_margin = 3
    sources = dict(exclusion_masks or {})
    if fat_exclusion_mask is not None:
        sources.setdefault("fat", fat_exclusion_mask)
    sources = {str(name): np.asarray(mask, dtype=bool).copy() for name, mask in sources.items()}
    if any(mask.shape != (H, W) for mask in sources.values()):
        raise ValueError("Exclusion annotations must match the current cube dimensions.")
    sources = {name: mask for name, mask in sources.items() if mask.any()}
    source_names = list(sources)
    source_index = source_names.index("fat") if "fat" in source_names else 0
    excluded_zone = exclusion_zone(sources[source_names[source_index]], exclusion_margin) if source_names else None
    margin_dragging = False
    opacity_dragging = False
    dropdown = None

    # ---------- Images de référence ----------
    if display_images is None:
        display_images = {"Reference": img8}
    else:
        clean_display_images = {}
        for name, im in display_images.items():
            if im is None:
                continue
            im = _normalize_to_uint8_bgr(im)
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            clean_display_images[str(name)] = im
        display_images = clean_display_images or {"Reference": img8}

    bg_names = list(display_images.keys())
    bg_index = 0

    # ---------- Overlay masque / final warp ----------
    mask_bgr = None
    mask_bool = None
    mask_source = None
    if mask_overlay is not None:
        try:
            mask_bgr, mask_bool, mask_source = mask_overlay
            mask_bgr = _normalize_to_uint8_bgr(mask_bgr)
            if mask_bgr.shape[:2] != (H, W):
                mask_bgr = cv2.resize(mask_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_bool = np.asarray(mask_bool).astype(bool)
            if mask_bool.shape[:2] != (H, W):
                mask_bool = cv2.resize(mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        except Exception as e:
            print(f"[ROI][MASK][WARN] Overlay invalide: {e}")
            mask_bgr, mask_bool, mask_source = None, None, None

    show_mask = False
    mask_alpha = 0.30
    roi_alpha = 0.30

    # ---------- Fenêtre ----------
    if max_size is None:
        try:
            import tkinter as tk
            r = tk.Tk(); r.withdraw()
            sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
            init_w, init_h = int(sw * 0.92), int(sh * 0.88)
        except Exception:
            init_w, init_h = 1280, 850
    else:
        # On respecte max_size comme taille initiale minimale, mais on force un panneau lisible.
        init_w = max(int(max_size[0]), 1180)
        init_h = max(int(max_size[1]), 780)

    PANEL_W = 310
    STATUS_H = 34
    TOP_H = 38
    MARGIN = 10

    win = "ROI Selection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, init_w, init_h)

    fullscreen = False
    show_help = True

    # ---------- État viewport ----------
    # scale en pixels écran par pixel image originale
    image_area_w0 = max(200, init_w - PANEL_W - 2 * MARGIN)
    image_area_h0 = max(200, init_h - TOP_H - STATUS_H - 2 * MARGIN)
    s_fit = min(image_area_w0 / W, image_area_h0 / H)
    s = min(1.0, s_fit)
    s_min = max(0.02, s_fit * 0.15)
    s_max = 30.0
    vx = 0.0  # offset dans l'image scaled
    vy = 0.0

    cached_key = None
    cached_scaled = None

    # ---------- ROIs ----------
    rois: list[np.ndarray] = []
    mode = "r"
    drawing_rect = False
    rect_start = (0, 0)
    cur = (0, 0)
    poly_pts: list[tuple[int, int]] = []
    brushing = False
    brush_r = 12
    brush_mask = np.zeros((H, W), np.uint8)
    wand_r = 48
    wand_tolerance = 12
    wand_anchor = None
    editing_roi = -1
    lab_cache = {}
    previous_point = (0, 0)
    erasing = False
    erase_before = None
    erase_changed = False
    eraser_mask = np.zeros((H, W), np.uint8)
    radius_dragging = False
    history = []  # Masks are immutable once committed; erasing uses copy-on-write.
    cancelled = False
    confirmed = False
    control_boxes = {}

    panning_r = panning_m = False
    pan_start = (0, 0)
    vx0 = vy0 = 0.0

    COLORS = [
        (0, 80, 255),
        (60, 210, 60),
        (255, 150, 0),
        (0, 220, 255),
        (220, 80, 255),
        (255, 255, 0),
    ]

    def get_window_size():
        try:
            _, _, ww, hh = cv2.getWindowImageRect(win)
            if ww > 100 and hh > 100:
                return int(ww), int(hh)
        except Exception:
            pass
        return init_w, init_h

    def layout():
        ww, hh = get_window_size()
        panel_x = max(ww - PANEL_W, 420)
        image_x = MARGIN
        image_y = TOP_H + MARGIN
        image_w = max(80, panel_x - 2 * MARGIN)
        image_h = max(80, hh - TOP_H - STATUS_H - 2 * MARGIN)
        return ww, hh, panel_x, image_x, image_y, image_w, image_h

    def composed_image():
        base = display_images[bg_names[bg_index]].copy()
        if show_mask and mask_bgr is not None and mask_bool is not None:
            mb = mask_bool
            base[mb] = ((1.0 - mask_alpha) * base[mb].astype(np.float32) + mask_alpha * mask_bgr[mb].astype(np.float32)).astype(np.uint8)
        return base

    def get_scaled():
        nonlocal cached_key, cached_scaled
        key = (bg_index, show_mask, round(mask_alpha, 3), round(s, 5), id(mask_bgr))
        if cached_key == key and cached_scaled is not None:
            return cached_scaled
        img = composed_image()
        new_w = max(1, int(round(W * s)))
        new_h = max(1, int(round(H * s)))
        cached_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if s <= 1 else cv2.INTER_LINEAR)
        cached_key = key
        return cached_scaled

    def clamp_view():
        nonlocal vx, vy
        _, _, _, _, _, area_w, area_h = layout()
        scaled = get_scaled()
        sw, sh = scaled.shape[1], scaled.shape[0]
        vx = clamp(vx, 0, max(0, sw - area_w))
        vy = clamp(vy, 0, max(0, sh - area_h))

    def image_origin_and_view():
        ww, hh, panel_x, image_x, image_y, area_w, area_h = layout()
        scaled = get_scaled()
        sw, sh = scaled.shape[1], scaled.shape[0]
        view_w = min(area_w, sw)
        view_h = min(area_h, sh)
        ox = image_x + max(0, (area_w - view_w) // 2)
        oy = image_y + max(0, (area_h - view_h) // 2)
        x1 = int(vx)
        y1 = int(vy)
        x2 = min(x1 + view_w, sw)
        y2 = min(y1 + view_h, sh)
        return ox, oy, area_w, area_h, x1, y1, x2, y2

    def screen_to_img(mx, my):
        ox, oy, _, _, _, _, _, _ = image_origin_and_view()
        ix = int(round((mx - ox + vx) / s))
        iy = int(round((my - oy + vy) / s))
        return clamp(ix, 0, W - 1), clamp(iy, 0, H - 1)

    def find_roi_at(ix, iy):
        for k in range(len(rois) - 1, -1, -1):
            if rois[k][iy, ix]:
                return k
        return -1

    def remember(before=None):
        history.append(tuple(rois) if before is None else before)
        if len(history) > 40:
            history.pop(0)

    def finish_stroke():
        nonlocal brushing, erasing, erase_before, erase_changed
        if brushing:
            add_brush_roi()
            brushing = False
        if erasing:
            if erase_changed:
                remember(erase_before)
            erasing = False
            erase_before = None
            erase_changed = False
            eraser_mask[:] = 0

    def set_mode(new_mode):
        nonlocal mode, drawing_rect
        finish_stroke()
        mode = new_mode
        drawing_rect = False
        poly_pts.clear()

    def undo_action():
        finish_stroke()
        if poly_pts:
            poly_pts.pop()
        elif history:
            rois[:] = history.pop()

    def clear_action():
        finish_stroke()
        if rois:
            remember()
            rois.clear()
        poly_pts.clear()
        brush_mask[:] = 0

    def radius_from_slider(mouse_x):
        nonlocal brush_r, wand_r
        bx, _, bw, _ = control_boxes["radius_slider"]
        fraction = clamp((mouse_x-bx-6)/max(1, bw-12), 0.0, 1.0)
        radius = int(round(1+199*fraction))
        if mode == "w":
            wand_r = radius
        else:
            brush_r = radius

    def margin_from_slider(mouse_x):
        nonlocal exclusion_margin, excluded_zone
        bx, _, bw, _ = control_boxes["exclusion_margin"]
        fraction = clamp((mouse_x-bx-6)/max(1, bw-12), 0., 1.)
        value = int(round(20*fraction))
        if value != exclusion_margin:
            exclusion_margin = value
            excluded_zone = exclusion_zone(sources[source_names[source_index]], exclusion_margin) if source_names else None

    def effective_mask(mask):
        return mask & ~excluded_zone if exclusion_enabled else mask

    def stamp_wand(point, previous=None):
        name = bg_names[bg_index]
        if name not in lab_cache:
            lab_cache[name] = wand_lab_image(display_images[name])
        lab = lab_cache[name]
        points = [point]
        if previous is not None:
            distance = np.hypot(point[0]-previous[0], point[1]-previous[1])
            steps = max(1, int(np.ceil(distance/max(1, wand_r//3))))
            points = [(int(round(previous[0]+(point[0]-previous[0])*i/steps)),
                       int(round(previous[1]+(point[1]-previous[1])*i/steps))) for i in range(1, steps+1)]
        for position in points:
            x0, y0, local = wand_region(lab, position, wand_anchor, wand_r, wand_tolerance, adaptive=True)
            h, w = local.shape
            brush_mask[y0:y0+h, x0:x0+w] |= local.astype(np.uint8)

    def stamp_eraser(point, previous):
        nonlocal erase_changed
        eraser_mask[:] = 0
        paint_stroke(eraser_mask, previous, point, brush_r)
        updated, changed = erase_roi_masks(rois, eraser_mask)
        if changed:
            rois[:] = updated
            erase_changed = True

    def add_polygon_roi(points):
        if len(points) < 3:
            return
        m = np.zeros((H, W), np.uint8)
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
        remember()
        rois.append(m.astype(bool))

    def add_rect_roi(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        rx, ry = min(x0, x1), min(y0, y1)
        rw, rh = abs(x1 - x0), abs(y1 - y0)
        if rw < 3 or rh < 3:
            return
        m = np.zeros((H, W), np.uint8)
        m[ry:ry + rh, rx:rx + rw] = 1
        remember()
        rois.append(m.astype(bool))

    def add_brush_roi():
        nonlocal brush_mask, editing_roi
        if brush_mask.sum() == 0:
            editing_roi = -1
            return
        remember()
        if 0 <= editing_roi < len(rois):
            rois[editing_roi] = rois[editing_roi] | (brush_mask > 0)
        else:
            rois.append((brush_mask > 0))
        editing_roi = -1
        brush_mask[:] = 0

    def ascii_ui(text):
        """Convert dynamic UI labels to characters supported by OpenCV."""
        normalized = unicodedata.normalize("NFKD", str(text))
        return normalized.encode("ascii", "ignore").decode("ascii")

    def draw_text(img, text, xy, scale=0.48, color=(235, 235, 235), thickness=1):
        cv2.putText(img, ascii_ui(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_button(img, x, y, w, h, text, active=False, warn=False):

        # Palette cohérente avec ton GUI
        BTN_BG        = (58, 58, 58)
        BTN_HOVER     = (70, 70, 70)
        BTN_ACTIVE    = (245, 165, 66)  # ⚠️ OpenCV = BGR → #42A5F5
        BTN_BORDER    = (90, 90, 90)
        BTN_TEXT      = (230, 230, 230)
        BTN_WARN      = (60, 60, 200)

        if warn:
            fill = BTN_BG
            border = BTN_WARN
        elif active:
            fill = BTN_ACTIVE
            border = BTN_ACTIVE
        else:
            fill = BTN_BG
            border = BTN_BORDER

        cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)

        label = ascii_ui(text)
        while label and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0] > w-18:
            label = label[:-1]
        if label != ascii_ui(text):
            label = label[:-3] + "..."
        draw_text(img, label, (x + 10, y + int(h * 0.65)), 0.48, BTN_TEXT, 1)

    def redraw():
        nonlocal control_boxes
        control_boxes = {}
        clamp_view()
        ww, hh, panel_x, image_x, image_y, area_w, area_h = layout()
        canvas = np.zeros((hh, ww, 3), dtype=np.uint8)

        BG_MAIN   = (30, 30, 30)
        BG_HEADER = (43, 43, 43)
        BG_PANEL  = (43, 43, 43)
        BG_STATUS = (43, 43, 43)
        BORDER    = (70, 70, 70)

        canvas[:] = BG_MAIN

        cv2.rectangle(canvas, (0, 0), (ww, TOP_H), BG_HEADER, -1)
        cv2.rectangle(canvas, (0, hh - STATUS_H), (ww, hh), BG_STATUS, -1)
        cv2.rectangle(canvas, (panel_x, TOP_H), (ww, hh - STATUS_H), BG_PANEL, -1)
        cv2.line(canvas, (panel_x, TOP_H), (panel_x, hh - STATUS_H), BORDER, 1)
        bg_label = bg_names[bg_index]
        background_text = ascii_ui(f"Background: {bg_label}")
        background_width = cv2.getTextSize(background_text, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]
        title_text = ascii_ui(prompt)
        title_width = max(50, panel_x-background_width-45)
        while title_text and cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0] > title_width:
            title_text = title_text[:-1]
        if title_text != ascii_ui(prompt):
            title_text = title_text[:-3] + "..."
        draw_text(canvas, title_text, (14, 25), .6, (255, 255, 255), 1)
        draw_text(canvas, background_text, (panel_x-background_width-12, 25), .5, (220, 230, 240), 1)

        # Image viewport avec ratio conservé
        scaled = get_scaled()
        ox, oy, _, _, x1, y1, x2, y2 = image_origin_and_view()
        cv2.rectangle(canvas, (image_x - 1, image_y - 1), (image_x + area_w + 1, image_y + area_h + 1), (15, 17, 18), -1)
        view = scaled[y1:y2, x1:x2]
        vh, vw = view.shape[:2]
        canvas[oy:oy + vh, ox:ox + vw] = view

        # Draw actual masks, including eraser holes; bound work to the viewport.
        def draw_roi_mask(mask, color, alpha=None):
            transform = np.array([[s, 0, -vx+(s-1)*.5], [0, s, -vy+(s-1)*.5]], dtype=np.float32)
            visible = cv2.warpAffine(mask.astype(np.uint8), transform, (vw, vh),
                                     flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            alpha = roi_alpha if alpha is None else alpha
            pixels = visible > 0
            target = canvas[oy:oy+vh, ox:ox+vw]
            target[pixels] = ((1-alpha)*target[pixels] + alpha*np.asarray(color)).astype(np.uint8)
            contours, _ = cv2.findContours(visible, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(target, contours, -1, color, 1)
            return visible

        labels = []
        if exclusion_enabled:
            draw_roi_mask(excluded_zone, (180, 60, 180))
        for i, mask in enumerate(rois, 1):
            color = COLORS[(i-1) % len(COLORS)]
            visible = draw_roi_mask(effective_mask(mask), color)
            points = cv2.findNonZero(visible)
            if points is not None:
                # Keep label coordinates local to the same viewport as the mask.
                # Pick an actual contour pixel nearest the bounding box top-left.
                points = points.reshape(-1, 2)
                left, top = points.min(axis=0)
                anchor = points[np.argmin(np.sum((points-[left, top])**2, axis=1))]
                label_w = max(28, cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]+10)
                lx = int(clamp(int(anchor[0]), 0, max(0, vw-label_w-1)))
                ly = int(clamp(int(anchor[1])-24, 0, max(0, vh-21)))
                labels.append((i, color, label_w, lx, ly, tuple(map(int, anchor))))
        if brushing:
            draw_roi_mask(effective_mask(brush_mask), (0, 255, 120))

        target = canvas[oy:oy+vh, ox:ox+vw]
        for i, color, label_w, lx, ly, anchor in labels:
            cv2.line(target, (lx+label_w//2, ly+20), anchor, color, 1)
            cv2.rectangle(target, (lx, ly), (lx+label_w, ly+20), (15, 15, 15), -1)
            cv2.rectangle(target, (lx, ly), (lx+label_w, ly+20), color, 1)
            draw_text(target, str(i), (lx+5, ly+15), .5, (255, 255, 255), 1)

        # Live previews
        if mode == "r" and drawing_rect:
            x0, y0 = rect_start
            x1i, y1i = cur
            X0, Y0 = int(x0 * s - vx + ox), int(y0 * s - vy + oy)
            X1, Y1 = int(x1i * s - vx + ox), int(y1i * s - vy + oy)
            cv2.rectangle(canvas, (X0, Y0), (X1, Y1), (0, 255, 120), 2)

        if mode == "p" and len(poly_pts) >= 1:
            pts = np.array(poly_pts + [cur], dtype=np.int32)
            pts_s = (pts.astype(np.float32) * s).astype(np.int32)
            pts_s[:, 0] = pts_s[:, 0] - int(vx) + ox
            pts_s[:, 1] = pts_s[:, 1] - int(vy) + oy
            cv2.polylines(canvas, [pts_s.reshape(-1, 1, 2)], False, (0, 255, 120), 2)

        if mode in ("b", "w", "e"):
            X, Y = int(cur[0] * s - vx + ox), int(cur[1] * s - vy + oy)
            rr = max(1, int((wand_r if mode == "w" else brush_r) * s))
            cv2.circle(canvas[oy:oy+vh, ox:ox+vw], (X-ox, Y-oy), rr,
                       (60, 60, 255) if mode == "e" else (0, 255, 120), 2)

        # Compact panel: all controls remain reachable at the initial window size.
        def button(name, x, y, w, h, text, active=False, warn=False):
            draw_button(canvas, x, y, w, h, text, active, warn)
            control_boxes[name] = (x, y, w, h)

        px = panel_x + 16
        width = PANEL_W-32

        def section(title, y):
            draw_text(canvas, title, (px, y), .45, (170, 190, 210))
            cv2.line(canvas, (px, y+7), (px+width, y+7), BORDER, 1)

        def slider(name, y, value, text):
            draw_text(canvas, text, (px, y), .43)
            y += 8
            control_boxes[name] = (px, y, width, 22)
            cv2.rectangle(canvas, (px, y), (px+width, y+22), (55, 55, 55), -1)
            cv2.line(canvas, (px+6, y+11), (px+width-6, y+11), (105, 105, 105), 3)
            knob = px+6+int(round(value*(width-12)))
            cv2.line(canvas, (px+6, y+11), (knob, y+11), (245, 165, 66), 3)
            cv2.circle(canvas, (knob, y+11), 6, (245, 165, 66), -1)

        section("AFFICHAGE", 62)
        button("reference_menu", px, 78, width, 28, f"{bg_names[bg_index]}   v", dropdown == "reference")
        button("overlay", px, 114, width, 25,
               f"Warp: {'ON' if show_mask else 'OFF'} (m)" if mask_bgr is not None else "Warp: indisponible",
               show_mask)
        slider("opacity", 160, roi_alpha, f"Opacite des masques : {roi_alpha:.0%}")

        section("DESSIN", 220)
        for key, offset, text in (("r", 0, "r Rect."), ("p", 96, "p Poly."), ("b", 192, "b Pinceau")):
            button("mode_"+key, px+offset, 236, 86, 27, text, mode == key)
        button("mode_w", px, 270, 134, 27, "w Baguette", mode == "w")
        button("mode_e", px+144, 270, 134, 27, "e Gomme", mode == "e")
        radius = wand_r if mode == "w" else brush_r
        draw_text(canvas, f"Rayon : {radius} px  [ / ]", (px, 320), .43)
        button("radius_minus", px, 330, 40, 25, "-")
        slider_x, slider_w = px+48, width-96
        cv2.rectangle(canvas, (slider_x, 330), (slider_x+slider_w, 355), (55, 55, 55), -1)
        cv2.line(canvas, (slider_x+6, 342), (slider_x+slider_w-6, 342), (105, 105, 105), 3)
        knob = slider_x+6+int(round((radius-1)/199*(slider_w-12)))
        cv2.circle(canvas, (knob, 342), 7, (245, 165, 66), -1)
        control_boxes["radius_slider"] = (slider_x, 330, slider_w, 25)
        button("radius_plus", px+width-40, 330, 40, 25, "+")
        draw_text(canvas, "Tolerance baguette  - / +", (px, 377), .43)
        button("tolerance_minus", px, 385, 40, 25, "-")
        button("tolerance_value", px+48, 385, width-96, 25, f"{wand_tolerance} Lab")
        button("tolerance_plus", px+width-40, 385, 40, 25, "+")

        section("EXCLUSION PAR CLASSE", 440)
        source_label = source_names[source_index] if source_names else "Aucune classe annotee"
        button("source_menu", px, 456, width, 28, f"Soustraire : {source_label}   v", dropdown == "source")
        button("exclusion_toggle", px, 492, width, 26,
               f"Exclusion : {'ON' if exclusion_enabled else 'OFF'} (x)", exclusion_enabled)
        slider("exclusion_margin", 541, exclusion_margin/20, f"Marge externe : {exclusion_margin} px (0-20)")
        draw_text(canvas, "Violet : classe exclue + marge", (px, 590), .39)
        button("undo", px, 606, 134, 26, "u/z Annuler")
        button("clear", px+144, 606, 134, 26, "c Tout effacer", warn=True)
        section("VALIDATION", hh-STATUS_H-100)
        button("confirm", px, hh-STATUS_H-86, width, 34, "Entree : valider", True)
        button("cancel_all", px, hh-STATUS_H-44, width, 34, "Echap : abandonner", warn=True)

        # Popup is drawn last, above controls; clicks are intercepted first.
        if dropdown is not None:
            names = bg_names if dropdown == "reference" else source_names
            box = control_boxes["reference_menu" if dropdown == "reference" else "source_menu"]
            row_h = min(28, max(16, (hh-TOP_H-STATUS_H-20)//max(1, len(names))))
            menu_y = min(box[1]+box[3], hh-STATUS_H-8-row_h*len(names))
            for index, name in enumerate(names):
                text = name
                if dropdown == "source":
                    text += f" ({np.count_nonzero(sources[name]):,} px)"
                button(f"choice_{index}", px, menu_y+index*row_h, width, row_h, text,
                       index == (bg_index if dropdown == "reference" else source_index))

        # Status bar
        draw_text(canvas, f"Mode: {mode.upper()} | ROIs: {len(rois)} | Zoom: {s:.2f}x",
                  (14, hh - 11), 0.46, (230, 235, 240), 1)
        help_text = "Clic dans ROI: agrandir | molette: zoom | clic droit: deplacer" if show_help else "g: reference | x: exclusion | h: aide | f: plein ecran"
        draw_text(canvas, help_text,
                  (max(300, panel_x - 555), hh - 11), 0.42, (210, 215, 220), 1)

        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal s, vx, vy, cur, drawing_rect, rect_start
        nonlocal panning_r, panning_m, pan_start, vx0, vy0
        nonlocal brushing, cached_key, brush_r, cancelled, confirmed
        nonlocal wand_r, wand_tolerance, wand_anchor, previous_point
        nonlocal erasing, erase_before, erase_changed, bg_index, show_mask
        nonlocal radius_dragging
        nonlocal editing_roi
        nonlocal exclusion_enabled, margin_dragging, opacity_dragging, roi_alpha, mask_alpha
        nonlocal dropdown, source_index, excluded_zone

        if opacity_dragging:
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                bx, _, bw, _ = control_boxes["opacity"]
                roi_alpha = mask_alpha = clamp((x-bx-6)/max(1, bw-12), 0., 1.)
                cached_key = None
                if event == cv2.EVENT_LBUTTONUP:
                    opacity_dragging = False
                return

        if event == cv2.EVENT_LBUTTONDOWN and dropdown is not None:
            opened = dropdown
            dropdown = None
            for name, (bx, by, bw, bh) in control_boxes.items():
                if name.startswith("choice_") and bx <= x < bx+bw and by <= y < by+bh:
                    finish_stroke()
                    index = int(name.split("_")[1])
                    if opened == "reference":
                        bg_index = index
                        cached_key = None
                        clamp_view()
                    else:
                        source_index = index
                        excluded_zone = exclusion_zone(sources[source_names[source_index]], exclusion_margin)
                    break
            return

        if margin_dragging:
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                margin_from_slider(x)
                if event == cv2.EVENT_LBUTTONUP:
                    margin_dragging = False
                return

        if radius_dragging:
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                radius_from_slider(x)
                if event == cv2.EVENT_LBUTTONUP:
                    radius_dragging = False
                return

        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (bx, by, bw, bh) in control_boxes.items():
                if bx <= x <= bx+bw and by <= y <= by+bh:
                    finish_stroke()
                    if name == "reference_menu":
                        dropdown = "reference"
                    elif name == "source_menu" and source_names:
                        dropdown = "source"
                    elif name == "opacity":
                        opacity_dragging = True
                        bx, _, bw, _ = control_boxes["opacity"]
                        roi_alpha = mask_alpha = clamp((x-bx-6)/max(1, bw-12), 0., 1.)
                        cached_key = None
                    elif name.startswith("mode_"):
                        set_mode(name[-1])
                    elif name.startswith("background_"):
                        bg_index = int(name.split("_")[-1])
                        cached_key = None
                        clamp_view()
                    elif name == "overlay" and mask_bgr is not None:
                        show_mask = not show_mask
                        cached_key = None
                    elif name in ("radius_minus", "radius_plus"):
                        step = -2 if name.endswith("minus") else 2
                        if mode == "w":
                            wand_r = clamp(wand_r+step, 1, 200)
                        else:
                            brush_r = clamp(brush_r+step, 1, 200)
                    elif name == "radius_slider":
                        radius_dragging = True
                        radius_from_slider(x)
                    elif name == "exclusion_toggle" and source_names:
                        exclusion_enabled = not exclusion_enabled
                    elif name == "exclusion_margin":
                        margin_dragging = True
                        margin_from_slider(x)
                    elif name in ("tolerance_minus", "tolerance_plus"):
                        wand_tolerance = clamp(wand_tolerance+(-2 if name.endswith("minus") else 2), 1, 80)
                    elif name == "undo":
                        undo_action()
                    elif name == "clear":
                        clear_action()
                    elif name == "confirm":
                        confirmed = True
                    elif name == "cancel_all":
                        cancelled = True
                    return

        ox, oy, _, _, x1, y1, x2, y2 = image_origin_and_view()
        inside = ox <= x < ox+(x2-x1) and oy <= y < oy+(y2-y1)
        ix, iy = screen_to_img(x, y)
        if inside:
            cur = (ix, iy)

        if event == cv2.EVENT_LBUTTONDOWN and inside:
            if mode == "r":
                rect_start = (ix, iy)
                drawing_rect = True
            elif mode == "p":
                poly_pts.append((ix, iy))
            elif mode in ("b", "w"):
                brushing = True
                editing_roi = find_roi_at(ix, iy)
                previous_point = (ix, iy)
                brush_mask[:] = 0
                if mode == "b":
                    paint_stroke(brush_mask, previous_point, previous_point, brush_r)
                else:
                    name = bg_names[bg_index]
                    if name not in lab_cache:
                        lab_cache[name] = wand_lab_image(display_images[name])
                    wand_anchor = wand_reference(lab_cache[name], previous_point)
                    stamp_wand(previous_point)
            elif mode == "e":
                erasing = True
                erase_before = tuple(rois)
                erase_changed = False
                previous_point = (ix, iy)
                stamp_eraser(previous_point, previous_point)

        elif event == cv2.EVENT_MOUSEMOVE:
            if panning_r or panning_m:
                vx = vx0-(x-pan_start[0])
                vy = vy0-(y-pan_start[1])
                clamp_view()
            elif inside:
                if brushing:
                    if mode == "w":
                        stamp_wand((ix, iy), previous_point)
                    else:
                        paint_stroke(brush_mask, previous_point, (ix, iy), brush_r)
                    previous_point = (ix, iy)
                elif erasing:
                    stamp_eraser((ix, iy), previous_point)
                    previous_point = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP:
            if mode == "r" and drawing_rect:
                add_rect_roi(rect_start, (ix, iy))
                drawing_rect = False
            finish_stroke()

        elif event == cv2.EVENT_RBUTTONDOWN and inside:
            if mode == "p" and len(poly_pts) >= 3:
                add_polygon_roi(poly_pts)
                poly_pts.clear()
            else:
                panning_r = True
                pan_start = (x, y)
                vx0, vy0 = vx, vy
        elif event == cv2.EVENT_RBUTTONUP:
            panning_r = False
        elif event == cv2.EVENT_MBUTTONDOWN and inside:
            panning_m = True
            pan_start = (x, y)
            vx0, vy0 = vx, vy
        elif event == cv2.EVENT_MBUTTONUP:
            panning_m = False
        elif event == cv2.EVENT_MOUSEWHEEL and inside:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 120 if flags > 0 else -120
            focus_x = (x-ox+vx)/s
            focus_y = (y-oy+vy)/s
            s_new = clamp(s*1.25**(delta/120.0), s_min, s_max)
            if abs(s_new-s) < 1e-6:
                return
            s = s_new
            cached_key = None
            vx = focus_x*s-(x-ox)
            vy = focus_y*s-(y-oy)
            clamp_view()

    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = redraw()
        cv2.imshow(win, canvas)
        k = cv2.waitKey(15) & 0xFF

        if k in (13, 10):
            finish_stroke()
            break
        if k == 27:
            if dropdown is not None:
                dropdown = None
                continue
            cancelled = True
            break
        if cancelled:
            break
        if confirmed:
            break

        if k in (ord('h'), ord('H')):
            show_help = not show_help

        elif k in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, prop)

        elif k in (ord('g'), ord('G')):
            finish_stroke()
            bg_index = (bg_index + 1) % len(bg_names)
            cached_key = None
            clamp_view()

        elif k in (ord('m'), ord('M')):
            if mask_bgr is not None:
                show_mask = not show_mask
                cached_key = None

        elif k in (ord('x'), ord('X')) and excluded_zone is not None:
            finish_stroke()
            exclusion_enabled = not exclusion_enabled

        elif k == ord(','):
            mask_alpha = max(0.0, mask_alpha - 0.05)
            roi_alpha = mask_alpha
            cached_key = None

        elif k == ord('.'):
            mask_alpha = min(1.0, mask_alpha + 0.05)
            roi_alpha = mask_alpha
            cached_key = None

        elif k in (ord('r'), ord('R'), ord('p'), ord('P'), ord('b'), ord('B'), ord('w'), ord('W'), ord('e'), ord('E')):
            set_mode(chr(k).lower())

        elif k in (ord('u'), ord('U'), ord('z'), ord('Z')):
            undo_action()

        elif k in (ord('c'), ord('C')):
            clear_action()

        elif k in (ord('d'), ord('D')):
            ix, iy = cur
            j = find_roi_at(ix, iy)
            if j >= 0:
                remember()
                rois.pop(j)

        elif k == ord('['):
            finish_stroke()
            if mode == "w": wand_r = max(1, wand_r - 2)
            else: brush_r = max(1, brush_r - 2)
        elif k == ord(']'):
            finish_stroke()
            if mode == "w": wand_r = min(200, wand_r + 2)
            else: brush_r = min(200, brush_r + 2)
        elif k in (ord('-'), ord('_'), ord('+'), ord('=')):
            finish_stroke()
            wand_tolerance = clamp(wand_tolerance+(-2 if k in (ord('-'), ord('_')) else 2), 1, 80)

        elif k in (8, 127):
            if poly_pts:
                poly_pts.pop()

    cv2.destroyWindow(win)
    if cancelled:
        return None
    if exclusion_enabled:
        result = [effective_mask(mask) for mask in rois]
        removed = sum(int(mask.sum()-trimmed.sum()) for mask, trimmed in zip(rois, result))
        print(f"[ROI] Class subtraction {source_names[source_index]!r} ON | outward margin={exclusion_margin} px | removed={removed} selected pixels")
        return [mask for mask in result if mask.any()]
    return rois


def _rect_to_mask(h: int, w: int, r: tuple[int,int,int,int]) -> np.ndarray:
    """converts (x,y,w,h) to boolean mask (h,w).
    args:
        h, w: dimensions of the mask  
        r: (x,y,w,h) rectangle
    returns:
        boolean mask (h,w) where True = inside rectangle"""
    x, y, ww, hh = r
    x2, y2 = min(x+ww, w), min(y+hh, h)
    m = np.zeros((h, w), np.uint8)
    m[y:y2, x:x2] = 1
    return m.astype(bool)

# ----------------------- Fonction principale -----------------------
def roi_select(
    cube_path: str | Path,
    out_dir: str | Path,
    class_names: list[str],
    *,
    display_band: int | None = None,
    keep_frac: float = 0.95,
    max_pix_per_roi: int | None = None,
    sat_thresh: int | float | str | None = "auto",
    dataset_name: str = "roi_dataset.npz",
    library_name: str | None = None,
    force_new_dataset: bool = False,
    update_only: bool = False,
    normalization_mode: str = "none",
    rgb_image_path: str | None = None,
    wavelengths: list[float] | np.ndarray | None = None,
    roi_background_wavelengths: tuple[int, ...] = (490, 560, 700),
    use_smoothed: bool = False,      # si True, utilise X_smooth si dispo
    smooth_window: int = 7,          # si X_smooth absent et use_smoothed True, lisse X
    smooth_poly: int = 2,
    save_spatial_meta: bool = True,
) -> dict:
    """
    ROI selection UI → builds/updates dataset + normalized library.
    - update_only=True : ne touche pas au cube. Lit seulement le dataset et régénère la librairie.
    - Compatibilité conservée : X et y restent présents et inchangés.
    """

    # -------- utils de normalisation (local, indépendants du cube) --------
    def _smooth_rows_savgol(X: np.ndarray, window: int, poly: int) -> np.ndarray:
        X = np.asarray(X, float)
        w = int(window)
        if w <= 1:
            return X
        if w % 2 == 0:
            w += 1
        B = X.shape[1]
        if B < 3:
            return X
        w = min(w, B if (B % 2 == 1) else B - 1)
        w = max(w, 3)
        p = int(poly)
        p = min(p, w - 1)
        return savgol_filter(X, window_length=w, polyorder=p, axis=1, mode="interp")

    def _snv_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / (sd + eps)

    def _snv_mp_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        # mean-preserving SNV: conserve la moyenne relative en ne divisant que par l'écart-type
        sd = X.std(axis=1, keepdims=True)
        return X / (sd + eps)

    def _snv_offset_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        mean_scalar = mu.squeeze(axis=1)
        scaled_mean = (mean_scalar - mean_scalar.min()) / (mean_scalar.max() - mean_scalar.min() + eps)
        offset = scaled_mean[:, None] * 10.0
        return ((X - mu) / (sd + eps)) + offset

    def _minmax_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        X = np.asarray(X, float)
        mn = X.min(axis=1, keepdims=True)
        mx = X.max(axis=1, keepdims=True)
        return (X - mn) / (mx - mn + eps)

    def _mean_center_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        mu = X.mean(axis=1, keepdims=True)
        return X - mu

    # -------- validation dossier sortie --------
    out_dir = Path(out_dir)
    if not str(out_dir).strip():
        raise ValueError("[ROI] out_dir is empty — cannot save dataset or library.")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(str(dataset_name)).name
    if not dataset_name.lower().endswith(".npz"):
        dataset_name += ".npz"
    dataset_path = out_dir / dataset_name

    if library_name and str(library_name).strip():
        resolved_library_name = Path(str(library_name).strip()).name
    else:
        resolved_library_name = f"{Path(dataset_name).stem}_library.json"
    library_path = out_dir / resolved_library_name

    # ===========================
    #   MODE UPDATE-ONLY
    # ===========================
    if update_only:
        normalization_mode = (normalization_mode or "none").lower().strip().replace("`", "")
  
        if not dataset_path.exists():
            raise SystemExit("[ROI] No ROI dataset found to update.")

        # Persist stable IDs in legacy datasets as soon as they are touched by
        # the current ROI workflow. The migration creates a one-time backup.
        if migrate_dataset_cube_ids(dataset_path, create_backup=True):
            print(f"[ROI][MIGRATION] Stable cube_ids added to {dataset_path.name}")

        data = np.load(str(dataset_path), allow_pickle=True)

        # >>> CHOIX DE LA SOURCE <<<
        if use_smoothed:
            if "X_smooth" in data:
                X = np.asarray(data["X_smooth"], np.float32)
                print("[ROI] Using X_smooth from dataset.")
            else:
                Xraw = np.asarray(data["X"], np.float32)
                X = _smooth_rows_savgol(Xraw, smooth_window, smooth_poly).astype(np.float32)
                
        else:
            X = np.asarray(data["X"], np.float32)

        y = np.asarray(data["y"], object)

        # normalisation globale selon le mode (par spectre / par ligne)
        if normalization_mode in ("snv", "snv_no_ratio", "snv_three_ratios"):
            Xn = _snv_rows(X)
        elif normalization_mode == "log":
            from shared.spectrum_normalization import normalize_spectra
            Xn = normalize_spectra(X, 'log')
        elif normalization_mode == "snv_mp":
            Xn = _snv_mp_rows(X)
        elif normalization_mode == "snv_offset":
            Xn = _snv_offset_rows(X)
        elif normalization_mode == "minmax":
            Xn = _minmax_rows(X)
        elif normalization_mode == "mean_center":
            Xn = _mean_center_rows(X)
        elif normalization_mode == "none":
            Xn = X
        else:
            raise ValueError(f"[ROI] Unknown normalization mode: {normalization_mode!r}")

        # recalcul médianes/std par classe (trim central keep_frac)
        library: dict[str, dict] = {}
        for cls in np.unique(y):
            mask = (y == cls)
            if not np.any(mask):
                continue

            Xi = Xn[mask]
            keep_mask = _central_trim_mask(Xi, keep_frac=keep_frac)
            Xi_kept = Xi[keep_mask]
            if Xi_kept.shape[0] == 0:
                Xi_kept = Xi  # fallback

            med = np.median(Xi_kept, axis=0).astype(np.float32)
            std = (
                Xi_kept.std(axis=0, ddof=1).astype(np.float32)
                if Xi_kept.shape[0] > 1 else np.zeros((Xi_kept.shape[1],), np.float32)
            )

            library[str(cls)] = {
                "median": med.tolist(),
                "std": std.tolist(),
                "n_total": int(mask.sum()),
                "n_kept": int(keep_mask.sum()),
                "normalization": normalization_mode,
            }

        library["_meta"] = {
            "normalization": normalization_mode,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        _write_library_json(library_path, library)

        print(f"[ROI] Library updated and saved to {library_path}")
        return library

    # ===========================
    #   MODE SÉLECTION ROI
    # ===========================
    # gestion optionnelle du reset dataset
    if force_new_dataset and dataset_path.exists():
        dataset_path.unlink(missing_ok=True)

    cube_path = Path(cube_path)
    if not cube_path.exists():
        raise SystemExit(f"[ROI] Cube not found: {cube_path}")

    from shared.envi import load_spectral_cube, resolve_envi_header
    cube_path = resolve_envi_header(cube_path)
    cube, imported_wavelengths = load_spectral_cube(cube_path)
    if wavelengths is None:
        wavelengths = imported_wavelengths
    H, W, N = cube.shape
    # Read statistics in blocks; do not copy a full 1024 x 1024 spectral cube.
    cube_min, cube_max = np.inf, -np.inf
    for row in range(0, H, 32):
        block = np.asarray(cube[row:row+32])
        finite = block[np.isfinite(block)]
        if finite.size:
            cube_min = min(cube_min, float(finite.min()))
            cube_max = max(cube_max, float(finite.max()))
    if np.isfinite(cube_min):
        print(f"[ROI] Scientific cube range: {cube_min:.3f}-{cube_max:.3f} (no 100% cap)")

    # L'axe spectral est sauvegardé dans chaque dataset. Il est d'abord lu
    # depuis --wavelengths, sinon inféré depuis les fichiers corr_XXXnm.
    current_wavelengths = _resolve_cube_wavelengths(
        cube_path=cube_path,
        n_bands=N,
        provided=wavelengths,
    )
    wavelengths = current_wavelengths
    print(f"[ROI] Wavelengths nm: {current_wavelengths.tolist()}")

    # Résolution robuste du seuil de saturation.
    # Avant, sat_thresh=255 était imposé par le main_GUI. Avec un cube 16-bit,
    # presque tous les pixels valides ont au moins une bande >255, donc ils
    # étaient tous rejetés comme "saturés" -> aucune ROI valide -> return code 1.
    def _resolve_sat_thresh(value, cube_arr: np.ndarray):
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("", "none", "off", "false", "no", "-1", "0"):
                return None
            if v != "auto":
                try:
                    f = float(v)
                    return None if f <= 0 else f
                except Exception:
                    print(f"[ROI][WARN] sat_thresh invalide {value!r}; fallback auto.")
        else:
            try:
                f = float(value)
                return None if f <= 0 else f
            except Exception:
                return None

        # Scientific floating-point values are not camera DN saturation levels.
        if np.issubdtype(cube_arr.dtype, np.floating):
            return None
        # AUTO: infère la dynamique réelle des valeurs du cube, sans supposer 8-bit.
        finite = np.asarray(cube_arr[np.isfinite(cube_arr)], dtype=np.float32)
        if finite.size == 0:
            return None
        # Échantillonnage pour éviter un coût énorme sur les gros cubes.
        if finite.size > 1_000_000:
            step = int(np.ceil(finite.size / 1_000_000))
            finite = finite[::step]
        vmax = float(np.nanmax(finite))

        if vmax <= 1.5:
            # Cube en réflectance 0-1: ne pas filtrer sur saturation DN.
            return None
        if vmax <= 255.5:
            return 255.0
        if vmax <= 4095.5:
            return 4095.0
        if vmax <= 65535.5:
            return 65535.0

        # Cas float arbitraire: pas de seuil DN fiable.
        return None

    sat_thresh = _resolve_sat_thresh(sat_thresh, cube)
    if sat_thresh is None:
        print("[ROI] Saturation filter: OFF")
    else:
        print(f"[ROI] Saturation filter: reject spectra with any band >= {sat_thresh:g} DN")

    # Always generate the default RGB from this ENVI, not a possibly unrelated
    # base_image_bgr.png left beside another acquisition.
    from shared.cube_preview import make_rgb_preview, save_preview_png, stretch_display_band
    image = None
    if rgb_image_path and Path(rgb_image_path).is_file():
        image = cv2.imdecode(np.fromfile(rgb_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (H, W):
            print('[ROI][WARN] Supplied preview does not match this cube; regenerating RGB.')
            image = None
    if image is None:
        key = hashlib.sha256(os.path.normcase(portable_project_path(cube_path)).encode()).hexdigest()[:12]
        preview_dir = Path(out_dir)/'previews'/f'{cube_path.stem}_{key}'
        image, actual = make_rgb_preview(cube, wavelengths)
        rgb_image_path = str(save_preview_png(preview_dir/'base_image_bgr.png', image))
        (preview_dir/'preview_source.json').write_text(json.dumps(dict(
            cube_path=portable_project_path(cube_path), path_base="project", shape=[H, W], rgb_wavelengths_nm=actual.tolist(),
            contrast='2-98 percentile per channel; display only', values='original spectra unchanged'),
            indent=2), encoding='utf-8')
        print(f'[ROI] RGB preview generated: {rgb_image_path}')
        print('[ROI] RGB wavelengths: ' + ' / '.join(f'{value:g}' for value in actual) + ' nm')
    img8 = image
    if display_band is not None:
        index = max(0, min(int(display_band), N-1))
        img8 = cv2.cvtColor(stretch_display_band(cube[:, :, index]), cv2.COLOR_GRAY2BGR)

    # Images de référence disponibles pendant la sélection ROI.
    # Touche g/G dans la fenêtre ROI pour alterner RGB synthétique, 490 nm, 560 nm, 700 nm.
    roi_display_images = _make_roi_display_images(
        cube=cube,
        cube_path=cube_path,
        rgb_image_path=rgb_image_path,
        img8_default=img8,
        wavelengths=wavelengths,
        roi_background_wavelengths=tuple(roi_background_wavelengths),
    )

    # Overlay optionnel du masque/warp final pendant la sélection ROI.
    # Touche m/M dans la fenêtre ROI pour l'afficher/masquer; ,/. pour l'opacité.
    roi_mask_overlay = _load_roi_mask_overlay(
        cube_path=cube_path,
        rgb_image_path=rgb_image_path,
        out_dir=out_dir,
        target_hw=(H, W),
    )

    # sélection ROIs par classe
    X_list, y_list = [], []
    rows_list, cols_list = [], []
    row_rel_list, col_rel_list = [], []
    cube_idx_list = []
    roi_id_list = []

    cube_2d = cube.reshape(-1, N)

    # ce run traite un seul cube. cube_id est indépendant du dossier absolu.
    current_cube_id = cube_id_from_path(cube_path)
    print(f"[ROI] Cube ID: {current_cube_id}")
    current_cube_ids = np.asarray([current_cube_id], dtype=object)
    current_cube_paths = np.asarray([portable_project_path(cube_path)], dtype=object)
    current_cube_index = 0

    # IMPORTANT:
    # roi_id doit identifier une ROI unique dans tout le dataset ajouté.
    # Avant, roi_idx recommençait à 0 pour chaque classe, donc roi_id=0 pouvait
    # exister pour fat, glandular, stroma, etc. Ensuite l'analyse croyait que
    # c'était une seule ROI contenant plusieurs classes.
    roi_global_id = 0

    # Keep full masks for this acquisition, before per-ROI subsampling.
    class_annotations = {}
    if dataset_path.exists():
        from shared.roi_drawing_tools import saved_class_masks
        with np.load(str(dataset_path), allow_pickle=True) as previous_dataset:
            class_annotations.update(saved_class_masks(previous_dataset, current_cube_id, (H, W)))
    # Fat must be annotated before meat so subtraction uses the current masks.
    selection_classes = list(class_names)
    if any(str(cls).strip().casefold() in ("meat", "muscle") for cls in selection_classes):
        selection_classes.sort(key=lambda cls: str(cls).strip().casefold() != "fat")

    for cls in selection_classes:
        roi_masks = select_rois_zoomable_shapes(
            img8,
            prompt=f"Select ROIs for: {cls}",
            max_size=(1200, 820),
            display_images=roi_display_images,
            mask_overlay=roi_mask_overlay,
            exclusion_masks={name: mask for name, mask in class_annotations.items()
                             if name.strip().casefold() != str(cls).strip().casefold()},
        )
        if roi_masks is None:
            # Nothing has been saved yet: all dataset/library writes occur only
            # after every class has completed its ROI selection.
            print("[ROI] Selection cancelled; no dataset or library was written.")
            return {"cancelled": True}
        if len(roi_masks) == 0:
            print(f"[INFO] Aucune ROI pour la classe '{cls}'.")
            continue

        annotation = class_annotations.setdefault(str(cls), np.zeros((H, W), bool))
        for mask in roi_masks:
            annotation |= mask

        for roi_idx, m in enumerate(roi_masks):
            # m est déjà (H,W) bool
            idx = np.where(m.ravel())[0]
            rr, cc = np.where(m)
            if idx.size == 0:
                continue

            if max_pix_per_roi is not None and max_pix_per_roi > 0 and idx.size > max_pix_per_roi:
                chosen = np.random.default_rng(0).choice(idx.size, size=max_pix_per_roi, replace=False)
                idx = idx[chosen]
                rr = rr[chosen]
                cc = cc[chosen]

            S = cube_2d[idx, :].astype(np.float32)

            if sat_thresh is not None:
                keep = (S < float(sat_thresh)).all(axis=1)
                if (~keep).sum() > 0:
                    print(f"[NOSAT] {cls}: -{int((~keep).sum())} spectres saturés")
                S = S[keep]
                rr = rr[keep]
                cc = cc[keep]
                if S.size == 0:
                    continue

            X_list.append(S)
            y_list.append(np.array([cls] * S.shape[0], dtype=object))

            if save_spatial_meta:
                rows_list.append(rr.astype(np.int32))
                cols_list.append(cc.astype(np.int32))
                row_rel_list.append((rr.astype(np.float32) / max(1, H - 1)).astype(np.float32))
                col_rel_list.append((cc.astype(np.float32) / max(1, W - 1)).astype(np.float32))
                cube_idx_list.append(np.full((S.shape[0],), current_cube_index, dtype=np.int32))
                roi_id_list.append(np.full((S.shape[0],), roi_global_id, dtype=np.int32))

            print(f"[ROI] Classe={cls!r} | roi_id={roi_global_id} | pixels gardés={S.shape[0]}")
            roi_global_id += 1

    if len(X_list) == 0:
        raise SystemExit("[STOP] Aucune donnée ajoutée (pas de ROI valides).")

    X_new = np.vstack(X_list).astype(np.float32)
    y_new = np.concatenate(y_list).astype(object)

    if save_spatial_meta:
        rows_new = np.concatenate(rows_list).astype(np.int32) if rows_list else np.empty((0,), dtype=np.int32)
        cols_new = np.concatenate(cols_list).astype(np.int32) if cols_list else np.empty((0,), dtype=np.int32)
        row_rel_new = np.concatenate(row_rel_list).astype(np.float32) if row_rel_list else np.empty((0,), dtype=np.float32)
        col_rel_new = np.concatenate(col_rel_list).astype(np.float32) if col_rel_list else np.empty((0,), dtype=np.float32)
        cube_idx_new = np.concatenate(cube_idx_list).astype(np.int32) if cube_idx_list else np.empty((0,), dtype=np.int32)
        roi_id_new = np.concatenate(roi_id_list).astype(np.int32)

    # append ou build dataset
    if dataset_path.exists():
        Z = np.load(str(dataset_path), allow_pickle=True)
        X_old = Z["X"].astype(np.float32)
        y_old = Z["y"].astype(object)

        old_wavelengths = _read_dataset_wavelengths(Z, n_bands=X_old.shape[1])
        if old_wavelengths is None:
            # Migration automatique des anciens datasets : si le nombre de bandes
            # correspond au cube courant, on ajoute désormais son axe spectral.
            if X_old.shape[1] == current_wavelengths.size:
                old_wavelengths = current_wavelengths.copy()
                print(
                    "[ROI][MIGRATION] Ancien dataset sans wavelengths; "
                    "axe du cube courant utilisé et sauvegardé."
                )
            else:
                raise SystemExit(
                    "[STOP] Impossible de déterminer les longueurs d'onde "
                    "de l'ancien dataset."
                )

        # Si on append à un dataset existant, les nouveaux roi_id doivent être
        # décalés pour rester uniques à l'échelle du dataset complet.
        if save_spatial_meta:
            if "roi_id" in Z.files and Z["roi_id"].size > 0:
                valid_old_roi = Z["roi_id"][Z["roi_id"] >= 0]
                roi_offset = int(valid_old_roi.max()) + 1 if valid_old_roi.size > 0 else 0
            else:
                roi_offset = 0
            roi_id_new = (roi_id_new + roi_offset).astype(np.int32)
            print(f"[ROI] Offset roi_id appliqué en append: +{roi_offset}")

        if X_old.shape[1] != X_new.shape[1]:
            raise SystemExit(f"[STOP] Incompatibilité: ancien dataset {X_old.shape[1]} bandes vs cube {X_new.shape[1]}.")

        if not np.allclose(old_wavelengths, current_wavelengths, rtol=0.0, atol=0.5):
            raise SystemExit(
                "[STOP] Incompatibilité des longueurs d'onde entre le dataset "
                f"et le cube.\nDataset: {old_wavelengths.tolist()}"
                f"\nCube: {current_wavelengths.tolist()}"
            )

        dataset_wavelengths = old_wavelengths.astype(np.float32)

        X = np.vstack([X_old, X_new])
        y = np.concatenate([y_old, y_new])
        print(f"[APPEND] Ancien: {X_old.shape[0]} px | Ajout: {X_new.shape[0]} px | Total: {X.shape[0]} px")
    else:
        X, y = X_new, y_new
        dataset_wavelengths = current_wavelengths.astype(np.float32)
        print(f"[BUILD] Nouveau dataset: {X.shape[0]} px")

    save_kwargs = {
        "X": X,
        "y": y,
        "wavelengths": dataset_wavelengths,
    }

    if save_spatial_meta:
        if dataset_path.exists():
            # recharger ancien dataset pour fusionner les métadonnées spatiales
            Z = np.load(str(dataset_path), allow_pickle=True)

            old_has_spatial = all(
                k in Z.files for k in ["rows", "cols", "row_rel", "col_rel", "sample_cube_index"]
            )

            if old_has_spatial:
                rows_old = Z["rows"].astype(np.int32)
                cols_old = Z["cols"].astype(np.int32)
                row_rel_old = Z["row_rel"].astype(np.float32)
                col_rel_old = Z["col_rel"].astype(np.float32)
                cube_idx_old = Z["sample_cube_index"].astype(np.int32)
                if "roi_id" in Z.files:
                    roi_id_old = Z["roi_id"].astype(np.int32)
                else:
                    roi_id_old = np.full((X_old.shape[0],), -1, dtype=np.int32)

                merged_cube_ids = [str(value) for value in cube_ids_from_dataset(Z).tolist()]
                merged_cube_paths = [
                    str(value)
                    for value in cube_paths_from_dataset(Z, len(merged_cube_ids)).tolist()
                ]
                current_path_str = str(cube_path)

                if current_cube_id in merged_cube_ids:
                    merged_current_idx = merged_cube_ids.index(current_cube_id)
                    # Refresh an obsolete absolute path without changing cube identity.
                    merged_cube_paths[merged_current_idx] = current_path_str
                else:
                    merged_cube_ids.append(current_cube_id)
                    merged_cube_paths.append(current_path_str)
                    merged_current_idx = len(merged_cube_ids) - 1

                rows = np.concatenate([rows_old, rows_new])
                cols = np.concatenate([cols_old, cols_new])
                row_rel = np.concatenate([row_rel_old, row_rel_new])
                col_rel = np.concatenate([col_rel_old, col_rel_new])
                sample_cube_index = np.concatenate([
                    cube_idx_old,
                    np.full((X_new.shape[0],), merged_current_idx, dtype=np.int32)
                ])
                roi_id = np.concatenate([roi_id_old, roi_id_new]).astype(np.int32)
                cube_ids_arr = np.asarray(merged_cube_ids, dtype=object)
                cube_paths_arr = np.asarray(merged_cube_paths, dtype=object)

            else:
                # ancien dataset sans info spatiale → placeholders
                n_old = X_old.shape[0]

                rows_old = np.full((n_old,), -1, dtype=np.int32)
                cols_old = np.full((n_old,), -1, dtype=np.int32)
                row_rel_old = np.full((n_old,), -1.0, dtype=np.float32)
                col_rel_old = np.full((n_old,), -1.0, dtype=np.float32)
                cube_idx_old = np.full((n_old,), -1, dtype=np.int32)
                roi_id_old = np.full((n_old,), -1, dtype=np.int32)

                rows = np.concatenate([rows_old, rows_new])
                cols = np.concatenate([cols_old, cols_new])
                row_rel = np.concatenate([row_rel_old, row_rel_new])
                col_rel = np.concatenate([col_rel_old, col_rel_new])

                sample_cube_index = np.concatenate([
                    cube_idx_old,
                    np.full((X_new.shape[0],), 0, dtype=np.int32)
                ])
                roi_id = np.concatenate([roi_id_old, roi_id_new]).astype(np.int32)
                cube_ids_arr = current_cube_ids
                cube_paths_arr = np.asarray([portable_project_path(cube_path)], dtype=object)
        else:
            # nouveau dataset
            rows = rows_new
            cols = cols_new
            row_rel = row_rel_new
            col_rel = col_rel_new
            sample_cube_index = cube_idx_new
            roi_id = roi_id_new.astype(np.int32)
            cube_ids_arr = current_cube_ids
            cube_paths_arr = current_cube_paths

        save_kwargs.update({
            "rows": rows,
            "cols": cols,
            "row_rel": row_rel,
            "col_rel": col_rel,
            "sample_cube_index": sample_cube_index,
            "cube_ids": cube_ids_arr,
            "cube_paths": np.asarray([portable_project_path(value) if value else "" for value in cube_paths_arr], dtype=object),
            "roi_id": roi_id
        })

    if dataset_path.exists():
        try:
            Z.close()
        except Exception:
            pass

    np.savez_compressed(str(dataset_path), **save_kwargs)
    print(f"[OK] Dataset -> {dataset_path}")
    finite_dataset = X[np.isfinite(X)]
    if finite_dataset.size:
        print(
            f"[ROI] Dataset range preserved: {float(finite_dataset.min()):.3f}-"
            f"{float(finite_dataset.max()):.3f} (no 100% cap)"
        )
    print(f"[ROI] Dataset cube IDs: {[str(value) for value in cube_ids_arr.tolist()]}")
    print(f"ROI_DATASET_PATH {dataset_path}")

    # ====== construction librairie selon normalisation ======
    normalization_mode = (normalization_mode or "none").lower().strip().replace("`", "")

    library: dict[str, dict] = {}
    all_classes = [str(c) for c in np.unique(y)]
    for cls in all_classes:
        mask = (y == cls)
        if not np.any(mask):
            continue

        Xc = np.asarray(X[mask], np.float32)

        if normalization_mode in ("snv", "snv_no_ratio", "snv_three_ratios"):
            Xc_norm = _snv_rows(Xc)
        elif normalization_mode == "log":
            from shared.spectrum_normalization import normalize_spectra
            Xc_norm = normalize_spectra(Xc, 'log')
        elif normalization_mode == "snv_mp":
            Xc_norm = _snv_mp_rows(Xc)
        elif normalization_mode == "snv_offset":
            Xc_norm = _snv_offset_rows(Xc)
        elif normalization_mode == "minmax":
            Xc_norm = _minmax_rows(Xc)
        elif normalization_mode == "mean_center":
            Xc_norm = _mean_center_rows(Xc)
        elif normalization_mode == "none":
            Xc_norm = Xc
        else:
            raise ValueError(f"[ROI] Unknown normalization mode: {normalization_mode!r}")

        keep_mask = _central_trim_mask(Xc_norm, keep_frac=keep_frac)
        Xk = Xc_norm[keep_mask]
        if Xk.shape[0] == 0:
            Xk = Xc_norm  # fallback si trim trop agressif

        med = np.median(Xk, axis=0).astype(np.float32)
        std = (
            Xk.std(axis=0, ddof=1).astype(np.float32)
            if Xk.shape[0] > 1 else np.zeros((Xk.shape[1],), np.float32)
        )

        library[cls] = {
            "median": med.tolist(),
            "std": std.tolist(),
            "n_total": int(Xc.shape[0]),
            "n_kept": int(Xk.shape[0]),
            "normalization": normalization_mode,
        }

    library["_meta"] = {
        "normalization": normalization_mode,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if library_path.exists():
        backup_path = library_path.with_suffix(".bak.json")
        shutil.copyfile(library_path, backup_path)
        print(f"[BACKUP] Sauvegarde -> {backup_path}")

    _write_library_json(library_path, library)
    print(f"[OK] Bibliothèque -> {library_path}")
    print(f"ROI_LIBRARY_PATH {library_path}")

    return library

# =====================================================================
# CLI entry point for the GUI -> ROI selection subprocess
# =====================================================================
if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="External ROI selection runner")
    parser.add_argument("--cube_path", required=False, default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--classes_json", required=False, default="[]")
    parser.add_argument("--class_names", nargs="*", default=None)
    parser.add_argument("--rgb_image_path", required=False, default=None)
    parser.add_argument("--display_band", type=int, default=None)
    parser.add_argument("--keep_frac", type=float, default=0.95)
    parser.add_argument(
        "--max_pix_per_roi",
        type=int,
        default=0,
        help="Optional ROI pixel cap; 0 keeps every selected pixel (default).",
    )
    parser.add_argument("--sat_thresh", default="auto", help="Saturation threshold: auto, none, or numeric full-scale DN")
    parser.add_argument("--dataset_name", default="roi_dataset.npz")
    parser.add_argument("--library_name", default=None)
    parser.add_argument("--normalization_mode", default="none")
    parser.add_argument("--update_only", action="store_true")
    parser.add_argument("--force_new_dataset", action="store_true")
    parser.add_argument("--use_smoothed", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--smooth_poly", type=int, default=2)
    parser.add_argument("--no_spatial_meta", action="store_true")
    parser.add_argument("--roi_background_wavelengths", nargs="*", type=int, default=[490, 560, 700])
    parser.add_argument(
        "--wavelengths",
        nargs="*",
        type=float,
        default=None,
        help="Optional explicit spectral axis. Otherwise inferred from corr_XXXnm files.",
    )

    args = parser.parse_args()

    if args.class_names:
        class_names = [str(c) for c in args.class_names if str(c).strip()]
    else:
        try:
            class_names = _json.loads(args.classes_json)
        except Exception as e:
            raise SystemExit(f"[ROI][CLI] Invalid --classes_json: {e}")
        class_names = [str(c) for c in class_names if str(c).strip() and str(c) != "_meta"]

    if not class_names and not args.update_only:
        raise SystemExit("[ROI][CLI] No class names were provided.")

    # print("[ROI][CLI] Starting external ROI selection")
    # print(f"[ROI][CLI] cube_path={args.cube_path}")
    # print(f"[ROI][CLI] out_dir={args.out_dir}")
    # print(f"[ROI][CLI] classes={', '.join(class_names) if class_names else '(none)'}")
    # print(f"[ROI][CLI] normalization={args.normalization_mode}")
    # print(f"[ROI][CLI] sat_thresh={args.sat_thresh}")
    # print(f"[ROI][CLI] update_only={args.update_only}")
    sys.stdout.flush()

    roi_select(
        cube_path=args.cube_path,
        out_dir=args.out_dir,
        class_names=class_names,
        display_band=args.display_band,
        keep_frac=args.keep_frac,
        max_pix_per_roi=None if args.max_pix_per_roi <= 0 else args.max_pix_per_roi,
        sat_thresh=args.sat_thresh,
        dataset_name=args.dataset_name,
        library_name=args.library_name,
        force_new_dataset=args.force_new_dataset,
        update_only=args.update_only,
        normalization_mode=args.normalization_mode,
        rgb_image_path=args.rgb_image_path,
        wavelengths=args.wavelengths,
        roi_background_wavelengths=tuple(args.roi_background_wavelengths),
        use_smoothed=args.use_smoothed,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        save_spatial_meta=not args.no_spatial_meta,
    )

    sys.stdout.flush()
