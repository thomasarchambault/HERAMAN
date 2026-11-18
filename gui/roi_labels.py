"""Place ROI callouts outside region bounds, in display coordinates."""
from PyQt5.QtCore import QRectF, QPointF


def _clip_segment(a, b, viewport):
    """Clip a leader's candidate edge to the visible image rectangle."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    lo, hi = 0., 1.
    for p, q in ((-dx, a[0]-viewport.left()), (dx, viewport.right()-a[0]),
                 (-dy, a[1]-viewport.top()), (dy, viewport.bottom()-a[1])):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            lo = max(lo, q/p)
        else:
            hi = min(hi, q/p)
        if lo > hi:
            return None
    return ((a[0]+lo*dx, a[1]+lo*dy), (a[0]+hi*dx, a[1]+hi*dy))


def nearest_outline_point(points, point, viewport=None):
    """Connect to the actual polygon edge, rather than its bounding box."""
    best, distance = None, float('inf')
    for a, b in zip(points, [*points[1:], points[0]]):
        if viewport is not None:
            clipped = _clip_segment(a, b, viewport)
            if clipped is None:
                continue
            a, b = clipped
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = dx*dx+dy*dy
        t = max(0., min(1., ((point.x()-a[0])*dx+(point.y()-a[1])*dy)/length)) if length else 0.
        candidate = QPointF(a[0]+t*dx, a[1]+t*dy)
        squared = (candidate.x()-point.x())**2+(candidate.y()-point.y())**2
        if squared < distance:
            best, distance = candidate, squared
    return best


def roi_outline_paths(roi, pixel_edges=False):
    """Native image coordinates for labels on actual ROI boundaries."""
    import math
    if roi['shape'] == 'circle':
        cx, cy = roi['vertices_xy'][0]
        radius = roi['radius_px']
        return [[(cx+radius*math.cos(i*math.tau/180), cy+radius*math.sin(i*math.tau/180))
                 for i in range(180)]]
    if roi['shape'] == 'polygon':
        return [roi['vertices_xy']]
    if roi['shape'] == 'mask':
        import cv2
        import numpy as np
        xy = roi['coordinates_xy']
        x0, y0 = xy.min(axis=0)
        width, height = xy.max(axis=0) - (x0, y0) + 1
        mask = np.zeros((height, width), np.uint8)
        mask[xy[:, 1]-y0, xy[:, 0]-x0] = 1
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return [(contour[:, 0, :] + (x0, y0)).tolist() for contour in contours]
    xy = roi['coordinates_xy']
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    if pixel_edges:
        x0, y0, x1, y1 = x0-.5, y0-.5, x1+.5, y1+.5
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


def roi_leader_endpoints(paths, label_rect, viewport):
    """Connect a label to the nearest visible outline, never to a bounding box."""
    center = label_rect.center()
    candidates = [nearest_outline_point(points, center, viewport) for points in paths]
    candidates = [point for point in candidates if point is not None]
    if not candidates:
        return None
    start = min(candidates, key=lambda p: (p.x()-center.x())**2+(p.y()-center.y())**2)
    end = QPointF(max(label_rect.left(), min(label_rect.right(), start.x())),
                 max(label_rect.top(), min(label_rect.bottom(), start.y())))
    return start, end


def place_roi_labels(bounds, sizes, viewport):
    """Return label rectangles and leader endpoints; avoid regions and labels."""
    occupied = []
    result = []
    for target, (width, height) in zip(bounds, sizes):
        if not viewport.intersects(target):
            result.append(None)
            continue
        candidates = []
        for ring in range(1, 16):
            gap = 8 + (ring-1)*(height+4)
            for x, y in ((target.right()+gap, target.top()-height-gap),
                         (target.right()+gap, target.center().y()-height/2),
                         (target.center().x()-width/2, target.top()-height-gap),
                         (target.left()-width-gap, target.center().y()-height/2),
                         (target.center().x()-width/2, target.bottom()+gap),
                         (target.left()-width-gap, target.top()-height-gap),
                         (target.right()+gap, target.bottom()+gap),
                         (target.left()-width-gap, target.bottom()+gap)):
                rect = QRectF(x, y, width, height)
                if not viewport.contains(rect):
                    continue
                overlap = 0.
                for obstacle in [*bounds, *occupied]:
                    intersection = rect.intersected(obstacle.adjusted(-3, -3, 3, 3))
                    overlap += max(0., intersection.width())*max(0., intersection.height())
                candidates.append((overlap, ring, rect))
            if candidates and min(c[0] for c in candidates) == 0:
                break
        if not candidates:
            # A zoomed ROI may extend beyond every viewport edge. Keep its
            # label in view even when no position outside the bounds fits.
            for x in (viewport.left(), viewport.right()-width, viewport.center().x()-width/2):
                for y in (viewport.top(), viewport.bottom()-height, viewport.center().y()-height/2):
                    rect = QRectF(x, y, width, height)
                    if viewport.contains(rect):
                        overlap = sum(max(0., rect.intersected(obstacle).width()) *
                                      max(0., rect.intersected(obstacle).height())
                                      for obstacle in occupied)
                        candidates.append((overlap, 16, rect))
            if not candidates:
                result.append(None)
                continue
        rect = min(candidates, key=lambda c: (c[0], c[1]))[2]
        occupied.append(rect)
        center = rect.center()
        start = QPointF(max(target.left(), min(target.right(), center.x())),
                        max(target.top(), min(target.bottom(), center.y())))
        end = QPointF(max(rect.left(), min(rect.right(), start.x())),
                      max(rect.top(), min(rect.bottom(), start.y())))
        result.append((rect, start, end))
    return result
