# Exporting and reloading ROIs

ROIs in the **Visualisation** view let you extract spectra and restore a selection on its source cube. Annotated training datasets are managed separately in **ROI selection**.

## Selecting and exporting

Under **Extract**, choose **Pixel**, **Rectangle**, or **Polygon**. Draw a rectangle by dragging with the left mouse button. For a polygon, place vertices with left-clicks and finish with a right-click; Escape cancels the current drawing. **Clear selection** clears the selection. Middle-button panning and zoom remain available.

Each ROI has an identifier displayed on the image. **Export selected pixels…** opens a preview to compare mean spectra and their standard deviations before saving as CSV or NPZ. The suggested folder is `extracted spectra/`, next to the cube.

## Output files

| File | Contents |
| --- | --- |
| `*_pixel_spectra.csv` or `.npz` | Coordinates and full spectra of selected pixels, including ROI membership. The filename can be customized. |
| `*_ROI_positions.csv` | ROI index: identifier, shape, cube, pixel count, bounds, and centroid. |
| `*_ROI_mean_spectra.csv` | Mean and standard deviation of each ROI's spectra, by wavelength. |
| `*_ROI_geometry.json` | Shapes, vertices, and metadata for restoring geometry. |
| `*_image_with_ROIs.png` | Annotated view of ROI positions. |

Coordinates are zero-based and bounds are inclusive. The pixel file contains each pixel once; `roi_ids` lists every ROI containing that pixel, separated by `;` in CSV. Each ROI's statistics include all its pixels, including those shared with another ROI, and ignore non-finite values separately for each band.

In the pixel CSV, `cube_name` contains only the cube filename. An empty cell refers to the previous cube. JSON and NPZ files may retain the full cube path: review their contents before sharing.

## Reloading from a folder

1. Click **Load ROIs…**. The folder picker opens in `ENVI files/`.
2. Choose the folder containing the cube. The GUI searches for CSV exports in that folder and its `extracted spectra/` subfolder. You can also select the `extracted spectra/` folder directly.
3. If multiple exports are present, choose a selection from the list, sorted newest first.
4. The associated cube opens automatically and the ROIs are displayed. If the cube cannot be found, a dialog lets you locate its `.hdr` header.

No cube needs to be open beforehand. Replacing an existing selection requires confirmation. The cube filename, spectral axis, and coordinates are checked before loading; recent exports also verify dimensions through the JSON file. These checks do not guarantee the identity of two cubes with the same filename.

Keep the pixel CSV and its companion files together to recover the original shapes. Without the geometry JSON, pixels and overlaps can still be recovered, but ROIs are displayed as masks. Displayed and re-exported spectra are recalculated from the associated cube.

Reloading currently supports **CSV exports**. The mean spectra CSV alone does not contain the required coordinates; an NPZ export cannot replace the CSV for this feature.
