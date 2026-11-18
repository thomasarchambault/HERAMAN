# Exporting and reloading ROIs

ROIs in the **Visualisation** view let you extract spectra and restore a selection on its source cube. Annotated training datasets are managed separately in **ROI selection**.

## One selection, one CSV

Each export creates **one CSV file** containing the selected pixel spectra, ROI identifiers, shapes, vertices, and a reference to the source cube. There are no automatically generated position, mean-spectrum, geometry, or image companion files. NPZ is no longer offered for ROI exports.

The default filename is `*_pixel_spectra.csv` in `extracted spectra/`, next to the cube. You can customize the filename. Keep the source ENVI header and binary available to view the cube and reload the selection.

## Selecting and exporting

Under **Extract**, choose **Pixel**, **Rectangle**, or **Polygon**. Draw a rectangle by dragging with the left mouse button. For a polygon, place vertices with left-clicks and finish with a right-click; Escape cancels the current drawing. **Clear selection** clears the selection. Middle-button panning and zoom remain available.

Each ROI has an identifier displayed on the image. **Export selected pixels…** opens a preview to compare mean spectra and their standard deviations. Click **Export ROI CSV…** to save the complete selection.

**Save image…** is a separate, optional action that creates an annotated PNG. The image is not required to reload the selection.

## Reloading from a folder

1. Click **Load ROIs…**. The folder picker opens in `ENVI files/`.
2. Choose the folder containing the cube. The GUI searches for CSV exports in that folder and its `extracted spectra/` subfolder. You can also select the `extracted spectra/` folder directly.
3. If multiple exports are present, choose a selection from the list, sorted newest first.
4. The associated cube opens automatically and the ROIs are displayed. If the cube cannot be found, a dialog lets you locate its `.hdr` header.

No cube needs to be open beforehand. Replacing an existing selection requires confirmation. The cube filename, spectral axis, dimensions, and coordinates are checked before loading. These checks do not guarantee the identity of two cubes with the same filename.

The reference to the source cube is stored relative to the CSV location. Moving the cube and its export folder together preserves that reference. If a relative reference cannot be recorded across drives, only the cube filename is stored and the GUI can ask you to locate it. New exports do not store an absolute machine-specific cube path.

## CSV contents

Each row represents one unique pixel. Columns contain:

- `cube_name`, `x`, and `y`: source filename and zero-based pixel coordinates. A blank cube name refers to the previous row's cube.
- One column per wavelength in nm: the original pixel values.
- `selection_metadata`: a versioned JSON value stored inside the first pixel row only, containing the cube reference, cube dimensions, coordinate convention, and each ROI's shape and vertices. Keep this column with the pixel data when reloading a selection.
- `roi_ids`: every ROI containing the pixel, separated by `;` when regions overlap.

All geometry is contained in the CSV; no external JSON is needed. Masks imported from older exports retain their exact pixel membership. The viewer recalculates mean spectra and population standard deviations from the associated cube, including shared pixels in each relevant ROI and ignoring non-finite values separately for each band.

## Existing exports

Older pixel CSVs remain readable. Keep their `*_ROI_geometry.json` companion, if available, to recover original shapes; without it, the pixels and overlaps are restored as masks. A legacy `*_ROI_positions.csv` index can still point to its pixel CSV during folder discovery. A mean spectra CSV alone does not contain enough information to restore the selection.

Re-exporting a loaded selection creates the new self-contained CSV. Old companion files are not deleted automatically. Embedded metadata in a new CSV takes precedence over any old geometry file with the same name.

The internal NPZ training datasets used by **ROI selection** are separate from these exports and remain supported.
