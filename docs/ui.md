# UI Manual

This page documents the stock browser UI served by `tensor-viz` and by the
`@tensor-viz/viewer-demo` package.

## Layout

The app has four main areas:

- Ribbon: `File`, `Display`, and `Widgets` menus.
- Tab strip: shown when a session contains multiple tabs.
- Viewport: the main 2D or 3D tensor scene.
- Sidebar: widget panels such as `Tensor View`, `Inspector`, and `Selection`.

You can resize the sidebar by dragging the splitter between the viewport and the
widgets.

## File Menu

- `Open Tensor`: load one local `.npy` file.
- `Save Tensor`: save the active tensor as `.npy`.

`Save Tensor` writes the currently active tensor, not every loaded tensor.

## Display Menu

- `Display as 2D`: switch to the orthographic 2D layout.
- `Display as 3D`: switch to the 3D layout.
- `Toggle Heatmap`: render values as a grayscale heatmap instead of the default
  base color.
- `Toggle Dimension Lines`: show or hide the dimension guides and labels.
- `Toggle Tensor Names`: show or hide the tensor name label above each tensor in
  2D.

### Advanced Submenu

`Display -> Advanced` contains quick display toggles:

- `Set Contiguous Axis Family Mapping`
- `Set Z-Order Axis Family Mapping`
- `Toggle Block Gaps`
- `Toggle Collapse Hidden Axes`
- `Toggle Log Scale`

Use the `Advanced Settings` widget when you need the numeric `Block Gap Scale`
control in addition to these quick toggles.

## Widgets Menu

- `Toggle Tensor View`
- `Toggle Inspector`
- `Toggle Selection`
- `Toggle Advanced Settings`

These commands only show or hide the corresponding sidebar panels. They do not
change tensor data or viewer state beyond panel visibility.

## Navigation

### 2D View

- Mouse wheel zooms.
- Right-drag pans.
- Left click selects the hovered tensor as active.
- Left-drag starts box selection when selection is available.

Selection is only available in `2D` with `Contiguous` axis family mapping.

### 3D View

- Left-drag rotates the camera.
- Right-drag pans the camera.
- Mouse wheel zooms.
- Click selects the hovered tensor as active.

Cell selection is disabled in 3D.

## Selection

Selection only works in `2D` with `Contiguous` mapping.

- Left-drag draws a selection box.
- As you drag, cells inside the box are preview-tinted live.
- Releasing the mouse applies the selection.
- Hold `Shift` while dragging to add cells to the current selection.
- Hold `Ctrl` while dragging to remove cells from the current selection.

The `Selection` widget shows:

- highlighted cell count
- `min`
- `25th`
- `50th`
- `75th`
- `max`
- `mean`
- `std`

These statistics are computed only from selected cells that currently have
loaded numeric values.

## Tensor View Widget

The `Tensor View` widget controls the active tensor.

- `Tensor`: choose which loaded tensor is active for view editing.
- `View String`: edit the view grammar directly.
- `Preview`: shows the implied permute, reshape, and slice expression.
- Slice sliders: appear for lowercase slice tokens and let you change the
  active slice index.

View string rules:

- Uppercase tokens stay visible.
- Lowercase tokens become slices.
- `1` inserts a singleton dimension.
- An empty input resets to the default view.

## Inspector Widget

The `Inspector` widget shows metadata for the active tensor and hover
information for the currently hovered cell.

Fields include:

- `Hovered Tensor`
- `Layout Coord`
- `Tensor Coord`
- `Value`
- `DType`
- `Tensor Shape`
- `Rank`

`Tensor Shape` and `Rank` follow the hovered tensor when the pointer is over a
different tensor.

## Advanced Settings Widget

The `Advanced Settings` widget exposes lower-level layout controls:

- `Block Gap Scale`: numeric multiplier for higher-level block spacing
- `Axis Family Mapping`: `Z-Order` or `Contiguous`
- `Show Block Gaps`
- `Collapse Hidden Axes`
- `Log`

These settings affect layout, label placement, and heatmap interpretation.

## Colorbar Widget

The `Colorbar` widget appears when heatmap mode is enabled and numeric value
ranges are available. It shows the current per-tensor min/max range and whether
the viewer is using linear or signed-log scaling.

## Command Palette And Shortcuts

Press `?` to open the command palette.

Useful shortcuts:

- `Ctrl+O`: open tensor
- `Ctrl+S`: save tensor
- `Ctrl+2`: switch to 2D
- `Ctrl+3`: switch to 3D
- `Ctrl+H`: toggle heatmap
- `Ctrl+D`: toggle dimension lines
- `Ctrl+V`: toggle Tensor View widget
- `Ctrl+P`: open the tab picker when a session has multiple tabs

The command palette also exposes the same display and widget actions that
appear in the menus.
