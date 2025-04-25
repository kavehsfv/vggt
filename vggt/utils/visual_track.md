# visual_track.py

## Contents

### Function `color_from_xy`

Map (x, y) -> color in (R, G, B).
1) Normalize x,y to [0,1].
2) Combine them into a single scalar c in [0,1].
3) Use matplotlib's colormap to convert c -> (R,G,B).

You can customize step 2, e.g., c = (x + y)/2, or some function of (x, y).

### Function `get_track_colors_by_position`

Given all tracks in one sample (b), compute a (N,3) array of RGB color values
in [0,255]. The color is determined by the (x,y) position in the first
visible frame for each track.

Args:
    tracks_b: Tensor of shape (S, N, 2). (x,y) for each track in each frame.
    vis_mask_b: (S, N) boolean mask; if None, assume all are visible.
    image_width, image_height: used for normalizing (x, y).
    cmap_name: for matplotlib (e.g., 'hsv', 'rainbow', 'jet').

Returns:
    track_colors: np.ndarray of shape (N, 3), each row is (R,G,B) in [0,255].

### Function `visualize_tracks_on_images`

Visualizes frames in a grid layout with specified frames per row.
Each track's color is determined by its (x,y) position
in the first visible frame (or frame 0 if always visible).
Finally convert the BGR result to RGB before saving.
Also saves each individual frame as a separate PNG file.

Args:
    images: torch.Tensor (S, 3, H, W) if CHW or (S, H, W, 3) if HWC.
    tracks: torch.Tensor (S, N, 2), last dim = (x, y).
    track_vis_mask: torch.Tensor (S, N) or None.
    out_dir: folder to save visualizations.
    image_format: "CHW" or "HWC".
    normalize_mode: "[0,1]", "[-1,1]", or None for direct raw -> 0..255
    cmap_name: a matplotlib colormap name for color_from_xy.
    frames_per_row: number of frames to display in each row of the grid.
    save_grid: whether to save all frames in one grid image.

Returns:
    None (saves images in out_dir).