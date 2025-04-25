# Heads Directory

This directory contains task‑specific “head” modules that convert transformer tokens into final predictions for camera pose, dense outputs (depth and 3D points), and point tracking.

## Files & Classes

### camera_head.py
**Class:** `CameraHead`
- Predicts camera parameters (translation, quaternion rotation, field of view) via iterative refinement over transformer tokens.
- **Key methods:**
  - `forward(aggregated_tokens_list, num_iterations=4) -> List[Tensor]`: returns list of pose encodings per iteration.
  - `trunk_fn(pose_tokens, num_iterations)`: internal refinement loop.
- **Usage example:**
```python
from vggt.heads.camera_head import CameraHead
# tokens_list from the Aggregator (List[Tensor])
head = CameraHead(dim_in=2048)
pose_iter = head(tokens_list)
final_pose = pose_iter[-1]  # [B, 9]
```

### dpt_head.py
**Class:** `DPTHead`
- Implements Dense Prediction Transformer head for pixelwise regression (e.g., depth or 3D points + confidence).
- **Key methods:**
  - `forward(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=8) -> (pred, conf)`
  - `_forward_impl(...)`: processes token projections, feature fusion, and activation.
- **Usage example:**
```python
from vggt.heads.dpt_head import DPTHead
head = DPTHead(dim_in=2048, output_dim=4)
pts3d, pts3d_conf = head(tokens_list, images, patch_start_idx)
```

### head_act.py
- **Functions:**
  - `activate_pose(pred_pose_enc, trans_act, quat_act, fl_act)`: applies specified activations to pose components.
  - `activate_head(out, activation, conf_activation) -> (pts3d, conf)`: converts raw DPT outputs to coordinates and confidence.
- **Usage example:**
```python
from vggt.heads.head_act import activate_pose, activate_head
pose = activate_pose(raw_pose_enc, trans_act='linear')
pts3d, conf = activate_head(raw_map, activation='norm_exp')
```

### track_head.py
**Class:** `TrackHead`
- Combines `DPTHead(feature_only=True)` for per‑frame features with `BaseTrackerPredictor` for iterative point tracking.
- **Key methods:**
  - `forward(tokens_list, images, patch_start_idx, query_points, iters) -> (coords, vis, conf)`
- **Usage example:**
```python
from vggt.heads.track_head import TrackHead
head = TrackHead(dim_in=2048)
coords, vis, conf = head(tokens_list, images, patch_start_idx, query_points)
```

### track_modules/
Submodules for the tracking predictor:
- **base_track_predictor.py:** `BaseTrackerPredictor` builds feature pyramid, cost volumes, and GRU updates.
- **blocks.py:** convolutional & GRU blocks used in predictor.
- **modules.py:** feature warping and correlation utilities.
- **utils.py:** helper functions for tracking (e.g., indexing, warping).