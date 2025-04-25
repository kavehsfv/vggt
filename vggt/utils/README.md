# Utils Directory

Utility functions for image I/O, geometry, pose encodings, rotation conversions, and visualization.

## Files & Functions

### load_fn.py
- **Function:** `load_and_preprocess_images(image_path_list, mode='crop')`
  - Loads images, handles RGBA, resizes/crops/pads to multiples of 14 or 518, returns `torch.Tensor [N,3,H,W]`.
  - **Usage:**
```python
from vggt.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(['f0.png','f1.png'], mode='pad')
```

### geometry.py
- Projective geometry utilities:
  - `project_points`, `unproject_depth`, `depth_to_world_points`, etc.

### pose_enc.py
- **Functions:**
  - `extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw) -> Tensor [B,S,9]`
  - `pose_encoding_to_extri_intri(pose_encoding, image_size_hw) -> (extrinsics, intrinsics)`
  - **Usage:**
```python
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
encoding = extri_intri_to_pose_encoding(ext, intr, (H,W))
```

### rotation.py
- **Functions:** `quat_to_mat(quaternions)`, `mat_to_quat(matrices)`, `standardize_quaternion`
- **Usage:**
```python
from vggt.utils.rotation import quat_to_mat, mat_to_quat
R = quat_to_mat(quat)
quat2 = mat_to_quat(R)
```

### visual_track.py
- **Functions:**
  - `get_track_colors_by_position(tracks, ...)`
  - `visualize_tracks_on_images(images, tracks, ...)`
- **Usage:**
```python
from vggt.utils.visual_track import visualize_tracks_on_images
visualize_tracks_on_images(images, track_preds, vis_mask, out_dir='vis/')
```