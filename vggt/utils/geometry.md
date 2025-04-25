# geometry.py

## Contents

### Function `unproject_depth_map_to_point_map`

Unproject a batch of depth maps to 3D world coordinates.

Args:
    depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
    extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
    intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

Returns:
    np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)

### Function `depth_to_world_coords_points`

Convert a depth map to world coordinates.

Args:
    depth_map (np.ndarray): Depth map of shape (H, W).
    intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
    extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

Returns:
    tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).

### Function `depth_to_cam_coords_points`

Convert a depth map to camera coordinates.

Args:
    depth_map (np.ndarray): Depth map of shape (H, W).
    intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

Returns:
    tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)

### Function `closed_form_inverse_se3`

Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

If `R` and `T` are provided, they must correspond to the rotation and translation
components of `se3`. Otherwise, they will be extracted from `se3`.

Args:
    se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
    R (optional): Nx3x3 array or tensor of rotation matrices.
    T (optional): Nx3x1 array or tensor of translation vectors.

Returns:
    Inverted SE3 matrices with the same type and device as `se3`.

Shapes:
    se3: (N, 4, 4)
    R: (N, 3, 3)
    T: (N, 3, 1)