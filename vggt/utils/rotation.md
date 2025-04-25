# rotation.py

## Contents

### Function `quat_to_mat`

Quaternion Order: XYZW or say ijkr, scalar-last

Convert rotations given as quaternions to rotation matrices.
Args:
    quaternions: quaternions with real part last,
        as tensor of shape (..., 4).

Returns:
    Rotation matrices as tensor of shape (..., 3, 3).

### Function `mat_to_quat`

Convert rotations given as rotation matrices to quaternions.

Args:
    matrix: Rotation matrices as tensor of shape (..., 3, 3).

Returns:
    quaternions with real part last, as tensor of shape (..., 4).
    Quaternion Order: XYZW or say ijkr, scalar-last

### Function `_sqrt_positive_part`

Returns torch.sqrt(torch.max(0, x))
but with a zero subgradient where x is 0.

### Function `standardize_quaternion`

Convert a unit quaternion to a standard form: one in which the real
part is non negative.

Args:
    quaternions: Quaternions with real part last,
        as tensor of shape (..., 4).

Returns:
    Standardized quaternions as tensor of shape (..., 4).