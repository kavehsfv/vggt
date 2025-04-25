# head_act.py

## Contents

### Function `activate_pose`

Activate pose parameters with specified activation functions.

Args:
    pred_pose_enc: Tensor containing encoded pose parameters [translation, quaternion, focal length]
    trans_act: Activation type for translation component
    quat_act: Activation type for quaternion component
    fl_act: Activation type for focal length component

Returns:
    Activated pose parameters tensor

### Function `base_pose_act`

Apply basic activation function to pose parameters.

Args:
    pose_enc: Tensor containing encoded pose parameters
    act_type: Activation type ("linear", "inv_log", "exp", "relu")

Returns:
    Activated pose parameters

### Function `activate_head`

Process network output to extract 3D points and confidence values.

Args:
    out: Network output tensor (B, C, H, W)
    activation: Activation type for 3D points
    conf_activation: Activation type for confidence values

Returns:
    Tuple of (3D points tensor, confidence tensor)

### Function `inverse_log_transform`

Apply inverse log transform: sign(y) * (exp(|y|) - 1)

Args:
    y: Input tensor

Returns:
    Transformed tensor