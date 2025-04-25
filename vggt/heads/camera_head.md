# camera_head.py

## Contents

### Class `CameraHead`

CameraHead predicts camera parameters from token representations using iterative refinement.

It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.

### Function `modulate`

Modulate the input tensor using scaling and shifting parameters.