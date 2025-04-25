# load_fn.py

## Contents

### Function `load_and_preprocess_images`

A quick start function to load and preprocess images for model input.
This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

Args:
    image_path_list (list): List of paths to image files
    mode (str, optional): Preprocessing mode, either "crop" or "pad".
                         - "crop" (default): Sets width to 518px and center crops height if needed.
                         - "pad": Preserves all pixels by making the largest dimension 518px
                           and padding the smaller dimension to reach a square shape.

Returns:
    torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

Raises:
    ValueError: If the input list is empty or if mode is invalid

Notes:
    - Images with different dimensions will be padded with white (value=1.0)
    - A warning is printed when images have different shapes
    - When mode="crop": The function ensures width=518px while maintaining aspect ratio
      and height is center-cropped if larger than 518px
    - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
      and the smaller dimension is padded to reach a square shape (518x518)
    - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements