# utils.py

## Contents

### Function `position_grid_to_embed`

Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

Args:
    pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
    embed_dim: Output channel dimension for embeddings

Returns:
    Tensor of shape (H, W, embed_dim) with positional embeddings

### Function `make_sincos_pos_embed`

This function generates a 1D positional embedding from a given grid using sine and cosine functions.

Args:
- embed_dim: The embedding dimension.
- pos: The position to generate the embedding from.

Returns:
- emb: The generated 1D positional embedding.

### Function `create_uv_grid`

Create a normalized UV grid of shape (width, height, 2).

The grid spans horizontally and vertically according to an aspect ratio,
ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
corner is at (x_span, y_span), normalized by the diagonal of the plane.

Args:
    width (int): Number of points horizontally.
    height (int): Number of points vertically.
    aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
    dtype (torch.dtype, optional): Data type of the resulting tensor.
    device (torch.device, optional): Device on which the tensor is created.

Returns:
    torch.Tensor: A (width, height, 2) tensor of UV coordinates.