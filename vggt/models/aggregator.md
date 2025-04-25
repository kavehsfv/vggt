# aggregator.py

## Contents

### Class `Aggregator`

The Aggregator applies alternating-attention over input frames,
as described in VGGT: Visual Geometry Grounded Transformer.


Args:
    img_size (int): Image size in pixels.
    patch_size (int): Size of each patch for PatchEmbed.
    embed_dim (int): Dimension of the token embeddings.
    depth (int): Number of blocks.
    num_heads (int): Number of attention heads.
    mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
    num_register_tokens (int): Number of register tokens.
    block_fn (nn.Module): The block type used for attention (Block by default).
    qkv_bias (bool): Whether to include bias in QKV projections.
    proj_bias (bool): Whether to include bias in the output projection.
    ffn_bias (bool): Whether to include bias in MLP layers.
    patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
    aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
    aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
    qk_norm (bool): Whether to apply QK normalization.
    rope_freq (int): Base frequency for rotary embedding. -1 to disable.
    init_values (float): Init scale for layer scale.

### Function `slice_expand_and_flatten`

Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
1) Uses the first position (index=0) for the first frame only
2) Uses the second position (index=1) for all remaining frames (S-1 frames)
3) Expands both to match batch size B
4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
   followed by (S-1) second-position tokens
5) Flattens to (B*S, X, C) for processing

Returns:
    torch.Tensor: Processed tokens with shape (B*S, X, C)