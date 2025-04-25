# dpt_head.py

## Contents

### Class `DPTHead`

DPT  Head for dense prediction tasks.

This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
(https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
backbone and produces dense predictions by fusing multi-scale features.

Args:
    dim_in (int): Input dimension (channels).
    patch_size (int, optional): Patch size. Default is 14.
    output_dim (int, optional): Number of output channels. Default is 4.
    activation (str, optional): Activation type. Default is "inv_log".
    conf_activation (str, optional): Confidence activation type. Default is "expp1".
    features (int, optional): Feature channels for intermediate representations. Default is 256.
    out_channels (List[int], optional): Output channels for each intermediate layer.
    intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
    pos_embed (bool, optional): Whether to use positional embedding. Default is True.
    feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
    down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.

### Function `_make_fusion_block`

### Function `_make_scratch`

### Class `ResidualConvUnit`

Residual convolution module.

### Class `FeatureFusionBlock`

Feature fusion block.

### Function `custom_interpolate`

Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.