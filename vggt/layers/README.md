# Layers Directory

This directory provides core transformer building blocks and embedding layers used by VGGT's backbone and heads.

## Files & Classes

### patch_embed.py
**Class:** `PatchEmbed`
- Splits an image into non-overlapping patches and projects them to embedding vectors.
- **Usage example:**
```python
from vggt.layers.patch_embed import PatchEmbed
pe = PatchEmbed(img_size=518, patch_size=14, in_chans=3, embed_dim=1024)
tokens = pe(images)  # Tensor shape: [B, N, embed_dim]
```

### attention.py
**Classes:**
- `Attention`: standard multi-head self-attention with optional rotary embeddings.
- `MemEffAttention`: memory-efficient variant using xFormers.
- **Usage example:**
```python
from vggt.layers.attention import Attention
attn = Attention(dim=1024, num_heads=16)
out = attn(tokens)  # [B, N, 1024]
```

### block.py
**Class:** `Block`
- A transformer block combining LayerNorm, Attention, LayerScale, DropPath, and MLP.
- **Usage example:**
```python
from vggt.layers.block import Block
blk = Block(dim=1024, num_heads=16)
out = blk(tokens)
```

### mlp.py
**Class:** `Mlp`
- Two-layer feed-forward network with configurable activation and dropout.
- **Usage example:**
```python
from vggt.layers.mlp import Mlp
mlp = Mlp(in_features=1024, hidden_features=4096)
out = mlp(tokens)
```

### drop_path.py
**Class:** `DropPath`
- Implements stochastic depth regularization (randomly drops residual connections).

### layer_scale.py
**Class:** `LayerScale`
- Learnable per-channel scaling for residual branches.

### rope.py
**Classes:** `RotaryPositionEmbedding2D`, `PositionGetter`
- Implements 2D rotary positional embeddings applied to Q/K.

### swiglu_ffn.py
**Class:** `SwiGLUFFNFused`
- Fused feed-forward network using SwiGLU activation.

### vision_transformer.py
**Class:** `DinoVisionTransformer`
- End-to-end Vision Transformer backbone (DINO‑ViT variants) with register tokens.
- **Usage example:**
```python
from vggt.layers.vision_transformer import DinoVisionTransformer
vit = DinoVisionTransformer(img_size=518, patch_size=14)
tokens = vit(images)
```