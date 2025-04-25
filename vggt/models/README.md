# Models Directory

Contains high-level model definitions and orchestration of backbone and heads.

## Files & Classes

### vggt.py
**Class:** `VGGT`
- Main model class integrating:
  - `Aggregator` (backbone with alternating frame/global attention)
  - `CameraHead` for pose estimation
  - `DPTHead` for depth and 3D point prediction
  - `TrackHead` for point tracking
- **Usage example:**
```python
from vggt.models.vggt import VGGT
model = VGGT(img_size=518, patch_size=14, embed_dim=1024)
# images: [B, S, 3, H, W], query_points: [B, N, 2]
preds = model(images, query_points)
poses = preds['pose_enc']        # [B, S, 9]
depth, depth_conf = preds['depth'], preds['depth_conf']
pts3d, pts3d_conf = preds['world_points'], preds['world_points_conf']
track, vis, conf = preds.get('track'), preds.get('vis'), preds.get('conf')
```

### aggregator.py
**Class:** `Aggregator`
- Builds token sequence by:
  1. Patch embedding (conv or DINO‑ViT)
  2. Adding camera & register tokens
  3. Applying alternating attention blocks (frame vs. global)
- **Usage example:**
```python
from vggt.models.aggregator import Aggregator
agg = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
tokens_list, patch_start_idx = agg(images)
```