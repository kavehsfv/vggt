# drop_path.py

This module implements stochastic depth, also known as DropPath, a regularization technique that randomly drops entire residual branches during training.

## Functions

### `drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False)`

**Description:**  
Randomly zeroes entire samples (paths) in a batch with probability `drop_prob` during training. Scales the remaining paths by `1 / (1 - drop_prob)` to maintain the expected value.

**Args:**  
- `x` (`Tensor`): Input tensor of any shape, where the first dimension is batch size.  
- `drop_prob` (`float`): Probability of dropping each path.  
- `training` (`bool`): If `False`, no dropping is applied.

**Returns:**  
Tensor of the same shape as `x` with paths dropped and scaled.

## Classes

### `DropPath`

**Description:**  
`nn.Module` wrapper around the `drop_path` function for seamless integration into model definitions. Automatically respects the module's `training` flag.

**Constructor Args:**  
- `drop_prob` (`float`, optional): Probability of dropping each path.

**Usage Example:**
```python
from vggt.layers.drop_path import DropPath

dropper = DropPath(drop_prob=0.1)
output = dropper(input_tensor)
``` 