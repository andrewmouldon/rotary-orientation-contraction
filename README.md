# Rotary Orientation Contraction (ROC)

PyTorch implementation of **Rotary Orientation Contraction (ROC)**, a lightweight modification to Rotary Position Embeddings (RoPE).

Standard RoPE rotates query and key coordinate pairs according to token position, but those pairs already contain content-dependent orientations before rotation. ROC introduces learned contraction coefficients that control these orientations before applying standard RoPE.

ROC preserves the usual RoPE frequencies, positional rotations, and dimensionality.

## Usage

```python
import torch
from roc import ROC
from rope import RotaryEmbedding, apply_rope

B, T, H, D = 2, 1024, 16, 64

roc = ROC(
    num_q_heads=H,
    num_k_heads=H,
    head_dim=D,
).cuda()

rope = RotaryEmbedding(D).cuda()

q = torch.randn(B, T, H, D, device="cuda")
k = torch.randn(B, T, H, D, device="cuda")

positions = torch.arange(T, device="cuda")
cos, sin = rope(positions, q.device, q.dtype)

# ROC contraction and QK normalization
(q1, q2), (k1, k2) = roc(q, k)

# Standard RoPE
q = apply_rope(q1, q2, cos, sin)
k = apply_rope(k1, k2, cos, sin)
```

ROC applies orientation contraction followed by QK normalization and an optional learned per-query-head gain. Standard RoPE is then applied to the transformed pairs.

### Tensor Layouts

Both common attention layouts are supported:

- `[B, T, H, D]` — `layout="bthd"` (default)
- `[B, H, T, D]` — `layout="bhtd"`

For heads-first tensors:

```python
(q1, q2), (k1, k2) = roc(q, k, layout="bhtd")

q = apply_rope(q1, q2, cos, sin, layout="bhtd")
k = apply_rope(k1, k2, cos, sin, layout="bhtd")
```

Grouped-query attention is supported through separate `num_q_heads` and `num_k_heads` arguments.

Rotary frequencies can use positions of shape `[T]` or `[B, T]`. ROC uses interleaved rotary pairs.

## Configuration

Default settings:

- `init_alpha=0.5` — initial contraction strength
- `per_head=True` — separate coefficients per attention head
- `per_rotary_pair=True` — separate coefficients per rotary pair
- `share_qk_alpha=True` — shared query/key contraction coefficients
- `normalize=True` — QK normalization
- `per_head_gain=True` — learned query-head gain

The recommended configuration uses QK normalization, which was important for ROC's improvements in the experiments.
