import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions, device, dtype):
        """
        positions: [T] or [B, T]
        """
        positions = positions.to(device=self.inv_freq.device)

        freqs = positions.float().unsqueeze(-1) * self.inv_freq

        cos = freqs.cos().to(device=device, dtype=dtype)
        sin = freqs.sin().to(device=device, dtype=dtype)

        return cos, sin


def apply_rope(x1, x2, cos, sin, layout="bthd"):
    if layout not in ("bthd", "bhtd"):
        raise ValueError(f"unsupported layout: {layout}")

    if x1.ndim != 4 or x2.shape != x1.shape:
        raise ValueError("x1 and x2 must have matching rank-4 shapes")

    if cos.shape != sin.shape or cos.ndim not in (2, 3):
        raise ValueError("cos and sin must have shape [T, D/2] or [B, T, D/2]")

    if cos.shape[-1] != x1.shape[-1]:
        raise ValueError("frequency dimension must match rotary pair dimension")

    seq_axis = 1 if layout == "bthd" else 2

    if cos.shape[-2] != x1.shape[seq_axis]:
        raise ValueError("frequency sequence length must match x1 and x2")

    if cos.ndim == 3 and cos.shape[0] not in (1, x1.shape[0]):
        raise ValueError("frequency batch dimension must match input batch size")

    if layout == "bthd":
        if cos.ndim == 2:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            cos = cos[:, :, None, :]
            sin = sin[:, :, None, :]
    else:
        if cos.ndim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        else:
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    return torch.stack((y1, y2), dim=-1).flatten(-2)
