import math

import torch
import torch.nn as nn


class ROC(nn.Module):
    def __init__(
        self,
        num_q_heads,
        num_k_heads,
        head_dim,
        init_alpha=0.5,
        per_head=True,
        per_rotary_pair=True,
        share_qk_alpha=True,
        normalize=True,
        per_head_gain=True,
        eps=1e-12,
    ):
        super().__init__()

        assert num_q_heads % num_k_heads == 0
        assert head_dim % 2 == 0
        assert 0.0 < init_alpha < 1.0

        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.head_dim = head_dim
        self.num_pairs = head_dim // 2
        self.n_rep = num_q_heads // num_k_heads

        self.per_head = per_head
        self.share_qk_alpha = share_qk_alpha
        self.normalize = normalize
        self.per_head_gain = per_head_gain
        self.eps = eps

        alpha_pairs = self.num_pairs if per_rotary_pair else 1
        init_logit = math.log(init_alpha / (1 - init_alpha))

        if share_qk_alpha:
            alpha_heads = num_k_heads if per_head else 1

            self.alpha_logits = nn.Parameter(
                torch.full((alpha_heads, alpha_pairs), init_logit)
            )

            self.register_parameter("q_alpha_logits", None)
            self.register_parameter("k_alpha_logits", None)

        else:
            q_heads = num_q_heads if per_head else 1
            k_heads = num_k_heads if per_head else 1

            self.q_alpha_logits = nn.Parameter(
                torch.full((q_heads, alpha_pairs), init_logit)
            )

            self.k_alpha_logits = nn.Parameter(
                torch.full((k_heads, alpha_pairs), init_logit)
            )

            self.register_parameter("alpha_logits", None)

        if per_head_gain:
            self.q_gain = nn.Parameter(torch.ones(num_q_heads))
        else:
            self.register_parameter("q_gain", None)

    def forward(self, q, k, layout="bthd"):
        if layout not in ("bthd", "bhtd"):
            raise ValueError(f"unsupported layout: {layout}")

        head_axis = 2 if layout == "bthd" else 1

        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must be rank-4 tensors")

        if q.shape[head_axis] != self.num_q_heads:
            raise ValueError("q head count does not match num_q_heads")

        if k.shape[head_axis] != self.num_k_heads:
            raise ValueError("k head count does not match num_k_heads")

        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError("q and k head dimensions must match head_dim")

        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        if self.share_qk_alpha:
            alpha_k = torch.sigmoid(self.alpha_logits)

            alpha_q = (
                alpha_k.repeat_interleave(self.n_rep, dim=0)
                if self.per_head and self.n_rep != 1
                else alpha_k
            )
        else:
            alpha_q = torch.sigmoid(self.q_alpha_logits)
            alpha_k = torch.sigmoid(self.k_alpha_logits)

        if layout == "bthd":
            aq = alpha_q.view(1, 1, *alpha_q.shape)
            ak = alpha_k.view(1, 1, *alpha_k.shape)
        else:
            aq = alpha_q.view(1, alpha_q.shape[0], 1, alpha_q.shape[1])
            ak = alpha_k.view(1, alpha_k.shape[0], 1, alpha_k.shape[1])

        qr = torch.sqrt(q1.square() + q2.square() + self.eps)
        kr = torch.sqrt(k1.square() + k2.square() + self.eps)

        q1 = (1 - aq) * q1 + aq * qr
        q2 = (1 - aq) * q2

        k1 = (1 - ak) * k1 + ak * kr
        k2 = (1 - ak) * k2

        if self.normalize:
            norm_eps = torch.finfo(q1.dtype).eps

            q_inv = torch.rsqrt(
                (q1.square() + q2.square()).sum(-1, keepdim=True)
                / self.head_dim
                + norm_eps
            )

            k_inv = torch.rsqrt(
                (k1.square() + k2.square()).sum(-1, keepdim=True)
                / self.head_dim
                + norm_eps
            )

            q1 = q1 * q_inv
            q2 = q2 * q_inv

            k1 = k1 * k_inv
            k2 = k2 * k_inv

        if self.per_head_gain:
            if layout == "bthd":
                gain = self.q_gain.view(1, 1, self.num_q_heads, 1)
            else:
                gain = self.q_gain.view(1, self.num_q_heads, 1, 1)

            q1 = q1 * gain
            q2 = q2 * gain

        return (q1, q2), (k1, k2)
