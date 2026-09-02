from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add the fixed sinusoidal absolute encoding from *Attention Is All You Need*.

    Args:
        d_model: Width of the token embeddings.
        max_len: Number of positions cached initially.  The cache grows
            automatically when a longer sequence is encountered.
        dropout: Dropout applied after adding positional information.

    Input and output tensors have shape ``(batch, sequence_length, d_model)``.
    ``offset`` is useful when decoding one token at a time from a cached prefix.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._make_encoding(max_len), persistent=False)

    def _make_encoding(self, length: int) -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        frequencies = torch.exp(-torch.log(torch.tensor(10000.0)) * even_indices / self.d_model)

        encoding = torch.zeros(length, self.d_model, dtype=torch.float32)
        angles = positions * frequencies
        encoding[:, 0::2] = torch.sin(angles)
        
        encoding[:, 1::2] = torch.cos(angles[:, : self.d_model // 2])
        return encoding.unsqueeze(0)  

    def _ensure_length(self, length: int, device: torch.device) -> None:
        if length > self.pe.size(1):
            self.pe = self._make_encoding(length).to(device=device)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence_length, d_model)")
        if x.size(-1) != self.d_model:
            raise ValueError(f"expected embedding size {self.d_model}, got {x.size(-1)}")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        sequence_length = x.size(1)
        self._ensure_length(offset + sequence_length, x.device)
        positions = self.pe[:, offset : offset + sequence_length].to(device=x.device, dtype=x.dtype)
        return self.dropout(x + positions)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs: ``(x0, x1) -> (-x1, x0)``."""
    if x.size(-1) % 2:
        raise ValueError("the rotary dimension must be even")
    x = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: Optional[int] = None
) -> torch.Tensor:
    """Apply precomputed RoPE cosines and sines to ``x``.

    ``x`` must end in ``(sequence_length, head_dim)``. ``cos`` and ``sin``
    are broadcastable to ``x`` and normally have shape ``(batch, 1, T, D)``.
    Features past ``rotary_dim`` are intentionally left unchanged.
    """
    rotary_dim = x.size(-1) if rotary_dim is None else rotary_dim
    if not 0 < rotary_dim <= x.size(-1) or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive, even value no larger than head_dim")

    rotated, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat((rotated, remainder), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Apply Rotary Position Embeddings (RoPE) to attention queries and keys.

    The expected query/key layout is ``(batch, heads, sequence_length, head_dim)``,
    which is the layout produced in :mod:`src.models.attention`.  A single
    position vector or batch-specific ``position_ids`` of shape ``(batch, T)``
    may be supplied.  If omitted, positions begin at ``offset``.
    """

    def __init__(self, head_dim: int, max_len: int = 512, base: float = 10000.0, rotary_dim: Optional[int] = None):
        super().__init__()
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if not 0 < rotary_dim <= head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even, and no larger than head_dim")
        if max_len <= 0 or base <= 0:
            raise ValueError("max_len and base must be positive")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cache(max_len, device=inv_freq.device)

    def _set_cache(self, length: int, device: torch.device) -> None:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()

    def _ensure_cache(self, length: int, device: torch.device) -> None:
        if length > self.cos_cached.size(0) or self.cos_cached.device != device:
            self._set_cache(length, device)

    def _cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions")
        if position_ids.numel() == 0:
            raise ValueError("position_ids cannot be empty")
        if position_ids.min().item() < 0:
            raise ValueError("position_ids must be non-negative")

        self._ensure_cache(int(position_ids.max().item()) + 1, position_ids.device)
        cos = self.cos_cached[position_ids].to(dtype=dtype).unsqueeze(1)
        sin = self.sin_cached[position_ids].to(dtype=dtype).unsqueeze(1)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must have shape (batch, heads, sequence_length, head_dim)")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
            raise ValueError("q and k must have the same batch and sequence dimensions")
        if q.size(-1) != self.head_dim or k.size(-1) != self.head_dim:
            raise ValueError(f"q and k must have head_dim={self.head_dim}")
        if q.device != k.device:
            raise ValueError("q and k must be on the same device")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        batch_size, _, sequence_length, _ = q.shape
        if position_ids is None:
            position_ids = torch.arange(offset, offset + sequence_length, device=q.device).unsqueeze(0)
        else:
            position_ids = position_ids.to(device=q.device)
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.ndim != 2 or position_ids.size(1) != sequence_length:
                raise ValueError("position_ids must have shape (sequence_length,) or (batch, sequence_length)")
            if position_ids.size(0) not in (1, batch_size):
                raise ValueError("position_ids batch size must be 1 or match q and k")

        cos, sin = self._cos_sin(position_ids, q.dtype)
        return (
            apply_rotary_pos_emb(q, cos, sin, self.rotary_dim),
            apply_rotary_pos_emb(k, cos, sin, self.rotary_dim),
        )


__all__ = [
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]
