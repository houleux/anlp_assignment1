"""
Byte Latent Transformer (BLT) implementation — simplified, with entropy-based
dynamic patching on the source side.

Pipeline: raw source bytes -> DynamicByteLatentEncoder (entropy-scored,
variable-length patches) -> GlobalPatchTransformer (self-attention over
patch vectors, reuses EncoderLayer from transformer.py) -> ByteLatentDecoder
(fixed-size local GRU expansion back into raw target bytes).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import EncoderLayer
from .positional import SinusoidalPositionalEncoding


def _check_bytes(bytes_: torch.Tensor, vocab_size: int, name: str) -> None:
    """Validate a batch of raw byte IDs without synchronising in normal use."""
    if bytes_.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, sequence_length)")
    if bytes_.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer byte IDs")
    if bytes_.numel() and (bytes_.min().item() < 0 or bytes_.max().item() >= vocab_size):
        raise ValueError(f"{name} values must be in [0, {vocab_size - 1}]")


# =====================================================================
# Entropy scoring (tiny, jointly-trained causal byte LM)
# =====================================================================

class ByteEntropyModel(nn.Module):
    """Tiny causal byte-level LM used ONLY to score local next-byte entropy
    for patch-boundary decisions. This is NOT a separate large SLM — a
    single-layer GRU with a small hidden size (default 32), trained jointly
    with the rest of the network via a cheap self-supervised next-byte loss
    on the same stream it patches. A few thousand parameters total.
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, bytes_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (entropy, logits), both length L, computed causally.

        entropy[:, t] is the predictive entropy over the NEXT byte given
        bytes_[:, :t+1] as context (i.e. right after consuming byte t).
        logits[:, t] is that same next-byte prediction, used only by
        `auxiliary_loss` for training this tiny model.
        """
        emb = self.embedding(bytes_.long())
        hidden, _ = self.gru(emb)                 # (B, L, H); hidden[:, t] summarizes bytes_[:, :t+1]
        logits = self.output_proj(hidden)          # (B, L, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (B, L)
        return entropy, logits

    def auxiliary_loss(self, bytes_: torch.Tensor, byte_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-supervised next-byte prediction loss. Call this from train.py
        and add `weight * this` (e.g. weight=0.1) to the main loss so the
        entropy scorer keeps improving alongside the rest of the model.
        """
        _, logits = self.forward(bytes_)
        pred_logits = logits[:, :-1]                # (B, L-1, V): prediction made at t for byte t+1
        targets = bytes_[:, 1:].long()               # (B, L-1)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, self.vocab_size), targets.reshape(-1), reduction='none'
        ).view(bytes_.size(0), -1)
        if byte_mask is not None:
            valid = byte_mask[:, 1:].to(loss.dtype)
            return (loss * valid).sum() / valid.sum().clamp_min(1)
        return loss.mean()


def entropy_to_boundaries(
    entropy: torch.Tensor,
    byte_mask: torch.Tensor,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Convert per-byte entropy (B, L) into a boolean patch-start mask (B, L),
    using BLT's "monotonic" heuristic: start a new patch wherever entropy
    jumps by more than `threshold` versus the previous byte (a local
    surprise spike), rather than a fixed-size window. Position 0 of every
    sample always starts a patch. Padding positions never start a patch.

    `threshold` is the one knob to tune: raise it for fewer, longer patches
    (closer to the fixed-size baseline); lower it for more, shorter patches.
    Calibrate empirically against your C1-C4 average tokens/sequence (see
    earlier fairness discussion) so patch count is comparable across
    configs rather than confounding the ablation with a length mismatch.
    """
    B, L = entropy.shape
    boundary = torch.zeros(B, L, dtype=torch.bool, device=entropy.device)
    boundary[:, 0] = True
    if L > 1:
        delta = entropy[:, 1:] - entropy[:, :-1]
        boundary[:, 1:] = delta > threshold
    return boundary & byte_mask


# =====================================================================
# Dynamic (entropy-based) Local Encoder — SOURCE side
# =====================================================================

class DynamicByteLatentEncoder(nn.Module):
    """Entropy-based dynamic patching local encoder (source side of BLT).

    Patches are variable-length, decided per-sample by `entropy_to_boundaries`.
    Bytes are pooled into patches via a masked scatter-mean (rather than the
    fixed reshape used by a fixed-size patcher), since patch lengths differ
    both within and across samples in a batch.

    `max_patch_size` bounds the position-in-patch embedding table only; it
    does not forcibly split long patches. In the rare case a patch exceeds
    this length, position info beyond `max_patch_size - 1` is reused (last
    embedding row repeats) rather than crashing — an acceptable simplification
    at course scope, since the entropy threshold keeps this rare in practice.
    """

    def __init__(
        self,
        byte_dim: int,
        latent_dim: int,
        vocab_size: int = 256,
        entropy_hidden_dim: int = 32,
        entropy_threshold: float = 0.2,
        max_patch_size: int = 16,
    ):
        super().__init__()
        if byte_dim <= 0 or latent_dim <= 0 or vocab_size <= 0 or max_patch_size <= 0:
            raise ValueError("byte_dim, latent_dim, vocab_size, and max_patch_size must be positive")

        self.vocab_size = vocab_size
        self.max_patch_size = max_patch_size
        self.entropy_threshold = entropy_threshold

        self.entropy_model = ByteEntropyModel(vocab_size, entropy_hidden_dim)
        self.byte_embedding = nn.Embedding(vocab_size, byte_dim)
        self.position_embedding = nn.Embedding(max_patch_size, byte_dim)
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(byte_dim),
            nn.Linear(byte_dim, latent_dim),
            nn.GELU(),
        )

    def forward(
        self, bytes_: torch.Tensor, byte_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``bytes_`` of shape ``(B, L)`` into ``(B, P, latent_dim)``
        patches plus a ``(B, P)`` boolean patch_mask (P varies by call,
        sized to the longest patch count in the batch)."""
        _check_bytes(bytes_, self.vocab_size, "bytes_")
        batch_size, sequence_length = bytes_.shape
        if sequence_length == 0:
            raise ValueError("bytes_ must contain at least one byte")
        if byte_mask is None:
            byte_mask = torch.ones_like(bytes_, dtype=torch.bool)
        else:
            if byte_mask.shape != bytes_.shape:
                raise ValueError("byte_mask must have the same shape as bytes_")
            byte_mask = byte_mask.to(device=bytes_.device, dtype=torch.bool)

        # Boundary decision uses detached entropy: it's a discrete index, not
        # a differentiable quantity. The entropy model itself is still
        # trained, just via `auxiliary_loss` in the training loop, not
        # through this path.
        entropy, _ = self.entropy_model(bytes_)
        boundary = entropy_to_boundaries(entropy.detach(), byte_mask, self.entropy_threshold)

        patch_id = (torch.cumsum(boundary.long(), dim=1) - 1).clamp_min(0)   # (B, L), 0-indexed per row
        num_patches = patch_id.max(dim=1).values + 1                         # (B,)
        max_num_patches = int(num_patches.max().item())

        # Position-in-patch: distance since the most recent boundary,
        # clamped to the position-embedding table size.
        positions = torch.arange(sequence_length, device=bytes_.device).unsqueeze(0).expand(batch_size, -1)
        patch_start = torch.where(boundary, positions, torch.zeros_like(positions))
        patch_start = torch.cummax(patch_start, dim=1).values
        position_in_patch = (positions - patch_start).clamp(max=self.max_patch_size - 1)

        byte_vec = self.byte_embedding(bytes_.long()) + self.position_embedding(position_in_patch)
        byte_vec = byte_vec * byte_mask.unsqueeze(-1)   # zero out padding before scatter

        dim = byte_vec.size(-1)
        patch_sums = byte_vec.new_zeros((batch_size, max_num_patches, dim))
        patch_counts = byte_vec.new_zeros((batch_size, max_num_patches))

        scatter_index = patch_id.unsqueeze(-1).expand(-1, -1, dim)
        patch_sums.scatter_add_(1, scatter_index, byte_vec)
        patch_counts.scatter_add_(1, patch_id, byte_mask.to(byte_vec.dtype))

        patch_means = patch_sums / patch_counts.clamp_min(1).unsqueeze(-1)
        patches = self.patch_projection(patch_means)
        patch_mask = torch.arange(max_num_patches, device=bytes_.device).unsqueeze(0) < num_patches.unsqueeze(1)
        return patches, patch_mask

    def auxiliary_entropy_loss(self, bytes_: torch.Tensor, byte_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.entropy_model.auxiliary_loss(bytes_, byte_mask)


# =====================================================================
# Fixed-size Local Decoder — TARGET side (unchanged design, kept as-is)
# =====================================================================

class ByteLatentDecoder(nn.Module):
    """Fixed-size local expansion back to raw target bytes.

    Kept fixed-size (not entropy-based) deliberately: at generation time the
    model hasn't produced future target bytes yet, so a target-side entropy
    boundary would need bytes that don't exist yet. See module docstring.

    IMPORTANT: construct with vocab_size=257 to match dataset.py's
    convention (classes 0-255 = real UTF-8 bytes, class 256 = EOS). The
    decoder's own BOS token is then auto-assigned to 257 (= vocab_size),
    kept outside the valid raw-byte AND EOS range.
    """

    def __init__(self, patch_size: int, latent_dim: int, byte_dim: int, vocab_size: int = 257):
        super().__init__()
        if patch_size <= 0 or latent_dim <= 0 or byte_dim <= 0 or vocab_size <= 0:
            raise ValueError("patch_size, latent_dim, byte_dim, and vocab_size must be positive")

        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.bos_token_id = vocab_size  # kept outside the valid output range
        self.latent_projection = nn.Linear(latent_dim, byte_dim)
        self.position_embedding = nn.Embedding(patch_size, byte_dim)
        self.input_embedding = nn.Embedding(vocab_size + 1, byte_dim)
        self.local_decoder = nn.GRU(byte_dim, byte_dim, batch_first=True)
        self.output_projection = nn.Linear(byte_dim, vocab_size)

    def _base_inputs(self, latent_patches: torch.Tensor, target_length: int) -> torch.Tensor:
        if latent_patches.ndim != 3:
            raise ValueError("latent_patches must have shape (batch, num_patches, latent_dim)")
        if target_length <= 0:
            raise ValueError("target_length must be positive")

        batch_size, num_patches, _ = latent_patches.shape
        maximum_length = num_patches * self.patch_size
        if target_length > maximum_length:
            raise ValueError("target_length cannot exceed num_patches * patch_size")
        local_positions = self.position_embedding(
            torch.arange(self.patch_size, device=latent_patches.device)
        )
        base = self.latent_projection(latent_patches).unsqueeze(2) + local_positions.view(1, 1, self.patch_size, -1)
        return base.reshape(batch_size, maximum_length, -1)[:, :target_length]

    def forward(
        self,
        latent_patches: torch.Tensor,
        target_length: Optional[int] = None,
        target_bytes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return byte-vocabulary logits, optionally using teacher forcing."""
        if target_bytes is not None:
            _check_bytes(target_bytes, self.vocab_size, "target_bytes")
            if target_bytes.size(0) != latent_patches.size(0):
                raise ValueError("target_bytes and latent_patches must have the same batch size")
            if target_length is not None and target_length != target_bytes.size(1):
                raise ValueError("target_length must equal target_bytes length when both are provided")
            target_length = target_bytes.size(1)
        if target_length is None:
            target_length = latent_patches.size(1) * self.patch_size

        base = self._base_inputs(latent_patches, target_length)
        batch_size = base.size(0)
        decoder_input = torch.full(
            (batch_size, target_length), self.bos_token_id, device=base.device, dtype=torch.long
        )
        if target_bytes is not None and target_length > 1:
            decoder_input[:, 1:] = target_bytes[:, :-1].to(device=base.device, dtype=torch.long)
        hidden, _ = self.local_decoder(base + self.input_embedding(decoder_input))
        return self.output_projection(hidden)

    @torch.no_grad()
    def generate(self, latent_patches: torch.Tensor, target_length: int, temperature: float = 0.0) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        base = self._base_inputs(latent_patches, target_length)
        batch_size = base.size(0)
        previous = torch.full((batch_size,), self.bos_token_id, device=base.device, dtype=torch.long)
        hidden_state = None
        generated = []
        for step in range(target_length):
            step_input = base[:, step] + self.input_embedding(previous)
            step_hidden, hidden_state = self.local_decoder(step_input.unsqueeze(1), hidden_state)
            logits = self.output_projection(step_hidden[:, 0])
            previous = logits.argmax(dim=-1) if temperature == 0 else torch.multinomial(
                torch.softmax(logits / temperature, dim=-1), num_samples=1
            ).squeeze(1)
            generated.append(previous)
        return torch.stack(generated, dim=1)


# =====================================================================
# Global transformer over patch vectors (fixes the interface mismatch)
# =====================================================================

class GlobalPatchTransformer(nn.Module):
    """Self-attention stack over continuous patch vectors.

    Reuses EncoderLayer from transformer.py directly — patches are already
    embedded (produced by DynamicByteLatentEncoder), so there is no token
    embedding lookup here, unlike Transformer.encode(). This is the piece
    that was previously mismatched: ByteLatentTransformer was calling
    global_transformer(patches, patch_mask=...) against a module (the C1-C4
    Transformer) that expects integer token IDs, not float patch vectors.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        use_gqa: bool = False,
        num_groups: int = 4,
        use_rmsnorm: bool = False,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        self.positions = (
            nn.Identity() if use_rope else SinusoidalPositionalEncoding(d_model, max_sequence_length)
        )
        self.layers = nn.ModuleList(
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                use_gqa=use_gqa,
                num_groups=num_groups,
                use_rmsnorm=use_rmsnorm,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_rope=use_rope,
                max_sequence_length=max_sequence_length,
            )
            for _ in range(num_layers)
        )

    def forward(self, patches: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        x = self.positions(patches)
        key_mask = patch_mask.unsqueeze(1)  # (B, 1, P): broadcasts over heads and query positions
        for layer in self.layers:
            x = layer(x, key_mask)
        return x


# =====================================================================
# Full BLT model
# =====================================================================

class ByteLatentTransformer(nn.Module):
    """DynamicByteLatentEncoder -> GlobalPatchTransformer -> ByteLatentDecoder.

    See DynamicByteLatentEncoder / ByteLatentDecoder docstrings for the
    documented source-vs-target patching asymmetry.
    """

    def __init__(
        self,
        encoder: DynamicByteLatentEncoder,
        decoder: ByteLatentDecoder,
        global_transformer: Optional[GlobalPatchTransformer] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.global_transformer = global_transformer

    def _global(self, patches: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        if self.global_transformer is None:
            return patches
        return self.global_transformer(patches, patch_mask)

    def forward(
        self,
        source_bytes: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_bytes: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode source bytes, process patches globally, and return byte logits."""
        patches, patch_mask = self.encoder(source_bytes, source_mask)
        latents = self._global(patches, patch_mask)
        if target_bytes is None and target_length is None:
            target_length = source_bytes.size(1)
        return self.decoder(latents, target_length=target_length, target_bytes=target_bytes)

    def auxiliary_entropy_loss(self, source_bytes: torch.Tensor, source_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Call from train.py: total_loss = main_loss + 0.1 * this."""
        return self.encoder.auxiliary_entropy_loss(source_bytes, source_mask)

    @torch.no_grad()
    def generate(
        self, source_bytes: torch.Tensor, target_length: int, source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Greedily reconstruct or translate a byte sequence."""
        patches, patch_mask = self.encoder(source_bytes, source_mask)
        return self.decoder.generate(self._global(patches, patch_mask), target_length)


LocalEncoder = DynamicByteLatentEncoder
LocalDecoder = ByteLatentDecoder

__all__ = [
    "ByteEntropyModel",
    "entropy_to_boundaries",
    "DynamicByteLatentEncoder",
    "ByteLatentDecoder",
    "GlobalPatchTransformer",
    "ByteLatentTransformer",
    "LocalEncoder",
    "LocalDecoder",
]