"""
Transformer model implementation.
Can be used to construct any configs given in assignment.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention, CrossMultiHeadAttention, CrossGroupedQueryAttention
from .ffn import PositionwiseFeedForward
from .norm import RMSNorm, LayerNorm
from .positional import SinusoidalPositionalEncoding, RotaryPositionalEmbedding


@dataclass
class TransformerConfig:
    """Shared architecture settings for the tokenized C1--C4 models.

    ``attention_type`` and ``norm_type`` change C3 and C4 respectively.
    ``positional_encoding`` changes C1's sinusoidal embeddings to C2's RoPE;
    RoPE is passed into every self-attention layer and rotates projected query
    and key tensors there.
    """

    src_vocab_size: int
    tgt_vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_sequence_length: int = 512
    attention_type: Literal["mha", "gqa"] = "mha"
    num_groups: int = 4
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    positional_encoding: Literal["sinusoidal", "rope"] = "sinusoidal"

    def __post_init__(self) -> None:
        if self.src_vocab_size <= 0 or self.tgt_vocab_size <= 0:
            raise ValueError("vocabulary sizes must be positive")
        if self.d_model <= 0 or self.ffn_dim <= 0 or self.max_sequence_length <= 0:
            raise ValueError("model dimensions and max_sequence_length must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("at least one encoder and decoder layer is required")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_type not in {"mha", "gqa"}:
            raise ValueError("attention_type must be 'mha' or 'gqa'")
        if self.attention_type == "gqa" and (self.num_groups <= 0 or self.num_heads % self.num_groups):
            raise ValueError("num_heads must be divisible by a positive num_groups for GQA")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")
        if self.positional_encoding not in {"sinusoidal", "rope"}:
            raise ValueError("positional_encoding must be 'sinusoidal' or 'rope'")
        if self.positional_encoding == "rope" and (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if not 0 <= self.pad_token_id < self.src_vocab_size:
            raise ValueError("pad_token_id must be valid in the source vocabulary")
        for token_id, name in ((self.pad_token_id, "pad_token_id"), (self.bos_token_id, "bos_token_id"), (self.eos_token_id, "eos_token_id")):
            if not 0 <= token_id < self.tgt_vocab_size:
                raise ValueError(f"{name} must be valid in the target vocabulary")



class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_gqa: bool=False, num_groups: int=4, use_rmsnorm: bool=False, ffn_dim: int=2048, dropout: float=0.1, use_rope: bool=False, max_sequence_length: int=512):
        super().__init__()
        self.use_gqa = use_gqa
        self.use_rmsnorm = use_rmsnorm
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.ffn_dim = ffn_dim
        rope = RotaryPositionalEmbedding(d_model // num_heads, max_sequence_length) if use_rope else None

        if use_gqa:
            self.self_attn = GroupedQueryAttention(self.d_model, self.d_model, self.num_heads, num_groups=self.num_groups, rope=rope)
        else:
            self.self_attn = MultiHeadAttention(self.d_model, self.d_model, self.num_heads, rope=rope)
        if use_rmsnorm:
            self.norm1 = RMSNorm(self.d_model)
            self.norm2 = RMSNorm(self.d_model)
        else:
            self.norm1 = LayerNorm(self.d_model)
            self.norm2 = LayerNorm(self.d_model)
        self.ffn = PositionwiseFeedForward(self.d_model, self.ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x



class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_gqa: bool=False, num_groups: int=4, use_rmsnorm: bool=False, ffn_dim: int=2048, dropout: float=0.1, use_rope: bool=False, max_sequence_length: int=512):
        super().__init__()
        self.use_gqa = use_gqa
        self.use_rmsnorm = use_rmsnorm
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.ffn_dim = ffn_dim
        rope = RotaryPositionalEmbedding(d_model // num_heads, max_sequence_length) if use_rope else None

        if use_gqa:
            self.self_attn = GroupedQueryAttention(self.d_model, self.d_model, self.num_heads, num_groups=self.num_groups, rope=rope)
            self.cross_attn = CrossGroupedQueryAttention(
                self.d_model, self.d_model, self.d_model, self.num_heads, self.num_groups
            )
        else:
            self.self_attn = MultiHeadAttention(self.d_model, self.d_model, self.num_heads, rope=rope)
            self.cross_attn = CrossMultiHeadAttention(
                self.d_model, self.d_model, self.d_model, self.num_heads
            )
        if use_rmsnorm:
            self.norm1 = RMSNorm(self.d_model)
            self.norm2 = RMSNorm(self.d_model)
            self.norm3 = RMSNorm(self.d_model)
        else:
            self.norm1 = LayerNorm(self.d_model)
            self.norm2 = LayerNorm(self.d_model)
            self.norm3 = LayerNorm(self.d_model)
        self.ffn = PositionwiseFeedForward(self.d_model, ffn_dim=self.ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.norm1(x), tgt_mask)
        x = x + self.dropout(attn_output)
        cross_attn_output = self.cross_attn(self.norm2(x), enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        ffn_output = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_output)
        return x


class Transformer(nn.Module):
    """Configurable encoder-decoder Transformer built from the shared layers.

    Inputs are batch-first token IDs. ``tgt_input_ids`` is the shifted-right
    target sequence (beginning with BOS); the model returns logits for its next
    token at every position. Padding masks are inferred from ``pad_token_id``
    unless explicitly provided.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding_scale = config.d_model**0.5
        use_gqa = config.attention_type == "gqa"
        use_rmsnorm = config.norm_type == "rmsnorm"
        use_rope = config.positional_encoding == "rope"

        self.src_embedding = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx=config.pad_token_id)
        if config.positional_encoding == "sinusoidal":
            # RoPE is applied inside self-attention, so only sinusoidal is added here.
            self.src_positions = SinusoidalPositionalEncoding(config.d_model, config.max_sequence_length)
            self.tgt_positions = SinusoidalPositionalEncoding(config.d_model, config.max_sequence_length)
        else:
            self.src_positions = nn.Identity()
            self.tgt_positions = nn.Identity()
        self.embedding_dropout = nn.Dropout(config.dropout)

        layer_args = dict(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_gqa=use_gqa,
            num_groups=config.num_groups,
            use_rmsnorm=use_rmsnorm,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            use_rope=use_rope,
            max_sequence_length=config.max_sequence_length,
        )
        self.encoder_layers = nn.ModuleList(EncoderLayer(**layer_args) for _ in range(config.num_encoder_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(**layer_args) for _ in range(config.num_decoder_layers))
        self.output_norm = RMSNorm(config.d_model) if use_rmsnorm else LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

    def encode(
        self, src_ids: torch.Tensor, src_valid: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``src_ids`` of shape ``(B, S)`` and return memory plus mask."""
        self._check_token_ids(src_ids, self.config.src_vocab_size, "src_ids")
        src_valid = self._valid_mask(src_ids, src_valid)
        x = self.embedding_dropout(self.src_positions(self.src_embedding(src_ids) * self.embedding_scale))
        # ``(B, 1, S)`` masks source *keys* and broadcasts over heads/queries.
        source_key_mask = src_valid.unsqueeze(1)
        for layer in self.encoder_layers:
            x = layer(x, source_key_mask)
        return x, src_valid

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_valid: torch.Tensor,
        tgt_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode shifted-right IDs into logits of shape ``(B, T, V_tgt)``."""
        self._check_token_ids(tgt_input_ids, self.config.tgt_vocab_size, "tgt_input_ids")
        if memory.ndim != 3 or memory.size(0) != tgt_input_ids.size(0) or memory.size(-1) != self.config.d_model:
            raise ValueError("memory must have shape (batch, source_length, d_model)")
        if source_valid.shape != memory.shape[:2]:
            raise ValueError("source_valid must have shape (batch, source_length)")

        tgt_valid = self._valid_mask(tgt_input_ids, tgt_valid)
        x = self.embedding_dropout(self.tgt_positions(self.tgt_embedding(tgt_input_ids) * self.embedding_scale))
        target_length = tgt_input_ids.size(1)
        causal = torch.tril(torch.ones(target_length, target_length, dtype=torch.bool, device=x.device))
        # (B, T, T): causal constraint AND valid decoder keys.
        target_mask = causal.unsqueeze(0) & tgt_valid.unsqueeze(1)
        source_key_mask = source_valid.unsqueeze(1)
        for layer in self.decoder_layers:
            x = layer(x, memory, source_key_mask, target_mask)
        return self.output_projection(self.output_norm(x))

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_valid: Optional[torch.Tensor] = None,
        tgt_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return next-token logits for a batch of source/decoder-input IDs."""
        memory, source_valid = self.encode(src_ids, src_valid)
        return self.decode(tgt_input_ids, memory, source_valid, tgt_valid)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_new_tokens: int,
        src_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedily generate target IDs, excluding the initial BOS token."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        memory, source_valid = self.encode(src_ids, src_valid)
        generated = torch.full(
            (src_ids.size(0), 1), self.config.bos_token_id, dtype=torch.long, device=src_ids.device
        )
        finished = torch.zeros(src_ids.size(0), dtype=torch.bool, device=src_ids.device)
        for _ in range(max_new_tokens):
            logits = self.decode(generated, memory, source_valid)
            next_ids = logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, self.config.eos_token_id), next_ids)
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished |= next_ids.eq(self.config.eos_token_id)
            if finished.all():
                break
        return generated[:, 1:]

    def _valid_mask(self, token_ids: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        if valid is None:
            return token_ids.ne(self.config.pad_token_id)
        if valid.shape != token_ids.shape:
            raise ValueError("validity masks must have the same shape as their token IDs")
        return valid.to(device=token_ids.device, dtype=torch.bool)

    @staticmethod
    def _check_token_ids(token_ids: torch.Tensor, vocab_size: int, name: str) -> None:
        if token_ids.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, sequence_length)")
        if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{name} must contain integer token IDs")
        if token_ids.numel() and (token_ids.min().item() < 0 or token_ids.max().item() >= vocab_size):
            raise ValueError(f"{name} contains IDs outside its vocabulary")


__all__ = ["TransformerConfig", "Transformer", "EncoderLayer", "DecoderLayer"]
