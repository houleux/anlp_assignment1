import math

import torch
import torch.nn as nn
from typing import Optional

from .positional import RotaryPositionalEmbedding

## Scaled Dot Product Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, float('-1e9'))

    attention = torch.softmax(attn_logits, dim=-1)
    output = torch.matmul(attention, v)
    return output, attention


## Helper function for Multi-Head Attention
def expand_mask(mask):
    """Expand 3D/4D masks only. Do NOT pass2D (B, T) padding masks here."""
    assert mask.dim() >= 3, "expand_mask expects>=3D mask. For 2D (B,T) use: mask[:, None, None, :]"
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask

def make_causal_mask(seq_len: int, device) -> torch.Tensor:
    """(1, T, T) lower-triangular mask, 1 = attend. ndim==3 so expand_mask -> (1,1,T,T)."""
    return torch.tril(torch.ones(1, seq_len, seq_len, device=device))

####### MULTI_HEAD ATTENTION #######
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope = rope

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            # Handle 2D (B, T) padding masks explicitly
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # (B, T) -> (B, 1, 1, T)
            else:
                mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        if self.rope is not None:
            q, k = self.rope(q, k)

        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o



####### GROUPED QUERY ATTENTION #######
class GroupedQueryAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_groups, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_groups == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups
        self.rope = rope

        self.kv_proj = nn.Linear(input_dim, 2 * self.num_groups * self.head_dim)
        self.q_proj = nn.Linear(input_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            # Handle 2D (B, T) padding masks explicitly
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # (B, T) -> (B, 1, 1, T)
            else:
                mask = expand_mask(mask)

        kv = self.kv_proj(x)
        kv = kv.reshape(batch_size, seq_length, self.num_groups, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        q = self.q_proj(x)
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        if self.rope is not None:
            q, k = self.rope(q, k)

        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

######## CROSS MULTI-HEAD ATTENTION #######
class CrossMultiHeadAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.kv_proj = nn.Linear(kv_dim, 2 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, query_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, query, kv_input, mask=None, return_attention=False):
        """
        query:    (B, Tq, query_dim)
        kv_input: (B, Tk, kv_dim)
        mask:     (B, Tk) padding mask or (B, Tq, Tk) / (B, H, Tq, Tk) attention mask
        """
        batch_size, q_len, _ = query.size()
        kv_len = kv_input.size(1)
        if mask is not None:
            # Handle 2D (B, src_len) padding masks explicitly
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # (B, src_len) -> (B, 1, 1, src_len)
            else:
                mask = expand_mask(mask)

        q = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv_input).reshape(batch_size, kv_len, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, q_len, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


    
class CrossGroupedQueryAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, embed_dim, num_heads, num_groups):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_groups == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_groups

        self.kv_proj = nn.Linear(kv_dim, 2 * self.num_groups * self.head_dim)
        self.q_proj = nn.Linear(query_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, query_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, query, kv_input, mask=None, return_attention=False):
        batch_size, q_len, _ = query.size()
        kv_len = kv_input.size(1)
        if mask is not None:
            # Handle 2D (B, src_len) padding masks explicitly
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # (B, src_len) -> (B, 1, 1, src_len)
            else:
                mask = expand_mask(mask)

        kv = self.kv_proj(kv_input)
        kv = kv.reshape(batch_size, kv_len, self.num_groups, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        q = self.q_proj(query)
        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, q_len, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o
