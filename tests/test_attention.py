"""
Unit tests for attention.py modules
Tests mask handling, shape correctness, and basic functionality
"""

import torch
import pytest
from src.models.attention import (
    scaled_dot_product_attention,
    expand_mask,
    make_causal_mask,
    MultiHeadAttention,
    GroupedQueryAttention,
    CrossMultiHeadAttention,
    CrossGroupedQueryAttention,
)
from src.models.positional import RotaryPositionalEmbedding


#=====================================================================
# Test expand_mask and make_causal_mask
# =====================================================================

def test_expand_mask_3d():
    """Test expand_mask with 3D input (B, T, T)"""
    mask = torch.ones(2, 10, 10)
    expanded = expand_mask(mask)
    assert expanded.shape == (2, 1, 10, 10), f"Expected (2, 1, 10, 10), got {expanded.shape}"


def test_expand_mask_4d():
    """Test expand_mask with 4D input (already expanded)"""
    mask = torch.ones(2, 8, 10, 10)
    expanded = expand_mask(mask)
    assert expanded.shape == (2, 8, 10, 10), f"Expected (2, 8, 10, 10), got {expanded.shape}"


def test_expand_mask_rejects_2d():
    """Test that expand_mask rejects 2D masks with clear error"""
    mask = torch.ones(2, 10)
    with pytest.raises(AssertionError, match="expand_mask expects>=3D"):
        expand_mask(mask)


def test_make_causal_mask():
    """Test causal mask generation"""
    seq_len = 5
    mask = make_causal_mask(seq_len, 'cpu')
    
    assert mask.shape == (1, seq_len, seq_len)
    # Check lower triangular property
    expected = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],], dtype=torch.float32)
    
    assert torch.allclose(mask[0], expected), "Causal mask is not lower triangular"


# =====================================================================
# Test scaled_dot_product_attention
# =====================================================================

def test_scaled_dot_product_attention_no_mask():
    """Test attention without mask"""
    batch_size, num_heads, seq_len, head_dim = 2, 4, 10, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    output, attn = scaled_dot_product_attention(q, k, v, mask=None)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)
    # Check attention weights sum to 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-6)


def test_scaled_dot_product_attention_with_mask():
    """Test attention with causal mask"""
    batch_size, num_heads, seq_len, head_dim = 2, 4, 5, 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Create causal mask (lower triangular)
    mask = torch.tril(torch.ones(batch_size, num_heads, seq_len, seq_len))
    
    output, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Check that masked positions have near-zero attention
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn[0, 0, i, j] < 1e-6, f"Position ({i},{j}) should be masked"


# =====================================================================
# Test MultiHeadAttention
# =====================================================================

def test_multihead_attention_basic():
    """Test MultiHeadAttention forward pass"""
    batch_size, seq_len, input_dim = 2, 10, 512
    embed_dim, num_heads = 512, 8
    
    mha = MultiHeadAttention(input_dim, embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = mha(x)
    
    assert output.shape == (batch_size, seq_len, input_dim)


def test_multihead_attention_with_2d_padding_mask():
    """Test MultiHeadAttention with 2D (B, T) padding mask"""
    batch_size, seq_len, input_dim = 2, 10, 512
    embed_dim, num_heads = 512, 8
    
    mha = MultiHeadAttention(input_dim, embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create 2D padding mask (B, T)
    padding_mask = torch.ones(batch_size, seq_len)
    padding_mask[0, 7:] = 0  # Mask last 3 positions of first sample
    padding_mask[1, 9:] = 0  # Mask last position of second sample
    
    # Should not raise an error
    output = mha(x, mask=padding_mask)
    
    assert output.shape == (batch_size, seq_len, input_dim)


def test_multihead_attention_with_3d_causal_mask():
    """Test MultiHeadAttention with 3D causal mask"""
    batch_size, seq_len, input_dim = 2, 10, 512
    embed_dim, num_heads = 512, 8
    
    mha = MultiHeadAttention(input_dim, embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create 3D causal mask (B, T, T)
    causal_mask = make_causal_mask(seq_len, 'cpu').expand(batch_size, seq_len, seq_len)
    
    output = mha(x, mask=causal_mask)
    
    assert output.shape == (batch_size, seq_len, input_dim)


def test_multihead_attention_return_attention():
    """Test MultiHeadAttention returns attention weights"""
    batch_size, seq_len, input_dim = 2, 10, 512
    embed_dim, num_heads = 512, 8
    
    mha = MultiHeadAttention(input_dim, embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output, attn = mha(x, return_attention=True)
    
    assert output.shape == (batch_size, seq_len, input_dim)
    assert attn.shape == (batch_size, num_heads, seq_len, seq_len)


# =====================================================================
# Test GroupedQueryAttention
# =====================================================================

def test_grouped_query_attention_basic():
    """Test GroupedQueryAttention forward pass"""
    batch_size, seq_len, input_dim = 2, 10, 512
    embed_dim, num_heads, num_groups = 512, 8, 4
    
    gqa = GroupedQueryAttention(input_dim, embed_dim, num_heads, num_groups)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = gqa(x)
    
    assert output.shape == (batch_size, seq_len, input_dim)


def test_grouped_query_attention_with_2d_mask():
    """Test GroupedQueryAttention with 2D padding mask"""
    batch_size, seq_len, input_dim = 2, 10, 512
    embed_dim, num_heads, num_groups = 512, 8, 4
    
    gqa = GroupedQueryAttention(input_dim, embed_dim, num_heads, num_groups)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create 2D padding mask
    padding_mask = torch.ones(batch_size, seq_len)
    padding_mask[0, 7:] = 0
    output = gqa(x, mask=padding_mask)
    
    assert output.shape == (batch_size, seq_len, input_dim)


# =====================================================================
# Test CrossMultiHeadAttention
# =====================================================================

def test_cross_attention_basic():
    """Test CrossMultiHeadAttention forward pass"""
    batch_size, q_len, kv_len = 2, 10, 15
    query_dim, kv_dim, embed_dim, num_heads = 512, 512, 512, 8
    
    cross_attn = CrossMultiHeadAttention(query_dim, kv_dim, embed_dim, num_heads)
    
    query = torch.randn(batch_size, q_len, query_dim)
    kv_input = torch.randn(batch_size, kv_len, kv_dim)
    
    output = cross_attn(query, kv_input)
    
    assert output.shape == (batch_size, q_len, query_dim)


def test_cross_attention_with_2d_encoder_mask():
    """Test CrossMultiHeadAttention with 2D (B, src_len) encoder padding mask"""
    batch_size, q_len, kv_len = 2, 10, 15
    query_dim, kv_dim, embed_dim, num_heads = 512, 512, 512, 8
    
    cross_attn = CrossMultiHeadAttention(query_dim, kv_dim, embed_dim, num_heads)
    
    query = torch.randn(batch_size, q_len, query_dim)
    kv_input = torch.randn(batch_size, kv_len, kv_dim)
    
    # Create 2D encoder mask (B, src_len) - THIS IS THE BUG FIX TEST
    encoder_mask = torch.ones(batch_size, kv_len)
    encoder_mask[0, 12:] = 0  # Mask last 3 positions of encoder
    encoder_mask[1, 14:] = 0  # Mask last position of encoder
    
    # Should not raise shape errors
    output = cross_attn(query, kv_input, mask=encoder_mask)
    
    assert output.shape == (batch_size, q_len, query_dim)


def test_cross_attention_return_attention():
    """Test CrossMultiHeadAttention returns attention weights"""
    batch_size, q_len, kv_len = 2, 10, 15
    query_dim, kv_dim, embed_dim, num_heads = 512, 512, 512, 8
    
    cross_attn = CrossMultiHeadAttention(query_dim, kv_dim, embed_dim, num_heads)
    
    query = torch.randn(batch_size, q_len, query_dim)
    kv_input = torch.randn(batch_size, kv_len, kv_dim)
    
    output, attn = cross_attn(query, kv_input, return_attention=True)
    
    assert output.shape == (batch_size, q_len, query_dim)
    assert attn.shape == (batch_size, num_heads, q_len, kv_len)


# =====================================================================
# Test CrossGroupedQueryAttention
# =====================================================================

def test_cross_grouped_query_attention_basic():
    """Test CrossGroupedQueryAttention forward pass"""
    batch_size, q_len, kv_len = 2, 10, 15
    query_dim, kv_dim, embed_dim = 512, 512, 512
    num_heads, num_groups = 8, 4
    
    cross_gqa = CrossGroupedQueryAttention(query_dim, kv_dim, embed_dim, num_heads, num_groups)
    
    query = torch.randn(batch_size, q_len, query_dim)
    kv_input = torch.randn(batch_size, kv_len, kv_dim)
    
    output = cross_gqa(query, kv_input)
    
    assert output.shape == (batch_size, q_len, query_dim)


def test_cross_grouped_query_attention_with_2d_mask():
    """Test CrossGroupedQueryAttention with 2D encoder mask"""
    batch_size, q_len, kv_len = 2, 10, 15
    query_dim, kv_dim, embed_dim = 512, 512, 512
    num_heads, num_groups = 8, 4
    
    cross_gqa = CrossGroupedQueryAttention(query_dim, kv_dim, embed_dim, num_heads, num_groups)
    
    query = torch.randn(batch_size, q_len, query_dim)
    kv_input = torch.randn(batch_size, kv_len, kv_dim)
    
    # Create 2D encoder mask
    encoder_mask = torch.ones(batch_size, kv_len)
    encoder_mask[0, 12:] = 0
    
    output = cross_gqa(query, kv_input, mask=encoder_mask)
    
    assert output.shape == (batch_size, q_len, query_dim)


# =====================================================================
# Integration test: Full encoder-decoder scenario
# =====================================================================

def test_full_encoder_decoder_scenario():
    """Test realistic encoder-decoder scenario with proper mask shapes from dataset.py"""
    batch_size = 4
    src_len, tgt_len = 20, 15
    model_dim = 512
    num_heads = 8
    
    # Simulate encoder output
    encoder_output = torch.randn(batch_size, src_len, model_dim)
    decoder_input = torch.randn(batch_size, tgt_len, model_dim)
    
    # Create masks as they come from collate_fn_tokenized in dataset.py
    # encoder_mask: (B, src_len) - padding mask
    encoder_mask = torch.ones(batch_size, src_len)
    encoder_mask[0, 15:] = 0  # First sample has padding
    encoder_mask[1, 18:] = 0  # Second sample has padding
    
    # decoder_mask: (B, tgt_len, tgt_len) - causal + padding
    decoder_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).expand(batch_size, tgt_len, tgt_len)
    # cross_mask: (B, src_len) - same as encoder_mask
    cross_mask = encoder_mask
    
    # Decoder self-attention with causal mask (3D)
    decoder_self_attn = MultiHeadAttention(model_dim, model_dim, num_heads)
    decoder_hidden = decoder_self_attn(decoder_input, mask=decoder_mask)
    assert decoder_hidden.shape == (batch_size, tgt_len, model_dim)
    
    # Cross-attention with 2D encoder padding mask
    cross_attn = CrossMultiHeadAttention(model_dim, model_dim, model_dim, num_heads)
    decoder_output = cross_attn(decoder_hidden, encoder_output, mask=cross_mask)
    assert decoder_output.shape == (batch_size, tgt_len, model_dim)
    
    print("✓ Full encoder-decoder scenario passed!")


# =====================================================================
# Run all tests
# =====================================================================

if __name__ == "__main__":
    print("Running attention module tests...\n")
    
    # Test expand_mask and utilities
    print("Testing expand_mask and utilities...")
    test_expand_mask_3d()
    test_expand_mask_4d()
    test_expand_mask_rejects_2d()
    test_make_causal_mask()
    print("✓ expand_mask tests passed\n")
    
    # Test scaled_dot_product_attention
    print("Testing scaled_dot_product_attention...")
    test_scaled_dot_product_attention_no_mask()
    test_scaled_dot_product_attention_with_mask()
    print("✓ scaled_dot_product_attention tests passed\n")
    
    # Test MultiHeadAttention
    print("Testing MultiHeadAttention...")
    test_multihead_attention_basic()
    test_multihead_attention_with_2d_padding_mask()
    test_multihead_attention_with_3d_causal_mask()
    test_multihead_attention_return_attention()
    print("✓ MultiHeadAttention tests passed\n")
    
    # Test GroupedQueryAttention
    print("Testing GroupedQueryAttention...")
    test_grouped_query_attention_basic()
    test_grouped_query_attention_with_2d_mask()
    print("✓ GroupedQueryAttention tests passed\n")
    
    # Test CrossMultiHeadAttention
    print("Testing CrossMultiHeadAttention...")
    test_cross_attention_basic()
    test_cross_attention_with_2d_encoder_mask()
    test_cross_attention_return_attention()
    print("✓ CrossMultiHeadAttention tests passed\n")
    
    # Test CrossGroupedQueryAttention
    print("Testing CrossGroupedQueryAttention...")
    test_cross_grouped_query_attention_basic()
    test_cross_grouped_query_attention_with_2d_mask()
    print("✓ CrossGroupedQueryAttention tests passed\n")
    
    # Integration test
    print("Testing full encoder-decoder scenario...")
    test_full_encoder_decoder_scenario()
    print("✓ Integration test passed\n")
    
    print("=" * 50)
    print("ALL TESTS PASSED!✓")
    print("=" * 50)
