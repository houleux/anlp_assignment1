"""
Test script for Dataset classes
Validates TokenizedSeqDataset and ByteSeqDataset with their collate functions
"""

import sys
sys.path.insert(0, 'src')

import torch
from models.bpe import BPETokenizer
from dataset import (
    TokenizedSeqDataset, 
    ByteSeqDataset,
    SpecialTokens,
    collate_fn_tokenized,
    collate_fn_bytes,
    create_dataloaders,
    create_byte_dataloaders
)


def test_special_tokens():
    """Test special token definitions"""
    print("="*70)
    print("TEST 1: SPECIAL TOKENS")
    print("="*70)
    
    tokens = SpecialTokens.get_all()
    print(f"\nSpecial tokens defined:")
    for name, token_id in tokens.items():
        print(f"  <{name}> = {token_id}")
    
    # Validate: special tokens must be outside BPE vocab range [0, 1024)
    assert SpecialTokens.PAD_ID == 1024, "PAD must be 1024 (outside vocab range)"
    assert SpecialTokens.BOS_ID == 1026, "BOS must be 1026"
    assert SpecialTokens.EOS_ID == 1027, "EOS must be 1027"
    
    print("\n✓ Special tokens validation passed (no collision with vocabulary)")


def test_tokenized_dataset():
    """Test TokenizedSeqDataset"""
    print("\n" + "="*70)
    print("TEST 2: TOKENIZED DATASET")
    print("="*70)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    encoder = BPETokenizer.load("tokenizers/encoder.json")
    decoder = BPETokenizer.load("tokenizers/decoder.json")
    
    # Create dataset
    print("\nCreating TokenizedSeqDataset (train split)...")
    dataset = TokenizedSeqDataset(
        split_name='train',
        src_tokenizer=encoder,
        tgt_tokenizer=decoder,
        max_src_len=1024,
        max_tgt_len=640,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Inspect a few samples
    print(f"\nSample inspection (first 3):")
    for i in range(3):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    Source length: {sample['src_len']} tokens")
        print(f"    Target length: {sample['tgt_len']} tokens")
        print(f"    Source IDs (first 10): {sample['src_ids'][:10]}")
        print(f"    Target IDs (first 10): {sample['tgt_ids'][:10]}")
        
        # Validate special tokens
        assert sample['src_ids'][0] == SpecialTokens.BOS_ID, "First token should be BOS"
        assert sample['src_ids'][-1] == SpecialTokens.EOS_ID, "Last token should be EOS"
        assert sample['tgt_ids'][0] == SpecialTokens.BOS_ID, "First token should be BOS"
        assert sample['tgt_ids'][-1] == SpecialTokens.EOS_ID, "Last token should be EOS"
    
    print("\n✓ TokenizedSeqDataset validation passed")
    return dataset


def test_byte_dataset():
    """Test ByteSeqDataset"""
    print("\n" + "="*70)
    print("TEST 3: BYTE DATASET (BLT)")
    print("="*70)
    
    # Create dataset
    print("\nCreating ByteSeqDataset (train split)...")
    dataset = ByteSeqDataset(
        split_name='train',
        max_src_len=1024,
        max_tgt_len=640,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Inspect a few samples
    print(f"\nSample inspection (first 3):")
    for i in range(3):
        sample = dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    Source length: {sample['src_len']} bytes")
        print(f"    Target length: {sample['tgt_len']} bytes")
        print(f"    Source bytes (first 10): {sample['src_bytes'][:10]}")
        print(f"    Target bytes (first 10): {sample['tgt_bytes'][:10]}")
        
        # Validate special bytes
        assert sample['src_bytes'][0] == dataset.BOS_BYTE, "First byte should be BOS"
        assert sample['src_bytes'][-1] == dataset.EOS_BYTE, "Last byte should be EOS"
        assert sample['tgt_bytes'][0] == dataset.BOS_BYTE, "First byte should be BOS"
        assert sample['tgt_bytes'][-1] == dataset.EOS_BYTE, "Last byte should be EOS"
    
    print("\n✓ ByteSeqDataset validation passed")
    return dataset


def test_collate_tokenized(dataset):
    """Test collate function for tokenized data"""
    print("\n" + "="*70)
    print("TEST 4: TOKENIZED COLLATE FUNCTION")
    print("="*70)
    
    # Create a small batch
    batch = [dataset[i] for i in range(5)]
    
    print(f"\nBatch size: {len(batch)}")
    print(f"Sequence lengths (src): {[item['src_len'] for item in batch]}")
    print(f"Sequence lengths (tgt): {[item['tgt_len'] for item in batch]}")
    
    # Collate
    collated = collate_fn_tokenized(batch)
    
    print(f"\nCollated tensors:")
    print(f"  encoder_input shape: {collated['encoder_input'].shape}")
    print(f"  decoder_input shape: {collated['decoder_input'].shape}")
    print(f"  encoder_mask shape: {collated['encoder_mask'].shape}")
    print(f"  decoder_mask shape: {collated['decoder_mask'].shape}")
    print(f"  decoder_padding_mask shape: {collated['decoder_padding_mask'].shape}")
    print(f"  cross_mask shape: {collated['cross_mask'].shape}")
    
    # Validate shapes
    batch_size = len(batch)
    max_src_len = max(item['src_len'] for item in batch)
    max_tgt_len = max(item['tgt_len'] for item in batch)
    
    assert collated['encoder_input'].shape == (batch_size, max_src_len)
    assert collated['decoder_input'].shape == (batch_size, max_tgt_len)
    assert collated['encoder_mask'].shape == (batch_size, max_src_len)
    assert collated['decoder_mask'].shape == (batch_size, max_tgt_len, max_tgt_len)
    assert collated['decoder_padding_mask'].shape == (batch_size, max_tgt_len)
    
    # Check causal mask is lower triangular
    for i in range(batch_size):
        causal = collated['decoder_mask'][i]
        # Check if padded positions are masked
        for j in range(max_tgt_len):
            if collated['decoder_padding_mask'][i, j] == 0:  # Padding
                assert (causal[j, :] == 0).all(), "Padded positions should be masked"
                assert (causal[:, j] == 0).all(), "Padded positions should be masked"
    
    print("\n✓ Tokenized collate function validation passed")
    
    # Print sample
    print(f"\nSample from batch:")
    print(f"  Encoder input (first sequence):")
    print(f"    {collated['encoder_input'][0][:20]}")
    print(f"  Decoder input (first sequence):")
    print(f"    {collated['decoder_input'][0][:20]}")
    print(f"  Encoder mask (first sequence):")
    print(f"    {collated['encoder_mask'][0][:20]}")


def test_collate_bytes(dataset):
    """Test collate function for byte data"""
    print("\n" + "="*70)
    print("TEST 5: BYTE COLLATE FUNCTION")
    print("="*70)
    
    # Create a small batch
    batch = [dataset[i] for i in range(5)]
    
    print(f"\nBatch size: {len(batch)}")
    print(f"Sequence lengths (src): {[item['src_len'] for item in batch]}")
    print(f"Sequence lengths (tgt): {[item['tgt_len'] for item in batch]}")
    
    # Collate
    collated = collate_fn_bytes(batch)
    
    print(f"\nCollated tensors:")
    print(f"  encoder_input shape: {collated['encoder_input'].shape}")
    print(f"  decoder_input shape: {collated['decoder_input'].shape}")
    print(f"  encoder_mask shape: {collated['encoder_mask'].shape}")
    print(f"  decoder_mask shape: {collated['decoder_mask'].shape}")
    
    # Validate shapes
    batch_size = len(batch)
    max_src_len = max(item['src_len'] for item in batch)
    max_tgt_len = max(item['tgt_len'] for item in batch)
    
    assert collated['encoder_input'].shape == (batch_size, max_src_len)
    assert collated['decoder_input'].shape == (batch_size, max_tgt_len)
    assert collated['encoder_mask'].shape == (batch_size, max_src_len)
    assert collated['decoder_mask'].shape == (batch_size, max_tgt_len, max_tgt_len)
    
    print("\n✓ Byte collate function validation passed")
    
    # Print sample
    print(f"\nSample from batch:")
    print(f"  Encoder input (first sequence):")
    print(f"    {collated['encoder_input'][0][:20]}")
    print(f"  Decoder input (first sequence):")
    print(f"    {collated['decoder_input'][0][:20]}")


def test_dataloaders():
    """Test DataLoader creation"""
    print("\n" + "="*70)
    print("TEST 6: DATALOADERS")
    print("="*70)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    encoder = BPETokenizer.load("tokenizers/encoder.json")
    decoder = BPETokenizer.load("tokenizers/decoder.json")
    
    # Create tokenized dataloaders
    print("\nCreating tokenized dataloaders...")
    token_loaders = create_dataloaders(
        src_tokenizer=encoder,
        tgt_tokenizer=decoder,
        batch_size=32,
        max_src_len=1024,
        max_tgt_len=640,
    )
    
    print(f"Dataloaders created: {list(token_loaders.keys())}")
    
    # Test iteration
    print("\nTesting tokenized dataloader iteration...")
    train_loader = token_loaders['train']
    batch_count = 0
    token_count = 0
    
    for batch in train_loader:
        batch_count += 1
        token_count += batch['encoder_input'].shape[0]
        
        if batch_count == 1:
            print(f"\nFirst batch stats:")
            print(f"  Batch size: {batch['encoder_input'].shape[0]}")
            print(f"  Encoder max length: {batch['encoder_input'].shape[1]}")
            print(f"  Decoder max length: {batch['decoder_input'].shape[1]}")
            print(f"  Encoder mask sum: {batch['encoder_mask'].sum():.0f}")
            print(f"  Decoder padding mask sum: {batch['decoder_padding_mask'].sum():.0f}")
        
        if batch_count >= 3:
            break
    
    print(f"\nProcessed {batch_count} batches ({token_count} sequences)")
    
    # Create byte dataloaders
    print("\n\nCreating byte-level dataloaders...")
    byte_loaders = create_byte_dataloaders(
        batch_size=32,
        max_src_len=1024,
        max_tgt_len=640,
    )
    
    print(f"Dataloaders created: {list(byte_loaders.keys())}")
    
    # Test iteration
    print("\nTesting byte dataloader iteration...")
    train_loader = byte_loaders['train']
    batch_count = 0
    
    for batch in train_loader:
        batch_count += 1
        
        if batch_count == 1:
            print(f"\nFirst batch stats:")
            print(f"  Batch size: {batch['encoder_input'].shape[0]}")
            print(f"  Encoder max length: {batch['encoder_input'].shape[1]}")
            print(f"  Decoder max length: {batch['decoder_input'].shape[1]}")
        
        if batch_count >= 3:
            break
    
    print(f"\nProcessed {batch_count} batches")
    print("\n✓ Dataloaders validation passed")


if __name__ == "__main__":
    test_special_tokens()
    dataset_token = test_tokenized_dataset()
    dataset_byte = test_byte_dataset()
    test_collate_tokenized(dataset_token)
    test_collate_bytes(dataset_byte)
    test_dataloaders()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
