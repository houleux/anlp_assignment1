# Dataset Implementation Guide

## Overview

The `src/dataset.py` module implements two dataset classes for training transformer models on the cipher ↔ plaintext task:

1. **`TokenizedSeqDataset`** — Uses BPE tokenizers (C1-C4 configurations)
2. **`ByteSeqDataset`** — Uses raw bytes (C5 / BLT configuration)

Both classes include custom collate functions that:
- Dynamically pad sequences to batch's own max length (not global max)
- Create proper attention masks (padding, causal, cross-attention)
- Preserve special tokens (<pad>, <bos>, <eos>, <unk>)

---

## Special Tokens

All special tokens are defined in the `SpecialTokens` class:

```python
<pad> = 0      # Padding token
<unk> = 1      # Unknown token (BPE never produces this)
<bos> = 2      # Beginning of sequence
<eos> = 3      # End of sequence
```

These IDs are **consistent across all tokenizers** and configurations.

### Why These IDs?

- **<pad>=0**: Standard padding ID; unused by BPE base vocabulary
- **<unk>=1**: Reserved for robustness (BPE never generates unknown tokens by design)
- **<bos>=2, <eos>=3**: Clearly separated from base vocabulary (256+)

---

## `TokenizedSeqDataset` (for C1-C4)

### Purpose
Loads cipher/plaintext pairs, encodes them using separate BPE tokenizers, and prepares them for transformer training.

### Process

1. **Load splits**: Reads pickled cipher bytes and plaintext from split files
2. **Encode**:
   - Decode cipher bytes → string (Latin-1)
   - Tokenize with src_tokenizer (encoder) → token IDs
   - Tokenize with tgt_tokenizer (decoder) → token IDs
3. **Add special tokens**:
   - Prepend `<bos>` (ID 2)
   - Append `<eos>` (ID 3)
4. **Truncate**:
   - If length > `max_src_len`: truncate to `max_src_len-1`, then append `<eos>`
   - If length > `max_tgt_len`: truncate to `max_tgt_len-1`, then append `<eos>`

### Example

```python
from models.bpe import BPETokenizer
from dataset import TokenizedSeqDataset

# Load tokenizers
encoder = BPETokenizer.load("tokenizers/encoder.json")
decoder = BPETokenizer.load("tokenizers/decoder.json")

# Create dataset
train_dataset = TokenizedSeqDataset(
    split_name='train',
    src_tokenizer=encoder,
    tgt_tokenizer=decoder,
    max_src_len=1024,
    max_tgt_len=640,
)

# Access a sample
sample = train_dataset[0]
print(sample['src_ids'])      # [2, 19, 33, 46, ..., 3]  (BOS...EOS)
print(sample['tgt_ids'])      # [2, 82, 984, 116, ..., 3]
print(sample['src_len'])      # 564
print(sample['tgt_len'])      # 306
```

### Dataset Statistics (Train Split)

| Metric | Value |
|--------|-------|
| Total samples | 4,000 |
| Encoder tokens (mean) | 407.6 |
| Decoder tokens (mean) | 214.7 |
| Encoder range | 15-1,703 tokens |
| Decoder range | 8-911 tokens |

---

## `ByteSeqDataset` (for C5 / BLT)

### Purpose
Loads cipher/plaintext pairs as raw bytes (token-free processing) for byte-level transformer training.

### Process

1. **Load splits**: Reads pickled cipher bytes and plaintext
2. **Encode**:
   - Cipher: keep as-is (list of byte values 0-255)
   - Plaintext: encode to UTF-8 bytes
3. **Add special tokens**:
   - Prepend `<bos>` (ID 256)
   - Append `<eos>` (ID 257)
4. **Truncate**:
   - Same as tokenized version

### Example

```python
from dataset import ByteSeqDataset

# Create dataset
train_dataset = ByteSeqDataset(
    split_name='train',
    max_src_len=1024,
    max_tgt_len=640,
)

# Access a sample
sample = train_dataset[0]
print(sample['src_bytes'])    # [256, 19, 33, 46, ..., 257]  (BOS...EOS)
print(sample['tgt_bytes'])    # [256, 82, 111, 98, ...]  UTF-8 bytes
print(sample['src_len'])      # 790
print(sample['tgt_len'])      # 640
```

### Key Difference from Tokenized

- **No vocabulary limit**: Each byte value (0-255) is used directly
- **No compression**: Bytes are not merged (unlike BPE merges)
- **Simplicity**: Direct byte → embedding mapping

---

## Collate Functions

Both datasets use custom collate functions that create padded batches with attention masks.

### Key Features

✅ **Dynamic padding**: Pads to batch's own max length (not global 1024)
✅ **Efficiency**: Reduces GPU memory usage for variable-length sequences
✅ **Proper masking**: Creates attention masks for encoder, decoder, and cross-attention

### `collate_fn_tokenized` (for TokenizedSeqDataset)

Returns a dict with keys:

```python
{
    'encoder_input': torch.Tensor,         # (batch, max_src_len) — token IDs
    'decoder_input': torch.Tensor,         # (batch, max_tgt_len) — token IDs
    'encoder_mask': torch.Tensor,          # (batch, max_src_len) — 1 for real, 0 for pad
    'decoder_mask': torch.Tensor,          # (batch, max_tgt_len, max_tgt_len) — causal
    'decoder_padding_mask': torch.Tensor,  # (batch, max_tgt_len) — 1 for real, 0 for pad
    'cross_mask': torch.Tensor,            # (batch, max_src_len) — for cross-attention
}
```

### `collate_fn_bytes` (for ByteSeqDataset)

Same structure as above, but with byte values instead of token IDs.

### Mask Structures

**Encoder Padding Mask** (1D):
```
[1, 1, 1, 1, 0, 0]  # 1 = real token, 0 = padding
```

**Decoder Causal Mask** (2D, lower-triangular):
```
[[1, 0, 0, 0],      # Query pos 0 can attend to pos 0 only
 [1, 1, 0, 0],      # Query pos 1 can attend to pos 0,1
 [1, 1, 1, 0],      # Query pos 2 can attend to pos 0,1,2
 [0, 0, 0, 0]]      # Query pos 3 is padding (masked out)
```

**Cross-Attention Mask** (1D):
```
[1, 1, 1, 1, 0, 0]  # Same as encoder padding mask
```

### Usage Example

```python
from torch.utils.data import DataLoader
from dataset import TokenizedSeqDataset, collate_fn_tokenized

# Create dataset
dataset = TokenizedSeqDataset(...)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_fn_tokenized,
    shuffle=True,
)

# Iterate
for batch in loader:
    encoder_input = batch['encoder_input']          # (32, seq_len)
    encoder_mask = batch['encoder_mask']            # (32, seq_len)
    decoder_input = batch['decoder_input']          # (32, seq_len)
    decoder_mask = batch['decoder_mask']            # (32, seq_len, seq_len)
    
    # Pass to model
    output = model(encoder_input, decoder_input, 
                   encoder_mask, decoder_mask)
```

---

## Helper Functions

### `create_dataloaders()`

Creates train/val/test dataloaders for tokenized data.

```python
from models.bpe import BPETokenizer
from dataset import create_dataloaders

encoder = BPETokenizer.load("tokenizers/encoder.json")
decoder = BPETokenizer.load("tokenizers/decoder.json")

loaders = create_dataloaders(
    src_tokenizer=encoder,
    tgt_tokenizer=decoder,
    batch_size=32,
    max_src_len=1024,
    max_tgt_len=640,
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### `create_byte_dataloaders()`

Creates train/val/test dataloaders for byte-level data.

```python
from dataset import create_byte_dataloaders

loaders = create_byte_dataloaders(
    batch_size=32,
    max_src_len=1024,
    max_tgt_len=640,
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

---

## Dataset Characteristics

### TokenizedSeqDataset (Train)
- **Total sequences**: 4,000
- **Encoder tokens (mean)**: 407.6 (range: 15-1,703)
- **Decoder tokens (mean)**: 214.7 (range: 8-911)
- **Compression ratio**: 1.4-2.6x (bytes/chars → tokens)

### ByteSeqDataset (Train)
- **Total sequences**: 4,000
- **Encoder bytes (mean)**: 580 (range: 21-2,418)
- **Decoder bytes (mean)**: 580 (range: 21-2,418)
- **No compression**: Bytes pass through unchanged

---

## Validation Results

✅ **All 6 tests passed**:

1. ✓ Special tokens defined correctly
2. ✓ TokenizedSeqDataset loads and processes data
3. ✓ ByteSeqDataset loads and processes data
4. ✓ Collate function for tokenized data produces correct shapes
5. ✓ Collate function for bytes produces correct shapes
6. ✓ DataLoaders iterate correctly

---

## Usage in Training Loop

```python
from models.bpe import BPETokenizer
from dataset import create_dataloaders
import torch

# Setup
encoder = BPETokenizer.load("tokenizers/encoder.json")
decoder = BPETokenizer.load("tokenizers/decoder.json")
loaders = create_dataloaders(encoder, decoder, batch_size=32)

# Training
for epoch in range(10):
    for batch in loaders['train']:
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        cross_mask = batch['cross_mask'].to(device)
        
        # Forward pass
        output = model(
            encoder_input,
            decoder_input,
            src_key_padding_mask=(encoder_mask == 0),
            tgt_mask=decoder_mask,
            memory_key_padding_mask=(cross_mask == 0),
        )
        
        # Compute loss and backprop
        loss = criterion(output, decoder_input)
        loss.backward()
        optimizer.step()
```

---

## Next Steps

The dataset classes are now ready for:
1. **Model training** (C1-C5 configurations)
2. **Evaluation** on val/test splits
3. **Metric computation** (accuracy, BLEU, etc.)
