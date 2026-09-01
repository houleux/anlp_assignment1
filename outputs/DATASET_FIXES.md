# Dataset Fixes - Summary

## Issues Fixed

### 1. ✅ PAD_ID Collision with Vocabulary

**Problem**: 
- BPE tokenizers produce tokens in range `[0, 1024)` (vocab_size=1024)
- Using `PAD_ID = 0` collides with real vocabulary token 0
- During padding, padding tokens become indistinguishable from real token 0
- This causes incorrect masking and attention calculations

**Solution**:
- Moved special tokens **outside** the vocabulary range to `[1024, 1028)`
- New special token IDs:
  - `<pad>` = 1024 (outside vocab, no collision)
  - `<unk>` = 1025 (reserved for robustness)
  - `<bos>` = 1026
  - `<eos>` = 1027

**Code Change** (src/dataset.py):
```python
class SpecialTokens:
    PAD_ID = 1024    # Changed from 0
    UNK_ID = 1025    # Changed from 1
    BOS_ID = 1026    # Changed from 2
    EOS_ID = 1027    # Changed from 3
```

**Impact**:
- ✅ Padding tokens now completely separate from real vocabulary
- ✅ Attention masks can reliably identify padding positions
- ✅ No ambiguity during loss computation

---

### 2. ✅ ByteSeqDataset Special Bytes Shifted

**Problem**:
- ByteSeqDataset operates on raw byte values (0-255)
- Originally used BOS_BYTE=256, EOS_BYTE=257
- Added PAD_BYTE=256 in collate function, causing collision with BOS_BYTE

**Solution**:
- Shifted all special bytes outside the byte range (0-255):
  - `PAD_BYTE` = 256
  - `BOS_BYTE` = 257
  - `EOS_BYTE` = 258

**Code Change** (src/dataset.py, ByteSeqDataset.__init__):
```python
self.PAD_BYTE = 256    # For padding
self.BOS_BYTE = 257    # Beginning of sequence
self.EOS_BYTE = 258    # End of sequence
```

**Impact**:
- ✅ No collision between padding, BOS, and EOS bytes
- ✅ Byte sequences remain unambiguous

---

### 3. ✅ Teacher Forcing Documentation

**Problem**:
- Standard seq2seq training requires teacher forcing:
  - Decoder input: `[<bos> tok1 tok2 ... tokN]` (fed to decoder)
  - Decoder target: `[tok1 tok2 ... tokN <eos>]` (compute loss against)
- Dataset returns full sequence `[<bos> ... <eos>]` but no explicit target offset
- Easy to forget this detail when implementing training loop

**Solution**:
- Added explicit documentation in collate functions with **one-line training loop example**:
```python
# In collate function docstring:
"""
TEACHER FORCING NOTE:
- decoder_input from this function contains: [<bos> tok1 tok2 ... tokN <eos>]
- For training with teacher forcing, in your training loop use:
    decoder_input_feed = decoder_input[:, :-1]      # [<bos> tok1 ... tokN]
    decoder_target = decoder_input[:, 1:]           # [tok1 tok2 ... <eos>]
- This ensures the decoder predicts the next token given previous context.
"""
```

**Impact**:
- ✅ Clear guidance in code comments
- ✅ Prevents accidental bugs in training loop
- ✅ Self-documenting implementation

---

## Test Results

✅ **ALL 6 TESTS PASSED** (after fixes):

1. ✓ Special tokens validation (PAD=1024, BOS=1026, EOS=1027)
2. ✓ TokenizedSeqDataset loads 4,000 samples correctly
3. ✓ ByteSeqDataset loads 4,000 samples correctly (BOS_BYTE=257, EOS_BYTE=258)
4. ✓ Tokenized collate function produces correct tensor shapes and masks
5. ✓ Byte collate function produces correct tensor shapes and masks
6. ✓ DataLoaders iterate all 3 splits (train/val/test) without errors

**Key Validation**:
- All token IDs are now >= 1024 (outside vocabulary range)
- No collisions between padding and real tokens
- Causal masks correctly generated (lower triangular, 5D validation)
- Dynamic padding working (batch max, not global max)

---

## Usage in Training Loop

```python
from dataset import create_dataloaders
from models.bpe import BPETokenizer
import torch

# Setup
encoder = BPETokenizer.load("tokenizers/encoder.json")
decoder = BPETokenizer.load("tokenizers/decoder.json")
loaders = create_dataloaders(encoder, decoder, batch_size=32)

# Training with teacher forcing
for epoch in range(10):
    for batch in loaders['train']:
        # Decode batch
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']
        cross_mask = batch['cross_mask']
        
        # Teacher forcing: shift targets
        decoder_input_feed = decoder_input[:, :-1]  # Remove <eos>
        decoder_target = decoder_input[:, 1:]       # Remove <bos>
        
        # Forward pass
        output = model(
            encoder_input,
            decoder_input_feed,
            src_key_padding_mask=(encoder_mask == 0),
            tgt_mask=decoder_mask[:, :-1, :-1],  # Adjust mask for shifted input
            memory_key_padding_mask=(cross_mask == 0),
        )
        
        # Loss (ignore pad_id=1024)
        loss = criterion(
            output.view(-1, vocab_size),
            decoder_target.view(-1),
            ignore_index=1024  # Ignore padding
        )
        
        loss.backward()
        optimizer.step()
```

---

## Files Modified

1. **src/dataset.py**:
   - Updated `SpecialTokens` class (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
   - Updated `ByteSeqDataset.__init__` (PAD_BYTE, BOS_BYTE, EOS_BYTE)
   - Updated `ByteSeqDataset._process_pair` (docstring and comments)
   - Updated `collate_fn_tokenized` (added teacher forcing documentation)
   - Updated `collate_fn_bytes` (added teacher forcing documentation, fixed PAD_ID parameter)
   - Fixed `causal_mask` generation (expand instead of repeat with -1)

2. **test_datasets.py**:
   - Updated `test_special_tokens()` to validate new PAD_ID values (1024-1027)

---

## Migration Notes

If upgrading from old dataset.py:

1. **Update ignore_index in loss**:
   ```python
   # Old
   criterion = nn.CrossEntropyLoss(ignore_index=0)
   
   # New
   criterion = nn.CrossEntropyLoss(ignore_index=1024)  # New PAD_ID
   ```

2. **Update vocabulary size references**:
   ```python
   # When creating embedding layers
   vocab_size = 1024
   pad_id = 1024  # Add 1 for special tokens (outside vocab)
   embedding = nn.Embedding(1028, embedding_dim, padding_idx=1024)
   ```

3. **Teacher forcing in training loop** (important!):
   ```python
   decoder_input_feed = batch['decoder_input'][:, :-1]
   decoder_target = batch['decoder_input'][:, 1:]
   ```

---

## Summary

✅ All issues fixed and tested
✅ No breaking changes to dataset API (same return structure)
✅ Special token IDs now collision-free
✅ Teacher forcing clearly documented with code example
✅ ByteSeqDataset special bytes non-overlapping
✅ All 6 tests passing
