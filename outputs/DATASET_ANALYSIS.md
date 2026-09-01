# Dataset Analysis Report

## Overview
- **Total samples:** 5,000
- **Suggested split:** 4,000 (train) / 500 (val) / 500 (test)
- **Dataset type:** Binary encryption task (plaintext → ciphertext)
- **Source:** Brown Corpus English text

---

## Ciphertext (Binary Sequences) Statistics

### Size Metrics
- **Total bits:** 23,904,256 (≈23.9 MB as strings)
- **Total bytes:** 2,988,032 (≈2.85 MB after packing)
- **Min length:** 168 bits (21 bytes) — corresponds to ~21 character plaintext
- **Max length:** 21,360 bits (2,670 bytes) — corresponds to ~2,670 character plaintext
- **Mean length:** 4,780.85 bits (597.61 bytes)
- **Median length:** 4,432 bits (554 bytes)

### Length Distribution Percentiles
| Percentile | Bits | Bytes |
|-----------|------|-------|
| 10th | 1,032 | 129 |
| 25th | 2,632 | 329 |
| 50th (Median) | 4,432 | 554 |
| 75th | 6,592 | 824 |
| 90th | 8,688 | 1,086 |
| 95th | 10,176 | 1,272 |
| 99th | 13,304 | 1,663 |

### Bit Distribution
- **Total 0s:** 13,665,487 (57.17%)
- **Total 1s:** 10,238,769 (42.83%)
- ✅ **Slightly imbalanced but reasonable** for cryptographic data

### Unique Sequence Lengths
- **1,361 unique lengths** — highly variable dataset
- Most sequences are unique in length
- Only top 5 lengths account for ~1.5% of all sequences

---

## Plaintext Statistics

- **Min length:** 21 characters (from Wikipedia/Brown corpus entries)
- **Max length:** 2,670 characters
- **Mean length:** 597.61 characters
- **Total characters:** 2,988,032

---

## Cipher-Plaintext Alignment

### Key Finding
✅ **Perfect 1:1 byte-to-character ratio throughout dataset**

- **Bits-per-character ratio:** 8.00 (constant)
- **No exceptions:** Every single sequence follows this pattern
- **Interpretation:** Each plaintext character encodes exactly to 8 bits (1 byte)
  
This suggests:
1. The encryption is likely byte-level (each character → 1 byte)
2. No compression or expansion in the ciphertext
3. Simple substitution or position-preserving encryption

---

## 🎯 Byte-Packing for BLT Implementation

### ✅ EXCELLENT NEWS: Full Compatibility

**All conditions met for byte-packing:**
- ✅ All sequences are pure binary (only 0s and 1s)
- ✅ All sequences are byte-aligned (length % 8 == 0)
- ✅ No padding or special handling needed

### Benefits
- **8x memory reduction:** 23.9 MB → 2.85 MB
- **Simpler processing:** Direct byte values (0-255) instead of bit strings
- **Vocabulary:** 256 possible byte values (vs. 2 for binary)
- **Local encoder friendly:** Bytes can be embedded as continuous values

### Byte-Packing Strategy for C5 (BLT)
```
Binary sequence: "00010011001000010010111000110101..."
                    ↓ (pack every 8 bits)
Bytes: [0x13, 0x22, 0x5C, 0x35, ...]
                    ↓ (embed as vectors)
Local encoder: Extract byte-level patterns
```

---

## Padding Considerations for Variable Sequences

If you use **bucketing** or **batching with padding**, here are the tradeoffs:

| Max Bytes | # Over | Padding Waste |
|-----------|--------|---------------|
| 64 | 4,729 (94.6%) | 2.0% |
| 128 | 4,503 (90.1%) | 4.9% |
| 256 | 4,067 (81.3%) | 9.4% |
| **512** | 2,736 (54.7%) | 20.3% |
| **1024** | 637 (12.7%) | 44.8% |
| 2048 | 7 (0.1%) | 70.8% |
| 4096 | 0 (0.0%) | 85.4% |

### Recommendation
- **For efficiency:** Use max length of **512 bytes** (45% sequences exceed)
- **For simplicity:** Use max length of **1024 bytes** (minimal padding waste)
- **For precision:** Use dynamic batching or bucketing by length

---

## Vocabulary Comparison

### C1-C4 (Standard Subword Tokenization)
- **Binary vocabulary:** 2 tokens (0, 1)
- **Byte vocabulary:** 256 tokens (0-255)
- **Tokenization strategy:** Byte-level or subword BPE needed

### C5 (BLT - Token-Free)
- **No vocabulary:** Direct byte representation
- **Embedding:** Continuous values 0-255 (e.g., embedding_dim=256 → byte_embedding)
- **Local encoder:** Processes raw byte sequences directly
- **Computational advantage:** No vocabulary lookup, no embedding table

---

## Memory Analysis

### Training Dataset (5,000 samples)
| Representation | Size | Notes |
|---|---|---|
| Binary strings | 22.8 MB | Current format |
| Byte-packed | 2.85 MB | After int() conversion |
| Embedding (BLT, d=256) | ~5.7 MB | 2.85 MB + embeddings |

### Single Sample Average
- Binary: ~4.8 KB
- Byte-packed: ~0.6 KB
- With embeddings (d=256): ~1.2 KB

---

## Implementation Considerations

### For Data Loading
```python
# Binary representation (C1-C4)
cipher_bits = "001011100101..."  # String of 0s and 1s
vocab_size = 2

# Byte-packed representation (C5 / BLT)
cipher_bytes = bytes([0x5C, 0x75, ...])  # Raw byte values
vocab_size = 256  # (or continuous embedding)
```

### For Local Encoder (BLT Patch Encoding)
```python
# Input: chunk of bytes (e.g., 8-byte patches)
# Process: Conv1d(8 bytes) → latent_dim
# Output: patch embedding
```

---

## Dataset Quality Flags

✅ **All checks passed:**
- No missing lines
- Consistent pair alignment
- No null characters or encoding issues
- Byte-aligned perfectly
- No anomalies detected

---

## Recommendations for Implementation

### Phase 1: Data Preparation
1. Split into train/val/test (4000/500/500)
2. For C1-C4: Keep as binary strings or tokenize
3. For C5: Convert to byte arrays immediately

### Phase 2: Tokenization
- **C1-C4:** Use simple 2-token vocabulary or byte-level BPE
- **C5:** Skip tokenization, feed bytes directly

### Phase 3: Model Configuration
- **Sequence length:** Use 1024 bytes max with padding (or bucketing)
- **Embedding dim:** 
  - C1-C4: `vocab_size=256` (byte tokens)
  - C5: `embedding_dim=256` (continuous embeddings)

### Phase 4: Local Encoder (C5 only)
- **Patch size:** 8 bytes (1 byte per patch) or 16 bytes (optimal for Conv1d)
- **Patch embedding dim:** 64-128 (reduces 16→64)
- **Total latent reduction:** 8x → 64-128x overall

---

## Questions to Consider

1. **Byte vs. Binary Representation:**
   - Why not always use bytes? Cleaner, faster, requires no tokenization
   - Only keep binary if explicitly required by assignment

2. **Local Encoder Design (C5):**
   - What patch size? (8, 16, 32 bytes)
   - Single layer or multi-layer local transformer?
   - Should C1-C4 also have equivalent bottleneck for fair comparison?

3. **Sequence Bucketing:**
   - Group by length to minimize padding?
   - Use dynamic batching?
   - Impact on training reproducibility

4. **Token-Free Interpretation for C5:**
   - Are you treating bytes as discrete (embedding) or continuous (regression)?
   - How to decode: byte prediction vs. bit regression?

---

*Report Generated: Dataset Analysis for ANLP Assignment 1*
