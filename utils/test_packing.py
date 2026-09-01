"""
Quick test of bit-packing and loading functionality
"""

import sys
sys.path.insert(0, 'src')

from dataset import BitPacker, DatasetLoader, load_packed_split, show_packing_example

print("="*70)
print("1. TESTING BIT-PACKING")
print("="*70)

# Test unpacking a few examples
print("\nUnpacking first 3 train samples (from packed format):\n")
cipher_packed, plain_text = load_packed_split(
    splits_dir="data.nosync/splits_packed",
    split_name="train"
)

for i in range(3):
    bits = BitPacker.bytes_to_bits(cipher_packed[i])
    print(f"Sample {i+1}:")
    print(f"  Bytes:        {cipher_packed[i]}")
    print(f"  Unpacked bits: {bits[:80]}... (total: {len(bits)} bits)")
    print(f"  Plaintext:     '{plain_text[i][:80]}...'")
    print()

print("="*70)
print("2. LOADING STATISTICS")
print("="*70)

loader = DatasetLoader(splits_dir="data.nosync/splits")
for split in ['train', 'val', 'test']:
    stats = loader.get_stats(split)
    print(f"\n{split.upper()}:")
    print(f"  Samples:          {stats['num_samples']}")
    print(f"  Cipher bits:      min={stats['cipher_bits_min']:,} max={stats['cipher_bits_max']:,} mean={stats['cipher_bits_mean']:.0f}")
    print(f"  Cipher bytes:     min={stats['cipher_bytes_min']:,} max={stats['cipher_bytes_max']:,} mean={stats['cipher_bytes_mean']:.0f}")
    print(f"  Plaintext chars:  min={stats['plain_chars_min']:,} max={stats['plain_chars_max']:,} mean={stats['plain_chars_mean']:.0f}")

print("\n" + "="*70)
print("3. VERIFICATION")
print("="*70)

# Verify round-trip conversion
print("\nRound-trip test (binary → bytes → binary):")
original = "1101001000110101010100010101011100010110"  # 40 bits
packed = BitPacker.bits_to_bytes(original)
restored = BitPacker.bytes_to_bits(packed)

print(f"  Original:  {original}")
print(f"  Packed:    {packed}")
print(f"  Restored:  {restored}")
print(f"  Match:     {'✓ YES' if original == restored else '✗ NO'}")

print("\n✓ All tests passed!")
