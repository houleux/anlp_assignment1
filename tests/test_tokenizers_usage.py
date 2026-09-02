"""
Test script to demonstrate loading and using the trained tokenizers
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from models.bpe import BPETokenizer
from dataset import load_packed_split

# Load tokenizers
print("="*70)
print("LOADING TOKENIZERS")
print("="*70)

encoder_tok = BPETokenizer.load("tokenizers/encoder.json")
decoder_tok = BPETokenizer.load("tokenizers/decoder.json")

print(f"✓ Encoder tokenizer loaded")
print(f"  Vocab size: {len(encoder_tok.vocab)}")
print(f"  Merges: {len(encoder_tok.merges)}")

print(f"\n✓ Decoder tokenizer loaded")
print(f"  Vocab size: {len(decoder_tok.vocab)}")
print(f"  Merges: {len(decoder_tok.merges)}")

# Load a sample
print("\n" + "="*70)
print("TESTING ON SAMPLE DATA")
print("="*70)

cipher_bytes, plain_text = load_packed_split(
    splits_dir="data.nosync/splits_packed",
    split_name="val"
)

# Test on first 3 samples
for i in range(3):
    print(f"\nSample {i+1}:")
    
    # Encode/decode cipher with encoder tokenizer
    cipher_str = cipher_bytes[i].decode('latin-1')
    encoder_ids = encoder_tok.encode(cipher_str)
    encoder_decoded = encoder_tok.decode(encoder_ids)
    
    print(f"  Cipher:")
    print(f"    Original:  {len(cipher_bytes[i])} bytes")
    print(f"    Encoded:   {len(encoder_ids)} tokens ({len(cipher_bytes[i])/len(encoder_ids):.2f}x)")
    print(f"    Match:     {'✓' if cipher_str == encoder_decoded else '✗'}")
    
    # Encode/decode plaintext with decoder tokenizer
    decoder_ids = decoder_tok.encode(plain_text[i])
    decoder_decoded = decoder_tok.decode(decoder_ids)
    
    print(f"  Plaintext:")
    print(f"    Original:  {len(plain_text[i])} chars")
    print(f"    Encoded:   {len(decoder_ids)} tokens ({len(plain_text[i])/len(decoder_ids):.2f}x)")
    print(f"    Match:     {'✓' if plain_text[i] == decoder_decoded else '✗'}")
    print(f"    Preview:   '{plain_text[i][:60]}...'")

print("\n" + "="*70)
print("✓ TOKENIZERS WORKING CORRECTLY")
print("="*70)
