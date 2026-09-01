"""
Training script for encoder and decoder tokenizers
- Encoder tokenizer: trained on cipher bytes
- Decoder tokenizer: trained on plain text
"""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from dataset import load_packed_split, BitPacker
from models.bpe import BPETokenizer


def prepare_cipher_training_data(cipher_bytes_list):
    """
    Convert list of bytes objects into training text.
    Each byte value is represented as a character in range 0-255.
    
    This avoids UTF-8 decoding issues and treats each byte as a unique token.
    """
    # Approach 1: Use raw bytes with Latin-1 encoding (1-to-1 mapping for 0-255)
    cipher_text = ''.join(
        bytes_obj.decode('latin-1') for bytes_obj in cipher_bytes_list
    )
    return cipher_text


def train_tokenizers(
    splits_dir: str = "data.nosync/splits_packed",
    output_dir: str = "tokenizers",
    encoder_vocab_size: int = 1024,
    decoder_vocab_size: int = 1024,
    verbose: bool = True
):
    """
    Train encoder and decoder tokenizers.
    
    Args:
        splits_dir: Directory containing packed splits
        output_dir: Directory to save tokenizers
        encoder_vocab_size: Vocabulary size for encoder (trained on cipher bytes)
        decoder_vocab_size: Vocabulary size for decoder (trained on plain text)
        verbose: Whether to print training progress
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("LOADING TRAINING DATA")
    print("="*70)
    
    # Load train split
    cipher_bytes, plain_text = load_packed_split(
        splits_dir=splits_dir,
        split_name="train"
    )
    
    print(f"✓ Loaded {len(cipher_bytes)} training pairs")
    print(f"  Cipher: {len(cipher_bytes)} byte sequences")
    print(f"  Plain: {len(plain_text)} text sequences")
    
    # Prepare training data
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    
    print("\nPreparing cipher data (encoder)...")
    cipher_training_text = prepare_cipher_training_data(cipher_bytes)
    print(f"  Cipher text length: {len(cipher_training_text):,} characters")
    print(f"  Unique bytes used: {len(set(ord(c) for c in cipher_training_text))}")
    
    print("\nPreparing plaintext data (decoder)...")
    plain_training_text = '\n'.join(plain_text)
    print(f"  Plain text length: {len(plain_training_text):,} characters")
    print(f"  Unique characters: {len(set(plain_training_text))}")
    
    # Train encoder tokenizer
    print("\n" + "="*70)
    print("TRAINING ENCODER TOKENIZER (on cipher bytes)")
    print("="*70)
    print(f"\nVocab size: {encoder_vocab_size}")
    print(f"Num merges: {encoder_vocab_size - 256}\n")
    
    encoder_tokenizer = BPETokenizer()
    encoder_tokenizer.train(
        cipher_training_text,
        vocab_size=encoder_vocab_size,
        verbose=False  # Set to True to see all merge operations
    )
    
    print(f"✓ Encoder tokenizer trained")
    print(f"  Vocab size: {len(encoder_tokenizer.vocab)}")
    print(f"  Merges learned: {len(encoder_tokenizer.merges)}")
    
    # Train decoder tokenizer
    print("\n" + "="*70)
    print("TRAINING DECODER TOKENIZER (on plain text)")
    print("="*70)
    print(f"\nVocab size: {decoder_vocab_size}")
    print(f"Num merges: {decoder_vocab_size - 256}\n")
    
    decoder_tokenizer = BPETokenizer()
    decoder_tokenizer.train(
        plain_training_text,
        vocab_size=decoder_vocab_size,
        verbose=False
    )
    
    print(f"✓ Decoder tokenizer trained")
    print(f"  Vocab size: {len(decoder_tokenizer.vocab)}")
    print(f"  Merges learned: {len(decoder_tokenizer.merges)}")
    
    # Save tokenizers
    print("\n" + "="*70)
    print("SAVING TOKENIZERS")
    print("="*70)
    
    encoder_path = output_path / "encoder.json"
    decoder_path = output_path / "decoder.json"
    
    encoder_tokenizer.save(str(encoder_path))
    decoder_tokenizer.save(str(decoder_path))
    
    print(f"\n✓ Encoder tokenizer saved to: {encoder_path}")
    print(f"✓ Decoder tokenizer saved to: {decoder_path}")
    
    # Test tokenizers
    print("\n" + "="*70)
    print("TESTING TOKENIZERS")
    print("="*70)
    
    # Test encoder on a sample
    sample_cipher_idx = 0
    sample_cipher_bytes = cipher_bytes[sample_cipher_idx]
    sample_cipher_text = sample_cipher_bytes.decode('latin-1')
    
    encoded_ids = encoder_tokenizer.encode(sample_cipher_text)
    decoded_text = encoder_tokenizer.decode(encoded_ids)
    
    print(f"\nEncoder tokenizer test:")
    print(f"  Input (cipher bytes): {sample_cipher_bytes[:20]}... ({len(sample_cipher_bytes)} bytes)")
    print(f"  Encoded: {len(encoded_ids)} tokens")
    print(f"  Compression ratio: {len(sample_cipher_bytes) / len(encoded_ids):.2f}x")
    print(f"  Decoded match: {'✓' if sample_cipher_text == decoded_text else '✗'}")
    
    # Test decoder on a sample
    sample_plain_idx = 0
    sample_plain = plain_text[sample_plain_idx]
    
    decoded_ids = decoder_tokenizer.encode(sample_plain)
    decoded_plain = decoder_tokenizer.decode(decoded_ids)
    
    print(f"\nDecoder tokenizer test:")
    print(f"  Input (plain text): '{sample_plain[:60]}...' ({len(sample_plain)} chars)")
    print(f"  Encoded: {len(decoded_ids)} tokens")
    print(f"  Compression ratio: {len(sample_plain) / len(decoded_ids):.2f}x")
    print(f"  Decoded match: {'✓' if sample_plain == decoded_plain else '✗'}")
    
    print("\n" + "="*70)
    print("✓ TOKENIZER TRAINING COMPLETE")
    print("="*70)
    
    return encoder_tokenizer, decoder_tokenizer


if __name__ == "__main__":
    encoder_tok, decoder_tok = train_tokenizers(
        splits_dir="data.nosync/splits_packed",
        output_dir="tokenizers",
        encoder_vocab_size=1024,
        decoder_vocab_size=1024,
        verbose=False
    )
