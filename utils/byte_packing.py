"""
Dataset utilities for ANLP Assignment 1
Handles loading, bit-packing, and tokenization of cipher/plaintext pairs
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import pickle


class BitPacker:
    """Utility class for packing/unpacking bits to/from bytes"""
    
    @staticmethod
    def bits_to_bytes(bit_string: str) -> bytes:
        """
        Convert binary string (e.g., "00010011") to bytes.
        
        Args:
            bit_string: String of 0s and 1s, must have length divisible by 8
            
        Returns:
            Bytes object where each byte represents 8 bits
            
        Example:
            "00010011" → bytes([19])  # 00010011 in binary = 19 in decimal
        """
        assert len(bit_string) % 8 == 0, f"Bit string length must be divisible by 8, got {len(bit_string)}"
        assert all(c in '01' for c in bit_string), "Bit string must contain only 0s and 1s"
        
        byte_list = []
        for i in range(0, len(bit_string), 8):
            byte_val = int(bit_string[i:i+8], 2)
            byte_list.append(byte_val)
        
        return bytes(byte_list)
    
    @staticmethod
    def bytes_to_bits(byte_data: bytes) -> str:
        """
        Convert bytes back to binary string.
        
        Args:
            byte_data: Bytes object
            
        Returns:
            String of 0s and 1s with length = len(byte_data) * 8
            
        Example:
            bytes([19]) → "00010011"
        """
        bit_list = []
        for byte_val in byte_data:
            bit_list.append(format(byte_val, '08b'))
        
        return ''.join(bit_list)
    
    @staticmethod
    def validate_and_pack(bit_string: str) -> bytes:
        """
        Validate and pack a bit string, with error handling.
        
        Args:
            bit_string: Binary string
            
        Returns:
            Packed bytes
            
        Raises:
            ValueError: If bit string is invalid
        """
        if not bit_string:
            raise ValueError("Empty bit string")
        if len(bit_string) % 8 != 0:
            raise ValueError(f"Bit string length {len(bit_string)} not divisible by 8")
        if not all(c in '01' for c in bit_string):
            raise ValueError("Bit string contains non-binary characters")
        
        return BitPacker.bits_to_bytes(bit_string)


class DatasetLoader:
    """Load and manage cipher/plaintext pairs from split files"""
    
    def __init__(self, splits_dir: str = "data.nosync/splits"):
        """
        Initialize dataset loader.
        
        Args:
            splits_dir: Path to directory containing train/val/test splits
        """
        self.splits_dir = Path(splits_dir)
        self.splits = {}
        self._load_splits()
    
    def _load_splits(self):
        """Load all available splits"""
        for split_name in ['train', 'val', 'test']:
            cipher_path = self.splits_dir / f"{split_name}_cipher.txt"
            plain_path = self.splits_dir / f"{split_name}_plain.txt"
            
            if cipher_path.exists() and plain_path.exists():
                with open(cipher_path, 'r') as f:
                    cipher_lines = [line.rstrip('\n') for line in f.readlines()]
                with open(plain_path, 'r') as f:
                    plain_lines = [line.rstrip('\n') for line in f.readlines()]
                
                self.splits[split_name] = {
                    'cipher': cipher_lines,
                    'plain': plain_lines
                }
                print(f"✓ Loaded {split_name}: {len(cipher_lines)} pairs")
    
    def get_split(self, split_name: str) -> Tuple[List[str], List[str]]:
        """Get cipher and plaintext for a split"""
        if split_name not in self.splits:
            raise ValueError(f"Split '{split_name}' not found. Available: {list(self.splits.keys())}")
        
        data = self.splits[split_name]
        return data['cipher'], data['plain']
    
    def get_stats(self, split_name: str) -> dict:
        """Get statistics for a split"""
        cipher, plain = self.get_split(split_name)
        cipher_lengths = [len(c) for c in cipher]
        plain_lengths = [len(p) for p in plain]
        
        return {
            'num_samples': len(cipher),
            'cipher_bits_total': sum(cipher_lengths),
            'cipher_bits_min': min(cipher_lengths),
            'cipher_bits_max': max(cipher_lengths),
            'cipher_bits_mean': sum(cipher_lengths) / len(cipher_lengths),
            'cipher_bytes_min': min(cipher_lengths) // 8,
            'cipher_bytes_max': max(cipher_lengths) // 8,
            'cipher_bytes_mean': sum(cipher_lengths) // 8 / len(cipher_lengths),
            'plain_chars_min': min(plain_lengths),
            'plain_chars_max': max(plain_lengths),
            'plain_chars_mean': sum(plain_lengths) / len(plain_lengths),
        }


def pack_dataset_to_bytes(
    input_splits_dir: str = "data.nosync/splits",
    output_dir: str = "data.nosync/splits_packed"
):
    """
    Convert all split files from binary strings to byte-packed format.
    
    Args:
        input_splits_dir: Directory with original train/val/test splits
        output_dir: Directory to save byte-packed splits
        
    Output structure:
        splits_packed/
        ├── train_cipher.pkl  (list of bytes objects)
        ├── train_plain.txt   (plaintext, unchanged)
        ├── val_cipher.pkl
        ├── val_plain.txt
        ├── test_cipher.pkl
        └── test_plain.txt
    """
    
    input_path = Path(input_splits_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Packing dataset from: {input_path}")
    print(f"Output directory: {output_path}\n")
    
    for split_name in ['train', 'val', 'test']:
        cipher_file = input_path / f"{split_name}_cipher.txt"
        plain_file = input_path / f"{split_name}_plain.txt"
        
        if not cipher_file.exists() or not plain_file.exists():
            print(f"⚠ Skipping {split_name}: files not found")
            continue
        
        print(f"Processing {split_name}...")
        
        # Read cipher (binary strings)
        with open(cipher_file, 'r') as f:
            cipher_lines = [line.rstrip('\n') for line in f.readlines()]
        
        # Read plain (no change needed)
        with open(plain_file, 'r') as f:
            plain_lines = [line.rstrip('\n') for line in f.readlines()]
        
        # Pack cipher bytes
        cipher_packed = []
        for i, bit_string in enumerate(cipher_lines):
            try:
                packed = BitPacker.bits_to_bytes(bit_string)
                cipher_packed.append(packed)
            except Exception as e:
                print(f"  ✗ Error on line {i+1}: {e}")
                raise
        
        # Save packed cipher as pickle
        cipher_out = output_path / f"{split_name}_cipher.pkl"
        with open(cipher_out, 'wb') as f:
            pickle.dump(cipher_packed, f)
        
        # Save plaintext (unchanged)
        plain_out = output_path / f"{split_name}_plain.txt"
        with open(plain_out, 'w') as f:
            for line in plain_lines:
                f.write(line + '\n')
        
        print(f"  ✓ {split_name}: {len(cipher_packed)} pairs")
        print(f"    Cipher: {cipher_out.name}")
        print(f"    Plain:  {plain_out.name}")
    
    print(f"\n✓ Packing complete!")


def load_packed_split(
    splits_dir: str = "data.nosync/splits_packed",
    split_name: str = "train"
) -> Tuple[List[bytes], List[str]]:
    """
    Load a byte-packed split.
    
    Args:
        splits_dir: Directory containing packed splits
        split_name: 'train', 'val', or 'test'
        
    Returns:
        (cipher_bytes_list, plaintext_list) where cipher_bytes_list is a list of bytes objects
    """
    cipher_file = Path(splits_dir) / f"{split_name}_cipher.pkl"
    plain_file = Path(splits_dir) / f"{split_name}_plain.txt"
    
    # Load packed cipher
    with open(cipher_file, 'rb') as f:
        cipher_packed = pickle.load(f)
    
    # Load plaintext
    with open(plain_file, 'r') as f:
        plain_text = [line.rstrip('\n') for line in f.readlines()]
    
    return cipher_packed, plain_text


def show_packing_example():
    """Demonstrate bit-packing conversion"""
    print("="*70)
    print("BIT-PACKING DEMONSTRATION")
    print("="*70)
    
    # Example 1: Simple 8-bit sequence
    bit_seq_1 = "00010011"
    packed_1 = BitPacker.bits_to_bytes(bit_seq_1)
    unpacked_1 = BitPacker.bytes_to_bits(packed_1)
    
    print(f"\nExample 1: Single byte")
    print(f"  Original bits: {bit_seq_1}")
    print(f"  Packed bytes:  {packed_1}")
    print(f"  Byte value:    {packed_1[0]} (decimal)")
    print(f"  Unpacked bits: {unpacked_1}")
    print(f"  Match:         {bit_seq_1 == unpacked_1}")
    
    # Example 2: Multiple bytes
    bit_seq_2 = "00010011001011100011010101000000"  # 32 bits = 4 bytes
    packed_2 = BitPacker.bits_to_bytes(bit_seq_2)
    unpacked_2 = BitPacker.bytes_to_bits(packed_2)
    
    print(f"\nExample 2: Multiple bytes")
    print(f"  Original bits ({len(bit_seq_2)}): {bit_seq_2}")
    print(f"  Packed bytes ({len(packed_2)}):   {packed_2}")
    print(f"  Byte values:        {[b for b in packed_2]}")
    print(f"  Unpacked bits:      {unpacked_2}")
    print(f"  Match:              {bit_seq_2 == unpacked_2}")
    
    print(f"\n✓ Memory reduction: {len(bit_seq_2)} bits → {len(packed_2)} bytes (8x smaller)")


if __name__ == "__main__":
    import sys
    
    # Show demo
    show_packing_example()
    
    # Pack dataset
    print("\n" + "="*70)
    print("PACKING DATASET")
    print("="*70 + "\n")
    
    pack_dataset_to_bytes(
        input_splits_dir="data.nosync/splits",
        output_dir="data.nosync/splits_packed"
    )
