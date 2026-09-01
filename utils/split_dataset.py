"""
Dataset split script for ANLP Assignment 1
Splits the cipher-plaintext pairs into train/val/test sets
"""

import os
from pathlib import Path

def split_dataset(
    cipher_path: str = "data.nosync/brown_cipher.txt",
    plain_path: str = "data.nosync/brown_plain.txt",
    output_dir: str = "data.nosync/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Split dataset into train/val/test with separate plain/cipher files.
    
    Args:
        cipher_path: Path to cipher text file
        plain_path: Path to plain text file
        output_dir: Directory to save split files
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for testing (default 0.1)
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
    
    Output files:
        splits/
        ├── train_plain.txt
        ├── train_cipher.txt
        ├── val_plain.txt
        ├── val_cipher.txt
        ├── test_plain.txt
        └── test_cipher.txt
    """
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total_ratio}"
    
    # Read files
    print(f"Reading cipher text from: {cipher_path}")
    with open(cipher_path, 'r') as f:
        cipher_lines = [line.rstrip('\n') for line in f.readlines()]
    
    print(f"Reading plain text from: {plain_path}")
    with open(plain_path, 'r') as f:
        plain_lines = [line.rstrip('\n') for line in f.readlines()]
    
    # Validate alignment
    assert len(cipher_lines) == len(plain_lines), \
        f"Mismatch: {len(cipher_lines)} cipher lines vs {len(plain_lines)} plain lines"
    
    num_samples = len(cipher_lines)
    print(f"Total samples: {num_samples}")
    
    # Create indices and optionally shuffle
    indices = list(range(num_samples))
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(indices)
        print(f"Shuffled with seed={seed}")
    
    # Calculate split points
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Helper function to write split
    def write_split(split_name, split_indices):
        cipher_out = output_path / f"{split_name}_cipher.txt"
        plain_out = output_path / f"{split_name}_plain.txt"
        
        with open(cipher_out, 'w') as f:
            for idx in split_indices:
                f.write(cipher_lines[idx] + '\n')
        
        with open(plain_out, 'w') as f:
            for idx in split_indices:
                f.write(plain_lines[idx] + '\n')
        
        print(f"✓ Wrote {split_name}: {len(split_indices)} samples")
        return cipher_out, plain_out
    
    # Write splits
    print("\nWriting split files...")
    write_split("train", train_indices)
    write_split("val", val_indices)
    write_split("test", test_indices)
    
    print(f"\n✓ Dataset split complete!")
    print(f"  Train: {len(train_indices)} samples ({100*len(train_indices)/num_samples:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples ({100*len(val_indices)/num_samples:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples ({100*len(test_indices)/num_samples:.1f}%)")
    
    return {
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'output_dir': str(output_path)
    }


if __name__ == "__main__":
    # Default: 80/10/10 split, no shuffling (deterministic)
    stats = split_dataset(
        cipher_path="data.nosync/brown_cipher.txt",
        plain_path="data.nosync/brown_plain.txt",
        output_dir="data.nosync/splits",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        shuffle=False,  # Set to True if you want random shuffling
        seed=42
    )
