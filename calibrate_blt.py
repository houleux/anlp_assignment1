"""
Calibration script for BLT entropy threshold tuning.

Runs DynamicByteLatentEncoder over a batch with different entropy_threshold
values and reports the average number of patches produced for each.
"""

import sys
from pathlib import Path

# Add project root to Python path so imports work
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.models.blt import DynamicByteLatentEncoder
from src.dataset import create_byte_dataloaders


def calibrate_entropy_thresholds(
    byte_dim: int = 128,
    latent_dim: int = 256,
    vocab_size: int = 256,
    splits_dir: str = "data.nosync/splits_packed",
    batch_size: int = 8,
    thresholds: list[float] = None,
):
    """Test different entropy thresholds and report average patch counts."""

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.5, 1.0]

    # Load real data from your dataset
    print("Loading dataset...")
    dataloaders = create_byte_dataloaders(
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=0,
    )

    # Get one batch from the validation set
    val_loader = dataloaders['val']
    batch = next(iter(val_loader))

    bytes_ = batch['encoder_input']  # (B, L)
    byte_mask = batch['encoder_mask']  # (B, L) bool

    batch_size_actual, sequence_length = bytes_.shape
    print(f"Calibrating BLT encoder on real batch: {batch_size_actual} sequences, up to {sequence_length} bytes each\n")
    print("=" * 60)

    for threshold in thresholds:
        encoder = DynamicByteLatentEncoder(
            byte_dim=byte_dim,
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            entropy_threshold=threshold,
        )
        encoder.eval()

        with torch.no_grad():
            patches, patch_mask = encoder(bytes_, byte_mask)
            # patches shape: (B, P, latent_dim)
            # patch_mask shape: (B, P) - bool mask

            # Count actual patches per sample (sum of patch_mask)
            num_patches = patch_mask.sum(dim=1).float()  # (B,)
            avg_patches = num_patches.mean().item()

            # Calculate average actual sequence length (non-padded bytes)
            actual_lengths = byte_mask.sum(dim=1).float()  # (B,)
            avg_length = actual_lengths.mean().item()

            print(f"Entropy threshold: {threshold:4.1f}")
            print(f"  Average patches per sequence: {avg_patches:6.2f}")
            print(f"  Min patches: {num_patches.min().item():.0f}")
            print(f"  Max patches: {num_patches.max().item():.0f}")
            print(f"  Avg sequence length: {avg_length:.1f} bytes")
            print(f"  Compression ratio: {avg_length / avg_patches:.2f}x")
            print()


if __name__ == "__main__":
    print("BLT Entropy Threshold Calibration")
    print("=" * 60)
    print()

    calibrate_entropy_thresholds()

    print("\nInterpretation:")
    print("- Lower threshold = more patches (more sensitive to entropy changes)")
    print("- Higher threshold = fewer patches (only large entropy spikes create boundaries)")
    print("- Calibrate to match your target avg tokens/sequence for fair comparison")
