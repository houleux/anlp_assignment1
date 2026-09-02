"""
Calibration sweep for DynamicByteLatentEncoder's `keep_fraction` (top-k
boundary rule). Run this AFTER pretrain_entropy_model.py.

Unlike the delta-threshold rule, keep_fraction has a closed-form starting
estimate:

    keep_fraction ≈ target_mean_patches / mean_sequence_length

so this script sweeps a small range around that estimate to fine-tune,
rather than blind-searching an open-ended nats value.
"""

import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.blt import ByteEntropyModel, entropy_to_boundaries_topk

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# From your C1-C4 BPE tokenizer stats (train split, encoder side).
TARGET_MEAN_PATCHES = 407.61
MEAN_SOURCE_LEN = 597.61  # mean packed cipher-byte length, train split
CLOSED_FORM_ESTIMATE = TARGET_MEAN_PATCHES / MEAN_SOURCE_LEN


class RawByteDataset(Dataset):
    def __init__(self, splits_dir: str, split_name: str, max_len: int = 1024):
        path = Path(splits_dir) / f"{split_name}_cipher.pkl"
        with open(path, "rb") as f:
            cipher_bytes_list = pickle.load(f)
        self.samples = [list(b)[:max_len] for b in cipher_bytes_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return padded, mask


@torch.no_grad()
def patch_counts_for_fraction(
    entropy_model: ByteEntropyModel,
    loader: DataLoader,
    keep_fraction: float,
) -> torch.Tensor:
    all_counts = []
    for bytes_, mask in loader:
        bytes_, mask = bytes_.to(device), mask.to(device)
        entropy, _ = entropy_model(bytes_)
        boundary = entropy_to_boundaries_topk(entropy, mask, keep_fraction=keep_fraction)
        patch_id = (torch.cumsum(boundary.long(), dim=1) - 1).clamp_min(0)
        patch_id_masked = patch_id.masked_fill(~mask, -1)
        num_patches = patch_id_masked.max(dim=1).values + 1
        all_counts.append(num_patches.cpu())
    return torch.cat(all_counts)


def sweep(
    splits_dir: str = "data.nosync/splits_packed",
    checkpoint_path: str = "checkpoints/entropy_model.pt",
    hidden_dim: int = 32,
    batch_size: int = 32,
    n_steps: int = 7,
    span: float = 0.15,
):
    """Sweeps n_steps values of keep_fraction centered on the closed-form
    estimate, +/- span (e.g. span=0.15 around 0.682 covers 0.532..0.832)."""
    entropy_model = ByteEntropyModel(vocab_size=256, hidden_dim=hidden_dim).to(device)
    entropy_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    entropy_model.requires_grad_(False)
    entropy_model.eval()

    train_ds = RawByteDataset(splits_dir, "train")
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    lo = max(0.01, CLOSED_FORM_ESTIMATE - span)
    hi = min(1.0, CLOSED_FORM_ESTIMATE + span)
    fractions = torch.linspace(lo, hi, n_steps).tolist()

    print(f"Closed-form estimate: keep_fraction ≈ {CLOSED_FORM_ESTIMATE:.4f} "
          f"({TARGET_MEAN_PATCHES} / {MEAN_SOURCE_LEN})")
    print(f"Sweeping {n_steps} values in [{lo:.3f}, {hi:.3f}]\n")
    print(f"{'keep_fraction':>13} | {'mean patches':>13} | {'median':>8} | {'p10':>6} | {'p90':>6} | {'Δ from target':>14}")
    print("-" * 75)

    results = []
    for kf in fractions:
        counts = patch_counts_for_fraction(entropy_model, loader, kf).float()
        mean_val = counts.mean().item()
        median_val = counts.median().item()
        p10 = counts.quantile(0.10).item()
        p90 = counts.quantile(0.90).item()
        delta = mean_val - TARGET_MEAN_PATCHES
        results.append((kf, mean_val, delta))
        print(f"{kf:>13.4f} | {mean_val:>13.1f} | {median_val:>8.1f} | {p10:>6.1f} | {p90:>6.1f} | {delta:>+14.1f}")

    best_kf, best_mean, best_delta = min(results, key=lambda r: abs(r[2]))
    print("-" * 75)
    print(f"Closest to target: keep_fraction={best_kf:.4f} "
          f"(mean patches={best_mean:.1f}, Δ={best_delta:+.1f})")
    print("If the plateau/coarseness of this sweep isn't tight enough, "
          "narrow `span` around best_kf and increase `n_steps`, then re-run.")
    return results


if __name__ == "__main__":
    sweep()