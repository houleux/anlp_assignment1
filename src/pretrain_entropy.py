import pickle
import sys
from pathlib import Path

# Add project root so the script can run directly as:
#   python src/pretrain_entropy.py
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from src.models.blt import ByteEntropyModel
except ImportError:  # pragma: no cover - fallback for direct execution from src/
    from models.blt import ByteEntropyModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class RawByteDataset(Dataset):
    """Just source bytes, no BOS/EOS, no target — this model only ever sees
    the cipher stream, matching what DynamicByteLatentEncoder will feed it."""

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


def pretrain_entropy_model(
    splits_dir: str = "data.nosync/splits_packed",
    hidden_dim: int = 32,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "checkpoints/entropy_model.pt",
) -> ByteEntropyModel:
    train_ds = RawByteDataset(splits_dir, "train")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = ByteEntropyModel(vocab_size=256, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for bytes_, mask in train_loader:
            bytes_, mask = bytes_.to(device), mask.to(device)
            loss = model.auxiliary_loss(bytes_, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        # Cross-entropy in nats -> bits for an intuitive read: log(256)=5.545
        # nats is the "no better than random" ceiling; watch this fall.
        print(f"epoch {epoch + 1}/{epochs}  avg next-byte CE loss = {avg_loss:.4f} nats "
              f"(random baseline ≈ {torch.log(torch.tensor(256.0)):.4f})")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved pretrained entropy model to {save_path}")
    return model


if __name__ == "__main__":
    pretrain_entropy_model()