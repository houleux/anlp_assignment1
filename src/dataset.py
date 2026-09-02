"""
Dataset classes for encoder-decoder transformers
- TokenizedSeqDataset: Uses BPE tokenizers for C1-C4
- ByteSeqDataset: Uses raw bytes for C5 (BLT)
"""

import torch
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from models.bpe import BPETokenizer


# =====================================================================
# Special Token IDs (Tokenized path, C1-C4)
# =====================================================================

class SpecialTokens:
    """
    Special token IDs consistent across all BPE tokenizers.

    BPE tokenizers produce tokens in range [0, vocab_size) = [0, 1024).
    Special tokens must be outside this range to avoid collisions with real tokens.

    NOTE: If you use these IDs, make sure src_vocab_size / tgt_vocab_size in
    TransformerConfig are set >= 1028 (to cover EOS_ID = 1027), or the
    embedding layers and _check_token_ids in transformer.py will reject them.
    """
    PAD_ID = 1024
    UNK_ID = 1025
    BOS_ID = 1026
    EOS_ID = 1027

    @classmethod
    def get_all(cls):
        return {
            'pad': cls.PAD_ID,
            'unk': cls.UNK_ID,
            'bos': cls.BOS_ID,
            'eos': cls.EOS_ID,
        }


# =====================================================================
# TokenizedSeqDataset (for C1-C4: with tokenizers)
# =====================================================================

class TokenizedSeqDataset(Dataset):
    """
    Dataset that tokenizes cipher and plaintext using BPE tokenizers.
    Suitable for C1-C4 configurations.
    """

    def __init__(
        self,
        split_name: str,
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        splits_dir: str = "data.nosync/splits_packed",
        max_src_len: int = 1024,
        max_tgt_len: int = 640,
    ):
        self.split_name = split_name
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        cipher_bytes_list, plaintext_list = self._load_split(splits_dir, split_name)

        self.samples = []
        for cipher_bytes, plaintext in zip(cipher_bytes_list, plaintext_list):
            sample = self._process_pair(cipher_bytes, plaintext)
            if sample is not None:
                self.samples.append(sample)

        print(f"✓ Loaded {split_name} split: {len(self.samples)} samples")

    def _load_split(self, splits_dir: str, split_name: str):
        splits_path = Path(splits_dir)
        cipher_path = splits_path / f"{split_name}_cipher.pkl"
        with open(cipher_path, 'rb') as f:
            cipher_bytes_list = pickle.load(f)
        plaintext_path = splits_path / f"{split_name}_plain.txt"
        with open(plaintext_path, 'r') as f:
            plaintext_list = [line.rstrip('\n') for line in f.readlines()]
        return cipher_bytes_list, plaintext_list

    def _process_pair(self, cipher_bytes: bytes, plaintext: str) -> Optional[Dict]:
        try:
            cipher_str = cipher_bytes.decode('latin-1')
        except Exception as e:
            print(f"Warning: Failed to decode cipher bytes: {e}")
            return None

        src_ids = self.src_tokenizer.encode(cipher_str)
        tgt_ids = self.tgt_tokenizer.encode(plaintext)

        src_ids = [SpecialTokens.BOS_ID] + src_ids + [SpecialTokens.EOS_ID]
        tgt_ids = [SpecialTokens.BOS_ID] + tgt_ids + [SpecialTokens.EOS_ID]

        if len(src_ids) > self.max_src_len:
            src_ids = src_ids[:self.max_src_len - 1] + [SpecialTokens.EOS_ID]
        if len(tgt_ids) > self.max_tgt_len:
            tgt_ids = tgt_ids[:self.max_tgt_len - 1] + [SpecialTokens.EOS_ID]

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# =====================================================================
# ByteSeqDataset (for C5: BLT, token-free)
# =====================================================================

class ByteSeqDataset(Dataset):
    """
    Dataset for byte-level processing (C5 - Byte Latent Transformer).

    Design notes (matches src/models/blt.py):
    - ByteLatentEncoder has vocab_size=256 and no BOS/EOS concept — sequence
      boundaries are carried entirely by the padding mask, so SOURCE bytes
      are NOT given any BOS/EOS marker. Adding one would push a value >= 256
      into the encoder and crash its range check / embedding lookup.
    - ByteLatentDecoder injects its own BOS internally (bos_token_id =
      vocab_size, shifted into decoder_input by forward()). So TARGET bytes
      must NOT be prepended with BOS either — only a single trailing EOS
      *class* is added, which must live inside the decoder's own vocab_size
      (use ByteLatentDecoder(vocab_size=257): classes 0-255 = real bytes,
      class 256 = EOS).
    """

    def __init__(
        self,
        split_name: str,
        splits_dir: str = "data.nosync/splits_packed",
        max_src_len: int = 1024,
        max_tgt_len: int = 640,
    ):
        self.split_name = split_name
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # EOS is an extra output *class*, not a raw byte value -> lives at
        # index 256, matching ByteLatentDecoder(vocab_size=257).
        self.EOS_CLASS = 256

        cipher_bytes_list, plaintext_list = self._load_split(splits_dir, split_name)

        self.samples = []
        for cipher_bytes, plaintext in zip(cipher_bytes_list, plaintext_list):
            sample = self._process_pair(cipher_bytes, plaintext)
            if sample is not None:
                self.samples.append(sample)

        print(f"✓ Loaded {split_name} split (byte-level): {len(self.samples)} samples")

    def _load_split(self, splits_dir: str, split_name: str):
        splits_path = Path(splits_dir)
        cipher_path = splits_path / f"{split_name}_cipher.pkl"
        with open(cipher_path, 'rb') as f:
            cipher_bytes_list = pickle.load(f)
        plaintext_path = splits_path / f"{split_name}_plain.txt"
        with open(plaintext_path, 'r') as f:
            plaintext_list = [line.rstrip('\n') for line in f.readlines()]
        return cipher_bytes_list, plaintext_list

    def _process_pair(self, cipher_bytes: bytes, plaintext: str) -> Optional[Dict]:
        """
        - src_bytes: raw cipher bytes, values in [0, 255], NO special tokens.
        - tgt_bytes: raw plaintext UTF-8 bytes, values in [0, 255], with a
          single trailing EOS_CLASS (256) appended. NO BOS is prepended;
          ByteLatentDecoder.forward() injects its own BOS via a right-shift.
        """
        src_bytes = list(cipher_bytes)

        try:
            tgt_bytes = list(plaintext.encode('utf-8'))
        except Exception as e:
            print(f"Warning: Failed to encode plaintext: {e}")
            return None

        tgt_bytes = tgt_bytes + [self.EOS_CLASS]

        # Truncate (reserve last slot for EOS_CLASS on the target side).
        if len(src_bytes) > self.max_src_len:
            src_bytes = src_bytes[:self.max_src_len]
        if len(tgt_bytes) > self.max_tgt_len:
            tgt_bytes = tgt_bytes[:self.max_tgt_len - 1] + [self.EOS_CLASS]

        return {
            'src_bytes': src_bytes,
            'tgt_bytes': tgt_bytes,
            'src_len': len(src_bytes),
            'tgt_len': len(tgt_bytes),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# =====================================================================
# Collate Functions
# =====================================================================

def collate_fn_tokenized(
    batch: List[Dict],
    pad_id: int = SpecialTokens.PAD_ID,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for TokenizedSeqDataset. Dynamically pads to batch max.

    TEACHER FORCING NOTE:
        decoder_input_feed = decoder_input[:, :-1]
        decoder_target     = decoder_input[:, 1:]
    """
    max_src_len_batch = max(item['src_len'] for item in batch)
    max_tgt_len_batch = max(item['tgt_len'] for item in batch)
    batch_size = len(batch)

    encoder_input = torch.full((batch_size, max_src_len_batch), pad_id, dtype=torch.long)
    decoder_input = torch.full((batch_size, max_tgt_len_batch), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        src_ids = item['src_ids']
        tgt_ids = item['tgt_ids']
        encoder_input[i, :len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        decoder_input[i, :len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)

    encoder_mask = (encoder_input != pad_id).float()
    decoder_padding_mask = (decoder_input != pad_id).float()

    causal_mask = torch.tril(torch.ones(max_tgt_len_batch, max_tgt_len_batch))
    # FIX: .expand() shares storage across the batch dim (stride 0) — writing
    # causal_mask[i, ...] = 0 in the loop below would silently overwrite the
    # SAME underlying matrix for every sample. .repeat() allocates real,
    # independent memory per batch index.
    causal_mask = causal_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    for i in range(batch_size):
        causal_mask[i, :, decoder_padding_mask[i] == 0] = 0
        causal_mask[i, decoder_padding_mask[i] == 0, :] = 0

    cross_mask = encoder_mask

    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'encoder_mask': encoder_mask,
        'decoder_mask': causal_mask,
        'decoder_padding_mask': decoder_padding_mask,
        'cross_mask': cross_mask,
    }


def collate_fn_bytes(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for ByteSeqDataset.

    IMPORTANT: byte values legitimately span the full [0, 255] range, so
    padding cannot be identified by "value == pad_id" the way token IDs can.
    Instead we pad with a harmless in-range placeholder (0) and build the
    padding masks directly from each sample's true length (src_len/tgt_len),
    which is known *before* padding and unambiguous.

    ByteLatentEncoder expects exactly this: raw bytes in [0,255] plus a
    separate boolean byte_mask (see blt.py) rather than an implicit pad ID.

    ByteLatentDecoder is GRU-based (causal by construction via recurrence),
    so no attention-style causal_mask is needed for it — we still return
    decoder_padding_mask so the training loop can mask padded positions out
    of the loss.
    """
    max_src_len_batch = max(item['src_len'] for item in batch)
    max_tgt_len_batch = max(item['tgt_len'] for item in batch)
    batch_size = len(batch)

    # Placeholder fill value is arbitrary (masked out downstream regardless);
    # 0 is a valid byte, kept only because it's a harmless, in-range default.
    PLACEHOLDER = 0

    encoder_input = torch.full((batch_size, max_src_len_batch), PLACEHOLDER, dtype=torch.long)
    decoder_input = torch.full((batch_size, max_tgt_len_batch), PLACEHOLDER, dtype=torch.long)
    encoder_mask = torch.zeros((batch_size, max_src_len_batch), dtype=torch.bool)
    decoder_padding_mask = torch.zeros((batch_size, max_tgt_len_batch), dtype=torch.bool)

    for i, item in enumerate(batch):
        src_bytes = item['src_bytes']
        tgt_bytes = item['tgt_bytes']

        encoder_input[i, :len(src_bytes)] = torch.tensor(src_bytes, dtype=torch.long)
        decoder_input[i, :len(tgt_bytes)] = torch.tensor(tgt_bytes, dtype=torch.long)

        # Mask built from true length, NOT from comparing against a pad value.
        encoder_mask[i, :len(src_bytes)] = True
        decoder_padding_mask[i, :len(tgt_bytes)] = True

    return {
        'encoder_input': encoder_input,   # (B, S) raw bytes in [0,255]
        'decoder_input': decoder_input,   # (B, T) raw bytes in [0,255], last real pos per row = EOS_CLASS (256)
        'encoder_mask': encoder_mask,     # (B, S) bool, True = real byte -> pass as byte_mask to ByteLatentEncoder
        'decoder_padding_mask': decoder_padding_mask,  # (B, T) bool, True = real position -> use for loss masking
        'cross_mask': encoder_mask,
    }


# =====================================================================
# Helper functions to create dataloaders
# =====================================================================

def create_dataloaders(
    src_tokenizer: BPETokenizer,
    tgt_tokenizer: BPETokenizer,
    splits_dir: str = "data.nosync/splits_packed",
    batch_size: int = 32,
    max_src_len: int = 1024,
    max_tgt_len: int = 640,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    for split_name in ['train', 'val', 'test']:
        dataset = TokenizedSeqDataset(
            split_name=split_name,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            splits_dir=splits_dir,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            collate_fn=collate_fn_tokenized,
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloaders


def create_byte_dataloaders(
    splits_dir: str = "data.nosync/splits_packed",
    batch_size: int = 32,
    max_src_len: int = 1024,
    max_tgt_len: int = 640,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    dataloaders = {}
    for split_name in ['train', 'val', 'test']:
        dataset = ByteSeqDataset(
            split_name=split_name,
            splits_dir=splits_dir,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        )
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            collate_fn=collate_fn_bytes,
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloaders