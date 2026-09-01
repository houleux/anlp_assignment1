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
# Special Token IDs
# =====================================================================

class SpecialTokens:
    """
    Special token IDs consistent across all tokenizers.
    
    BPE tokenizers produce tokens in range [0, vocab_size) = [0, 1024).
    Special tokens must be outside this range to avoid collisions with real tokens.
    
    We use:
    - <pad> = 1024 (outside vocab range, no collision with real tokens)
    - <unk> = 1025 (reserved, though BPE never produces <unk>)
    - <bos> = 1026
    - <eos> = 1027
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
    
    @classmethod
    def validate_tokenizers(cls, tokenizer_src, tokenizer_tgt):
        """
        Validate that special tokens are consistent across tokenizers.
        
        For BPE, we assume IDs are assigned sequentially, so we just check
        that both tokenizers can handle the special token IDs.
        """
        print("✓ Special tokens defined:")
        for name, token_id in cls.get_all().items():
            print(f"  <{name}> = {token_id}")
        return True


# =====================================================================
# TokenizedSeqDataset (for C1-C4: with tokenizers)
# =====================================================================

class TokenizedSeqDataset(Dataset):
    """
    Dataset that tokenizes cipher and plaintext using BPE tokenizers.
    
    Suitable for C1-C4 configurations.
    
    Process:
    1. Load cipher bytes and plaintext from split files
    2. Tokenize with src_tokenizer (encoder) and tgt_tokenizer (decoder)
    3. Add special tokens: <bos> ... <eos>
    4. Truncate to max_src_len and max_tgt_len
    5. Return (encoder_ids, decoder_ids, src_len, tgt_len)
    
    Collate function pads dynamically per batch.
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
        """
        Initialize TokenizedSeqDataset.
        
        Args:
            split_name: 'train', 'val', or 'test'
            src_tokenizer: BPE tokenizer for cipher (encoder source)
            tgt_tokenizer: BPE tokenizer for plaintext (decoder target)
            splits_dir: Path to packed splits
            max_src_len: Max encoder input length (after tokenization)
            max_tgt_len: Max decoder output length (after tokenization)
        """
        self.split_name = split_name
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Load split data
        cipher_bytes_list, plaintext_list = self._load_split(splits_dir, split_name)
        
        # Tokenize and prepare
        self.samples = []
        for cipher_bytes, plaintext in zip(cipher_bytes_list, plaintext_list):
            sample = self._process_pair(cipher_bytes, plaintext)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"✓ Loaded {split_name} split: {len(self.samples)} samples")
    
    def _load_split(self, splits_dir: str, split_name: str):
        """Load cipher bytes and plaintext from split files."""
        splits_path = Path(splits_dir)
        
        # Load cipher bytes
        cipher_path = splits_path / f"{split_name}_cipher.pkl"
        with open(cipher_path, 'rb') as f:
            cipher_bytes_list = pickle.load(f)
        
        # Load plaintext
        plaintext_path = splits_path / f"{split_name}_plain.txt"
        with open(plaintext_path, 'r') as f:
            plaintext_list = [line.rstrip('\n') for line in f.readlines()]
        
        return cipher_bytes_list, plaintext_list
    
    def _process_pair(self, cipher_bytes: bytes, plaintext: str) -> Optional[Dict]:
        """
        Tokenize and process a cipher-plaintext pair.
        
        Process:
        1. Convert cipher bytes to string (latin-1)
        2. Tokenize both
        3. Add BOS/EOS tokens
        4. Truncate if needed
        5. Return dict with token IDs and lengths
        """
        # Decode cipher bytes to string
        try:
            cipher_str = cipher_bytes.decode('latin-1')
        except Exception as e:
            print(f"Warning: Failed to decode cipher bytes: {e}")
            return None
        
        # Tokenize
        src_ids = self.src_tokenizer.encode(cipher_str)
        tgt_ids = self.tgt_tokenizer.encode(plaintext)
        
        # Add special tokens: <bos> ... <eos>
        src_ids = [SpecialTokens.BOS_ID] + src_ids + [SpecialTokens.EOS_ID]
        tgt_ids = [SpecialTokens.BOS_ID] + tgt_ids + [SpecialTokens.EOS_ID]
        
        # Truncate
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
    
    No tokenization; feeds raw byte sequences directly.
    Each byte value (0-255) becomes an input feature.
    
    Process:
    1. Load cipher bytes (stays as bytes)
    2. Convert plaintext to bytes (UTF-8)
    3. Add special tokens: <bos> (256) and <eos> (257)
    4. Truncate to max lengths
    5. Return (encoder_bytes, decoder_bytes, src_len, tgt_len)
    
    Collate function converts to tensors and pads.
    """
    
    def __init__(
        self,
        split_name: str,
        splits_dir: str = "data.nosync/splits_packed",
        max_src_len: int = 1024,
        max_tgt_len: int = 640,
    ):
        """
        Initialize ByteSeqDataset.
        
        Args:
            split_name: 'train', 'val', or 'test'
            splits_dir: Path to packed splits
            max_src_len: Max encoder input length (in bytes)
            max_tgt_len: Max decoder output length (in bytes)
        """
        self.split_name = split_name
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Special byte tokens (outside 0-255 range to avoid collisions)
        self.PAD_BYTE = 256    # For padding (unique from BOS/EOS)
        self.BOS_BYTE = 257
        self.EOS_BYTE = 258
        
        # Load split data
        cipher_bytes_list, plaintext_list = self._load_split(splits_dir, split_name)
        
        # Process
        self.samples = []
        for cipher_bytes, plaintext in zip(cipher_bytes_list, plaintext_list):
            sample = self._process_pair(cipher_bytes, plaintext)
            if sample is not None:
                self.samples.append(sample)
        
        print(f"✓ Loaded {split_name} split (byte-level): {len(self.samples)} samples")
    
    def _load_split(self, splits_dir: str, split_name: str):
        """Load cipher bytes and plaintext from split files."""
        splits_path = Path(splits_dir)
        
        # Load cipher bytes
        cipher_path = splits_path / f"{split_name}_cipher.pkl"
        with open(cipher_path, 'rb') as f:
            cipher_bytes_list = pickle.load(f)
        
        # Load plaintext
        plaintext_path = splits_path / f"{split_name}_plain.txt"
        with open(plaintext_path, 'r') as f:
            plaintext_list = [line.rstrip('\n') for line in f.readlines()]
        
        return cipher_bytes_list, plaintext_list
    
    def _process_pair(self, cipher_bytes: bytes, plaintext: str) -> Optional[Dict]:
        """
        Process a cipher-plaintext pair at byte level.
        
        Process:
        1. Keep cipher as bytes (list of 0-255 values)
        2. Convert plaintext to UTF-8 bytes
        3. Add BOS/EOS special bytes (257/258, outside byte range)
        4. Truncate if needed
        5. Return dicts with byte values
        """
        # Cipher: convert to list of byte values
        src_bytes = list(cipher_bytes)
        
        # Plaintext: encode to UTF-8 bytes
        try:
            tgt_bytes = list(plaintext.encode('utf-8'))
        except Exception as e:
            print(f"Warning: Failed to encode plaintext: {e}")
            return None
        
        # Add special tokens (outside 0-255 range to avoid collision)
        src_bytes = [self.BOS_BYTE] + src_bytes + [self.EOS_BYTE]
        tgt_bytes = [self.BOS_BYTE] + tgt_bytes + [self.EOS_BYTE]
        
        # Truncate
        if len(src_bytes) > self.max_src_len:
            src_bytes = src_bytes[:self.max_src_len - 1] + [self.EOS_BYTE]
        if len(tgt_bytes) > self.max_tgt_len:
            tgt_bytes = tgt_bytes[:self.max_tgt_len - 1] + [self.EOS_BYTE]
        
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for TokenizedSeqDataset.
    
    Dynamically pads batch to its own max lengths (not global fixed max).
    
    TEACHER FORCING NOTE:
    - decoder_input from this function contains: [<bos> tok1 tok2 ... tokN <eos>]
    - For training with teacher forcing, in your training loop use:
        decoder_input_feed = decoder_input[:, :-1]      # [<bos> tok1 ... tokN]
        decoder_target = decoder_input[:, 1:]           # [tok1 tok2 ... <eos>]
    - This ensures the decoder predicts the next token given previous context.
    
    Returns:
        encoder_input: (batch_size, max_src_len) padded token IDs
        decoder_input: (batch_size, max_tgt_len) padded token IDs
        encoder_mask: (batch_size, max_src_len) - 1 for real, 0 for padding
        decoder_mask: (batch_size, max_tgt_len, max_tgt_len) - causal mask
        cross_mask: (batch_size, max_src_len) - 1 for real, 0 for padding (for cross-attn)
    """
    
    # Get dynamic max lengths for this batch
    max_src_len_batch = max(item['src_len'] for item in batch)
    max_tgt_len_batch = max(item['tgt_len'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    encoder_input = torch.full((batch_size, max_src_len_batch), pad_id, dtype=torch.long)
    decoder_input = torch.full((batch_size, max_tgt_len_batch), pad_id, dtype=torch.long)
    
    # Fill in token IDs
    for i, item in enumerate(batch):
        src_ids = item['src_ids']
        tgt_ids = item['tgt_ids']
        
        encoder_input[i, :len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        decoder_input[i, :len(tgt_ids)] = torch.tensor(tgt_ids, dtype=torch.long)
    
    # Create padding masks (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_input != pad_id).float()  # (batch_size, max_src_len)
    decoder_padding_mask = (decoder_input != pad_id).float()  # (batch_size, max_tgt_len)
    
    # Create causal mask for decoder self-attention (lower triangular)
    # Shape: (batch_size, max_tgt_len, max_tgt_len)
    causal_mask = torch.tril(torch.ones(max_tgt_len_batch, max_tgt_len_batch))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, max_tgt_len_batch, max_tgt_len_batch)
    
    # Apply padding to causal mask: set padded positions to 0
    # For decoder self-attention: position i can attend to j iff i >= j and j is not padded
    for i in range(batch_size):
        # Set columns (key positions) to 0 where there's padding
        causal_mask[i, :, decoder_padding_mask[i] == 0] = 0
        # Set rows (query positions) to 0 where there's padding
        causal_mask[i, decoder_padding_mask[i] == 0, :] = 0
    
    # Cross-attention mask (encoder padding)
    # Shape: (batch_size, max_src_len)
    # Used to mask out padding positions in encoder for cross-attention
    cross_mask = encoder_mask
    
    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'encoder_mask': encoder_mask,
        'decoder_mask': causal_mask,
        'decoder_padding_mask': decoder_padding_mask,
        'cross_mask': cross_mask,
    }


def collate_fn_bytes(
    batch: List[Dict],
    pad_id: int = 256,  # Use 256 (outside byte range 0-255) to avoid collision with real data
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for ByteSeqDataset.
    
    Converts byte values to tensors and creates masks.
    
    TEACHER FORCING NOTE:
    - decoder_input from this function contains: [<bos_byte> byte1 byte2 ... byteN <eos_byte>]
    - For training with teacher forcing, in your training loop use:
        decoder_input_feed = decoder_input[:, :-1]      # [<bos_byte> byte1 ... byteN]
        decoder_target = decoder_input[:, 1:]           # [byte1 byte2 ... <eos_byte>]
    - This ensures the decoder predicts the next byte given previous context.
    
    Returns:
        encoder_input: (batch_size, max_src_len) padded byte values
        decoder_input: (batch_size, max_tgt_len) padded byte values
        encoder_mask: (batch_size, max_src_len) - 1 for real, 0 for padding
        decoder_mask: (batch_size, max_tgt_len, max_tgt_len) - causal mask
    """
    
    # Get dynamic max lengths
    max_src_len_batch = max(item['src_len'] for item in batch)
    max_tgt_len_batch = max(item['tgt_len'] for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors
    encoder_input = torch.full((batch_size, max_src_len_batch), pad_id, dtype=torch.long)
    decoder_input = torch.full((batch_size, max_tgt_len_batch), pad_id, dtype=torch.long)
    
    # Fill in byte values
    for i, item in enumerate(batch):
        src_bytes = item['src_bytes']
        tgt_bytes = item['tgt_bytes']
        
        encoder_input[i, :len(src_bytes)] = torch.tensor(src_bytes, dtype=torch.long)
        decoder_input[i, :len(tgt_bytes)] = torch.tensor(tgt_bytes, dtype=torch.long)
    
    # Create masks
    encoder_mask = (encoder_input != pad_id).float()
    decoder_padding_mask = (decoder_input != pad_id).float()
    
    # Causal mask for decoder
    causal_mask = torch.tril(torch.ones(max_tgt_len_batch, max_tgt_len_batch))
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, max_tgt_len_batch, max_tgt_len_batch)
    
    # Apply padding to causal mask
    for i in range(batch_size):
        causal_mask[i, :, decoder_padding_mask[i] == 0] = 0
        causal_mask[i, decoder_padding_mask[i] == 0, :] = 0
    
    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'encoder_mask': encoder_mask,
        'decoder_mask': causal_mask,
        'decoder_padding_mask': decoder_padding_mask,
        'cross_mask': encoder_mask,
    }


# =====================================================================
# Helper function to create dataloaders
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
    """
    Create dataloaders for train, val, test splits.
    
    Args:
        src_tokenizer: BPE tokenizer for encoder (cipher)
        tgt_tokenizer: BPE tokenizer for decoder (plaintext)
        splits_dir: Path to split files
        batch_size: Batch size for dataloaders
        max_src_len: Max encoder length
        max_tgt_len: Max decoder length
        num_workers: Number of workers for data loading
        
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    
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
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            collate_fn=collate_fn_tokenized,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders


def create_byte_dataloaders(
    splits_dir: str = "data.nosync/splits_packed",
    batch_size: int = 32,
    max_src_len: int = 1024,
    max_tgt_len: int = 640,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create byte-level dataloaders (for C5/BLT).
    
    Args:
        splits_dir: Path to split files
        batch_size: Batch size
        max_src_len: Max encoder length
        max_tgt_len: Max decoder length
        num_workers: Number of workers
        
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    
    dataloaders = {}
    
    for split_name in ['train', 'val', 'test']:
        dataset = ByteSeqDataset(
            split_name=split_name,
            splits_dir=splits_dir,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            collate_fn=collate_fn_bytes,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders
