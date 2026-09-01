"""
Byte Pair Encoding implementation (used AI):

Usage:
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=1000)
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)

    tokenizer.save("tokenizer.json")
    tokenizer2 = BPETokenizer.load("tokenizer.json")
"""

import heapq
import json
import regex as re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# GPT-2 style pre-tokenization pattern: splits text into words/punctuation/
# whitespace chunks so merges never cross word boundaries.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self):
        # id -> bytes
        self.vocab: Dict[int, bytes] = {}
        # (id1, id2) -> merged_id, in the order merges were learned
        self.merges: Dict[Tuple[int, int], int] = {}
        self.pattern = re.compile(GPT2_SPLIT_PATTERN)
        self.special_tokens: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size must be at least 256"
        num_merges = vocab_size - 256

        # 1. Pre-tokenize into words, count occurrences of each unique word.
        words = self.pattern.findall(text)
        word_counts = Counter(words)

        # 2. Represent each unique word as a tuple of byte-ids (0-255).
        #    seqs[i] = list of current token ids for unique word i
        #    counts[i] = how many times that word occurs in the corpus
        seqs: List[List[int]] = [list(w.encode("utf-8")) for w in word_counts]
        counts: List[int] = list(word_counts.values())

        # 3. Build initial pair -> total frequency, and pair -> set of word
        #    indices that contain it (so we know what to update on a merge).
        pair_freq: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_words: Dict[Tuple[int, int], set] = defaultdict(set)

        def add_word_pairs(idx: int):
            seq = seqs[idx]
            c = counts[idx]
            for a, b in zip(seq, seq[1:]):
                pair_freq[(a, b)] += c
                pair_words[(a, b)].add(idx)

        for i in range(len(seqs)):
            add_word_pairs(i)

        # 4. Max-heap of (-freq, pair), with lazy deletion via a version map.
        heap = [(-f, p) for p, f in pair_freq.items()]
        heapq.heapify(heap)

        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: Dict[Tuple[int, int], int] = {}

        next_id = 256
        merges_done = 0

        while merges_done < num_merges and heap:
            neg_freq, pair = heapq.heappop(heap)
            freq = -neg_freq

            # Lazy deletion: skip stale heap entries whose freq no longer
            # matches the authoritative count.
            if pair not in pair_freq or pair_freq[pair] != freq or freq <= 0:
                continue

            a, b = pair
            new_id = next_id
            vocab[new_id] = vocab[a] + vocab[b]
            merges[pair] = new_id

            if verbose:
                print(f"merge {merges_done+1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}) had {freq} occurrences")

            # Apply this merge to every word that contains the pair.
            affected = list(pair_words.get(pair, ()))
            for idx in affected:
                seq = seqs[idx]
                c = counts[idx]
                if len(seq) < 2:
                    continue

                # Remove old pair contributions for this word.
                for x, y in zip(seq, seq[1:]):
                    pair_freq[(x, y)] -= c
                    pair_words[(x, y)].discard(idx)

                # Merge occurrences of (a, b) -> new_id in this word's sequence.
                new_seq = []
                i = 0
                n = len(seq)
                while i < n:
                    if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                seqs[idx] = new_seq

                # Add new pair contributions and push updated heap entries.
                for x, y in zip(new_seq, new_seq[1:]):
                    pair_freq[(x, y)] += c
                    pair_words[(x, y)].add(idx)
                    if pair_freq[(x, y)] > 0:
                        heapq.heappush(heap, (-pair_freq[(x, y)], (x, y)))

            del pair_freq[pair]
            pair_words.pop(pair, None)
            next_id += 1
            merges_done += 1

        self.vocab = vocab
        self.merges = merges

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def _encode_word(self, word_bytes: bytes) -> List[int]:
        ids = list(word_bytes)
        if len(ids) < 2:
            return ids

        while True:
            # Find the eligible pair with the lowest merge rank (i.e. the
            # one learned earliest during training).
            best_pair = None
            best_rank = None
            for a, b in zip(ids, ids[1:]):
                rank = self.merges.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                break

            a, b = best_pair
            new_id = self.merges[best_pair]
            new_ids = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in self.pattern.findall(text):
            ids.extend(self._encode_word(word.encode("utf-8")))
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = bytearray()
        for i in ids:
            parts.extend(self.vocab[i])
        return parts.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "vocab_size": 256 + len(self.merges),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {i: bytes([i]) for i in range(256)}
        tok.merges = {}
        for a, b, new_id in data["merges"]:
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok