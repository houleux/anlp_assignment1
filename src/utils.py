"""
Evaluation metrics for Assignment 1 (Section 4): Bit-Level Accuracy,
Sequence Accuracy, Levenshtein Distance, BLEU, and ROUGE.

All metrics are computed from GREEDY DECODING (model.generate()), as the
assignment specifies, on the actual reconstructed plaintext -- not on
teacher-forced logits. Works generically across:
  - C1-C4 (Transformer, BPE token IDs)
  - C5 (ByteLatentTransformer, raw UTF-8 bytes)

BLEU/ROUGE are only meaningful for tokenized models (C1-C4) per the
assignment's Section 4 note ("for tokenized models only") -- for C5 these
are reported as None.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import Levenshtein as _fast_levenshtein  # pip install python-Levenshtein
except Exception:  # pragma: no cover
    _fast_levenshtein = None


# =====================================================================
# Core string/byte metrics
# =====================================================================

def levenshtein_distance(a: str, b: str) -> int:
    """Character-level edit distance. Uses the C-implemented `Levenshtein`
    package if installed (`pip install python-Levenshtein --break-system-packages`)
    -- much faster for long plaintext sequences (up to ~2670 chars here).
    Falls back to a pure-Python O(n*m) time, O(min(n,m)) space DP.
    """
    if _fast_levenshtein is not None:
        return _fast_levenshtein.distance(a, b)

    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (ca != cb)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row
    return previous_row[-1]


def bit_level_accuracy(pred_text: str, ref_text: str) -> float:
    """Percentage of exact bit matches, at the raw UTF-8 byte level.

    Predicted/reference are padded (with zero bytes) to the same length
    before comparing bit-by-bit -- a length mismatch therefore costs
    accuracy over the padded region rather than being silently ignored,
    which matches "exact bit matches" as a strict reconstruction metric.
    """
    pred_bytes = pred_text.encode("utf-8", errors="replace")
    ref_bytes = ref_text.encode("utf-8", errors="replace")
    max_len = max(len(pred_bytes), len(ref_bytes))
    if max_len == 0:
        return 100.0  # both empty: trivially identical

    pred_bytes = pred_bytes.ljust(max_len, b"\x00")
    ref_bytes = ref_bytes.ljust(max_len, b"\x00")

    matching_bits = 0
    total_bits = max_len * 8
    for pb, rb in zip(pred_bytes, ref_bytes):
        xor = pb ^ rb
        matching_bits += 8 - bin(xor).count("1")
    return 100.0 * matching_bits / total_bits


def sequence_match(pred_text: str, ref_text: str) -> bool:
    """Exact reconstruction: True iff prediction == reference verbatim."""
    return pred_text == ref_text


# =====================================================================
# BLEU (self-contained, no nltk dependency)
# =====================================================================

def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(candidate: Sequence[str], reference: Sequence[str], n: int) -> Tuple[int, int]:
    cand_ngrams = _ngram_counts(candidate, n)
    ref_ngrams = _ngram_counts(reference, n)
    clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
    total = max(1, sum(cand_ngrams.values()))
    return clipped, total


def bleu_score(candidate_text: str, reference_text: str, max_n: int = 4) -> float:
    """Single-reference BLEU (0-100), up to 4-grams, uniform weights, with
    brevity penalty. Uses additive (+1) smoothing on n-gram precision so a
    single missing higher-order n-gram doesn't zero out the whole score --
    appropriate for single-sentence-level scoring on possibly-short/imperfect
    reconstructions, unlike corpus-level BLEU which doesn't need it as much.
    Word-level tokenization via whitespace split (Brown-corpus English text;
    a simplification, not a proper tokenizer).
    """
    candidate = candidate_text.split()
    reference = reference_text.split()
    if not candidate or not reference:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        clipped, total = _modified_precision(candidate, reference, n)
        precisions.append((clipped + 1) / (total + 1))  # +1 smoothing

    geo_mean = 1.0
    for p in precisions:
        geo_mean *= p
    geo_mean = geo_mean ** (1.0 / max_n)

    bp = 1.0 if len(candidate) > len(reference) else torch.exp(
        torch.tensor(1.0 - len(reference) / max(1, len(candidate)))
    ).item()
    return 100.0 * bp * geo_mean


# =====================================================================
# ROUGE (self-contained: ROUGE-1 and ROUGE-L, F1)
# =====================================================================

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def rouge1_f1(candidate_text: str, reference_text: str) -> float:
    candidate = candidate_text.split()
    reference = reference_text.split()
    if not candidate or not reference:
        return 0.0
    cand_counts = Counter(candidate)
    ref_counts = Counter(reference)
    overlap = sum(min(c, ref_counts.get(w, 0)) for w, c in cand_counts.items())
    precision = overlap / len(candidate)
    recall = overlap / len(reference)
    if precision + recall == 0:
        return 0.0
    return 100.0 * 2 * precision * recall / (precision + recall)


def rougeL_f1(candidate_text: str, reference_text: str) -> float:
    candidate = candidate_text.split()
    reference = reference_text.split()
    if not candidate or not reference:
        return 0.0
    lcs = _lcs_length(candidate, reference)
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return 100.0 * 2 * precision * recall / (precision + recall)


# =====================================================================
# Decoding helpers
# =====================================================================

def _decode_tokenized_ids(ids: List[int], tokenizer, eos_id: int, special_ids: set) -> str:
    """Truncate at first EOS, drop any stray special-token IDs the model
    might emit mid-sequence (not expected post-training, but decode() may
    not be defined for IDs outside the tokenizer's real trained vocab), then
    detokenize."""
    if eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    ids = [i for i in ids if i not in special_ids]
    if not ids:
        return ""
    return tokenizer.decode(ids)


def _decode_byte_ids(ids: List[int], eos_class: int = 256) -> str:
    if eos_class in ids:
        ids = ids[: ids.index(eos_class)]
    ids = [i for i in ids if 0 <= i <= 255]
    return bytes(ids).decode("utf-8", errors="replace")


# =====================================================================
# Main evaluation entry point
# =====================================================================

@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    dataloader,
    model_name: str,
    device: torch.device,
    tgt_tokenizer=None,
    max_new_tokens: int = 640,
    max_batches: Optional[int] = None,
    compute_bleu_rouge: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run greedy decoding over `dataloader` and compute all Section-4
    metrics. Pass `max_batches` (e.g. 2-3) for a cheap periodic snapshot
    during training; leave it None for the full final test-set evaluation.

    `compute_bleu_rouge` defaults to True for C1-C4 and False for C5,
    matching the assignment's "for tokenized models only" note -- override
    only if you deliberately want a word-level BLEU/ROUGE on C5's decoded
    byte-output text for exploratory comparison (not the assignment's
    required number).
    """
    if compute_bleu_rouge is None:
        compute_bleu_rouge = model_name in {"C1", "C2", "C3", "C4"}

    model.eval()
    bit_accs, seq_matches, lev_dists, bleus, rouge1s, rougeLs = [], [], [], [], [], []
    n_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if model_name in {"C1", "C2", "C3", "C4"}:
            src_ids = batch["encoder_input"].to(device)
            src_valid = batch["encoder_mask"].bool().to(device)
            tgt_ids = batch["decoder_input"]  # reference, still on CPU is fine

            eos_id = model.config.eos_token_id
            pad_id = model.config.pad_token_id
            special_ids = {model.config.pad_token_id, model.config.bos_token_id, model.config.eos_token_id}

            generated = model.generate(src_ids, max_new_tokens=max_new_tokens, src_valid=src_valid)
            generated = generated.cpu().tolist()

            for row_idx in range(len(generated)):
                pred_ids = generated[row_idx]
                ref_ids_full = tgt_ids[row_idx].tolist()
                # Reference stored as [BOS, ...real tokens..., EOS, PAD, PAD...];
                # strip BOS, then decode via the same helper (truncates at EOS).
                ref_ids = ref_ids_full[1:]

                pred_text = _decode_tokenized_ids(pred_ids, tgt_tokenizer, eos_id, special_ids)
                ref_text = _decode_tokenized_ids(ref_ids, tgt_tokenizer, eos_id, special_ids)
                _accumulate(pred_text, ref_text, bit_accs, seq_matches, lev_dists,
                            bleus, rouge1s, rougeLs, compute_bleu_rouge)
                n_samples += 1

        elif model_name == "C5":
            src_bytes = batch["encoder_input"].to(device)
            src_mask = batch["encoder_mask"].bool().to(device)
            tgt_bytes = batch["decoder_input"]  # reference, CPU

            target_length = min(src_bytes.size(1), max_new_tokens)
            generated = model.generate(src_bytes, target_length=target_length, source_mask=src_mask)
            generated = generated.cpu().tolist()

            for row_idx in range(len(generated)):
                pred_ids = generated[row_idx]
                ref_ids = tgt_bytes[row_idx].tolist()  # no BOS prepended on this side, see dataset.py

                pred_text = _decode_byte_ids(pred_ids)
                ref_text = _decode_byte_ids(ref_ids)
                _accumulate(pred_text, ref_text, bit_accs, seq_matches, lev_dists,
                            bleus, rouge1s, rougeLs, compute_bleu_rouge=False)
                n_samples += 1
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    if n_samples == 0:
        return {"n_samples": 0}

    result: Dict[str, Any] = {
        "n_samples": n_samples,
        "bit_level_accuracy": sum(bit_accs) / n_samples,
        "sequence_accuracy": 100.0 * sum(seq_matches) / n_samples,
        "levenshtein_distance": sum(lev_dists) / n_samples,
    }
    if compute_bleu_rouge:
        result["bleu"] = sum(bleus) / n_samples
        result["rouge1"] = sum(rouge1s) / n_samples
        result["rougeL"] = sum(rougeLs) / n_samples
    else:
        result["bleu"] = None
        result["rouge1"] = None
        result["rougeL"] = None
    return result


def _accumulate(
    pred_text: str,
    ref_text: str,
    bit_accs: List[float],
    seq_matches: List[bool],
    lev_dists: List[int],
    bleus: List[float],
    rouge1s: List[float],
    rougeLs: List[float],
    compute_bleu_rouge: bool,
) -> None:
    bit_accs.append(bit_level_accuracy(pred_text, ref_text))
    seq_matches.append(sequence_match(pred_text, ref_text))
    lev_dists.append(levenshtein_distance(pred_text, ref_text))
    if compute_bleu_rouge:
        bleus.append(bleu_score(pred_text, ref_text))
        rouge1s.append(rouge1_f1(pred_text, ref_text))
        rougeLs.append(rougeL_f1(pred_text, ref_text))


__all__ = [
    "levenshtein_distance",
    "bit_level_accuracy",
    "sequence_match",
    "bleu_score",
    "rouge1_f1",
    "rougeL_f1",
    "evaluate_metrics",
]