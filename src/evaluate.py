"""Evaluate trained C1-C5 models and save metrics/plots under outputs/."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset import (  # noqa: E402
    ByteSeqDataset,
    SpecialTokens,
    TokenizedSeqDataset,
    collate_fn_bytes,
    collate_fn_tokenized,
)
from src.models.bpe import BPETokenizer  # noqa: E402
from src.train import build_model, get_device  # noqa: E402

TOKENIZED_MODELS = {"C1", "C2", "C3", "C4"}
ALL_MODELS = ["C1", "C2", "C3", "C4", "C5"]


def strip_token_ids(ids: Sequence[int]) -> List[int]:
    result = []
    for token_id in ids:
        if token_id == SpecialTokens.EOS_ID:
            break
        if token_id not in {
            SpecialTokens.PAD_ID,
            SpecialTokens.BOS_ID,
            SpecialTokens.UNK_ID,
        }:
            result.append(int(token_id))
    return result


def strip_byte_ids(ids: Sequence[int]) -> List[int]:
    result = []
    for value in ids:
        if value == 256:
            break
        if 0 <= value <= 255:
            result.append(int(value))
    return result


def levenshtein_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_value in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_value in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + (ref_value != hyp_value),
            ))
        previous = current
    return previous[-1]


def bit_accuracy(reference: bytes, hypothesis: bytes) -> float:
    compared = min(len(reference), len(hypothesis))
    matching_bits = sum(
        8 - (reference[index] ^ hypothesis[index]).bit_count()
        for index in range(compared)
    )
    total_bits = max(len(reference), len(hypothesis)) * 8
    return matching_bits / total_bits if total_bits else 1.0


def ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(tuple(tokens[index:index + order]) for index in range(len(tokens) - order + 1))


def corpus_bleu(references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]], max_order: int = 4) -> float:
    matches = [0] * max_order
    possible = [0] * max_order
    reference_length = sum(len(reference) for reference in references)
    hypothesis_length = sum(len(hypothesis) for hypothesis in hypotheses)

    for reference, hypothesis in zip(references, hypotheses):
        for order in range(1, max_order + 1):
            reference_counts = ngrams(reference, order)
            hypothesis_counts = ngrams(hypothesis, order)
            matches[order - 1] += sum((reference_counts & hypothesis_counts).values())
            possible[order - 1] += max(0, len(hypothesis) - order + 1)

    if hypothesis_length == 0:
        return 0.0
    # Add-one smoothing keeps BLEU informative for short test sequences where
    # higher-order n-grams do not exist.
    precisions = [
        (matches[index] + 1) / (possible[index] + 1)
        for index in range(max_order)
    ]
    log_precision = sum(math.log(value) for value in precisions) / max_order
    if hypothesis_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - reference_length / hypothesis_length)
    return brevity_penalty * math.exp(log_precision)


def rouge_l_f1(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    previous = [0] * columns
    for ref_token in reference:
        current = [0]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            if ref_token == hyp_token:
                current.append(previous[hyp_index - 1] + 1)
            else:
                current.append(max(previous[hyp_index], current[-1]))
        previous = current
    lcs = previous[-1]
    if not reference or not hypothesis or lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall)


def load_checkpoint(model_name: str, checkpoint_dir: Path, device: torch.device, encoder_tokenizer, decoder_tokenizer):
    checkpoint_path = checkpoint_dir / model_name / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        latest_path = checkpoint_dir / model_name / f"{model_name}_latest.pt"
        if latest_path.exists():
            checkpoint_path = latest_path
        else:
            raise FileNotFoundError(f"No checkpoint found for {model_name} in {checkpoint_dir / model_name}")
    model = build_model(model_name, encoder_tokenizer, decoder_tokenizer, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint_path


def evaluate_tokenized(model, loader, tokenizer, device, max_new_tokens: int, max_samples: int):
    references_text: List[str] = []
    hypotheses_text: List[str] = []
    bit_scores: List[float] = []
    distances: List[int] = []
    processed = 0

    with torch.no_grad():
        for batch in loader:
            if processed >= max_samples:
                break
            keep = min(batch["encoder_input"].size(0), max_samples - processed)
            src = batch["encoder_input"][:keep].to(device)
            src_valid = batch["encoder_mask"][:keep].bool().to(device)
            target = batch["decoder_input"][:keep]
            target_valid = batch["decoder_padding_mask"][:keep].bool()
            generated = model.generate(src, max_new_tokens=max_new_tokens, src_valid=src_valid).cpu()

            for index in range(keep):
                reference_ids = target[index][target_valid[index]].tolist()
                hypothesis_ids = strip_token_ids(generated[index].tolist())
                reference_text = tokenizer.decode(strip_token_ids(reference_ids))
                hypothesis_text = tokenizer.decode(hypothesis_ids)
                reference_bytes = reference_text.encode("utf-8")
                hypothesis_bytes = hypothesis_text.encode("utf-8")
                references_text.append(reference_text)
                hypotheses_text.append(hypothesis_text)
                bit_scores.append(bit_accuracy(reference_bytes, hypothesis_bytes))
                distances.append(levenshtein_distance(reference_text, hypothesis_text))
                processed += 1
                if processed % 10 == 0 or processed == max_samples:
                    print(f"  Processed {processed}/{max_samples} samples", flush=True)

    reference_tokens = [text.split() for text in references_text]
    hypothesis_tokens = [text.split() for text in hypotheses_text]
    exact = sum(reference == hypothesis for reference, hypothesis in zip(references_text, hypotheses_text))
    return {
        "samples": processed,
        "bit_level_accuracy": sum(bit_scores) / max(1, len(bit_scores)),
        "sequence_accuracy": exact / max(1, processed),
        "levenshtein_distance": sum(distances) / max(1, len(distances)),
        "bleu": corpus_bleu(reference_tokens, hypothesis_tokens),
        "rouge_l": sum(
            rouge_l_f1(reference, hypothesis)
            for reference, hypothesis in zip(reference_tokens, hypothesis_tokens)
        ) / max(1, len(reference_tokens)),
    }


def evaluate_bytes(model, loader, device, max_samples: int):
    bit_scores: List[float] = []
    distances: List[int] = []
    exact = 0
    processed = 0

    with torch.no_grad():
        for batch in loader:
            if processed >= max_samples:
                break
            keep = min(batch["encoder_input"].size(0), max_samples - processed)
            source = batch["encoder_input"][:keep].to(device)
            source_mask = batch["encoder_mask"][:keep].bool().to(device)
            target = batch["decoder_input"][:keep]
            target_valid = batch["decoder_padding_mask"][:keep]
            target_lengths = target_valid.sum(dim=1).tolist()
            generated = model.generate(
                source,
                target_length=max(target_lengths),
                source_mask=source_mask,
            ).cpu()

            for index, target_length in enumerate(target_lengths):
                reference = strip_byte_ids(target[index, :target_length].tolist())
                hypothesis = strip_byte_ids(generated[index].tolist())
                bit_scores.append(bit_accuracy(bytes(reference), bytes(hypothesis)))
                distances.append(levenshtein_distance(reference, hypothesis))
                exact += reference == hypothesis
                processed += 1
                if processed % 10 == 0 or processed == max_samples:
                    print(f"  Processed {processed}/{max_samples} samples", flush=True)

    return {
        "samples": processed,
        "bit_level_accuracy": sum(bit_scores) / max(1, len(bit_scores)),
        "sequence_accuracy": exact / max(1, processed),
        "levenshtein_distance": sum(distances) / max(1, len(distances)),
        "bleu": None,
        "rouge_l": None,
    }


def save_plots(results: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    model_names = list(results)
    numeric_metrics = ["bit_level_accuracy", "sequence_accuracy", "bleu", "rouge_l"]
    for metric in numeric_metrics:
        values = [results[name][metric] for name in model_names if results[name][metric] is not None]
        names = [name for name in model_names if results[name][metric] is not None]
        if not values:
            continue
        plt.figure(figsize=(8, 5))
        plt.bar(names, values)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} by model")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close()

    values = [results[name]["levenshtein_distance"] for name in model_names]
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, values)
    plt.ylabel("Average Levenshtein distance")
    plt.title("Levenshtein distance by model")
    plt.tight_layout()
    plt.savefig(output_dir / "levenshtein_distance.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained C1-C5 checkpoints.")
    parser.add_argument("--model", choices=["all", *ALL_MODELS], default="all")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--splits-dir", default="data.nosync/splits_packed")
    parser.add_argument("--tokenizers-dir", default="tokenizers")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    args = parser.parse_args()

    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_tokenizer = BPETokenizer.load(str(Path(args.tokenizers_dir) / "encoder.json"))
    decoder_tokenizer = BPETokenizer.load(str(Path(args.tokenizers_dir) / "decoder.json"))
    model_names = ALL_MODELS if args.model == "all" else [args.model]
    results: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        print(f"\nEvaluating {model_name}...", flush=True)
        model, checkpoint_path = load_checkpoint(
            model_name,
            checkpoint_dir,
            device,
            encoder_tokenizer if model_name in TOKENIZED_MODELS else None,
            decoder_tokenizer if model_name in TOKENIZED_MODELS else None,
        )
        if model_name in TOKENIZED_MODELS:
            dataset = TokenizedSeqDataset(
                "test", encoder_tokenizer, decoder_tokenizer,
                splits_dir=args.splits_dir, max_src_len=1024, max_tgt_len=args.max_new_tokens,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_tokenized)
            metrics = evaluate_tokenized(model, loader, decoder_tokenizer, device, args.max_new_tokens, args.max_samples)
        else:
            dataset = ByteSeqDataset("test", splits_dir=args.splits_dir, max_src_len=1024, max_tgt_len=args.max_new_tokens)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_bytes)
            metrics = evaluate_bytes(model, loader, device, args.max_samples)
        metrics["checkpoint"] = str(checkpoint_path)
        metrics["device"] = str(device)
        results[model_name] = metrics
        print(model_name, json.dumps(metrics, indent=2))

    results_path = output_dir / "evaluation_metrics.json"
    results_path.write_text(json.dumps(results, indent=2))
    save_plots(results, output_dir)
    print(f"Saved metrics to {results_path}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
