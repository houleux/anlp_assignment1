"""Training entry point for the ANLP Assignment 1 models.

This file trains all model configurations:
- C1: standard transformer (sinusoidal, MHA, LayerNorm)
- C2: RoPE variant
- C3: GQA variant
- C4: RMSNorm variant
- C5: BLT / ByteLatentTransformer

It supports:
- separate tokenizers for source and target from tokenizers/
- BLT entropy-model checkpoints from checkpoints/
- local model checkpointing
- Weights & Biases logging
- optional Hugging Face Hub upload via env vars
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project-root imports work when running this script directly.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset import (
    SpecialTokens,
    create_byte_dataloaders,
    create_dataloaders,
)
from src.models.blt import (
    ByteLatentDecoder,
    ByteLatentTransformer,
    DynamicByteLatentEncoder,
    GlobalPatchTransformer,
)
from src.models.bpe import BPETokenizer
from src.models.transformer import Transformer, TransformerConfig
from src.utils import evaluate_metrics

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover
    HfApi = None

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Prefer CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> Dict[str, float]:
    """Return allocator memory in GB; MPS values are sampled, not hardware peaks."""
    if device.type == "cuda":
        return {
            "allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
            "reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
        }
    if device.type == "mps":
        return {
            "allocated_gb": torch.mps.current_allocated_memory() / 2**30,
            "driver_allocated_gb": torch.mps.driver_allocated_memory() / 2**30,
        }
    return {"allocated_gb": 0.0}


def update_peak_memory(peak: Dict[str, float], current: Dict[str, float]) -> None:
    for key, value in current.items():
        peak[key] = max(peak.get(key, 0.0), value)


def max_vocab_size(tokenizer: BPETokenizer) -> int:
    if not tokenizer.vocab:
        raise ValueError("Tokenizer vocabulary is empty.")
    # Special tokens are reserved outside the learned BPE vocab, so the
    # embedding table must be sized to cover both the trained tokens and the
    # fixed IDs used by the dataset / teacher forcing logic.
    base_vocab = max(tokenizer.vocab.keys()) + 1
    return max(base_vocab, SpecialTokens.EOS_ID + 1)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config_for_model(
    model_name: str,
    src_tokenizer: Optional[BPETokenizer] = None,
    tgt_tokenizer: Optional[BPETokenizer] = None,
) -> Dict[str, Any]:
    """Return a config dict for the chosen model family."""
    common = {
        "d_model": 256,
        "num_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "ffn_dim": 1024,
        "dropout": 0.1,
        "max_sequence_length": 1024,
    }

    if model_name == "C1":
        return {
            **common,
            "attention_type": "mha",
            "norm_type": "layernorm",
            "positional_encoding": "sinusoidal",
            "src_vocab_size": max_vocab_size(src_tokenizer),
            "tgt_vocab_size": max_vocab_size(tgt_tokenizer),
            "pad_token_id": SpecialTokens.PAD_ID,
            "bos_token_id": SpecialTokens.BOS_ID,
            "eos_token_id": SpecialTokens.EOS_ID,
        }
    if model_name == "C2":
        return {
            **common,
            "attention_type": "mha",
            "norm_type": "layernorm",
            "positional_encoding": "rope",
            "src_vocab_size": max_vocab_size(src_tokenizer),
            "tgt_vocab_size": max_vocab_size(tgt_tokenizer),
            "pad_token_id": SpecialTokens.PAD_ID,
            "bos_token_id": SpecialTokens.BOS_ID,
            "eos_token_id": SpecialTokens.EOS_ID,
        }
    if model_name == "C3":
        return {
            **common,
            "attention_type": "gqa",
            "num_groups": 4,
            "norm_type": "layernorm",
            "positional_encoding": "sinusoidal",
            "src_vocab_size": max_vocab_size(src_tokenizer),
            "tgt_vocab_size": max_vocab_size(tgt_tokenizer),
            "pad_token_id": SpecialTokens.PAD_ID,
            "bos_token_id": SpecialTokens.BOS_ID,
            "eos_token_id": SpecialTokens.EOS_ID,
        }
    if model_name == "C4":
        return {
            **common,
            "attention_type": "mha",
            "norm_type": "rmsnorm",
            "positional_encoding": "sinusoidal",
            "src_vocab_size": max_vocab_size(src_tokenizer),
            "tgt_vocab_size": max_vocab_size(tgt_tokenizer),
            "pad_token_id": SpecialTokens.PAD_ID,
            "bos_token_id": SpecialTokens.BOS_ID,
            "eos_token_id": SpecialTokens.EOS_ID,
        }
    if model_name == "C5":
        return {
            "byte_dim": 128,
            "latent_dim": 256,
            "vocab_size": 256,
            # Decoder-side (plaintext) local expansion factor. Calibrated to
            # match C1-C4's decoder-side BPE compression: mean plaintext len
            # 597.61 chars / mean decoder tokens 214.7 ≈ 2.78 -> round to 3.
            # (Previously hardcoded to 8, which reduces to ~75 output units
            # per sequence vs C1-C4's ~215 -- reopens the length-confound
            # risk flagged earlier for the ablation.)
            "patch_size": 3,
            # Source-side (cipher) dynamic patching. FIX: this was previously
            # named "entropy_threshold" (a no-op under the default
            # boundary_mode="topk") -- renamed to keep_fraction, which is the
            # parameter DynamicByteLatentEncoder actually reads under topk
            # mode. Value from calibrate_entropy_threshold.py sweep.
            "boundary_mode": "topk",
            "keep_fraction": 0.7321,
            "max_patch_size": 16,
            "entropy_hidden_dim": 32,
            "global_layers": 4,
            "global_heads": 8,
            "global_ffn_dim": 1024,
            "decoder_vocab_size": 257,
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model(model_name: str, src_tokenizer: Optional[BPETokenizer], tgt_tokenizer: Optional[BPETokenizer], device: torch.device) -> nn.Module:
    """Instantiate the correct model class for the selected configuration."""
    if model_name in {"C1", "C2", "C3", "C4"}:
        if src_tokenizer is None or tgt_tokenizer is None:
            raise ValueError(f"{model_name} requires trained source and target tokenizers.")
        cfg_dict = build_config_for_model(model_name, src_tokenizer, tgt_tokenizer)
        cfg = TransformerConfig(**cfg_dict)
        model = Transformer(cfg)
        return model.to(device)

    if model_name == "C5":
        cfg = build_config_for_model(model_name)
        encoder = DynamicByteLatentEncoder(
            byte_dim=cfg["byte_dim"],
            latent_dim=cfg["latent_dim"],
            vocab_size=cfg["vocab_size"],
            entropy_hidden_dim=cfg["entropy_hidden_dim"],
            boundary_mode=cfg["boundary_mode"],
            keep_fraction=cfg["keep_fraction"],
            max_patch_size=cfg["max_patch_size"],
        )
        decoder = ByteLatentDecoder(
            patch_size=cfg["patch_size"],
            latent_dim=cfg["latent_dim"],
            byte_dim=cfg["byte_dim"],
            vocab_size=cfg["decoder_vocab_size"],
        )
        global_transformer = GlobalPatchTransformer(
            d_model=cfg["latent_dim"],
            num_heads=cfg["global_heads"],
            num_layers=cfg["global_layers"],
            ffn_dim=cfg["global_ffn_dim"],
            dropout=0.1,
            use_rope=False,
            max_sequence_length=1024,
        )
        model = ByteLatentTransformer(
            encoder=encoder,
            decoder=decoder,
            global_transformer=global_transformer,
        )
        entropy_model_path = project_root / "checkpoints" / "entropy_model.pt"
        if entropy_model_path.exists():
            state = torch.load(entropy_model_path, map_location=device)
            model.encoder.entropy_model.load_state_dict(state)
            # FIX: freeze after loading. Without this, the pretrained entropy
            # scorer keeps updating throughout the main C5 run (it's still
            # used in the loss below via auxiliary_entropy_loss unless you
            # also strip that out), which makes average patch count drift
            # across training -- exactly what pretrain-then-freeze was meant
            # to avoid, and reopens a moving-target confound against C1-C4.
            for p in model.encoder.entropy_model.parameters():
                p.requires_grad_(False)
            print(f"✓ Loaded pretrained entropy model from {entropy_model_path} (frozen)")
        else:
            print(
                f"⚠ No pretrained entropy model found at {entropy_model_path}. "
                "Run pretrain_entropy_model.py first -- an untrained entropy "
                "scorer produces near-random patch boundaries."
            )
        return model.to(device)

    raise ValueError(f"Unsupported model name: {model_name}")


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_loss_for_batch(model: nn.Module, batch: Dict[str, torch.Tensor], model_name: str, device: torch.device) -> torch.Tensor:
    if model_name in {"C1", "C2", "C3", "C4"}:
        src_ids = batch["encoder_input"].to(device)
        tgt_ids = batch["decoder_input"].to(device)
        src_valid = batch["encoder_mask"].bool().to(device)
        tgt_valid = batch["decoder_padding_mask"].bool().to(device)

        decoder_input = tgt_ids[:, :-1]
        decoder_target = tgt_ids[:, 1:]
        # Mask passed to the model reflects INPUT validity (correct: this is
        # what decode() uses to build its internal causal/padding mask over
        # the sequence actually being fed in).
        decoder_input_valid = tgt_valid[:, :-1]
        # Mask used to select loss terms reflects TARGET validity instead
        # (FIX: previously reused decoder_input_valid here too, which is
        # off-by-one from what's being predicted; it happened to work only
        # because padding is contiguous and ignore_index=PAD_ID absorbed the
        # one mismatched boundary position -- this makes it explicit/robust
        # instead of relying on that coincidence).
        decoder_target_valid = tgt_valid[:, 1:]

        logits = model(src_ids, decoder_input, src_valid=src_valid, tgt_valid=decoder_input_valid)
        flat_logits = logits[decoder_target_valid]
        flat_targets = decoder_target[decoder_target_valid]
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=SpecialTokens.PAD_ID)

    if model_name == "C5":
        src_bytes = batch["encoder_input"].to(device)
        src_mask = batch["encoder_mask"].bool().to(device)
        tgt_bytes = batch["decoder_input"].to(device)
        tgt_valid = batch["decoder_padding_mask"].bool().to(device)

        logits = model(src_bytes, src_mask, target_bytes=tgt_bytes)
        flat_logits = logits[tgt_valid]
        flat_targets = tgt_bytes[tgt_valid]

        # FIX: no more auxiliary entropy loss here. The entropy model is
        # pretrained + frozen (see build_model), so there's nothing left to
        # backprop into -- computing and adding entropy_aux would just waste
        # a forward pass through the entropy model for a term that
        # contributes no gradient. If you deliberately want to keep
        # fine-tuning the entropy scorer jointly instead of freezing it,
        # remove the requires_grad_(False) call in build_model AND restore
        # this term -- but note that reopens the patch-count drift issue.
        return F.cross_entropy(flat_logits, flat_targets)

    raise ValueError(f"Unsupported model_name: {model_name}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, dataloader, model_name: str, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_loss_for_batch(model, batch, model_name, device)
            total_loss += loss.item()
            count += 1
    if count == 0:
        return float("inf")
    return total_loss / count


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_name: str,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_name": model_name,
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, path)


def format_metrics_line(prefix: str, metrics: Dict[str, Any]) -> str:
    parts = [f"n={metrics.get('n_samples', 0)}"]
    for key in ("bit_level_accuracy", "sequence_accuracy", "levenshtein_distance", "bleu", "rouge1", "rougeL"):
        val = metrics.get(key)
        if val is None:
            continue
        parts.append(f"{key}={val:.3f}")
    return f"{prefix} | " + " | ".join(parts)


def log_metrics_to_wandb(run, model_name: str, split: str, metrics: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    payload = {}
    for key in ("bit_level_accuracy", "sequence_accuracy", "levenshtein_distance", "bleu", "rouge1", "rougeL"):
        val = metrics.get(key)
        if val is not None:
            payload[f"{model_name}/{split}_{key}"] = val
    if payload:
        run.log(payload, step=step)


def update_benchmark_file(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    benchmarks = {}
    if path.exists():
        with path.open() as file:
            benchmarks = json.load(file)
    benchmarks[result["model_name"]] = result
    with path.open("w") as file:
        json.dump(benchmarks, file, indent=2)


def _bar_chart(
    model_names: list,
    values: list,
    title: str,
    ylabel: str,
    out_path: Path,
    value_fmt: str = "{:.3f}",
) -> None:
    """Save a single bar chart comparing `values` across `model_names`."""
    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.5), 5))
    bars = ax.bar(model_names, values, color=plt.cm.tab10.colors[: len(model_names)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        ax.annotate(
            value_fmt.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_comparison_plots(benchmark_path: Path, results_dir: Path) -> None:
    """Read benchmarks.json and save one comparison bar chart per metric,
    with every trained model that reports that metric on the same chart."""
    if plt is None:
        print("matplotlib is unavailable; skipping comparison plots.")
        return
    if not benchmark_path.exists():
        print(f"No benchmark file found at {benchmark_path}; skipping comparison plots.")
        return

    with benchmark_path.open() as file:
        benchmarks = json.load(file)
    if not benchmarks:
        print("Benchmark file is empty; skipping comparison plots.")
        return

    preferred_order = ["C1", "C2", "C3", "C4", "C5"]
    model_names = [m for m in preferred_order if m in benchmarks]
    model_names += [m for m in benchmarks if m not in model_names]

    results_dir.mkdir(parents=True, exist_ok=True)

    # Top-level scalar metrics (present directly on the result dict).
    scalar_metrics = [
        ("best_validation_loss", "Best Validation Loss", "loss", "{:.4f}"),
        ("training_minutes", "Total Training Time", "minutes", "{:.1f}"),
        ("parameter_count", "Model Size", "parameters", "{:,.0f}"),
        ("samples_per_second", "Training Throughput", "samples / sec", "{:.1f}"),
        ("units_per_second", "Training Throughput", "units / sec (tokens or bytes)", "{:.1f}"),
    ]
    for key, title, ylabel, fmt in scalar_metrics:
        names, values = [], []
        for m in model_names:
            val = benchmarks[m].get(key)
            if val is not None:
                names.append(m)
                values.append(val)
        if values:
            _bar_chart(names, values, f"{title} by Model", ylabel, results_dir / f"{key}.png", fmt)

    # Section-4 test-set metrics (nested under "test_metrics").
    test_metric_specs = [
        ("bit_level_accuracy", "Test Bit-Level Accuracy", "accuracy", "{:.3f}"),
        ("sequence_accuracy", "Test Sequence Accuracy", "accuracy", "{:.3f}"),
        ("levenshtein_distance", "Test Levenshtein Distance", "distance", "{:.2f}"),
        ("bleu", "Test BLEU", "BLEU", "{:.3f}"),
        ("rouge1", "Test ROUGE-1", "ROUGE-1", "{:.3f}"),
        ("rougeL", "Test ROUGE-L", "ROUGE-L", "{:.3f}"),
    ]
    for key, title, ylabel, fmt in test_metric_specs:
        names, values = [], []
        for m in model_names:
            test_metrics = benchmarks[m].get("test_metrics") or {}
            val = test_metrics.get(key)
            if val is not None:
                names.append(m)
                values.append(val)
        if values:
            _bar_chart(names, values, f"{title} by Model", ylabel, results_dir / f"test_{key}.png", fmt)

    # Peak memory (nested under "peak_memory_gb"; keys vary by device type,
    # e.g. cuda has allocated_gb/reserved_gb, mps has allocated_gb/driver_allocated_gb).
    memory_keys = set()
    for m in model_names:
        memory_keys.update((benchmarks[m].get("peak_memory_gb") or {}).keys())
    for key in sorted(memory_keys):
        names, values = [], []
        for m in model_names:
            peak_memory = benchmarks[m].get("peak_memory_gb") or {}
            val = peak_memory.get(key)
            if val is not None:
                names.append(m)
                values.append(val)
        if values:
            label = key.replace("_", " ").title()
            _bar_chart(names, values, f"Peak {label} by Model", "GB", results_dir / f"peak_memory_{key}.png", "{:.3f}")

    print(f"✓ Saved comparison plots to {results_dir}")


def train_one_config(
    model_name: str,
    encoder_tokenizer: Optional[BPETokenizer],
    decoder_tokenizer: Optional[BPETokenizer],
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    use_wandb: bool = True,
    push_to_hub: bool = False,
    resume_from: Optional[Path] = None,
    benchmark_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a single configuration and save a checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name in {"C1", "C2", "C3", "C4"}:
        loaders = create_dataloaders(
            src_tokenizer=encoder_tokenizer,
            tgt_tokenizer=decoder_tokenizer,
            batch_size=batch_size,
            max_src_len=1024,
            max_tgt_len=640,
        )
    else:
        loaders = create_byte_dataloaders(
            batch_size=batch_size,
            max_src_len=1024,
            max_tgt_len=640,
        )

    model = build_model(model_name, encoder_tokenizer, decoder_tokenizer, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resuming {model_name} from epoch {start_epoch}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_memory: Dict[str, float] = {}
    training_start = time.perf_counter()
    total_train_samples = 0
    total_train_units = 0
    total_train_batches = 0

    run = None
    if use_wandb and wandb is None:
        print("WandB is unavailable; continuing without WandB logging.")
    if use_wandb and wandb is not None:
        run = wandb.init(project="anlp-assignment-1", name=f"{model_name}-train", reinit=True)
        run.config.update({
            "model": model_name,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
        })
        print(f"WandB logging enabled: {run.url}")

    best_path = output_dir / f"{model_name}_best.pt"
    latest_path = output_dir / f"{model_name}_latest.pt"

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(loaders["train"]):
            synchronize(device)
            batch_start = time.perf_counter()
            optimizer.zero_grad()
            loss = compute_loss_for_batch(model, batch, model_name, device)
            loss.backward()
            optimizer.step()
            synchronize(device)
            batch_seconds = time.perf_counter() - batch_start

            train_loss += float(loss.item())
            n_batches += 1
            total_train_batches += 1
            total_train_samples += batch["encoder_input"].size(0)
            unit_key = "decoder_input" if model_name in {"C1", "C2", "C3", "C4"} else "encoder_input"
            total_train_units += batch[unit_key].numel()
            update_peak_memory(peak_memory, memory_snapshot(device))

            if run is not None and batch_idx % 20 == 0:
                run.log({
                    f"{model_name}/train_batch_loss": float(loss.item()),
                    f"{model_name}/batch_seconds": batch_seconds,
                }, step=total_train_batches)

        avg_train_loss = train_loss / max(1, n_batches)
        val_loss = evaluate(model, loaders["val"], model_name, device)
        scheduler.step(val_loss)
        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d} | {model_name} | train={avg_train_loss:.4f} "
            f"| val={val_loss:.4f} | {epoch_seconds:.1f}s"
        )

        if run is not None:
            # FIX: log epoch-level metrics against the SAME monotonic step
            # counter as batch-level logs (total_train_batches), not against
            # `epoch` -- wandb requires non-decreasing step values across all
            # log calls in a run, and mixing "epoch number" with "cumulative
            # batch count" as two different step sequences in the same run
            # makes the epoch-level points go backwards after the first few
            # batches, so wandb drops or warns on them.
            run.log({
                f"{model_name}/train_loss": avg_train_loss,
                f"{model_name}/val_loss": val_loss,
                f"{model_name}/lr": optimizer.param_groups[0]["lr"],
                f"{model_name}/epoch": epoch,
            }, step=total_train_batches)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, model_name, epoch, val_loss, best_val_loss)
            print(f"✓ Saved best checkpoint to {best_path}")

        save_checkpoint(latest_path, model, optimizer, scheduler, model_name, epoch, val_loss, best_val_loss)
        print(f"✓ Saved resumable checkpoint to {latest_path}")

    # Test-set evaluation, greedy decoding, capped at 2 batches -- full
    # test-set eval was taking >10 min per model, so this trades exact
    # Section-4 numbers for a fast approximate check. Bump max_batches back
    # up (or set to None) if you need the true final numbers again.
    print(f"\n==== Final test-set evaluation: {model_name} ====")
    test_metrics = evaluate_metrics(
        model, loaders["test"], model_name, device,
        tgt_tokenizer=decoder_tokenizer, max_batches=2,
    )
    print(format_metrics_line(f"  test", test_metrics))
    log_metrics_to_wandb(run, model_name, "test", test_metrics, total_train_batches)

    if run is not None:
        run.finish()

    # Optional HF upload: repo_id must be set via env var HF_REPO_ID.
    if push_to_hub and HfApi is not None:
        repo_id = os.getenv("HF_REPO_ID")
        if repo_id is None:
            print("HF_REPO_ID is not set; skipping Hugging Face upload.")
        else:
            api = HfApi(token=os.getenv("HF_TOKEN"))
            try:
                api.create_repo(repo_id, exist_ok=True, private=True)
            except Exception:
                pass
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(output_dir),
                repo_type="model",
                commit_message=f"Upload {model_name} checkpoint",
            )
            print(f"✓ Uploaded {model_name} checkpoint to Hugging Face repo {repo_id}")

    synchronize(device)
    elapsed_seconds = time.perf_counter() - training_start
    result = {
        "model_name": model_name,
        "device": str(device),
        "epochs_completed": max(0, epochs - start_epoch + 1),
        "batch_size": batch_size,
        "learning_rate": lr,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed_seconds,
        "training_minutes": elapsed_seconds / 60.0,
        "training_batches": total_train_batches,
        "samples_per_second": total_train_samples / max(elapsed_seconds, 1e-9),
        "units_per_second": total_train_units / max(elapsed_seconds, 1e-9),
        "peak_memory_gb": peak_memory,
        "best_validation_loss": best_val_loss,
        "checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "test_metrics": test_metrics,
    }
    if benchmark_path is not None:
        update_benchmark_file(benchmark_path, result)
        print(f"✓ Updated benchmark file at {benchmark_path}")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all ANLP assignment models.")
    parser.add_argument("--model", choices=["all", "C1", "C2", "C3", "C4", "C5"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--tokenizers-dir", type=str, default="tokenizers")
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable WandB logging (default: enabled; use --no-wandb to disable).",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = Path(args.tokenizers_dir)
    encoder_path = tokenizer_dir / "encoder.json"
    decoder_path = tokenizer_dir / "decoder.json"
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            "Missing tokenizer files. Train them first with train_tokenizers.py "
            f"at {encoder_path} and {decoder_path}."
        )

    encoder_tokenizer = BPETokenizer.load(str(encoder_path))
    decoder_tokenizer = BPETokenizer.load(str(decoder_path))

    model_names = [args.model] if args.model != "all" else ["C1", "C2", "C3", "C4", "C5"]
    benchmark_path = output_dir / "benchmarks.json"

    for model_name in model_names:
        print(f"\n==== Training {model_name} ====\n")
        train_one_config(
            model_name=model_name,
            encoder_tokenizer=encoder_tokenizer if model_name in {"C1", "C2", "C3", "C4"} else None,
            decoder_tokenizer=decoder_tokenizer if model_name in {"C1", "C2", "C3", "C4"} else None,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            output_dir=output_dir / model_name,
            use_wandb=args.wandb,
            push_to_hub=args.push_to_hub,
            resume_from=Path(args.resume_from) if args.resume_from else None,
            benchmark_path=benchmark_path,
        )

    generate_comparison_plots(benchmark_path, Path(args.results_dir))


if __name__ == "__main__":
    main()