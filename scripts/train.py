"""Train the LSTM language model on word-level Tiny Shakespeare.

Usage:
    python scripts/train.py --run-dir runs/default [--max-steps 20000]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jspace.config import ModelConfig, Paths, TrainConfig
from jspace.data import Corpus, sample_batch
from jspace.model import LSTMLanguageModel


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup over the first 2% of steps, then cosine to min_lr."""
    warmup = max(1, cfg.max_steps // 50)
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.max_steps - warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, ids, cfg: TrainConfig, generator, device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = sample_batch(ids, cfg.seq_len, cfg.batch_size, generator)
        logits = model(x.to(device))
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    paths = Paths(run_dir=args.run_dir)
    corpus = Corpus.load(paths.corpus_file)
    print(f"vocab={len(corpus.vocab)} train_tokens={len(corpus.train_ids)} device={device}")

    model_cfg = ModelConfig(
        vocab_size=len(corpus.vocab),
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.lr / 10,
        seed=args.seed,
    )
    model = LSTMLanguageModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(train_cfg.seed)

    run_dir = Path(paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    started = time.monotonic()

    model.train()
    for step in range(train_cfg.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, train_cfg)
        x, y = sample_batch(
            corpus.train_ids, train_cfg.seq_len, train_cfg.batch_size, generator
        )
        logits = model(x.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if (step + 1) % train_cfg.eval_every == 0 or step + 1 == train_cfg.max_steps:
            val = evaluate(model, corpus.val_ids, train_cfg, generator, device)
            speed = (step + 1) / (time.monotonic() - started)
            print(
                f"step {step + 1}/{train_cfg.max_steps} "
                f"train_loss={loss.item():.3f} val_loss={val:.3f} "
                f"val_ppl={math.exp(val):.1f} ({speed:.1f} steps/s)",
                flush=True,
            )
            if val < best_val:
                best_val = val
                torch.save(
                    {"model": model.state_dict(), "config": model_cfg.to_dict()},
                    paths.checkpoint,
                )

    (run_dir / "train_summary.json").write_text(
        json.dumps({"best_val_loss": best_val, "best_val_ppl": math.exp(best_val)})
    )
    print(f"done: best val_loss={best_val:.3f} (ppl {math.exp(best_val):.1f})")
    print(f"checkpoint: {paths.checkpoint}")


if __name__ == "__main__":
    main()
