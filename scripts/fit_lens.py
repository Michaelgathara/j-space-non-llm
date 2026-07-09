"""Fit the horizon-indexed Jacobian lens on a trained checkpoint.

Usage:
    python scripts/fit_lens.py --run-dir runs/default [--max-horizon 16]
"""

from __future__ import annotations

import argparse

import torch

from jspace.config import FitConfig, Paths
from jspace.data import Corpus
from jspace.fitting import fit_lens
from jspace.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = Paths(run_dir=args.run_dir)
    corpus = Corpus.load(paths.corpus_file)
    model = load_checkpoint(paths.checkpoint, device)

    cfg = FitConfig(
        max_horizon=args.max_horizon,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    lens = fit_lens(model, corpus.train_ids, cfg, device=device)
    lens.save(paths.lens_file)

    norms = lens.transports.flatten(1).norm(dim=1)
    print(f"saved lens to {paths.lens_file}")
    print("transport Frobenius norm by horizon (memory-decay profile):")
    for k, n in enumerate(norms, start=1):
        print(f"  k={k:>2}: {n:.4f}")


if __name__ == "__main__":
    main()
