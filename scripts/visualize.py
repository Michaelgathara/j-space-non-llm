"""Render the timestep × horizon lens grid for a prompt.

Usage:
    python scripts/visualize.py --run-dir runs/default \\
        [--prompt "ROMEO : What lady is that"] [--blocks h_top c1] [--max-horizon 8]

Without --prompt, a window from the validation split is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jspace.config import Paths
from jspace.data import Corpus, tokenize
from jspace.lens import HorizonLens
from jspace.model import load_checkpoint
from jspace.viz import build_grid_data, render_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--prompt", default=None, help="text to analyze (word-level)")
    parser.add_argument("--val-offset", type=int, default=0, help="window start in val split")
    parser.add_argument("--num-tokens", type=int, default=48)
    parser.add_argument(
        "--blocks", nargs="+", default=["state", "h_top", "c1", "h0", "c0"]
    )
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=5,
        help="hide tokens rarer than this in the training corpus (0 = show all)",
    )
    parser.add_argument("--out", default=None, help="output HTML path (default: run dir)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = Paths(run_dir=args.run_dir)
    corpus = Corpus.load(paths.corpus_file)
    model = load_checkpoint(paths.checkpoint, device)
    lens = HorizonLens.load(paths.lens_file, device=device)

    if args.max_horizon > lens.max_horizon:
        parser.error(f"lens was fitted with max_horizon={lens.max_horizon}")

    if args.prompt is not None:
        token_ids = corpus.vocab.encode(tokenize(args.prompt))
    else:
        token_ids = corpus.val_ids[args.val_offset : args.val_offset + args.num_tokens]
    token_ids = token_ids.to(device)

    data = build_grid_data(
        model,
        lens,
        corpus.vocab,
        token_ids,
        args.blocks,
        args.max_horizon,
        min_token_count=args.min_token_count,
        token_counts=torch.bincount(corpus.train_ids, minlength=len(corpus.vocab)),
    )
    out = Path(args.out or f"{paths.run_dir}/viz.html")
    out.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {out} ({len(token_ids)} positions × horizons 0..{args.max_horizon})")


if __name__ == "__main__":
    main()
