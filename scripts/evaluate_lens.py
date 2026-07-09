"""Quantify lens quality against the model's own future computation.

The transport J_k approximates the model's logits k steps ahead, so the
primary metric is **agreement**: how often the horizon-k readout (in "taylor"
mode, the full first-order approximation mean_logits + W_U J_k (s - mean)) at
position p recovers the model's actual top-1 output at position p+k, and how
often that output lands in the readout's top-5. The zeroth-order baseline —
predicting the corpus-mean logits everywhere — is printed alongside: the
lens's value is its excess over that constant predictor.

Secondary metrics ground everything in the text: how often readout top-1
matches the token the corpus actually realizes at p+k+1, next to the model's
own top-1 accuracy (the ceiling: the lens cannot predict text better than the
model whose future it approximates) and a unigram baseline (the floor).

Usage:
    python scripts/evaluate_lens.py --run-dir runs/default [--num-windows 32]
"""

from __future__ import annotations

import argparse

import torch

from jspace.config import Paths
from jspace.data import Corpus, sample_batch
from jspace.lens import HorizonLens
from jspace.model import load_checkpoint


def print_table(title: str, values: dict[tuple[str, int], float], blocks, horizons):
    print(title)
    print("block    " + "".join(f"  k={k:<4}" for k in horizons))
    for block in blocks:
        print(f"{block:<9}" + "".join(f"  {values[(block, k)]:.3f}" for k in horizons))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--num-windows", type=int, default=32)
    parser.add_argument("--window-len", type=int, default=128)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["state", "h_top", "c1", "h0", "c0"],
        help='"state" is the full-state readout; named blocks isolate components',
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = Paths(run_dir=args.run_dir)
    corpus = Corpus.load(paths.corpus_file)
    model = load_checkpoint(paths.checkpoint, device)
    lens = HorizonLens.load(paths.lens_file, device=device)
    lens.transports = lens.transports.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    windows, _ = sample_batch(
        corpus.val_ids, args.window_len, args.num_windows, generator
    )
    horizons = list(range(1, lens.max_horizon + 1))

    freq = torch.bincount(corpus.train_ids, minlength=len(corpus.vocab))
    unigram_top1 = freq.argmax().item()

    agree1: dict[tuple[str, int], int] = {}
    agree5: dict[tuple[str, int], int] = {}
    realized1: dict[tuple[str, int], int] = {}
    denom: dict[int, int] = dict.fromkeys(horizons, 0)
    model_top1_correct = unigram_correct = positions = 0
    zeroth_agree1 = zeroth_agree5 = zeroth_positions = 0
    mean_top5 = lens.mean_logits.to(device).topk(5).indices

    for window in windows:
        window = window.to(device)
        lens_logits, model_logits = lens.apply(
            model, window, args.blocks, horizons, mode="taylor"
        )
        model_top1 = model_logits.argmax(dim=-1)  # (T,)
        T = len(window)

        # Zeroth-order baseline: predict the corpus-mean logits everywhere.
        zeroth_agree1 += int((model_top1 == mean_top5[0]).sum())
        zeroth_agree5 += int((model_top1[:, None] == mean_top5[None, :]).any(-1).sum())
        zeroth_positions += T

        # Model ceiling and unigram floor (logits at p predict token p+1).
        model_top1_correct += int((model_top1[:-1] == window[1:]).sum())
        unigram_correct += int((window[1:] == unigram_top1).sum())
        positions += T - 1

        for k in horizons:
            n = T - k - 1  # positions p with both a model output and a token at p+k+1
            denom[k] += n
            future_model = model_top1[k : k + n]  # model top-1 at p+k
            future_token = window[k + 1 :]  # realized token at p+k+1
            for block in args.blocks:
                top5 = lens_logits[(block, k)][:n].topk(5, dim=-1).indices
                key = (block, k)
                agree1[key] = agree1.get(key, 0) + int((top5[:, 0] == future_model).sum())
                agree5[key] = agree5.get(key, 0) + int(
                    (top5 == future_model[:, None]).any(dim=-1).sum()
                )
                realized1[key] = realized1.get(key, 0) + int(
                    (top5[:, 0] == future_token).sum()
                )

    print(
        f"model top-1 accuracy (ceiling): {model_top1_correct / positions:.3f}   "
        f"unigram top-1 (floor): {unigram_correct / positions:.3f}"
    )
    print(
        "zeroth-order baseline (constant mean-logits prediction): "
        f"agreement@1={zeroth_agree1 / zeroth_positions:.3f} "
        f"agreement@5={zeroth_agree5 / zeroth_positions:.3f}\n"
    )
    as_rate = lambda hits: {  # noqa: E731
        key: hits[key] / denom[key[1]] for key in hits
    }
    print_table(
        "lens/model agreement@1 (readout top-1 == model top-1 at p+k):",
        as_rate(agree1), args.blocks, horizons,
    )
    print_table(
        "lens/model agreement@5 (model top-1 in readout top-5):",
        as_rate(agree5), args.blocks, horizons,
    )
    print_table(
        "realized-text hit rate (readout top-1 == token at p+k+1):",
        as_rate(realized1), args.blocks, horizons,
    )


if __name__ == "__main__":
    main()
