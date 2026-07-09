"""Fitting the horizon lens: corpus-averaged state-transition Jacobians.

For a sequence with recurrent states s_0, s_1, ..., s_T the one-step Jacobian

    A_t = d s_t / d s_{t-1}                       (state_size x state_size)

is computed exactly with ``torch.func.jacrev`` through the model's
``step_flat`` (the consumed token is held fixed, so A_t is well-defined under
teacher forcing). The horizon-k transport from source position t is then the
chained product restricted to the rows of the top hidden block:

    d h_top(t+k) / d s(t) = [A_{t+k} · A_{t+k-1} · ... · A_{t+1}]_{h_top rows}

Rather than materializing full products, we iterate over *target* positions
tau and right-multiply the (d_model x state_size) row slice through a ring
buffer of the last K one-step Jacobians:

    V_1 = A_tau[h_top, :],   V_j = V_{j-1} @ A_{tau-j+1}

so each target position contributes one exact sample to every horizon
k = 1..K at cost O(K * d_model * state_size^2). Samples are averaged over
positions and sequences, mirroring the corpus averaging that distinguishes
the J-lens from a single-context linearization.
"""

from __future__ import annotations

import time
from collections import deque

import torch
from torch import Tensor
from torch.func import jacrev

from jspace.config import FitConfig
from jspace.data import sample_batch
from jspace.lens import HorizonLens
from jspace.model import LSTMLanguageModel


def one_step_jacobian(model: LSTMLanguageModel, state: Tensor, x_emb: Tensor) -> Tensor:
    """Exact Jacobian of one recurrence step w.r.t. the flat state: (S, S)."""
    return jacrev(lambda s: model.step_flat(s, x_emb))(state).detach()


def fit_lens_on_sequences(
    model: LSTMLanguageModel,
    sequences: Tensor,
    cfg: FitConfig,
    log_every: int = 8,
) -> HorizonLens:
    """Fit horizon transports on explicit token sequences (N, T).

    Deterministic given the sequences; ``fit_lens`` handles corpus sampling.
    """
    model = model.eval()
    device = model.embed.weight.device
    top = model.block_slices()["h_top"]
    d, s_size, k_max = model.cfg.d_model, model.state_size, cfg.max_horizon

    sums = torch.zeros(k_max, d, s_size, device=device)
    counts = torch.zeros(k_max, device=device)
    state_sum = torch.zeros(s_size, device=device)
    logits_sum = torch.zeros(model.cfg.vocab_size, device=device)
    state_count = 0
    started = time.monotonic()

    for seq_idx, seq in enumerate(sequences):
        seq = seq.to(device)
        with torch.no_grad():
            emb = model.embed(seq)
        states = model.states_over_sequence(seq)  # (T+1, S)
        state_sum += states[cfg.burn_in :].sum(0)
        with torch.no_grad():
            logits_sum += model.logits_from_state(states[cfg.burn_in :]).sum(0)
        state_count += len(states) - cfg.burn_in

        # ring[j] = A_{tau-j}; freshest one-step Jacobian at the front.
        ring: deque[Tensor] = deque(maxlen=k_max)
        for tau in range(1, len(seq) + 1):
            ring.appendleft(one_step_jacobian(model, states[tau - 1], emb[tau - 1]))

            with torch.no_grad():
                v = ring[0][top, :]
                for k in range(1, len(ring) + 1):
                    if k > 1:
                        v = v @ ring[k - 1]
                    if tau - k >= cfg.burn_in:  # source state s_{tau-k}
                        sums[k - 1] += v
                        counts[k - 1] += 1

        if (seq_idx + 1) % log_every == 0 or seq_idx + 1 == len(sequences):
            elapsed = time.monotonic() - started
            print(
                f"[fit] sequence {seq_idx + 1}/{len(sequences)} "
                f"({elapsed:.0f}s elapsed, {elapsed / (seq_idx + 1):.1f}s/seq)",
                flush=True,
            )

    if (counts == 0).any():
        raise RuntimeError(
            "Some horizons received no samples; increase seq_len or lower "
            "burn_in/max_horizon."
        )

    transports = (sums / counts[:, None, None]).cpu()
    blocks = {
        name: (blk.start, blk.stop) for name, blk in model.block_slices().items()
    }
    return HorizonLens(
        transports=transports,
        mean_state=(state_sum / state_count).cpu(),
        mean_logits=(logits_sum / state_count).cpu(),
        blocks=blocks,
        model_config=model.cfg.to_dict(),
        fit_config=cfg.to_dict(),
    )


def fit_lens(
    model: LSTMLanguageModel,
    corpus_ids: Tensor,
    cfg: FitConfig,
    device: str | torch.device = "cpu",
    log_every: int = 8,
) -> HorizonLens:
    """Fit horizon transports on ``cfg.num_sequences`` random corpus windows."""
    model = model.to(device)
    generator = torch.Generator().manual_seed(cfg.seed)
    sequences, _ = sample_batch(corpus_ids, cfg.seq_len, cfg.num_sequences, generator)
    return fit_lens_on_sequences(model, sequences, cfg, log_every=log_every)
