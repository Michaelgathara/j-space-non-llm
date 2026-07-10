"""Correctness of the transport computation.

The critical property: the ring-buffer chain of one-step Jacobians must equal
the Jacobian of the k-step unrolled map computed directly by autograd, and the
fitted lens must equal a brute-force average of those direct Jacobians.
"""

import torch
from torch.func import jacrev

from jspace.config import FitConfig
from jspace.fitting import fit_lens, fit_lens_on_sequences, one_step_jacobian
from jspace.model import LSTMLanguageModel


def unrolled_jacobian(
    model: LSTMLanguageModel, states: torch.Tensor, emb: torch.Tensor, t: int, k: int
) -> torch.Tensor:
    """d pre_logits(t+k) / d s(t) via autograd through the k-step unrolled map."""
    readout = model.readout_state_matrix()

    def unroll(s: torch.Tensor) -> torch.Tensor:
        for i in range(t + 1, t + k + 1):
            s = model.step_flat(s, emb[i - 1])
        return readout @ s

    return jacrev(unroll)(states[t]).detach()


def chained_transport(model, states, emb, t, k) -> torch.Tensor:
    """d s(t+k) / d s(t) by chaining one-step Jacobians."""
    prod = torch.eye(model.state_size)
    for i in range(t + 1, t + k + 1):
        prod = one_step_jacobian(model, states[i - 1], emb[i - 1]) @ prod
    return prod


def test_chained_equals_unrolled(tiny_model, tokens):
    with torch.no_grad():
        emb = tiny_model.embed(tokens)
    states = tiny_model.states_over_sequence(tokens)
    readout = tiny_model.readout_state_matrix()

    t, k = 3, 4
    chain = readout @ chained_transport(tiny_model, states, emb, t, k)
    direct = unrolled_jacobian(tiny_model, states, emb, t, k)
    torch.testing.assert_close(chain, direct, rtol=1e-4, atol=1e-5)


def test_fitted_lens_equals_brute_force_average(tiny_model):
    torch.manual_seed(2)
    seq = torch.randint(0, 50, (10,))
    cfg = FitConfig(max_horizon=3, num_sequences=1, seq_len=10, burn_in=2)

    lens = fit_lens_on_sequences(tiny_model, seq[None], cfg, log_every=100)

    with torch.no_grad():
        emb = tiny_model.embed(seq)
    states = tiny_model.states_over_sequence(seq)
    T = len(seq)
    for k in range(1, cfg.max_horizon + 1):
        samples = [
            unrolled_jacobian(tiny_model, states, emb, t, k)
            for t in range(cfg.burn_in, T - k + 1)
        ]
        expected = torch.stack(samples).mean(0)
        torch.testing.assert_close(
            lens.transports[k - 1], expected, rtol=1e-4, atol=1e-5
        )


def test_fit_lens_end_to_end_shapes(tiny_model):
    torch.manual_seed(3)
    corpus = torch.randint(0, 50, (200,))
    cfg = FitConfig(max_horizon=3, num_sequences=2, seq_len=16, burn_in=2)
    lens = fit_lens(tiny_model, corpus, cfg, log_every=100)
    assert lens.transports.shape == (3, 8, tiny_model.state_size)
    assert lens.max_horizon == 3
    assert set(lens.blocks) == {"h0", "c0", "h1", "c1", "h_top"}
