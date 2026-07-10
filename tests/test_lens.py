"""HorizonLens semantics, persistence, and interventions."""

import pytest
import torch

from jspace.config import FitConfig, ModelConfig
from jspace.fitting import fit_lens_on_sequences
from jspace.interventions import (
    ablation_hook,
    patch_lens_coordinates,
    run_with_state_hook,
    steering_hook,
)
from jspace.lens import HorizonLens
from jspace.model import LSTMLanguageModel


@pytest.fixture()
def fitted(tiny_model, tokens):
    cfg = FitConfig(max_horizon=3, num_sequences=1, seq_len=len(tokens), burn_in=2)
    return fit_lens_on_sequences(tiny_model, tokens[None], cfg, log_every=100)


def test_save_load_roundtrip(fitted, tmp_path):
    path = tmp_path / "lens.pt"
    fitted.save(path)
    loaded = HorizonLens.load(path)
    torch.testing.assert_close(loaded.transports, fitted.transports)
    torch.testing.assert_close(loaded.mean_state, fitted.mean_state)
    torch.testing.assert_close(loaded.mean_logits, fitted.mean_logits)
    assert loaded.blocks == fitted.blocks
    assert loaded.model_config == fitted.model_config
    assert loaded.fit_config == fitted.fit_config


def test_horizon_zero_is_depth_lens(fitted, tiny_model, tokens):
    """k=0 reads a hidden block's direct residual-stream contribution W_U h_l."""
    states = tiny_model.states_over_sequence(tokens)[1:]
    for block in ("h_top", "h0"):
        start, stop = fitted.blocks[block]
        expected = states[:, start:stop] @ tiny_model.unembed.weight.detach().T
        readout = fitted.readout(tiny_model, states, block, horizon=0, mode="raw")
        torch.testing.assert_close(readout, expected)
    with pytest.raises(ValueError, match="hidden"):
        fitted.readout(tiny_model, states, "c0", horizon=0)
    with pytest.raises(ValueError, match="hidden"):
        fitted.readout(tiny_model, states, "state", horizon=0)


def test_apply_model_logits_match_forward(fitted, tiny_model, tokens):
    _, model_logits = fitted.apply(tiny_model, tokens, blocks=["c0"], horizons=[1])
    torch.testing.assert_close(
        model_logits, tiny_model(tokens[None])[0], rtol=1e-5, atol=1e-5
    )


def test_readout_matches_lens_matrix(fitted, tiny_model, tokens):
    states = tiny_model.states_over_sequence(tokens)[1:]
    start, stop = fitted.blocks["c0"]
    via_matrix = states[:, start:stop] @ fitted.lens_matrix(tiny_model, "c0", 2).T
    torch.testing.assert_close(
        fitted.readout(tiny_model, states, "c0", 2, mode="raw"), via_matrix
    )


def test_readout_modes_are_consistent(fitted, tiny_model, tokens):
    """Centered = raw minus the constant mean-state contribution (linearity);
    taylor = centered plus the zeroth-order mean-logits term."""
    states = tiny_model.states_over_sequence(tokens)[1:]
    raw = fitted.readout(tiny_model, states, "c1", 2, mode="raw")
    centered = fitted.readout(tiny_model, states, "c1", 2, mode="centered")
    taylor = fitted.readout(tiny_model, states, "c1", 2, mode="taylor")
    mean_offset = fitted.readout(tiny_model, fitted.mean_state[None], "c1", 2, mode="raw")
    torch.testing.assert_close(centered, raw - mean_offset, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        taylor, centered + fitted.mean_logits, rtol=1e-4, atol=1e-5
    )
    with pytest.raises(ValueError, match="mode"):
        fitted.readout(tiny_model, states, "c1", 2, mode="bogus")


def test_state_readout_is_sum_of_block_readouts(fitted, tiny_model, tokens):
    """The blocks partition the state, so by linearity the full-state readout
    must equal the sum of the per-block contributions."""
    states = tiny_model.states_over_sequence(tokens)[1:]
    total = fitted.readout(tiny_model, states, "state", 2, mode="raw")
    parts = sum(
        fitted.readout(tiny_model, states, b, 2, mode="raw")
        for b in ["h0", "c0", "h1", "c1"]
    )
    torch.testing.assert_close(total, parts, rtol=1e-4, atol=1e-5)


def test_apply_shapes_and_horizon_bounds(fitted, tiny_model, tokens):
    lens_logits, model_logits = fitted.apply(
        tiny_model, tokens, blocks=["h0", "c1"], horizons=[1, 3]
    )
    vocab = tiny_model.cfg.vocab_size
    assert model_logits.shape == (len(tokens), vocab)
    assert set(lens_logits) == {("h0", 1), ("h0", 3), ("c1", 1), ("c1", 3)}
    assert all(v.shape == (len(tokens), vocab) for v in lens_logits.values())
    with pytest.raises(ValueError, match="horizon"):
        fitted.lens_matrix(tiny_model, "h0", 4)


def test_model_config_mismatch_rejected(fitted):
    other = LSTMLanguageModel(
        ModelConfig(vocab_size=50, d_model=8, num_layers=3, dropout=0.0)
    )
    with pytest.raises(ValueError, match="does not match"):
        fitted.lens_matrix(other, "h0", 1)


def test_steering_raises_target_disposition(fitted, tiny_model, tokens):
    block, horizon, target = "c0", 2, 7
    baseline_states, _ = run_with_state_hook(tiny_model, tokens, lambda p, s: s)
    steered_states, _ = run_with_state_hook(
        tiny_model,
        tokens,
        steering_hook(fitted, tiny_model, block, horizon, target, alpha=5.0),
    )
    base = fitted.readout(tiny_model, baseline_states, block, horizon)[:, target]
    steered = fitted.readout(tiny_model, steered_states, block, horizon)[:, target]
    assert (steered > base).all()


def test_ablation_removes_lens_directions(fitted, tiny_model, tokens):
    block, horizon, targets = "h0", 1, [3, 9, 20]
    hook = ablation_hook(fitted, tiny_model, block, horizon, targets)
    states, _ = run_with_state_hook(tiny_model, tokens, hook)
    start, stop = fitted.blocks[block]
    dictionary = fitted.lens_matrix(tiny_model, block, horizon)[targets]
    residual = states[:, start:stop] @ dictionary.T
    torch.testing.assert_close(
        residual, torch.zeros_like(residual), rtol=0, atol=1e-4
    )


def test_patch_lens_coordinates_swaps_and_preserves_complement():
    torch.manual_seed(4)
    d, m = 16, 3
    dictionary = torch.linalg.qr(torch.randn(d, m))[0].T  # orthonormal rows
    coords = torch.tensor([1.0, -2.0, 0.5])
    orthogonal = torch.randn(d)
    orthogonal -= dictionary.T @ (dictionary @ orthogonal)
    block = dictionary.T @ coords + orthogonal

    patched = patch_lens_coordinates(block, dictionary, permutation=[1, 0, 2])

    torch.testing.assert_close(
        dictionary @ patched, coords[[1, 0, 2]], rtol=1e-5, atol=1e-5
    )
    residual = patched - dictionary.T @ (dictionary @ patched)
    torch.testing.assert_close(residual, orthogonal, rtol=1e-5, atol=1e-5)
