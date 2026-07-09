"""The manual step path must reproduce the fused nn.LSTM path exactly:
every lens quantity is derived from ``step_flat``, so parity with the
training-time forward is the foundation everything else rests on."""

from itertools import pairwise

import torch

from jspace.model import LSTMLanguageModel


def test_step_flat_matches_nn_lstm(tiny_model: LSTMLanguageModel, tokens: torch.Tensor):
    fused = tiny_model(tokens[None])[0]  # (T, vocab)
    states = tiny_model.states_over_sequence(tokens)
    stepped = tiny_model.logits_from_state(states[1:])
    torch.testing.assert_close(stepped, fused, rtol=1e-5, atol=1e-5)


def test_block_slices_partition_state(tiny_model: LSTMLanguageModel):
    blocks = tiny_model.block_slices()
    d, L = tiny_model.cfg.d_model, tiny_model.cfg.num_layers
    assert tiny_model.state_size == 2 * L * d
    covered = sorted(
        (blk.start, blk.stop) for name, blk in blocks.items() if name != "h_top"
    )
    assert covered[0][0] == 0 and covered[-1][1] == tiny_model.state_size
    assert all(a[1] == b[0] for a, b in pairwise(covered))
    assert blocks["h_top"] == blocks[f"h{L - 1}"]


def test_initial_state_is_zero(tiny_model: LSTMLanguageModel):
    assert tiny_model.initial_state().abs().sum() == 0
