"""The manual step path must reproduce the fused nn.LSTM path exactly:
every lens quantity is derived from ``step_flat``, so parity with the
training-time forward is the foundation everything else rests on."""

from itertools import pairwise

import pytest
import torch

from jspace.config import ModelConfig
from jspace.model import LSTMLanguageModel


@pytest.mark.parametrize("residual", [True, False])
def test_step_flat_matches_fused_forward(tokens: torch.Tensor, residual: bool):
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=50, d_model=8, num_layers=2, dropout=0.0, residual=residual
    )
    model = LSTMLanguageModel(cfg).eval()
    fused = model(tokens[None])[0]  # (T, vocab)
    _, stepped = model.logits_over_sequence(tokens)
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


def test_readout_matrix_matches_architecture(tiny_model: LSTMLanguageModel):
    r = tiny_model.readout_state_matrix()
    blocks = tiny_model.block_slices()
    state = torch.randn(tiny_model.state_size)
    if tiny_model.cfg.residual:
        expected = sum(
            state[blocks[f"h{layer}"]] for layer in range(tiny_model.cfg.num_layers)
        )
    else:
        expected = state[blocks["h_top"]]
    torch.testing.assert_close(r @ state, expected)


def test_initial_state_is_zero(tiny_model: LSTMLanguageModel):
    assert tiny_model.initial_state().abs().sum() == 0
