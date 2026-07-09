"""LensApp request validation and grid computation (no HTTP required)."""

import pytest
import torch

from jspace.config import FitConfig
from jspace.data import Vocab
from jspace.fitting import fit_lens_on_sequences
from jspace.server import LensApp
from jspace.viz import render_app_shell


@pytest.fixture()
def app(tiny_model, tokens):
    cfg = FitConfig(max_horizon=3, num_sequences=1, seq_len=len(tokens), burn_in=2)
    lens = fit_lens_on_sequences(tiny_model, tokens[None], cfg, log_every=100)
    vocab = Vocab(id_to_token=("<unk>", *[f"w{i}" for i in range(1, 50)]))
    return LensApp(
        model=tiny_model,
        lens=lens,
        vocab=vocab,
        token_counts=torch.randint(1, 100, (50,)),
        device="cpu",
    )


def test_grid_happy_path(app):
    data = app.grid(
        {"prompt": "w1 w2 w3 w4 w5", "blocks": ["state", "c1"], "max_horizon": 2}
    )
    assert data["tokens"] == ["w1", "w2", "w3", "w4", "w5"]
    assert [s["block"] for s in data["sections"]] == ["state", "c1"]
    assert [r["k"] for r in data["sections"][1]["rows"]] == [1, 2]


def test_grid_rejects_bad_input(app):
    with pytest.raises(ValueError, match="empty"):
        app.grid({"prompt": "   ", "blocks": ["state"]})
    with pytest.raises(ValueError, match="blocks"):
        app.grid({"prompt": "w1 w2", "blocks": ["bogus"]})
    with pytest.raises(ValueError, match="max_horizon"):
        app.grid({"prompt": "w1 w2", "blocks": ["state"], "max_horizon": 99})
    with pytest.raises(ValueError, match="too long"):
        app.grid({"prompt": "w1 " * 300, "blocks": ["state"]})


def test_unknown_words_map_to_unk(app):
    data = app.grid({"prompt": "w1 notaword w2", "blocks": ["c1"], "max_horizon": 1})
    assert data["tokens"] == ["w1", "<unk>", "w2"]


def test_app_shell_renders(app):
    shell = render_app_shell(app.config())
    assert shell.startswith("<!doctype html>")
    for needle in ("APP_CONFIG", "/api/grid", 'id="prompt"', "prefers-color-scheme"):
        assert needle in shell
