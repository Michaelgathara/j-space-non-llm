"""Smoke tests for the grid builder and HTML renderer."""

from jspace.config import FitConfig
from jspace.data import Vocab
from jspace.fitting import fit_lens_on_sequences
from jspace.viz import N_BINS, build_grid_data, render_html


def test_grid_and_html(tiny_model, tokens):
    cfg = FitConfig(max_horizon=3, num_sequences=1, seq_len=len(tokens), burn_in=2)
    lens = fit_lens_on_sequences(tiny_model, tokens[None], cfg, log_every=100)
    vocab = Vocab(id_to_token=tuple(f"w{i}<&>" for i in range(50)))

    data = build_grid_data(
        tiny_model, lens, vocab, tokens, blocks=["h_top", "c0"], max_horizon=3
    )

    assert [s["block"] for s in data["sections"]] == ["h_top", "c0"]
    h_top_rows, c0_rows = (s["rows"] for s in data["sections"])
    assert [r["k"] for r in h_top_rows] == [0, 1, 2, 3]  # k=0 only on h_top
    assert [r["k"] for r in c0_rows] == [1, 2, 3]
    for row in h_top_rows:
        assert len(row["cells"]) == len(tokens)
        for cell in row["cells"]:
            assert len(cell["top"]) == 5
            assert 0 <= cell["bin"] < N_BINS
            probs = [t["prob"] for t in cell["top"]]
            assert probs == sorted(probs, reverse=True)

    html_doc = render_html(data)
    assert html_doc.startswith("<!doctype html>")
    assert "GRID_DATA" in html_doc
    first_input = f"w{int(tokens[0])}&lt;&amp;&gt;"
    assert first_input in html_doc  # header token text is escaped
    assert "prefers-color-scheme: dark" in html_doc

    body_only = render_html(data, standalone=False)
    assert not body_only.startswith("<!doctype html>")
