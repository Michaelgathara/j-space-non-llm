# J-Space for Recurrent Networks

A port of Anthropic's **Jacobian lens / J-space** interpretability technique
([*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/),
[reference code](https://github.com/anthropics/jacobian-lens)) from transformers
to recurrent architectures, demonstrated on a word-level LSTM language model.

## The idea

The transformer J-lens transports a residual-stream activation at layer ℓ into
the final-layer basis with a corpus-averaged Jacobian, then decodes it with the
model's own unembedding — reading out which tokens the activation is *disposed
to make the model say*.

An RNN has no deep layer stack, but it has something better suited to the
metaphor: a **persistent recurrent state** that every timestep reads and
rewrites — a literal workspace. This repo replaces the *layer* axis with a
**time-horizon** axis. With flat recurrent state
`s_t = (h⁰_t, c⁰_t, …, h^{L−1}_t, c^{L−1}_t)`, we fit for each horizon
`k = 1..K` the corpus-averaged transport

```
J_k = E_{t, corpus}[ ∂ h_top(t+k) / ∂ s(t) ]        (d_model × state_size)
```

so that `W_U · J_k · s_t` answers: **which tokens is the recurrent state at
time t disposed to make the model emit k steps in the future?**

Two things fall out of the temporal framing that the transformer version
doesn't have:

- **A time-to-verbalization axis.** A concept can be watched entering the
  state, riding silently for k steps, then surfacing as output.
- **A memory-decay profile.** Chained one-step Jacobians of a stable RNN
  contract, so `‖J_k‖` vs. k measures which content the state protects over
  time — connecting the lens to the dynamical-systems view of RNNs
  (fixed-point linearization à la Sussillo & Barak, 2013, with the single-point
  linearization replaced by Anthropic's corpus averaging).

The state decomposes into named blocks (`h0`, `c0`, `h1`, `c1`, … — hidden and
cell state per layer), and the lens reads each block separately. Whether the
*cell* state (the LSTM's protected memory) carries more long-horizon
verbalizable content than the hidden state is exactly the kind of question the
lens exists to ask.

## How it's computed

For a sequence with states `s_0 … s_T`, the one-step Jacobian
`A_t = ∂s_t/∂s_{t−1}` is computed exactly with `torch.func.jacrev` through the
model's `step_flat` (a hand-written LSTM step, unit-tested for parity with the
fused `nn.LSTM` path; the consumed token is held fixed, so `A_t` is
well-defined under teacher forcing). Horizon transports are chained products

```
∂h_top(t+k)/∂s(t) = [A_{t+k} · … · A_{t+1}]_{h_top rows}
```

accumulated with a ring buffer over target positions — each position
contributes one **exact** sample to every horizon at
`O(K · d_model · state_size²)` cost, then samples are averaged over positions
and sequences. No approximations; a test verifies the chained result against
direct autograd through the k-step unrolled map, and the fitted average against
a brute-force reference.

`k = 0` on the top hidden block is the ordinary logit lens (`W_U h_top`) and
needs no transport.

### Readout modes

A transformer's final LayerNorm recenters activations before unembedding; an
LSTM has no such stage, so the raw product `W_U J_k s` is dominated by the
corpus-typical component of the state. The lens therefore stores the
corpus-mean state and mean logits from fitting and offers three readouts:

- **`centered`** (default) — `W_U J_k (s − E[s])`: the state-*specific*
  disposition. Use for reading content (the visualizer uses this).
- **`taylor`** — `E[logits] + W_U J_k (s − E[s])`: the full first-order
  Taylor approximation of the model's future logits. Use for prediction.
- **`raw`** — `W_U J_k s`: no reference point, for analysis.

Per-block readouts isolate one component's contribution; the pseudo-block
`"state"` reads the full transport (blocks partition the state, so it equals
the sum of the block readouts).

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # 14 tests, a few seconds

python scripts/train.py                           # word-level Tiny Shakespeare LSTM
python scripts/fit_lens.py                        # fit J_k for k = 1..16 (~90s on GPU)
python scripts/evaluate_lens.py                   # agreement vs the model's own future
python scripts/visualize.py                       # static timestep × horizon HTML grid
python scripts/serve.py                           # live viewer at http://127.0.0.1:8731/
```

Artifacts land in `runs/default/` (`model.pt`, `lens.pt`, `viz.html`). The
corpus (~1.1 MB) downloads to `data/` on first use. Word-level tokenization is
deliberate: the lens decodes into vocabulary space, and the J-space
construction is only interesting when the token dictionary is overcomplete
relative to the model dimension (vocab ≈ 12.7k ≫ d_model = 256).

## Visualization

`scripts/visualize.py` renders a self-contained HTML grid: columns are sequence
positions, row k shows each state's top disposed token k steps ahead, shaded by
its percentile strength within the grid, with hover tooltips for the top-5 and
a ring where the readout is confirmed by the token the text actually contains
at the predicted position — one grid per state block. Rare tokens (which have
outsized unembedding norms and read as noise) are hidden from the decoded
top-k by a display-only frequency filter (`--min-token-count`).

### Live viewer

`scripts/serve.py` runs the same grid as a local web app: the checkpoint and
lens load once (CUDA if available), then any prompt you type is read through
the lens interactively — a readout is one LSTM pass plus a few small matmuls
(~50 ms typical). Stdlib-only HTTP server, bound to `127.0.0.1`; choose
blocks, horizon, and the rare-token filter from the page.

## Interventions

`jspace.interventions` implements the paper's three intervention families
against the recurrent state, applied through a step-by-step replay hook:

- **Steering** — add α · (unit lens vector for a token) to a block:
  inject/suppress a concept's disposition-to-be-emitted.
- **Ablation** — project a block out of the span of chosen lens directions.
- **Patching in lens coordinates** — least-squares coordinates over a small
  lens dictionary, permute, reconstruct; the orthogonal complement is
  preserved exactly.

## Layout

```
src/jspace/
  model.py          LSTM LM; exact differentiable step_flat + flat-state blocks
  fitting.py        one-step Jacobians, ring-buffer chaining, corpus averaging
  lens.py           HorizonLens: transports, readout modes, persistence
  interventions.py  steering / ablation / lens-coordinate patching
  viz.py            timestep × horizon HTML grid renderer
  data.py           word-level Tiny Shakespeare corpus
scripts/            train.py, fit_lens.py, evaluate_lens.py, visualize.py
tests/              step parity, Jacobian correctness, lens semantics
```

## First results (Tiny Shakespeare LSTM, d=128, 2 layers)

- **Memory-decay profile**: `‖J_k‖` drops sharply k=1→2 then decays ever more
  slowly (per-step retention 0.37 → 0.92 by k=16) — a protected long-lived
  subspace in the recurrent state.
- **Agreement with the model's own future** (taylor mode, top-5): 0.60 at k=1
  vs a 0.44 zeroth-order baseline, decaying to baseline by k≈16 in step with
  the transport contraction. The lens's information lives in the ranking; the
  argmax is dominated by the constant term.
- **The cell state is where the workspace lives**: the one-step signal routes
  through `c1` (the top cell), not `h1` — and at long horizons the *layer-0
  cell* `c0` retains excess agreement longest. The hidden states are the
  scratch I/O; the cells are the protected memory.

## What this is not

This is the *tool* (the lens) ported to a new architecture class, plus the
infrastructure to run the paper's intervention batteries. It does not by
itself establish that LSTMs have a global workspace in the paper's sense —
that requires the five functional-property experiments (verbal report,
directed modulation, internal reasoning, flexible generalization,
selectivity), for which this repo provides the primitives.
