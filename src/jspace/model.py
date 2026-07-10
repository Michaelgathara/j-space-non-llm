"""Residual stacked LSTM language model with an exact single-step API.

Architecture (``residual=True``, the default): equal-width LSTM layers wired
through a residual stream across depth,

    x_0 = embed(token),   x_{l+1} = x_l + h_l,   logits = W_U (x_L - x_0)

so the pre-logit vector is exactly ``sum_l h_l`` — what the layers *wrote*
into the stream. Every layer's hidden state contributes directly and
additively to the logits, like writers into a transformer's residual stream;
that structure is what makes the k=0 depth lens trivial (decode ``W_U h_l``
per layer) and makes depth trainable. The raw input embedding is excluded
from the readout deliberately: with tied embeddings, ``W_U embed(w)`` gives
every token a ~||e_w||^2 self-logit at initialization — an "echo the input"
trap the model would first have to unlearn. With ``residual=False`` the model
degenerates to the classic stack (input to layer l+1 is h_l; logits read
h_top only).

Training uses fused single-layer ``nn.LSTM`` kernels over the whole sequence;
lens fitting uses ``step_flat``, a hand-written per-timestep cell that shares
the same weights. A unit test asserts parity between the two paths.

Flat-state convention
---------------------
The full recurrent state is flattened as

    s = concat(h_0, c_0, h_1, c_1, ..., h_{L-1}, c_{L-1})      (size 2*L*d)

``block_slices()`` maps names ("h0", "c0", ...) to slices; "h_top" aliases the
last hidden block. ``readout_state_matrix()`` gives the constant R (d x S)
with ``pre_logits = R @ s``: R sums the h blocks (residual) or selects h_top
(non-residual).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from jspace.config import ModelConfig


class LSTMLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            nn.LSTM(cfg.d_model, cfg.d_model, num_layers=1, batch_first=True)
            for _ in range(cfg.num_layers)
        )
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.unembed.weight = self.embed.weight
        self.dropout = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # Training / batch path
    # ------------------------------------------------------------------

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens (batch, T) -> logits (batch, T, vocab)."""
        x0 = self.dropout(self.embed(tokens))
        x = x0
        for lstm in self.layers:
            h, _ = lstm(x)
            x = x + self.dropout(h) if self.cfg.residual else self.dropout(h)
        return self.unembed(x - x0 if self.cfg.residual else x)

    # ------------------------------------------------------------------
    # Flat-state geometry
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        return 2 * self.cfg.num_layers * self.cfg.d_model

    def block_slices(self) -> dict[str, slice]:
        d, blocks = self.cfg.d_model, {}
        for layer in range(self.cfg.num_layers):
            base = 2 * layer * d
            blocks[f"h{layer}"] = slice(base, base + d)
            blocks[f"c{layer}"] = slice(base + d, base + 2 * d)
        blocks["h_top"] = blocks[f"h{self.cfg.num_layers - 1}"]
        return blocks

    def initial_state(self, device: torch.device | None = None) -> Tensor:
        return torch.zeros(self.state_size, device=device or self.embed.weight.device)

    def readout_state_matrix(self) -> Tensor:
        """Constant R (d_model x state_size): the state's linear contribution
        to the pre-logit vector. Residual: sum of all h blocks (each layer
        writes directly into the output stream). Non-residual: h_top only."""
        d = self.cfg.d_model
        r = torch.zeros(d, self.state_size, device=self.embed.weight.device)
        eye = torch.eye(d, device=r.device)
        blocks = self.block_slices()
        if self.cfg.residual:
            for layer in range(self.cfg.num_layers):
                r[:, blocks[f"h{layer}"]] = eye
        else:
            r[:, blocks["h_top"]] = eye
        return r

    # ------------------------------------------------------------------
    # Exact single-step path (used for Jacobians and interventions)
    # ------------------------------------------------------------------

    def _layer_weights(self, layer: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lstm = self.layers[layer]
        return lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0

    def step_flat(self, state: Tensor, x_emb: Tensor) -> Tensor:
        """One recurrence step: flat state (S,) + input embedding (d,) -> flat state (S,).

        Standard LSTM cell per layer (gate order i, f, g, o — matching
        ``nn.LSTM``), with the layer input following the residual stream when
        ``cfg.residual``. No dropout: this is the evaluation-time function
        whose Jacobian defines the lens.
        """
        d = self.cfg.d_model
        x = x_emb
        new_blocks: list[Tensor] = []
        for layer in range(self.cfg.num_layers):
            base = 2 * layer * d
            h, c = state[base : base + d], state[base + d : base + 2 * d]
            w_ih, w_hh, b_ih, b_hh = self._layer_weights(layer)
            gates = w_ih @ x + b_ih + w_hh @ h + b_hh
            i, f, g, o = gates.chunk(4)
            c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h_new = torch.sigmoid(o) * torch.tanh(c_new)
            new_blocks += [h_new, c_new]
            x = x + h_new if self.cfg.residual else h_new
        return torch.cat(new_blocks)

    @torch.no_grad()
    def states_over_sequence(self, token_ids: Tensor) -> Tensor:
        """Run ``step_flat`` over a sequence.

        token_ids (T,) -> states (T+1, S); row 0 is the zero initial state and
        row t is the post-step state after consuming token t-1.
        """
        emb = self.embed(token_ids)
        states = [self.initial_state(token_ids.device)]
        for t in range(len(token_ids)):
            states.append(self.step_flat(states[-1], emb[t]))
        return torch.stack(states)

    def logits_from_state(self, states: Tensor) -> Tensor:
        """Next-token logits from flat states (..., S): ``W_U (R s)``."""
        return self.unembed(states @ self.readout_state_matrix().T)

    @torch.no_grad()
    def logits_over_sequence(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """token_ids (T,) -> (states (T, S), logits (T, vocab)), post-consumption."""
        states = self.states_over_sequence(token_ids)[1:]
        return states, self.logits_from_state(states)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[LSTMLanguageModel, dict]:
    """Load a model saved by ``scripts/train.py`` in eval mode.

    Returns the model and the raw payload (which carries e.g. the corpus
    name the checkpoint was trained on).
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    model = LSTMLanguageModel(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload
