"""LSTM language model with an exact, differentiable single-step API.

Training uses the fused ``nn.LSTM`` kernel. Lens fitting needs the Jacobian of
one recurrence step with respect to the *flattened* recurrent state, so the
model also exposes ``step_flat``: a pure function of ``(flat_state, x_emb)``
implementing the standard LSTM cell equations directly from the ``nn.LSTM``
weight tensors. A unit test asserts bit-for-bit-comparable parity between the
two paths.

Flat-state convention
---------------------
The full recurrent state is flattened as

    s = concat(h_0, c_0, h_1, c_1, ..., h_{L-1}, c_{L-1})      (size 2*L*d)

``block_slices()`` maps human-readable block names ("h0", "c0", ...) to slices
of this vector; "h_top" aliases the last hidden block, whose linear readout
``W_U @ h_top`` produces the logits.
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
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.unembed.weight = self.embed.weight
        self.io_dropout = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # Training / batch path
    # ------------------------------------------------------------------

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens (batch, T) -> logits (batch, T, vocab)."""
        out, _ = self.lstm(self.io_dropout(self.embed(tokens)))
        return self.unembed(self.io_dropout(out))

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

    # ------------------------------------------------------------------
    # Exact single-step path (used for Jacobians and interventions)
    # ------------------------------------------------------------------

    def _layer_weights(self, layer: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        g = self.lstm
        return (
            getattr(g, f"weight_ih_l{layer}"),
            getattr(g, f"weight_hh_l{layer}"),
            getattr(g, f"bias_ih_l{layer}"),
            getattr(g, f"bias_hh_l{layer}"),
        )

    def step_flat(self, state: Tensor, x_emb: Tensor) -> Tensor:
        """One recurrence step: flat state (S,) + input embedding (d,) -> flat state (S,).

        Implements the standard LSTM cell (gate order i, f, g, o — matching
        ``nn.LSTM``) per layer, feeding each layer's new hidden state to the
        next layer within the same timestep. No dropout: this is the
        evaluation-time function whose Jacobian defines the lens.
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
            x = h_new
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

    def logits_from_state(self, state: Tensor) -> Tensor:
        """Next-token logits read from a flat state's top hidden block."""
        return self.unembed(state[..., self.block_slices()["h_top"]])


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> LSTMLanguageModel:
    """Load a model saved by ``scripts/train.py`` in eval mode."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = LSTMLanguageModel(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()
