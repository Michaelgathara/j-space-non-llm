"""The horizon-indexed Jacobian lens.

A ``HorizonLens`` stores, for each temporal horizon k = 1..K, the
corpus-averaged linear transport

    J_k = E_{t, corpus}[ d h_top(t+k) / d s(t) ]        (d_model x state_size)

mapping a perturbation of the full recurrent state at time t to its
first-order effect on the top hidden state — and hence, through the model's
own unembedding, on the logits — k steps later. Reading a state through the
lens answers: *which tokens is this state disposed to make the model emit
k steps from now?*

This is the recurrent analog of the transformer J-lens, with the layer axis
replaced by a time-horizon axis. The k = 0 readout of the top hidden block is
the ordinary logit lens (``W_U @ h_top``) and needs no transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from jspace.model import LSTMLanguageModel


@dataclass
class HorizonLens:
    """Fitted transports plus the metadata needed to apply them safely.

    Attributes:
        transports: (K, d_model, state_size); ``transports[k-1]`` is J_k.
        mean_state: (state_size,) corpus-mean flat state — the point the
            first-order readout linearizes around.
        mean_logits: (vocab,) corpus-mean output logits — the zeroth-order
            Taylor term for the "taylor" readout mode.
        blocks: block name -> (start, stop) into the flat state vector.
        model_config: config dict of the model the lens was fitted on.
        fit_config: fitting hyperparameters, for provenance.

    Readout modes:
        "centered" — ``W_U J_k (s - mean_state)``: the state-specific
            disposition, for reading *content*. An LSTM has no final
            normalization to recenter activations the way a transformer's
            last LayerNorm does, so the raw product is dominated by the
            corpus-typical component of the state; centering removes it.
        "taylor" — ``mean_logits + W_U J_k (s - mean_state)``: the full
            first-order Taylor approximation of the model's future logits,
            for *predicting* what the model will output.
        "raw" — ``W_U J_k s``: no reference point; mainly for analysis.
    """

    transports: Tensor
    mean_state: Tensor
    mean_logits: Tensor
    blocks: dict[str, tuple[int, int]]
    model_config: dict[str, Any]
    fit_config: dict[str, Any]

    @property
    def max_horizon(self) -> int:
        return self.transports.shape[0]

    def _check_model(self, model: LSTMLanguageModel) -> None:
        if model.cfg.to_dict() != self.model_config:
            raise ValueError(
                "Model config does not match the config this lens was fitted on: "
                f"{model.cfg.to_dict()} != {self.model_config}"
            )

    def _block(self, name: str) -> slice:
        # "state" is the full-state pseudo-block: the complete transport
        # J_k @ s, i.e. the lens's best approximation of the future logits.
        # Named blocks restrict to one component's contribution.
        if name == "state":
            return slice(0, self.transports.shape[2])
        start, stop = self.blocks[name]
        return slice(start, stop)

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------

    def lens_matrix(self, model: LSTMLanguageModel, block: str, horizon: int) -> Tensor:
        """Dictionary of lens vectors for (block, horizon): (vocab, d_block).

        Row v is the direction in ``block``'s state space whose activation
        promotes emission of vocabulary token v after ``horizon`` steps.
        """
        self._check_model(model)
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(f"horizon must be in [1, {self.max_horizon}], got {horizon}")
        transport = self.transports[horizon - 1][:, self._block(block)]
        return model.unembed.weight.detach() @ transport

    def readout(
        self,
        model: LSTMLanguageModel,
        states: Tensor,
        block: str,
        horizon: int,
        mode: str = "centered",
    ) -> Tensor:
        """Disposition logits for states (..., state_size) -> (..., vocab).

        ``mode`` is one of "centered", "taylor", "raw" (see class docstring).
        ``horizon=0`` is supported for the top hidden block only, where it is
        the plain logit lens (the model's own output, mode-independent).
        """
        self._check_model(model)
        if mode not in ("centered", "taylor", "raw"):
            raise ValueError(f"unknown readout mode: {mode!r}")
        if horizon == 0:
            if block != "h_top" and self.blocks.get(block) != self.blocks["h_top"]:
                raise ValueError("horizon=0 is only defined for the top hidden block")
            return model.logits_from_state(states)
        blk = self._block(block)
        values = states[..., blk]
        if mode != "raw":
            values = values - self.mean_state.to(values.device)[blk]
        transported = values @ self.transports[horizon - 1][:, blk].T
        logits = transported @ model.unembed.weight.detach().T
        if mode == "taylor":
            logits = logits + self.mean_logits.to(logits.device)
        return logits

    @torch.no_grad()
    def apply(
        self,
        model: LSTMLanguageModel,
        token_ids: Tensor,
        blocks: list[str],
        horizons: list[int],
        mode: str = "centered",
    ) -> tuple[dict[tuple[str, int], Tensor], Tensor]:
        """Run the model over ``token_ids`` (T,) and read every state through the lens.

        Returns:
            lens_logits: (block, horizon) -> (T, vocab). Entry ``[p]`` is read
                from the state *after* consuming ``token_ids[p]``; at horizon k
                it approximates the model's logits at position p+k, i.e. its
                disposition toward the token at position p+k+1.
            model_logits: (T, vocab), the model's actual output logits.
        """
        self._check_model(model)
        states = model.states_over_sequence(token_ids)[1:]  # drop zero initial state
        model_logits = model.logits_from_state(states)
        lens_logits = {
            (block, k): self.readout(model, states, block, k, mode=mode)
            for block in blocks
            for k in horizons
        }
        return lens_logits, model_logits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "transports": self.transports.cpu(),
                "mean_state": self.mean_state.cpu(),
                "mean_logits": self.mean_logits.cpu(),
                "blocks": self.blocks,
                "model_config": self.model_config,
                "fit_config": self.fit_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> HorizonLens:
        payload = torch.load(path, map_location=device, weights_only=True)
        return cls(
            transports=payload["transports"],
            mean_state=payload["mean_state"],
            mean_logits=payload["mean_logits"],
            blocks=payload["blocks"],
            model_config=payload["model_config"],
            fit_config=payload["fit_config"],
        )
