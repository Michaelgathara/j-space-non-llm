"""Horizon-indexed Jacobian lens for recurrent networks.

An adaptation of Anthropic's J-lens / J-space technique ("Verbalizable
Representations Form a Global Workspace in Language Models",
transformer-circuits.pub/2026/workspace/) to recurrent architectures.

Where the transformer lens transports residual-stream activations across
*layers* into the unembedding basis, the recurrent lens transports the
persistent recurrent state across *time*:

    J_k = E[ d h_top(t+k) / d s(t) ]

so that ``W_U @ J_k @ s_t`` reads out which vocabulary tokens the state at
time ``t`` is disposed to make the model emit ``k`` steps in the future.
"""

from jspace.config import FitConfig, ModelConfig, TrainConfig
from jspace.lens import HorizonLens
from jspace.model import LSTMLanguageModel

__all__ = [
    "FitConfig",
    "HorizonLens",
    "LSTMLanguageModel",
    "ModelConfig",
    "TrainConfig",
]
