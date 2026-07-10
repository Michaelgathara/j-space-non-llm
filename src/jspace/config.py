"""Configuration dataclasses shared by scripts and library code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the LSTM language model."""

    vocab_size: int
    d_model: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    # Tie unembedding to the input embedding: standard for word-level LMs on
    # small corpora, and halves the (dominant) embedding parameter count.
    tie_embeddings: bool = True
    # Residual connections between layers: x_{l+1} = x_l + h_l. Besides
    # making depth trainable, this creates a residual stream across layers —
    # the structure the transformer J-lens reads — so each layer's hidden
    # state has a *direct*, decodable contribution to the logits (the k=0
    # depth lens).
    residual: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**d)


@dataclass(frozen=True)
class TrainConfig:
    """Language-model training hyperparameters."""

    seq_len: int = 128
    batch_size: int = 64
    max_steps: int = 20_000
    lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 20
    seed: int = 0


@dataclass(frozen=True)
class FitConfig:
    """Lens-fitting hyperparameters.

    Attributes:
        max_horizon: largest temporal horizon K; transports are fitted for
            k = 1..K. (k = 0 for the top hidden block is the plain logit
            lens ``W_U @ h_top`` and needs no transport.)
        num_sequences: number of corpus sequences averaged over. Anthropic
            report saturation near ~100 prompts; we default comfortably above.
        seq_len: length of each sequence used for fitting.
        burn_in: source positions ``t < burn_in`` are excluded from the
            average so transports reflect a warmed-up recurrent state rather
            than the arbitrary zero initial state.
        seed: sampling seed for choosing corpus sequences.
    """

    max_horizon: int = 16
    num_sequences: int = 128
    seq_len: int = 128
    burn_in: int = 16
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FitConfig:
        return cls(**d)


@dataclass(frozen=True)
class Paths:
    """Canonical on-disk layout for artifacts, relative to the repo root."""

    data_dir: str = "data"
    run_dir: str = "runs/default"

    @property
    def checkpoint(self) -> str:
        return f"{self.run_dir}/model.pt"

    @property
    def lens_file(self) -> str:
        return f"{self.run_dir}/lens.pt"
