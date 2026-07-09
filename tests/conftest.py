import pytest
import torch

from jspace.config import ModelConfig
from jspace.model import LSTMLanguageModel


@pytest.fixture()
def tiny_model() -> LSTMLanguageModel:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, d_model=8, num_layers=2, dropout=0.0)
    return LSTMLanguageModel(cfg).eval()


@pytest.fixture()
def tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 50, (12,))
