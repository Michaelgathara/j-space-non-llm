"""Word-level Tiny Shakespeare corpus.

Word-level (rather than character-level) tokenization is a deliberate choice:
the lens decodes states into *vocabulary* space, and the J-space construction
is only interesting when the token dictionary is overcomplete relative to the
model dimension (n_vocab >> d_model). Tiny Shakespeare yields a ~12k-word
vocabulary against d_model of a few hundred.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CorpusSpec:
    url: str
    # Cap the download (HTTP Range) — TinyStories is ~2GB; a slice of a few
    # tens of MB is already 10-20x more tokens than all of Tiny Shakespeare.
    max_bytes: int | None = None


CORPORA: dict[str, CorpusSpec] = {
    "tinyshakespeare": CorpusSpec(
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt",
    ),
    "tinystories": CorpusSpec(
        url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStories-train.txt",
        max_bytes=25 * 1024 * 1024,
    ),
}

UNK = "<unk>"

# Words (with internal apostrophes, e.g. "don't", "'tis" -> "'", "tis"), or a
# single non-alphanumeric symbol. Case is preserved: Shakespeare's proper
# nouns are the most legible concepts the lens can surface.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def download_corpus(name: str, data_dir: str | Path = "data") -> str:
    """Return corpus text, downloading and caching it at ``data_dir/{name}.txt``."""
    spec = CORPORA[name]
    path = Path(data_dir) / f"{name}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(spec.url)
        if spec.max_bytes is not None:
            request.add_header("Range", f"bytes=0-{spec.max_bytes - 1}")
        with urllib.request.urlopen(request, timeout=120) as resp:
            raw = resp.read(spec.max_bytes) if spec.max_bytes else resp.read()
        # A Range slice may end mid-line; drop the final partial line.
        text = raw.decode("utf-8", errors="ignore")
        if spec.max_bytes is not None and len(raw) >= spec.max_bytes:
            text = text[: text.rfind("\n")]
        path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Vocab:
    """Bidirectional token <-> id mapping. Id 0 is reserved for ``<unk>``."""

    id_to_token: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_token_to_id", {t: i for i, t in enumerate(self.id_to_token)}
        )

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str]) -> torch.Tensor:
        table: dict[str, int] = self._token_to_id  # type: ignore[attr-defined]
        return torch.tensor([table.get(t, 0) for t in tokens], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    @classmethod
    def build(cls, tokens: list[str], min_count: int = 1) -> Vocab:
        # Sorted by frequency for readability of the vocab file; determinism
        # is guaranteed by the secondary lexicographic key. Words rarer than
        # ``min_count`` fold into <unk> — with tied embeddings a
        # seen-once word never learns a useful vector anyway.
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        ordered = sorted(
            (t for t, c in counts.items() if c >= min_count),
            key=lambda t: (-counts[t], t),
        )
        return cls(id_to_token=(UNK, *ordered))


@dataclass(frozen=True)
class Corpus:
    """Encoded corpus with a contiguous train/validation split."""

    vocab: Vocab
    train_ids: torch.Tensor
    val_ids: torch.Tensor

    @classmethod
    def load(
        cls,
        name: str = "tinystories",
        data_dir: str | Path = "data",
        val_fraction: float = 0.05,
        vocab_min_count: int = 3,
    ) -> Corpus:
        tokens = tokenize(download_corpus(name, data_dir))
        split = int(len(tokens) * (1.0 - val_fraction))
        vocab = Vocab.build(tokens[:split], min_count=vocab_min_count)
        return cls(
            vocab=vocab,
            train_ids=vocab.encode(tokens[:split]),
            val_ids=vocab.encode(tokens[split:]),
        )


def sample_batch(
    ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(inputs, targets)`` of shape (batch, seq_len), targets shifted by one."""
    starts = torch.randint(
        0, len(ids) - seq_len - 1, (batch_size,), generator=generator
    )
    offsets = torch.arange(seq_len)
    idx = starts[:, None] + offsets[None, :]
    return ids[idx], ids[idx + 1]
