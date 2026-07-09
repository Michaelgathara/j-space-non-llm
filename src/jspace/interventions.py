"""Causal interventions on the recurrent state, in lens coordinates.

These mirror the three intervention families from the J-space paper:

* **Steering** — add a scaled lens vector to a state block, injecting or
  suppressing a concept's disposition-to-be-emitted.
* **Ablation** — project a state block out of the span of chosen lens
  directions, removing those concepts' first-order route to the output.
* **Patching in lens coordinates** — express a state block as coordinates
  over a small lens-vector dictionary, permute the coordinates, and
  reconstruct: exchanges concepts while leaving the orthogonal complement of
  the dictionary untouched.

All interventions are applied through ``run_with_state_hook``, which replays
the model step by step and lets a hook rewrite the flat state after each step.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from jspace.lens import HorizonLens
from jspace.model import LSTMLanguageModel

# hook(position, flat_state) -> flat_state; position p is the index of the
# token just consumed.
StateHook = Callable[[int, Tensor], Tensor]


@torch.no_grad()
def run_with_state_hook(
    model: LSTMLanguageModel, token_ids: Tensor, hook: StateHook
) -> tuple[Tensor, Tensor]:
    """Run ``token_ids`` (T,) through the model, rewriting states via ``hook``.

    Returns (states (T, S), logits (T, vocab)) — post-hook, so downstream
    computation sees the intervened trajectory.
    """
    emb = model.embed(token_ids)
    state = model.initial_state(token_ids.device)
    states = []
    for p in range(len(token_ids)):
        state = hook(p, model.step_flat(state, emb[p]))
        states.append(state)
    stacked = torch.stack(states)
    return stacked, model.logits_from_state(stacked)


def token_lens_vector(
    lens: HorizonLens,
    model: LSTMLanguageModel,
    block: str,
    horizon: int,
    token_id: int,
) -> Tensor:
    """The lens vector for one token: the direction in ``block``'s state space
    that most increases the model's disposition to emit ``token_id`` after
    ``horizon`` steps. Returned unnormalized (its norm is meaningful: it is
    the first-order logit gain per unit of state perturbation)."""
    return lens.lens_matrix(model, block, horizon)[token_id]


def steering_hook(
    lens: HorizonLens,
    model: LSTMLanguageModel,
    block: str,
    horizon: int,
    token_id: int,
    alpha: float,
    positions: range | None = None,
) -> StateHook:
    """Hook that adds ``alpha`` units of the (unit-normalized) lens vector for
    ``token_id`` to ``block`` at each position in ``positions`` (default: all)."""
    start, stop = lens.blocks[block]
    v = token_lens_vector(lens, model, block, horizon, token_id)
    v = v / v.norm()

    def hook(p: int, state: Tensor) -> Tensor:
        if positions is not None and p not in positions:
            return state
        state = state.clone()
        state[start:stop] += alpha * v.to(state.device)
        return state

    return hook


def ablation_hook(
    lens: HorizonLens,
    model: LSTMLanguageModel,
    block: str,
    horizon: int,
    token_ids: list[int],
    positions: range | None = None,
) -> StateHook:
    """Hook that projects ``block`` out of the span of the lens vectors for
    ``token_ids`` (the recurrent analog of top-k J-direction ablation)."""
    start, stop = lens.blocks[block]
    dictionary = lens.lens_matrix(model, block, horizon)[token_ids]  # (m, d_block)
    q, _ = torch.linalg.qr(dictionary.T)  # orthonormal basis, (d_block, m)

    def hook(p: int, state: Tensor) -> Tensor:
        if positions is not None and p not in positions:
            return state
        state = state.clone()
        blk = state[start:stop]
        basis = q.to(state.device)
        state[start:stop] = blk - basis @ (basis.T @ blk)
        return state

    return hook


def patch_lens_coordinates(
    state_block: Tensor, dictionary: Tensor, permutation: list[int]
) -> Tensor:
    """Swap concepts within a state block, in lens coordinates.

    Solves least-squares coordinates c of ``state_block`` over the rows of
    ``dictionary`` (m, d_block), permutes them, and adds back the difference:

        block' = block + D^T (P c - c)

    Components of the block orthogonal to the dictionary rows are preserved
    exactly.
    """
    coords = torch.linalg.lstsq(dictionary.T, state_block).solution  # (m,)
    permuted = coords[torch.tensor(permutation, device=coords.device)]
    return state_block + dictionary.T @ (permuted - coords)
