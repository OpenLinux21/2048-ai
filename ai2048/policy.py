"""Pluggable board-evaluation policies for the GA.

A policy scores BOARD STATES (not state-action pairs); the action is chosen by
applying each of the four moves (without spawning) and picking the best-scoring
result among the legal ones. The genome is a flat float32 vector and ONE weight
vector is supported per environment slot, so a whole population can be
evaluated in a single batched rollout.

Policies:
  LinearPolicy - features(board) . weights          (default, interpretable)
  MLPPolicy    - tiny per-board MLP over exponents  (larger genome, opt-in)
"""

from __future__ import annotations

import torch

from ai2048.vec_env import DIRECTIONS, apply_move

ALL_FEATURES = ("value16", "empty", "max_exp", "corner_max", "monotonic4", "smoothness")


def feature_size(features: list[str] | tuple[str, ...]) -> int:
    sizes = {"value16": 16, "empty": 1, "max_exp": 1, "corner_max": 1,
             "monotonic4": 4, "smoothness": 1}
    unknown = [f for f in features if f not in sizes]
    if unknown:
        raise ValueError(f"unknown features: {unknown}; valid: {ALL_FEATURES}")
    return sum(sizes[f] for f in features)


def extract_features(board: torch.Tensor, features: list[str] | tuple[str, ...]) -> torch.Tensor:
    """uint8 board [B, 4, 4] (log2 exponents) -> float32 features [B, F].

    Feature order matches `features`; genomes must use the same order.
    """
    exp = board.to(torch.float32)
    occupied = board != 0
    value = torch.where(occupied, torch.exp2(exp), torch.zeros_like(exp))
    b = board.shape[0]
    cols: list[torch.Tensor] = []

    for name in features:
        if name == "value16":
            cols.append(value.reshape(b, 16))
        elif name == "empty":
            cols.append((board == 0).sum(dim=(1, 2), dtype=torch.float32).unsqueeze(1))
        elif name == "max_exp":
            cols.append(exp.reshape(b, -1).max(dim=1, keepdim=True).values)
        elif name == "corner_max":
            corners = value[:, [0, 0, 3, 3], [0, 3, 0, 3]]
            cols.append(corners.max(dim=1, keepdim=True).values)
        elif name == "monotonic4":
            # Per-direction penalty: how far each row/col is from being sorted
            # (in exponent space). GA learns the sign/scale.
            row_dec = (exp[:, :, :-1] - exp[:, :, 1:]).clamp(min=0).sum(dim=(1, 2)).unsqueeze(1)
            row_inc = (exp[:, :, 1:] - exp[:, :, :-1]).clamp(min=0).sum(dim=(1, 2)).unsqueeze(1)
            col_dec = (exp[:, :-1, :] - exp[:, 1:, :]).clamp(min=0).sum(dim=(1, 2)).unsqueeze(1)
            col_inc = (exp[:, 1:, :] - exp[:, :-1, :]).clamp(min=0).sum(dim=(1, 2)).unsqueeze(1)
            cols.extend([row_dec, row_inc, col_dec, col_inc])
        elif name == "smoothness":
            h = (exp[:, :, :-1] - exp[:, :, 1:]).abs().sum(dim=(1, 2)).unsqueeze(1)
            v = (exp[:, :-1, :] - exp[:, 1:, :]).abs().sum(dim=(1, 2)).unsqueeze(1)
            cols.append(-(h + v))
    return torch.cat(cols, dim=1)


class BaseBoardPolicy:
    """Shared greedy action selection over the four post-move candidate boards."""

    param_size: int

    def evaluate(self, board: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def candidate_scores(
        self, board: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores [B, 4] of the four post-move boards + legality mask [B, 4]."""
        scores = torch.empty(
            (board.shape[0], len(DIRECTIONS)), dtype=torch.float32, device=board.device
        )
        moved = torch.empty_like(scores, dtype=torch.bool)
        for d in DIRECTIONS:
            new_board, _, mv = apply_move(board, d)
            scores[:, d] = self.evaluate(new_board, weights)
            moved[:, d] = mv
        scores = torch.where(moved, scores, torch.full_like(scores, -torch.inf))
        return scores, moved

    @torch.no_grad()
    def act(self, board: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy 1-ply: best-scoring legal move per game. Returns (int64 [B], moved)."""
        candidates, moved = self.candidate_scores(board, weights)
        return candidates.argmax(dim=1), moved


class LinearPolicy(BaseBoardPolicy):
    """Board scorer: score = features(board) . weights. Genome = weights [F]."""

    def __init__(self, features: list[str] | tuple[str, ...] = ("value16", "empty")):
        self.features = tuple(features)
        self.param_size = feature_size(self.features)

    def evaluate(self, board: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """board [B,4,4], weights [B,F] (per-game genomes) -> scores [B]."""
        feats = extract_features(board, self.features)
        return (feats * weights).sum(dim=1)


class MLPPolicy(BaseBoardPolicy):
    """Per-board tiny MLP: 16 normalized exponents -> H tanh -> scalar score.

    Genome layout (per environment slot): [16*H | H | H | 1] =
    W1 (16xH) + b1 (H) + W2 (H) + b2. With per-game genomes this evaluates a
    full population with plain einsums - no nn.Module needed.
    """

    def __init__(self, hidden: int = 32):
        self.hidden = hidden
        self.param_size = 16 * hidden + hidden + hidden + 1

    def evaluate(self, board: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        b = board.shape[0]
        h_dim = self.hidden
        x = board.to(torch.float32).reshape(b, 16) / 15.0
        w1 = weights[:, : 16 * h_dim].reshape(b, 16, h_dim)
        b1 = weights[:, 16 * h_dim : 17 * h_dim]
        w2 = weights[:, 17 * h_dim : 17 * h_dim + h_dim]
        b2 = weights[:, -1]
        hidden = torch.tanh(torch.einsum("bf,bfh->bh", x, w1) + b1)
        return torch.einsum("bh,bh->b", hidden, w2) + b2


def make_policy(policy_type: str, features, mlp_hidden: int = 32) -> BaseBoardPolicy:
    if policy_type == "linear":
        return LinearPolicy(features)
    if policy_type == "mlp":
        return MLPPolicy(mlp_hidden)
    raise ValueError(f"unknown policy type: {policy_type}")
