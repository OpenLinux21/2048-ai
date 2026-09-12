"""2-ply expectimax action selection (optional, opt-in quality upgrade).

Depth 1 (the default everywhere): pick the move whose immediate result scores
best - spawns are handled implicitly by the evolved weights.

Depth 2 (this module): for each candidate move, average the post-spawn value
over sampled spawns, where the post-spawn value is the best response move's
score. Chance sampling: `samples` uniformly chosen empty cells per candidate,
each tried as a 2 (weight 0.9) and a 4 (weight 0.1).

Cost control: all chance branches of one direction are stacked into a single
[b * samples * 2] mega-batch, so a decision costs ~2 depth-1 sweeps over a
bigger batch instead of dozens of tiny sweeps - necessary for this to be
usable at all outside toy sizes.
"""

from __future__ import annotations

import torch

from ai2048.vec_env import DIRECTIONS, apply_move

_CHANCE = ((1, 0.9), (2, 0.1))  # (spawn exponent, probability)


@torch.no_grad()
def expectimax2_actions(
    policy,
    board: torch.Tensor,
    weights: torch.Tensor,
    samples: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """board [B,4,4], weights [B,F] -> (actions int64[B], moved_any bool[B])."""
    b = board.shape[0]
    device = board.device
    n_branch = samples * len(_CHANCE)
    # Branch v of env i lives at mega-batch index v*b + i.
    branch_vals = torch.tensor([v for v, _ in _CHANCE for _ in range(samples)], device=device)
    branch_w = torch.tensor([w for _, w in _CHANCE for _ in range(samples)], device=device)

    values = torch.full((b, len(DIRECTIONS)), -torch.inf, dtype=torch.float32, device=device)
    moved_any = torch.zeros(b, dtype=torch.bool, device=device)

    for d in DIRECTIONS:
        after_move, _, moved = apply_move(board, d)
        if not bool(moved.any()):
            continue
        moved_any |= moved
        empties = (after_move == 0).reshape(b, -1)
        n_empty = empties.sum(dim=1)

        # One mega-batch holding every (sample x chance-value) spawn variant:
        # layout [b * n_branch, 4, 4], variant v of env i at index v*b + i.
        spawned = after_move.repeat(n_branch, 1, 1)
        mask_moved = moved.repeat(n_branch)
        for v in range(n_branch):
            r = torch.randint(0, 16, (b,), device=device, generator=generator)
            rank = torch.cumsum(empties.to(torch.int64), dim=1)
            target = (r % n_empty.clamp(min=1)).unsqueeze(1) + 1
            pick = empties & (rank == target)
            idx = pick.float().argmax(dim=1)  # first empty with matching rank
            row, col = idx // 4, idx % 4
            lo, hi = v * b, (v + 1) * b
            m = mask_moved[lo:hi]
            spawned[lo:hi][m, row[m], col[m]] = branch_vals[v].to(torch.uint8)

        # Best response: one sweep over 4 moves on the whole mega-batch.
        best = torch.full((b * n_branch,), -torch.inf, dtype=torch.float32, device=device)
        mega_weights = weights.repeat(n_branch, 1)
        for d2 in DIRECTIONS:
            nb, _, mv2 = apply_move(spawned, d2)
            sc = policy.evaluate(nb, mega_weights)
            best = torch.where(mv2, torch.maximum(best, sc), best)
        best = best.view(n_branch, b)

        weighted = (best * branch_w.unsqueeze(1) * moved.unsqueeze(0)).sum(dim=0)
        wsum = (branch_w.unsqueeze(1) * moved.unsqueeze(0)).sum(dim=0)
        values[:, d] = torch.where(wsum > 0, weighted / wsum.clamp(min=1e-6), weighted)

    return values.argmax(dim=1), moved_any
