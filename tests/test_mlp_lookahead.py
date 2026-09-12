"""Tests for the MLP policy and the 2-ply expectimax selector."""

import torch

from ai2048.lookahead import expectimax2_actions
from ai2048.policy import LinearPolicy, MLPPolicy, make_policy


def test_mlp_param_size_and_eval():
    mlp = MLPPolicy(hidden=16)
    assert mlp.param_size == 16 * 16 + 16 + 16 + 1
    board = torch.randint(0, 11, (5, 4, 4), dtype=torch.uint8)
    w = torch.randn(5, mlp.param_size) * 0.05
    out = mlp.evaluate(board, w)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_mlp_per_genome_scoring_matches_solo():
    torch.manual_seed(1)
    mlp = MLPPolicy(hidden=8)
    board = torch.randint(0, 11, (4, 4, 4), dtype=torch.uint8)
    w = torch.randn(4, mlp.param_size) * 0.1
    scores, moved = mlp.candidate_scores(board, w)
    for i in range(4):
        solo, solo_moved = mlp.candidate_scores(board[i : i + 1], w[i : i + 1])
        assert torch.allclose(scores[i], solo[0], atol=1e-5)
        assert torch.equal(moved[i], solo_moved[0])


def test_make_policy_dispatch():
    assert isinstance(make_policy("linear", ["value16"]), LinearPolicy)
    assert isinstance(make_policy("mlp", []), MLPPolicy)
    try:
        make_policy("nope", [])
    except ValueError as e:
        assert "unknown policy" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_expectimax2_picks_legal_and_differs_from_greedy_sometimes():
    torch.manual_seed(3)
    lin = LinearPolicy(["value16", "empty", "corner_max", "monotonic4"])
    w = torch.randn(8, lin.param_size) * 0.3
    boards = torch.randint(0, 8, (8, 4, 4), dtype=torch.uint8)
    a2, moved = expectimax2_actions(lin, boards, w, samples=2)
    assert moved.all()
    greedy, moved_g = lin.act(boards, w)
    assert torch.equal(moved, moved_g.any(dim=1))
    # legality: replay each chosen action through apply_move_batch must move
    from ai2048.vec_env import apply_move_batch

    nb, _, mv = apply_move_batch(boards, a2)
    assert bool(mv.all())
    # 2-ply explores spawns, so it should disagree with greedy on some boards
    # (weak property; with these seeds it holds - if it ever fails, sampling
    # collapsed to greedy, which would deserve a look)
    assert not torch.equal(a2, greedy)


def test_expectimax2_deterministic_and_slice_reproducible():
    """Same generator seed -> same actions; a 1-env rerun reproduces itself."""
    lin = LinearPolicy(["value16", "empty"])
    torch.manual_seed(4)
    w = torch.randn(5, lin.param_size) * 0.2
    boards = torch.randint(0, 10, (5, 4, 4), dtype=torch.uint8)

    a1, _ = expectimax2_actions(lin, boards, w, samples=2,
                                generator=torch.Generator().manual_seed(7))
    a2, _ = expectimax2_actions(lin, boards, w, samples=2,
                                generator=torch.Generator().manual_seed(7))
    assert torch.equal(a1, a2)

    for i in range(5):
        a_i_1, _ = expectimax2_actions(lin, boards[i:i+1], w[i:i+1], samples=2,
                                       generator=torch.Generator().manual_seed(7))
        a_i_2, _ = expectimax2_actions(lin, boards[i:i+1], w[i:i+1], samples=2,
                                       generator=torch.Generator().manual_seed(7))
        assert torch.equal(a_i_1, a_i_2)
