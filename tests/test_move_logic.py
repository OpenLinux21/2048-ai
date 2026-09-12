"""Move/merge/score semantics, pinned by the 13 vectors shipped inside the
original binary (games/2048 test -> testSucceed()) plus hand cases.

Vectors are "slide toward index 0" semantics; in our Python row-major board
that is a LEFT move on the first row.
"""

import numpy as np
import pytest
import torch

from ai2048 import LEFT, UP, ShimGame2048
from ai2048.vec_env import apply_move, has_moves

# 4x in, 4x out, 1x points - verbatim from games/2048.c testSucceed()
C_VECTORS = [
    (0, 0, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 1, 1, 2, 0, 0, 0, 4),
    (0, 1, 0, 1, 2, 0, 0, 0, 4),
    (1, 0, 0, 1, 2, 0, 0, 0, 4),
    (1, 0, 1, 0, 2, 0, 0, 0, 4),
    (1, 1, 1, 0, 2, 1, 0, 0, 4),
    (1, 0, 1, 1, 2, 1, 0, 0, 4),
    (1, 1, 0, 1, 2, 1, 0, 0, 4),
    (1, 1, 1, 1, 2, 2, 0, 0, 8),
    (2, 2, 1, 1, 3, 2, 0, 0, 12),
    (1, 1, 2, 2, 2, 3, 0, 0, 12),
    (3, 0, 1, 1, 3, 2, 0, 0, 4),
    (2, 0, 1, 1, 2, 2, 0, 0, 4),
]


@pytest.mark.parametrize("case", C_VECTORS, ids=range(len(C_VECTORS)))
def test_vectors_torch(case):
    row_in, row_out, points = case[:4], case[4:8], case[8]
    board = torch.zeros(1, 4, 4, dtype=torch.uint8)
    board[0, 0, :] = torch.tensor(row_in, dtype=torch.uint8)
    new, gain, moved = apply_move(board, LEFT)
    assert new[0, 0, :].tolist() == list(row_out)
    assert int(gain[0]) == points
    assert bool(moved[0]) == (row_in != row_out)


@pytest.mark.parametrize("case", C_VECTORS, ids=range(len(C_VECTORS)))
def test_vectors_shim(case):
    row_in, row_out = case[:4], case[4:8]
    game = ShimGame2048()
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, :] = row_in
    out = game.moved_board(board, LEFT)
    assert out[0, :].tolist() == list(row_out)


def test_illegal_move_leaves_board_and_no_spawn():
    """main() only spawns after a legal move; the shim must match."""
    game = ShimGame2048()
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, :] = (2, 1, 0, 0)  # packed left -> LEFT is a no-op
    game.set_board(board, score=7)
    rng_before = game._rng.value
    moved, _ = game.step(LEFT)
    assert not moved
    assert (game.board() == board).all()
    assert game.score == 7
    assert game._rng.value == rng_before  # no RNG draws on illegal moves


def test_double_merge_prevented():
    """[2,2,2,2] -> [3,3,0,0] with 12 points, never [4,0,0,0]."""
    board = torch.zeros(2, 4, 4, dtype=torch.uint8)
    board[0, 0, :] = torch.tensor((2, 2, 2, 2), dtype=torch.uint8)
    board[1, 1, :] = torch.tensor((1, 1, 1, 1), dtype=torch.uint8)
    new, gain, _ = apply_move(board, LEFT)
    assert new[0, 0, :].tolist() == [3, 3, 0, 0]
    assert int(gain[0]) == 16  # two (2+2)->4 merges: 8 points each
    assert new[1, 1, :].tolist() == [2, 2, 0, 0]
    assert int(gain[1]) == 8  # (2+2)->4 plus (2+2)->4 on exp-1 tiles


def test_up_moves_columns_not_rows():
    """Transposition check: UP slides each column toward row 0."""
    board = torch.zeros(1, 4, 4, dtype=torch.uint8)
    board[0, 2, 0] = 1  # row 2, col 0
    board[0, 3, 0] = 1  # row 3, col 0 -> merges upward into 4 (exp 2)
    board[0, 0, 1] = 3  # already at top, stays
    new, gain, _ = apply_move(board, UP)
    assert new[0, 0, 0] == 2
    assert new[0, 1, 0] == 0
    assert new[0, 0, 1] == 3
    assert int(gain[0]) == 4  # merge score of the resulting 4-tile


def test_directions_are_all_consistent():
    board = torch.zeros(1, 4, 4, dtype=torch.uint8)
    board[0, 0, 0] = 1  # a single 2-tile in the top-left corner
    from ai2048 import DOWN, RIGHT
    from ai2048.vec_env import apply_move as am

    assert am(board, UP)[0][0, 0, 0] == 1
    assert am(board, LEFT)[0][0, 0, 0] == 1
    assert am(board, DOWN)[0][0, 3, 0] == 1
    assert am(board, RIGHT)[0][0, 0, 3] == 1


def test_has_moves_matches_shim_ended():
    from ai2048 import ShimGame2048

    # full board, no equal neighbours horizontally or vertically
    full = np.zeros((4, 4), dtype=np.uint8)
    for r in range(4):
        full[r] = np.roll(np.array([1, 2, 3, 4], dtype=np.uint8), r)
    board = torch.from_numpy(full).unsqueeze(0)
    assert not bool(has_moves(board)[0])
    assert ShimGame2048().game_ended(full)

    full[0, 0] = full[0, 1] = 5  # a mergeable pair revives the board
    board = torch.from_numpy(full).unsqueeze(0)
    assert bool(has_moves(board)[0])
    assert not ShimGame2048().game_ended(full)
