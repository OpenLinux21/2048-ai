"""ctypes binding to games/libshim2048.so (built by `make`).

This is the headless C environment: the unmodified games/2048.c move/merge/end
logic, driven by a seedable xorshift32 spawn RNG inside the shim.

Board conventions
-----------------
C side (2048.c):       board[x][y], x = column, y = row (transposed!).
Python side (here and everywhere else in the package): uint8 array of shape
[4, 4] indexed [row, col] = [y, x]. `board()` converts; `set_board()` writes.

Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT (module constants, used package-wide).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)
DIR_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}

_SIZE = 4
_DEFAULT_LIB = Path(__file__).resolve().parent.parent / "games" / "libshim2048.so"


def _write_board(buf, board: np.ndarray) -> None:
    """Copy a Python [row, col] board into a C [col][row] uint8[16] buffer.

    NB: board.T.ctypes.data alone would be wrong - .T is a view and shares the
    source memory - so the transposed copy must be materialized first.
    """
    dst = np.frombuffer(buf, dtype=np.uint8)
    dst[...] = np.ascontiguousarray(board.T).reshape(-1)


class ShimGame2048:
    """Single-game headless environment backed by the C shim.

    One instance owns its board/score/RNG buffers; instances are independent
    (the RNG state lives in this object, not in the library), so multiple
    games can interleave freely.
    """

    def __init__(self, lib_path: str | Path | None = None):
        path = Path(lib_path) if lib_path else _DEFAULT_LIB
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - build it first with `make` in the repo root"
            )
        self.lib = ctypes.CDLL(str(path))

        board_p = ctypes.POINTER(ctypes.c_uint8)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        self.lib.shim_new_game.argtypes = [board_p, u32p]
        self.lib.shim_step.argtypes = [board_p, u32p, ctypes.c_int, u32p]
        self.lib.shim_step.restype = ctypes.c_int
        self.lib.shim_moved_board.argtypes = [board_p, board_p, ctypes.c_int]
        self.lib.shim_moved_board.restype = ctypes.c_int
        self.lib.shim_game_ended.argtypes = [board_p]
        self.lib.shim_game_ended.restype = ctypes.c_uint8

        self._board_buf = (ctypes.c_uint8 * 16)()
        self._scratch_in = (ctypes.c_uint8 * 16)()
        self._scratch_out = (ctypes.c_uint8 * 16)()
        self._score = ctypes.c_uint32(0)
        self._rng = ctypes.c_uint32(0)
        self.steps = 0

    # -- state management -------------------------------------------------

    def reset(self, seed: int) -> np.ndarray:
        """Start a new game from `seed`; returns the initial board [row, col].

        The RNG state lives in this object and is advanced in place by the
        shim, so the stream continues coherently across reset/step.
        """
        self._rng.value = seed & 0xFFFFFFFF
        self.lib.shim_new_game(self._board_buf, ctypes.byref(self._rng))
        self._score.value = 0
        self.steps = 0
        return self.board()

    @property
    def score(self) -> int:
        return self._score.value

    def board(self) -> np.ndarray:
        """Current board as an owned uint8 array [row, col] of log2 exponents."""
        c_layout = np.frombuffer(self._board_buf, dtype=np.uint8).reshape(4, 4)
        # C layout is [col][row]; Python convention is [row][col].
        return np.ascontiguousarray(c_layout.T)

    def set_board(self, board: np.ndarray, score: int = 0) -> None:
        """Overwrite the env state from a Python-convention [row, col] board."""
        assert board.shape == (4, 4) and board.dtype == np.uint8
        _write_board(self._board_buf, board)
        self._score.value = score

    # -- interaction -------------------------------------------------------

    def step(self, action: int) -> tuple[bool, bool]:
        """Apply a move + spawn. Returns (moved, ended)."""
        flags = self.lib.shim_step(
            self._board_buf, ctypes.byref(self._score), action, ctypes.byref(self._rng)
        )
        if flags:
            self.steps += 1
        return bool(flags & 1), bool(flags & 2)

    # -- pure helpers (no env state) ----------------------------------------

    def moved_board(self, board: np.ndarray, action: int) -> np.ndarray:
        """Board after applying `action` WITHOUT the random spawn."""
        _write_board(self._scratch_in, board)
        self.lib.shim_moved_board(self._scratch_in, self._scratch_out, action)
        out = np.frombuffer(self._scratch_out, dtype=np.uint8).reshape(4, 4)
        return np.ascontiguousarray(out.T)

    def legal_moves(self, board: np.ndarray) -> list[bool]:
        return [self._moves_changed(board, d) for d in DIRECTIONS]

    def _moves_changed(self, board: np.ndarray, action: int) -> bool:
        _write_board(self._scratch_in, board)
        return bool(self.lib.shim_moved_board(self._scratch_in, self._scratch_out, action))

    def game_ended(self, board: np.ndarray) -> bool:
        _write_board(self._scratch_in, board)
        return bool(self.lib.shim_game_ended(self._scratch_in))
