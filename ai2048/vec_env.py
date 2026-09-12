"""Batched 2048 environment in pure torch ops (CPU or CUDA).

Bit-exact with the C shim (csrc/shim2048.c over games/2048.c):
  - identical xorshift32(13,17,5) spawn RNG, one uint32 state per game,
    seeded per episode (seed 0 remapped to 0x9E3779B9 exactly like the shim)
  - identical spawn policy: empty cells listed column-major (x=col outer,
    y=row inner), first draw picks the cell, second draw picks 2 (p=0.9)
    or 4 (p=0.1); draws happen only after a *legal* move
  - identical merge/score semantics, validated against the 13 test vectors
    built into the original binary.

Board convention: uint8 tensor [B, 4, 4] indexed [row, col] holding log2
exponents (0 = empty). Directions: 0=UP 1=RIGHT 2=DOWN 3=LEFT.

The RNG state is kept in int64 tensors holding uint32-range values, with
shifts masked to 32 bits - that reproduces C unsigned 32-bit arithmetic
exactly while avoiding torch's patchy native uint32 support.
"""

from __future__ import annotations

import torch

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)
SIZE = 4

_MASK32 = 0xFFFFFFFF
_SEED_ZERO = 0x9E3779B9  # xorshift32 is absorbing at 0; same remap as the shim

# ---------------------------------------------------------------------------
# Row lookup table: every 16-bit packed row (4 nibbles) -> slid row + score.
# ---------------------------------------------------------------------------

_LUT_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _xs32(state: torch.Tensor) -> torch.Tensor:
    """One xorshift32(13,17,5) step on int64 tensors holding uint32 values."""
    x = state
    x = x ^ ((x << 13) & _MASK32)
    x = x ^ (x >> 17)
    x = x ^ ((x << 5) & _MASK32)
    return x & _MASK32


def _slide_rows(cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized port of 2048.c slideArray/findTarget.

    cells: int64 [N, 4], one row per game-line, values 0..15 (nibbles).
    Returns (new_cells [N, 4], gained_score int64 [N]).
    """
    n = cells.shape[0]
    device = cells.device
    arr = cells.clone()
    score = torch.zeros(n, dtype=torch.int64, device=device)
    stop = torch.zeros(n, dtype=torch.int64, device=device)
    rows = torch.arange(n, device=device)

    for x in range(1, SIZE):  # x = 0 can never move
        tile = arr[:, x]
        nz = tile != 0
        t = torch.full((n,), x, dtype=torch.int64, device=device)
        resolved = torch.zeros(n, dtype=torch.bool, device=device)
        for p in range(x - 1, -1, -1):
            cand = nz & ~resolved & (p >= stop)
            col_p = arr[:, p]
            nonzero = cand & (col_p != 0)
            hit_stop = cand & ~nonzero & (p == stop)
            diff = nonzero & (col_p != tile)
            same = nonzero & (col_p == tile)
            t = torch.where(hit_stop | diff | same, torch.full_like(t, p), t)
            t = torch.where(diff, torch.full_like(t, p + 1), t)
            resolved = resolved | hit_stop | nonzero
        move = nz & (t != x)
        if not bool(move.any()):
            continue
        tgt = arr[rows, t]
        merged = move & (tgt == tile)  # equal neighbours -> merge
        slid = move & (tgt == 0)
        arr[rows, t] = torch.where(slid, tile, torch.where(merged, tgt + 1, tgt))
        # Clamp before packing: a 15-merge would need a 17th bit. Two 32768
        # tiles in one line cannot occur in practice; keep the table sane.
        arr[rows, t] = arr[rows, t].clamp(max=15)
        score += torch.where(merged, 1 << (tgt + 1), 0)  # 1 << resulting exponent
        stop = torch.where(merged, t + 1, stop)
        arr[:, x] = torch.where(move, torch.zeros_like(tile), tile)
    return arr, score


def _pack(cells: torch.Tensor) -> torch.Tensor:
    """[N, 4] int64 nibbles -> packed 16-bit row index (cell 0 is the MSB)."""
    return (
        (cells[:, 0] << 12) | (cells[:, 1] << 8) | (cells[:, 2] << 4) | cells[:, 3]
    )


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack -> [N, 4] int64 nibbles."""
    return torch.stack(
        [(packed >> s) & 15 for s in (12, 8, 4, 0)], dim=1
    )


def _build_lut(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(1 << 16, device=device, dtype=torch.int64)
    cells = _unpack(idx)
    new_cells, score = _slide_rows(cells)
    # Slide result can be identity for blocked rows; pack back to uint16 range.
    new_packed = _pack(new_cells.clamp(max=15))
    return new_packed, score


def get_lut(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.device(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = _build_lut(key)
    return _LUT_CACHE[key]


# ---------------------------------------------------------------------------
# Batched board transforms
# ---------------------------------------------------------------------------


def apply_move(
    board: torch.Tensor, direction: int, lut: tuple[torch.Tensor, torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move WITHOUT the random spawn (the policy lookahead primitive).

    board: uint8 [B, 4, 4] [row, col]. Returns (new_board uint8, gain int64[B],
    moved bool[B]).
    """
    lut_new, lut_score = lut if lut is not None else get_lut(board.device)
    b = board
    transposed = direction in (UP, DOWN)
    reversed_ = direction in (RIGHT, DOWN)
    if transposed:
        b = b.transpose(1, 2)
    rows = b.reshape(-1, SIZE)  # [B*4, 4] cells toward index 0
    if reversed_:
        rows = torch.flip(rows, dims=(1,))
    packed = _pack(rows.to(torch.int64))
    new_packed = lut_new[packed]
    gain = lut_score[packed].view(-1, SIZE).sum(dim=1)
    new_rows = _unpack(new_packed).to(torch.uint8)
    if reversed_:
        new_rows = torch.flip(new_rows, dims=(1,))
    new_board = new_rows.view(board.shape)
    if transposed:
        new_board = new_board.transpose(1, 2)
    new_board = new_board.contiguous()
    moved = (new_board != board).any(dim=(1, 2))
    return new_board, gain, moved


def apply_move_batch(
    board: torch.Tensor, actions: torch.Tensor, lut: tuple[torch.Tensor, torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-game directions: board [B,4,4], actions int64[B] in 0..3.

    Returns (new_board, gain int64[B], moved bool[B]).
    """
    new = board.clone()
    gains = torch.zeros(board.shape[0], dtype=torch.int64, device=board.device)
    moved = torch.zeros(board.shape[0], dtype=torch.bool, device=board.device)
    for d in DIRECTIONS:
        sel = actions == d
        if not bool(sel.any()):
            continue
        nb, g, mv = apply_move(board[sel], d, lut)
        new[sel] = nb
        gains[sel] = g
        moved[sel] = mv
    return new, gains, moved


def has_moves(board: torch.Tensor) -> torch.Tensor:
    """bool[B]: False where the game is dead (no empty cell, no equal pair)."""
    empty = (board == 0).any(dim=(1, 2))
    h_pair = (board[:, :, :-1] == board[:, :, 1:]).any(dim=(1, 2))
    v_pair = (board[:, :-1, :] == board[:, 1:, :]).any(dim=(1, 2))
    return empty | h_pair | v_pair


# ---------------------------------------------------------------------------
# The batched environment
# ---------------------------------------------------------------------------


class VecEnv2048:
    """N independent 2048 games stepped in lockstep on one device."""

    def __init__(
        self,
        n: int,
        device: str | torch.device = "cpu",
        seeds: torch.Tensor | int | None = None,
    ):
        self.device = torch.device(device)
        self.n = n
        self.lut = get_lut(self.device)
        self.boards = torch.zeros((n, SIZE, SIZE), dtype=torch.uint8, device=self.device)
        self.scores = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.steps_taken = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.alive = torch.ones(n, dtype=torch.bool, device=self.device)
        self.rng = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.reset(seeds)

    def reset(self, seeds: torch.Tensor | int | list[int] | None = None) -> torch.Tensor:
        """Restart all games. `seeds`: int tensor [n], a list, or one seed for all."""
        if seeds is None:
            seeds = torch.randint(1, 1 << 31, (self.n,), dtype=torch.int64)
        elif isinstance(seeds, int):
            seeds = torch.full((self.n,), seeds, dtype=torch.int64)
        else:
            seeds = torch.as_tensor(seeds, dtype=torch.int64)
        seeds = seeds.to(self.device).to(torch.int64) & _MASK32
        seeds = torch.where(seeds == 0, torch.full_like(seeds, _SEED_ZERO), seeds)
        self.rng = seeds.clone()
        self.boards.zero_()
        self.scores.zero_()
        self.steps_taken.zero_()
        self.alive.fill_(True)
        for _ in range(2):  # two starting tiles, exactly like initBoard()
            self._spawn_all(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self.boards

    @property
    def max_tile(self) -> torch.Tensor:
        """int64[B]: highest tile value (2**exponent) on each board."""
        return (1 << self.boards.view(self.n, -1).to(torch.int64).max(dim=1).values)

    # -- internals ---------------------------------------------------------

    def _spawn_all(self, which: torch.Tensor) -> None:
        """One addRandom() for every env where `which` is True.

        Draw order and cell listing order mirror shim_add_random() exactly so
        RNG streams stay aligned with the C shim.
        """
        if not bool(which.any()):
            return
        s1 = _xs32(self.rng)
        s2 = _xs32(s1)
        self.rng = torch.where(which, s2, self.rng)

        # Empty-cell mask listed column-major (x=col outer, y=row inner).
        empty = (self.boards == 0).to(torch.int64)
        col_major = empty.transpose(1, 2).reshape(self.n, -1)  # [B, 16]
        n_empty = col_major.sum(dim=1)
        pos = s1 % n_empty.clamp(min=1)  # position draw
        # rank r = number of empties up to and including this cell (col-major)
        rank = torch.cumsum(col_major, dim=1)
        pick = (col_major == 1) & (rank == pos.unsqueeze(1) + 1)
        flat_idx = pick.float().argmax(dim=1)  # first True; safe: pos < n_empty
        row, col = flat_idx % SIZE, flat_idx // SIZE  # back to [row, col]

        val = (s2 % 10) // 9 + 1  # 2 w.p. 0.9, 4 w.p. 0.1 (as in 2048.c)
        self.boards[which, row[which], col[which]] = val[which].to(torch.uint8)

    # -- interaction ---------------------------------------------------------

    def step(self, actions: torch.Tensor | int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one move per game. actions: int64 [B] (or scalar/list) in 0..3.

        Dead games ignore their action. Returns (moved bool[B], ended bool[B]);
        `ended` games flip `alive` off and stay frozen afterwards.
        """
        acts = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        if acts.ndim == 0:
            acts = acts.expand(self.n)
        new_boards, gains, moved_any = apply_move_batch(self.boards, acts, self.lut)
        moved = moved_any & self.alive
        self.boards = torch.where(moved.view(-1, 1, 1), new_boards, self.boards)
        self.scores += gains * moved
        self.steps_taken += moved
        self._spawn_all(moved)
        has = has_moves(self.boards)
        ended = moved & ~has
        self.alive &= ~ended
        return moved, ended
