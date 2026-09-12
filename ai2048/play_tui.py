"""Drive the REAL games/2048 binary through a pty: the AI reads the board the
game renders (ANSI frames) and plays by injecting arrow-key bytes, exactly like
a human at the keyboard. Used for demos and as an end-to-end sanity check that
the trained policy works on the untouched game.
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ai2048.shim_env import DOWN, LEFT, RIGHT, UP
from ai2048.vec_env import apply_move, get_lut

ARROW_KEYS = {UP: b"\x1b[A", DOWN: b"\x1b[B", RIGHT: b"\x1b[C", LEFT: b"\x1b[D"}
ACTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}

_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SCORE_RE = re.compile(r"2048\.c\s+(\d+)\s+pts")
_CELL_RE = re.compile(r"^(?:·|\d{1,5})$")


class GameSession:
    """One games/2048 child process on a pseudo-terminal."""

    def __init__(self, binary: str | Path = "games/2048", scheme: str = "blackwhite"):
        self.binary = str(binary)
        if not Path(self.binary).exists():
            raise FileNotFoundError(f"game binary not found: {self.binary}")
        master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            [self.binary, scheme],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        self.master = master
        self._buf = b""

    def read_available(self, timeout: float = 0.05) -> bytes:
        out = b""
        while True:
            r, _, _ = select.select([self.master], [], [], timeout)
            if not r:
                break
            try:
                chunk = os.read(self.master, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            timeout = 0.02  # drain quickly once data flows
        self._buf += out
        return out

    def latest_frame(self) -> tuple[np.ndarray, int] | None:
        """Parse the most recent COMPLETE frame; None while still drawing."""
        frames = self._buf.split(b"\x1b[H")
        if len(frames) < 2:
            return None
        text = frames[-1].decode("utf-8", errors="replace")
        board = parse_board(text)
        if board is None:
            return None
        m = _SCORE_RE.search(text)
        score = int(m.group(1)) if m else 0
        return board, score

    def game_over(self) -> bool:
        return b"GAME OVER" in self._buf

    def send(self, data: bytes) -> None:
        os.write(self.master, data)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send(b"q")  # quit? -> y
                self.send(b"y")
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        os.close(self.master)


def parse_board(text: str) -> np.ndarray | None:
    """Extract the 4x4 board from one rendered frame; None if incomplete.

    drawBoard() prints 4 row-groups of 3 lines; the middle line of each group
    is exactly four 7-char cells ("   ·   " or a right-aligned number).
    """
    clean = _CSI_RE.sub("", text).replace("\r", "")  # pty adds \r before \n
    lines = clean.split("\n")
    middles = []
    for line in lines:
        if len(line) != 28 or not line.strip():
            continue
        cells = [line[i * 7 : (i + 1) * 7].strip() for i in range(4)]
        if all(_CELL_RE.match(c) for c in cells):
            middles.append(cells)
    if len(middles) < 4:
        return None
    board = np.zeros((4, 4), dtype=np.uint8)
    for r, cells in enumerate(middles[-4:]):
        for c, cell in enumerate(cells):
            board[r, c] = 0 if cell == "·" else min(int(cell).bit_length() - 1, 15)
    return board


def choose_action(
    board: np.ndarray,
    weights: torch.Tensor,
    features: tuple[str, ...],
    policy=None,
    lookahead: int = 1,
) -> int | None:
    from ai2048.policy import LinearPolicy  # local import avoids cycles in tests

    policy = policy or LinearPolicy(features)
    board_t = torch.from_numpy(board).unsqueeze(0)
    if lookahead == 2:
        from ai2048.lookahead import expectimax2_actions

        action, moved = expectimax2_actions(policy, board_t, weights.unsqueeze(0))
        if not bool(moved[0]):
            return None
        return int(action[0])
    scores = torch.empty(4)
    moved = torch.zeros(4, dtype=torch.bool)
    lut = get_lut(board_t.device)
    for d in (UP, RIGHT, DOWN, LEFT):
        nb, _, mv = apply_move(board_t, d, lut)
        scores[d] = policy.evaluate(nb, weights.unsqueeze(0))[0]
        moved[d] = mv
    scores = torch.where(moved, scores, torch.full_like(scores, -torch.inf))
    if not bool(moved.any()):
        return None
    return int(scores.argmax())


def render_board(
    board: np.ndarray, score: int, moves: int, action: str | None = None
) -> None:
    """Clear-screen redraw of the parsed board, for `--render` watch mode."""
    print("\x1b[2J\x1b[H", end="")  # clear screen + cursor home
    header = f"2048 AI  |  score {score}  |  move {moves}"
    if action:
        header += f"  |  {action}"
    print(header)
    print("+-------+-------+-------+-------+")
    for r in range(4):
        cells = []
        for c in range(4):
            e = int(board[r, c])
            cells.append("     . " if e == 0 else f"{2 ** e:6d} ")
        print("|" + "|".join(cells) + "|")
        print("+-------+-------+-------+-------+")


def load_policy(path: str | None, features: tuple[str, ...], device: str = "cpu"):
    """Returns (policy, weights) - random linear policy when path is omitted."""
    from ai2048.policy import make_policy

    if path is None:
        policy = make_policy("linear", features)
        g = torch.Generator().manual_seed(0)
        return policy, torch.empty(policy.param_size).uniform_(-1, 1, generator=g)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    feats = tuple(ckpt.get("features", features))
    policy = make_policy(ckpt.get("policy_type", "linear"), feats,
                         ckpt.get("mlp_hidden", 32))
    return policy, ckpt["genome"].to(device).float()


def main() -> None:
    p = argparse.ArgumentParser(description="watch the AI play the real games/2048 via arrow keys")
    p.add_argument("--checkpoint", default=None, help="best.pt from a training run (omit = random policy)")
    p.add_argument("--features", nargs="*", default=["value16", "empty"])
    p.add_argument("--binary", default="games/2048")
    p.add_argument("--scheme", default="blackwhite", choices=["original", "blackwhite", "bluered", "whiteblack"])
    p.add_argument("--delay", type=float, default=0.25, help="seconds between moves (game sleeps 0.15 per move)")
    p.add_argument("--max-moves", type=int, default=100000)
    p.add_argument("--lookahead", type=int, default=1, choices=(1, 2),
                   help="2 = 2-ply expectimax (slower, stronger)")
    p.add_argument("--render", action="store_true",
                   help="redraw the parsed board every move (live watch mode)")
    p.add_argument("--quiet", action="store_true", help="only print the final result")
    args = p.parse_args()

    policy, weights = load_policy(args.checkpoint, tuple(args.features))
    session = GameSession(args.binary, args.scheme)
    moves = 0
    result = "incomplete"
    try:
        # wait for the first complete frame
        board = score = None
        for _ in range(100):
            session.read_available(0.05)
            frame = session.latest_frame()
            if frame:
                board, score = frame
                break
        if board is None:
            raise RuntimeError("never received a full frame from the game")
        if args.render:
            render_board(board, score, 0)
        elif not args.quiet:
            print(f"start: score={score} board=\n{board}")
        while moves < args.max_moves:
            action = choose_action(board, weights, tuple(args.features), policy, args.lookahead)
            if action is None:
                result = "no legal moves"
                break
            action_name = ACTION_NAMES[action]
            session.send(ARROW_KEYS[action])
            moves += 1
            deadline = time.time() + max(args.delay, 0.2)
            while time.time() < deadline:
                session.read_available(0.03)
                frame = session.latest_frame()
                if frame:
                    board, score = frame
            if session.game_over():
                result = "game over"
                break
            if args.render:
                render_board(board, score, moves, action_name)
            elif not args.quiet:
                print(f"move {moves:4d} {action_name:5s} score={score}")
    except KeyboardInterrupt:
        result = "interrupted by user"
    finally:
        session.close()

    max_exp = int(board.max()) if board is not None else 0
    print(f"result: {result} after {moves} moves | final score {score} | max tile {1 << max_exp}")


if __name__ == "__main__":
    main()
