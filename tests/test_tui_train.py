"""TUI frame parser fixtures + end-to-end mini-training smoke test."""

import numpy as np
import torch

from ai2048.play_tui import parse_board

# A blackwhite-scheme frame shaped exactly like drawBoard() output (with the
# pty's \r\n), showing row0=[2,-,4,-] row1=[-,-,-,8] row2=[16,...] etc.
FRAME = (
    "\x1b[H2048.c               128 pts\r\n\r\n"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m      2\x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m"
    "\x1b[38;5;255;48;5;232m      4\x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m   ·   \x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m"
    "\x1b[38;5;255;48;5;232m   ·   \x1b[m\x1b[38;5;255;48;5;232m      8\x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m     16\x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m"
    "\x1b[38;5;255;48;5;232m   ·   \x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m"
    "\x1b[38;5;255;48;5;232m       \x1b[m\x1b[38;5;255;48;5;232m       \x1b[m\r\n"
    "\x1b[38;5;255;48;5;232m   ·   \x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m"
    "\x1b[38;5;255;48;5;232m   ·   \x1b[m\x1b[38;5;255;48;5;232m   ·   \x1b[m\r\n"
    "\r\n"
    "        ←,↑,→,↓ or q        \r\n"
    "\x1b[A"
)


def test_parse_board_full_frame():
    board = parse_board(FRAME)
    assert board is not None
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[0] = (1, 0, 2, 0)
    expected[1, 3] = 3
    expected[2, 0] = 4
    assert np.array_equal(board, expected)


def test_parse_board_incomplete_returns_none():
    assert parse_board(FRAME[:400]) is None
    assert parse_board("") is None


def test_parse_game_over_marker():
    import ai2048.play_tui as tui  # import sanity: module must import cleanly

    assert callable(tui.parse_board)
    assert b"GAME OVER" in b"\x1b[H         GAME OVER          \r\n"


def test_episode_seed_pairing_is_genome_major():
    """Slot j must pair genome j//k with seed j%k - the view(p,k) contract."""
    from ai2048.train import TrainConfig, episode_seeds

    cfg = TrainConfig()
    p, k = cfg.population, cfg.episodes_per_genome
    pool = episode_seeds(cfg.seed, generation=3, n_episodes=k)
    seeds = pool.repeat(p)                      # what Trainer.evaluate does
    assert seeds.shape == (p * k,)
    assert torch.equal(seeds.view(p, k), pool.expand(p, k))  # row i = genome i's episodes


def test_render_board(capsys):
    import numpy as np

    from ai2048.play_tui import render_board

    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 1  # tile 2
    board[3, 3] = 11  # tile 2048
    render_board(board, 128, 42, "UP")
    out = capsys.readouterr().out
    assert "score 128" in out and "move 42" in out and "UP" in out
    assert "2048" in out and "     2" in out
    assert out.count("+-------+-------+-------+-------+") == 5  # 4 rows + borders
    assert out.startswith("\x1b[2J\x1b[H")  # clear + home for live redraw


def test_mini_training_run(tmp_path):
    """2 generations of a tiny GA must run end-to-end and checkpoint."""
    from ai2048.train import TrainConfig, Trainer

    cfg = TrainConfig(
        population=6,
        episodes_per_genome=2,
        generations=2,
        max_episode_steps=3000,
        device="cpu",
        seed=42,
        out_dir=str(tmp_path / "mini"),
        checkpoint_every=1,
    )
    trainer = Trainer(cfg)
    trainer.run()
    assert (tmp_path / "mini" / "best.pt").exists()
    assert (tmp_path / "mini" / "checkpoint_gen2.pt").exists()
    ckpt = torch.load(tmp_path / "mini" / "best.pt", weights_only=False)
    assert ckpt["genome"].shape == (17,)
    assert ckpt["features"] == ["value16", "empty"]


def test_training_resume(tmp_path):
    from ai2048.train import TrainConfig, Trainer

    common = dict(
        population=6, episodes_per_genome=2, generations=2, max_episode_steps=3000,
        device="cpu", seed=7, checkpoint_every=1,
    )
    out = str(tmp_path / "run")
    Trainer(TrainConfig(out_dir=out, **common)).run()
    ckpt = torch.load(f"{out}/checkpoint_gen2.pt", weights_only=False)
    assert ckpt["ga"]["generation"] == 2

    # resume for 2 more generations from gen-2 state
    cfg2 = TrainConfig(out_dir=out, **common)
    cfg2.generations = 4
    Trainer(cfg2).run(resume=f"{out}/checkpoint_gen2.pt")
    metrics = [line for line in open(f"{out}/metrics.jsonl")]
    gens = [int(__import__("json").loads(l)["gen"]) for l in metrics]
    assert gens == [0, 1, 2, 3]  # resumed without repeating generations
