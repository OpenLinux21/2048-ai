"""Differential equivalence: C shim vs torch VecEnv, single vs batched, CPU vs
CUDA - identical episode seeds must produce identical trajectories.
"""

import numpy as np
import pytest
import torch

from ai2048 import ShimGame2048
from ai2048.vec_env import VecEnv2048

N_EPISODES = 20
MAX_STEPS = 3000


def _scripted_actions(ep: int) -> list[int]:
    rng = np.random.default_rng(10_000 + ep)
    return rng.integers(0, 4, MAX_STEPS).tolist()


def _run_shim(seed: int, actions: list[int]):
    game = ShimGame2048()
    game.reset(seed)
    traj = [(game.board().copy(), game.score)]
    ended = False
    for a in actions:
        _, e = game.step(a)
        ended = e
        traj.append((game.board().copy(), game.score))
        if e:
            break
    return traj, ended


def _run_vec(seed: int, actions: list[int], device: str = "cpu"):
    env = VecEnv2048(1, device=device, seeds=seed)
    traj = [(env.boards[0].cpu().clone(), int(env.scores[0]))]
    ended = False
    for a in actions:
        _, e = env.step(torch.tensor([a], dtype=torch.int64))
        ended = bool(e[0])
        traj.append((env.boards[0].cpu().clone(), int(env.scores[0])))
        if ended:
            break
    return traj, ended


@pytest.mark.parametrize("ep", range(N_EPISODES))
def test_shim_matches_vec_cpu(ep):
    seed = 500 + ep
    actions = _scripted_actions(ep)
    c_traj, c_end = _run_shim(seed, actions)
    v_traj, v_end = _run_vec(seed, actions)
    assert c_end == v_end
    assert len(c_traj) == len(v_traj)
    for (cb, cs), (vb, vs) in zip(c_traj, v_traj):
        assert np.array_equal(cb, vb.numpy())
        assert cs == vs


def test_initial_boards_match_many_seeds():
    for seed in range(1, 120):
        cb = ShimGame2048().reset(seed)
        vb = VecEnv2048(1, "cpu", seeds=seed).boards[0].numpy()
        assert np.array_equal(cb, vb), f"seed {seed}"


def test_batched_matches_single():
    n = 32
    seeds = list(range(9000, 9000 + n))
    batch = VecEnv2048(n, "cpu", seeds=seeds)
    singles = [VecEnv2048(1, "cpu", seeds=s) for s in seeds]
    rng = np.random.default_rng(7)
    for _ in range(2500):
        acts = rng.integers(0, 4, n)
        batch.step(torch.from_numpy(acts).to(torch.int64))
        for i, s in enumerate(singles):
            s.step(int(acts[i]))
        assert torch.equal(batch.boards, torch.stack([s.boards[0] for s in singles]))
        assert torch.equal(batch.scores, torch.stack([s.scores[0] for s in singles]))
        if not bool(batch.alive.any()):
            break


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    seeds = list(range(400, 464))
    cpu = VecEnv2048(64, "cpu", seeds=seeds)
    gpu = VecEnv2048(64, "cuda", seeds=seeds)
    g = torch.Generator().manual_seed(3)
    for _ in range(2500):
        acts = torch.randint(0, 4, (64,), generator=g, dtype=torch.int64)
        cpu.step(acts)
        gpu.step(acts)
        assert torch.equal(gpu.boards.cpu(), cpu.boards)
        assert torch.equal(gpu.scores.cpu(), cpu.scores)
        if not bool(cpu.alive.any()):
            break
