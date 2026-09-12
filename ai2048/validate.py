"""Differential validation + throughput benchmarks.

Two jobs:
  1. env equivalence - the C shim (unmodified games/2048.c logic) and the
     torch VecEnv must produce bit-identical trajectories for identical
     episode seeds and identical action scripts.
  2. throughput - steps/s of each backend so batch sizes can be picked.

Usage:
  .venv/bin/python -m ai2048.validate                 # 200 episodes, CPU+GPU
  .venv/bin/python -m ai2048.validate --episodes 1000 --no-gpu
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ai2048 import ShimGame2048
from ai2048.vec_env import VecEnv2048


def run_shim_episode(seed: int, actions: list[int]) -> tuple[list, bool, int]:
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
    return traj, ended, game.score


def run_vec_episode(seed: int, actions: list[int], device: str) -> tuple[list, bool, int]:
    env = VecEnv2048(1, device=device, seeds=seed)
    traj = [(env.boards[0].cpu().clone(), int(env.scores[0]))]
    ended = False
    for a in actions:
        _, e = env.step(torch.tensor([a], dtype=torch.int64))
        ended = bool(e[0])
        traj.append((env.boards[0].cpu().clone(), int(env.scores[0])))
        if ended:
            break
    return traj, ended, int(env.scores[0])


def validate_equivalence(n_episodes: int, max_steps: int, devices: list[str]) -> bool:
    print(f"== equivalence: C shim vs VecEnv over {n_episodes} episodes "
          f"(<= {max_steps} steps each), devices={devices}")
    all_ok = True
    for device in devices:
        mismatches = 0
        total_steps = 0
        for ep in range(n_episodes):
            seed = 31_337 + ep
            rng = np.random.default_rng(555_000 + ep)
            actions = rng.integers(0, 4, max_steps).tolist()
            c_traj, c_end, _ = run_shim_episode(seed, actions)
            v_traj, v_end, _ = run_vec_episode(seed, actions, device)
            total_steps += len(c_traj) - 1
            same = (
                c_end == v_end
                and len(c_traj) == len(v_traj)
                and all(
                    np.array_equal(cb, vb.numpy()) and cs == vs
                    for (cb, cs), (vb, vs) in zip(c_traj, v_traj)
                )
            )
            if not same:
                mismatches += 1
                print(f"  MISMATCH device={device} episode={ep} seed={seed}")
        ok = mismatches == 0
        all_ok &= ok
        status = "OK" if ok else "FAILED"
        print(f"  [{device}] {n_episodes} episodes, {total_steps} steps -> {status} "
              f"({mismatches} mismatches)")
    return all_ok


def bench_shim(seconds: float = 2.0) -> float:
    game = ShimGame2048()
    game.reset(1)
    rng = np.random.default_rng(0)
    acts = iter(rng.integers(0, 4, 10_000_000).tolist())
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(2000):
            game.step(next(acts))
            n += 1
    return n / (time.perf_counter() - t0)


def bench_vec(device: str, batch: int, seconds: float = 2.0) -> float:
    env = VecEnv2048(batch, device=device)
    acts = torch.randint(0, 4, (batch,), device=device)
    for _ in range(10):
        env.step(acts)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(50):
            env.step(acts)
        n += 50 * batch
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description="differential validation + benchmarks")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--no-gpu", action="store_true", help="skip CUDA equivalence + bench")
    p.add_argument("--skip-equiv", action="store_true")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--batch", type=int, default=16384)
    args = p.parse_args()

    devices = ["cpu"] if args.no_gpu else ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    ok = True
    if not args.skip_equiv:
        ok = validate_equivalence(args.episodes, args.max_steps, devices)
    if not args.skip_bench:
        print("== throughput")
        print(f"  C shim (single thread): {bench_shim() / 1e6:.2f}M steps/s")
        print(f"  VecEnv cpu  B={args.batch}: {bench_vec('cpu', args.batch) / 1e6:.2f}M steps/s")
        if not args.no_gpu and torch.cuda.is_available():
            for b in (4096, args.batch, 65536):
                if b <= args.batch or args.batch == b:
                    print(f"  VecEnv cuda B={b}: {bench_vec('cuda', b) / 1e6:.2f}M steps/s")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
