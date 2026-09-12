"""Fixed-seed benchmark of a trained checkpoint.

Plays N batched episodes with seeds derived from a fixed evaluation seed
(disjoint from the training CRN stream) and reports score statistics and the
max-tile distribution.

  .venv/bin/python -m ai2048.evaluate --checkpoint runs/gpu/best.pt --games 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ai2048.policy import make_policy
from ai2048.train import rollout


def load_checkpoint(path: str, device: str) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    features = ckpt.get("features", ["value16", "empty"])
    policy = make_policy(ckpt.get("policy_type", "linear"), features,
                         ckpt.get("mlp_hidden", 32))
    genome = ckpt["genome"].to(device).float()
    return policy, genome, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="evaluate a trained 2048 policy")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-seed", type=int, default=20260911)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--lookahead", type=int, default=1, choices=(1, 2))
    p.add_argument("--samples", type=int, default=2, help="chance samples per 2-ply node")
    p.add_argument("--json", default=None, help="also write results to this JSON file")
    args = p.parse_args()

    policy, genome, ckpt = load_checkpoint(args.checkpoint, args.device)
    seeds = args.eval_seed + torch.arange(args.games, dtype=torch.int64) * 7919
    weights = genome.unsqueeze(0).expand(args.games, -1).contiguous()

    t0 = time.perf_counter()
    if args.lookahead == 1:
        scores, tiles, steps = rollout(weights, seeds, policy, args.device, args.max_steps)
    else:
        from ai2048.lookahead import expectimax2_actions
        from ai2048.vec_env import VecEnv2048

        env = VecEnv2048(args.games, device=args.device, seeds=seeds)
        gen = torch.Generator(device=args.device).manual_seed(args.eval_seed ^ 0xBEEF)
        for _ in range(args.max_steps):
            if not bool(env.alive.any()):
                break
            actions, _ = expectimax2_actions(policy, env.boards, weights, args.samples, gen)
            env.step(actions)
        scores, tiles, steps = env.scores, env.max_tile, env.steps_taken
    elapsed = time.perf_counter() - t0

    f = scores.to(torch.float32)
    tiers = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    dist = {str(t): int((tiles >= t).sum()) for t in tiers}
    results = {
        "checkpoint": args.checkpoint,
        "generation": ckpt.get("generation"),
        "games": args.games,
        "lookahead": args.lookahead,
        "score_mean": round(float(f.mean()), 1),
        "score_median": round(float(f.median()), 1),
        "score_min": int(scores.min()),
        "score_max": int(scores.max()),
        "score_std": round(float(f.std()) if args.games > 1 else 0.0, 1),
        "max_tile_reached": int(tiles.max()),
        "reach_1024": dist["1024"],
        "reach_2048": dist["2048"],
        "reach_4096": dist["4096"],
        "max_tile_distribution_ge": dist,
        "wall_time_s": round(elapsed, 1),
        "steps_per_s": int(int(steps.sum()) / max(elapsed, 1e-9)),
    }
    print(json.dumps(results, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
