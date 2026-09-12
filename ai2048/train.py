"""GA training loop: evaluate the whole population as one batched rollout.

Every generation plays P x K episodes simultaneously in a single VecEnv2048 on
the configured device (CPU or CUDA). All individuals of a generation face the
same episode seed set (common random numbers) so fitness differences reflect
the genomes, not spawn luck.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

from ai2048.ga import GAConfig, GeneticAlgorithm
from ai2048.policy import BaseBoardPolicy, feature_size, make_policy
from ai2048.vec_env import VecEnv2048


@dataclass
class FitnessConfig:
    mode: str = "score"            # "score" | "score_maxtile"
    maxtile_tier_bonus: float = 100.0  # fitness bonus per tile tier above 1024
    log_score: bool = False        # use log2(1+score) instead of raw score


@dataclass
class TrainConfig:
    population: int = 50
    episodes_per_genome: int = 4
    generations: int = 200
    max_episode_steps: int = 20000
    device: str = "cpu"
    seed: int = 0                  # master seed (GA + episode seeds derive from it)
    seed_stride: int = 7919        # episode-k seed stride; keep CRN across genomes
    features: list[str] = field(default_factory=lambda: ["value16", "empty"])
    policy_type: str = "linear"    # "linear" | "mlp"
    mlp_hidden: int = 32
    compile: bool = False          # torch.compile the policy scorer (optional)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    out_dir: str = "runs/train"
    checkpoint_every: int = 10
    log_every: int = 1

    def __post_init__(self):
        self.ga.population = self.population
        self.ga.seed = self.seed
        if len(self.features) == 0:
            raise ValueError("at least one feature required")


def rollout(
    weights: torch.Tensor,
    seeds: torch.Tensor,
    policy: BaseBoardPolicy,
    device: torch.device,
    max_steps: int,
    sync_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Play one episode per env slot to completion (or step cap).

    weights [B, F], seeds int64 [B] -> (scores int64[B], max_tiles int64[B],
    steps int64[B]). The all-dead check syncs only every `sync_chunk` steps so
    the GPU pipeline stays busy; stepping dead games is a harmless no-op.
    """
    env = VecEnv2048(weights.shape[0], device=device, seeds=seeds)
    for done in range(0, max_steps, sync_chunk):
        if not bool(env.alive.any()):
            break
        for _ in range(min(sync_chunk, max_steps - done)):
            actions, _ = policy.act(env.boards, weights)
            env.step(actions)
    return env.scores, env.max_tile, env.steps_taken


def episode_seeds(base_seed: int, generation: int, n_episodes: int, stride: int = 7919) -> torch.Tensor:
    """Common-random-number seed set: identical for every individual of a gen."""
    base = (base_seed * 1_000_003 + generation * 65_537) & 0x7FFFFFFF
    return base + torch.arange(n_episodes, dtype=torch.int64) * stride


def fitness_from_episodes(
    cfg: TrainConfig, scores: torch.Tensor, max_tiles: torch.Tensor
) -> torch.Tensor:
    """scores/max_tiles [P, K] -> fitness [P]."""
    if cfg.fitness.log_score:
        per_ep = torch.log2(scores.to(torch.float32) + 1)
    else:
        per_ep = scores.to(torch.float32)
    fit = per_ep.mean(dim=1)
    if cfg.fitness.mode == "score_maxtile":
        # one bonus unit per tile tier above 1024 (2^10), e.g. 2048 -> +1 tier
        tiers = (torch.log2(max_tiles.to(torch.float32).clamp(min=1)) - 10).clamp(min=0)
        fit = fit + tiers.mean(dim=1) * cfg.fitness.maxtile_tier_bonus
    return fit


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self.policy = make_policy(cfg.policy_type, cfg.features, cfg.mlp_hidden)
        if cfg.compile:
            try:
                self.policy.evaluate = torch.compile(self.policy.evaluate, dynamic=True)
                print("torch.compile enabled for policy.evaluate")
            except Exception as exc:  # pragma: no cover - depends on host toolchain
                print(f"torch.compile unavailable ({exc}); running eager")
        self.ga = GeneticAlgorithm(cfg.ga, self.policy.param_size, cfg.device)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"

    def evaluate(self, generation: int) -> dict:
        cfg = self.cfg
        k = cfg.episodes_per_genome
        p = cfg.population
        # genome-major layout: slot j plays genome j//k with episode seed j%k,
        # so scores.view(p, k) rows are exactly one genome's k episodes.
        seeds = episode_seeds(cfg.seed, generation, k, cfg.seed_stride).repeat(p)
        weights = self.ga.weights().repeat_interleave(k, dim=0)
        t0 = time.perf_counter()
        scores, max_tiles, steps = rollout(
            weights, seeds, self.policy, self.device, cfg.max_episode_steps
        )
        elapsed = time.perf_counter() - t0
        fitness = fitness_from_episodes(cfg, scores.view(p, k), max_tiles.view(p, k))
        stats = {
            "gen": generation,
            "elapsed_s": round(elapsed, 3),
            "steps_total": int(steps.sum()),
            "steps_per_s": int(steps.sum() / max(elapsed, 1e-9)),
            "fitness_best": float(fitness.max()),
            "fitness_mean": float(fitness.mean()),
            "score_best": int(scores.view(p, k).max()),
            "score_mean": float(scores.view(p, k).to(torch.float32).mean()),
            "maxtile_max": int(max_tiles.max()),
            "maxtile_mean": float(max_tiles.to(torch.float32).mean()),
            "n_2048_episodes": int((max_tiles >= 2048).sum()),
        }
        return stats | {"fitness_vector": fitness}

    def best_genome(self) -> tuple[int, torch.Tensor]:
        idx = int(self.ga.fitness.argmax())
        return idx, self.ga.population[idx]

    def save_checkpoint(self, tag: str) -> Path:
        path = self.out_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "ga": self.ga.state_dict(),
                "cfg": asdict(self.cfg),
                "features": list(self.policy.features),
            },
            path,
        )
        best = self.best_genome()
        torch.save(
            {
                "genome": best[1].detach().cpu(),
                "fitness": float(self.ga.fitness[best[0]]),
                "generation": self.ga.generation,
                "features": list(self.policy.features) if hasattr(self.policy, "features") else [],
                "policy_type": self.cfg.policy_type,
                "mlp_hidden": self.cfg.mlp_hidden,
                "ga_cfg": asdict(self.cfg.ga),
            },
            self.out_dir / "best.pt",
        )
        return path

    def log(self, stats: dict) -> None:
        row = {k: v for k, v in stats.items() if k != "fitness_vector"}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def run(self, resume: Path | None = None) -> Path:
        cfg = self.cfg
        start_gen = 0
        if resume:
            state = torch.load(resume, map_location=self.device, weights_only=False)
            self.ga.load_state_dict(state["ga"])
            start_gen = self.ga.generation
            print(f"resumed from {resume} at generation {start_gen}")
        torch.manual_seed(cfg.seed)

        last = None
        for gen in range(start_gen, cfg.generations):
            stats = self.evaluate(gen)
            fitness = stats.pop("fitness_vector")
            self.ga.step(fitness)
            if gen % cfg.log_every == 0 or gen == cfg.generations - 1:
                self.log(stats)
                print(
                    f"gen {stats['gen']:4d}  best {stats['fitness_best']:9.1f}  "
                    f"mean {stats['fitness_mean']:9.1f}  maxtile {stats['maxtile_max']:5d}  "
                    f"{stats['steps_per_s'] / 1e6:.2f}M steps/s"
                )
            if (gen + 1) % cfg.checkpoint_every == 0 or gen == cfg.generations - 1:
                last = self.save_checkpoint(f"gen{gen + 1}")
        return last if last else self.save_checkpoint("final")


def load_config(path: str | Path, overrides: dict | None = None) -> TrainConfig:
    raw: dict = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw.update(overrides)
    ga_raw = raw.pop("ga", {}) or {}
    fit_raw = raw.pop("fitness", {}) or {}
    cfg = TrainConfig(
        ga=GAConfig(**ga_raw),
        fitness=FitnessConfig(**fit_raw),
        **raw,
    )
    cfg.ga.population = cfg.population
    cfg.ga.seed = cfg.seed
    return cfg


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="2048 GA trainer")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | cuda")
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None, help="episodes per genome")
    p.add_argument("--out", default=None, help="output dir override")
    p.add_argument("--resume", default=None, help="checkpoint path to resume")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    overrides = {
        k: v
        for k, v in {
            "generations": args.generations,
            "device": args.device,
            "population": args.population,
            "episodes_per_genome": args.episodes,
            "out_dir": args.out,
            "seed": args.seed,
        }.items()
        if v is not None
    }
    cfg = load_config(args.config, overrides)
    trainer = Trainer(cfg)
    print(
        f"training on {trainer.device}: P={cfg.population} K={cfg.episodes_per_genome} "
        f"features={cfg.features} ({feature_size(cfg.features)} params)"
    )
    trainer.run(Path(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
