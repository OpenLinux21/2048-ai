"""Genetic algorithm over flat parameter genomes, fully tensorized.

All randomness comes from one torch.Generator so runs are reproducible on CPU
and CUDA. The genome is `param_size` policy weights plus (optionally) one
self-adaptive mutation-sigma gene appended at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class GAConfig:
    population: int = 50
    elite_count: int = 2                # genomes copied unchanged every gen
    tournament_k: int = 4
    crossover: str = "uniform"          # "uniform" | "blx"
    blx_alpha: float = 0.5
    crossover_rate: float = 0.9         # per pair; else child copies parent A
    mutation_rate: float = 0.1          # per gene
    mutation_sigma: float = 0.25
    adaptive_sigma: bool = False        # append a sigma gene (log-space)
    sigma_gene_tau: float = 0.1         # mutation std for the sigma gene itself
    rank_selection: bool = False        # rank-proportional instead of tournament
    hof_size: int = 5
    init_range: float = 1.0             # U(-init_range, init_range) at gen 0
    seed: int = 0


class GeneticAlgorithm:
    def __init__(self, cfg: GAConfig, param_size: int, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.param_size = param_size
        self.genome_size = param_size + (1 if cfg.adaptive_sigma else 0)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(cfg.seed)
        self.generation = 0
        self.population = self._random_population()
        self.fitness = torch.zeros(cfg.population, dtype=torch.float32, device=self.device)
        self.hof: list[tuple[float, torch.Tensor]] = []  # (fitness, genome) best-first

    # -- setup ---------------------------------------------------------------

    def _random_population(self) -> torch.Tensor:
        cfg = self.cfg
        pop = torch.empty(
            (cfg.population, self.genome_size), dtype=torch.float32, device=self.device
        ).uniform_(-cfg.init_range, cfg.init_range, generator=self.generator)
        if cfg.adaptive_sigma:
            pop[:, -1] = math.log(cfg.mutation_sigma)
        return pop

    def weights(self) -> torch.Tensor:
        """Policy weight matrix [P, param_size] (strips the sigma gene)."""
        return self.population[:, : self.param_size]

    # -- operators ------------------------------------------------------------

    def _tournament_indices(self, n: int) -> torch.Tensor:
        """Winners of n tournaments -> parent indices [n]."""
        cfg = self.cfg
        cand = torch.randint(
            0, cfg.population, (n, cfg.tournament_k),
            device=self.device, generator=self.generator,
        )
        fits = self.fitness[cand]
        win = cand.gather(1, fits.argmax(dim=1, keepdim=True)).squeeze(1)
        return win

    def _rank_indices(self, n: int) -> torch.Tensor:
        """Rank-proportional parent sampling (best rank = highest fitness)."""
        order = self.fitness.argsort(descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(order), device=self.device)
        probs = (len(order) - ranks).to(torch.float64)  # P .. 1
        probs /= probs.sum()
        return torch.multinomial(probs, n, replacement=True, generator=self.generator)

    def _crossover(self, pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        n, g = pa.shape
        do = (
            torch.rand((n, 1), device=self.device, generator=self.generator)
            < cfg.crossover_rate
        )
        if cfg.crossover == "uniform":
            mask = torch.rand((n, g), device=self.device, generator=self.generator) < 0.5
            children = torch.where(mask, pa, pb)
        elif cfg.crossover == "blx":
            lo = torch.minimum(pa, pb) - cfg.blx_alpha * (pa - pb).abs()
            hi = torch.maximum(pa, pb) + cfg.blx_alpha * (pa - pb).abs()
            u = torch.rand((n, g), device=self.device, generator=self.generator)
            children = lo + (hi - lo) * u
        else:
            raise ValueError(f"unknown crossover: {cfg.crossover}")
        return torch.where(do, children, pa)

    def _mutate(self, children: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        n, g = children.shape
        gene_mask = (
            torch.rand((n, g), device=self.device, generator=self.generator)
            < cfg.mutation_rate
        )
        if cfg.adaptive_sigma:
            sigmas = torch.exp(children[:, -1]).unsqueeze(1)  # per-genome sigma
        else:
            sigmas = torch.full((n, 1), cfg.mutation_sigma, device=self.device)
        noise = torch.randn((n, g), device=self.device, generator=self.generator) * sigmas
        children = children + torch.where(gene_mask, noise, torch.zeros_like(noise))
        if cfg.adaptive_sigma:
            tau_mask = (
                torch.rand((n, 1), device=self.device, generator=self.generator)
                < cfg.mutation_rate
            )
            tau_noise = torch.randn((n, 1), device=self.device, generator=self.generator)
            children[:, -1] += torch.where(
                tau_mask, tau_noise * cfg.sigma_gene_tau, torch.zeros_like(tau_noise)
            ).squeeze(1)
        return children

    # -- evolution loop ---------------------------------------------------------

    def step(self, fitness: torch.Tensor) -> None:
        """Evaluate the current population, then breed the next generation."""
        cfg = self.cfg
        assert fitness.shape == (cfg.population,)
        self.fitness = fitness.to(self.device)

        order = self.fitness.argsort(descending=True)
        for idx in order[: cfg.elite_count].tolist():
            self._hof_push(float(self.fitness[idx]), self.population[idx])

        n_children = cfg.population - cfg.elite_count
        pick = self._rank_indices if cfg.rank_selection else self._tournament_indices
        pa = self.population[pick(n_children)]
        pb = self.population[pick(n_children)]
        children = self._mutate(self._crossover(pa, pb))
        self.population = torch.cat([self.population[order[: cfg.elite_count]], children])
        self.generation += 1

    def _hof_push(self, fitness: float, genome: torch.Tensor) -> None:
        if any(g is genome for _, g in self.hof):
            return
        self.hof.append((fitness, genome.detach().clone()))
        self.hof.sort(key=lambda t: t[0], reverse=True)
        del self.hof[self.cfg.hof_size:]

    # -- persistence ---------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "cfg": vars(self.cfg),
            "generation": self.generation,
            "population": self.population.detach().cpu(),
            "fitness": self.fitness.detach().cpu(),
            "hof": [(f, g.detach().cpu()) for f, g in self.hof],
            "generator_state": self.generator.get_state().detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.cfg.__dict__.update(state["cfg"])
        self.generation = state["generation"]
        self.population = state["population"].to(self.device)
        self.fitness = state["fitness"].to(self.device)
        self.hof = [(f, g.to(self.device)) for f, g in state["hof"]]
        self.generator.set_state(state["generator_state"].to(self.device))
