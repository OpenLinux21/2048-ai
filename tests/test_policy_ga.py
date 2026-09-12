import torch

from ai2048.ga import GAConfig, GeneticAlgorithm
from ai2048.policy import LinearPolicy, extract_features, feature_size


def test_feature_sizes():
    assert feature_size(["value16", "empty"]) == 17
    assert feature_size(list(extract_features.__globals__["ALL_FEATURES"])) == 24


def test_extract_features_shape_and_order():
    torch.manual_seed(0)
    board = torch.randint(0, 12, (5, 4, 4), dtype=torch.uint8)
    feats = ["value16", "empty", "max_exp", "corner_max", "monotonic4", "smoothness"]
    out = extract_features(board, feats)
    assert out.shape == (5, 24)
    # empty feature counts zeros
    board0 = torch.zeros(1, 4, 4, dtype=torch.uint8)
    out0 = extract_features(board0, feats)
    assert out0[0, 16] == 16  # all 16 cells empty
    assert out0[0, 0].sum() == 0  # value16 all zero
    assert out0[0, 17] == 0  # max_exp of empty board


def test_policy_masks_illegal_moves():
    torch.manual_seed(1)
    board = torch.zeros(1, 4, 4, dtype=torch.uint8)
    board[0, 0, 0] = 5  # single tile top-left: only RIGHT and DOWN legal
    policy = LinearPolicy(["value16", "empty"])
    weights = torch.randn(1, policy.param_size)
    scores, moved = policy.candidate_scores(board, weights)
    assert moved.tolist() == [[False, True, True, False]]  # UP, RIGHT, DOWN, LEFT
    assert torch.isinf(scores[0, 0]) and scores[0, 0] < 0
    action, _ = policy.act(board, weights)
    assert action.item() in (1, 2)


def test_policy_matches_per_env_weights():
    """One batch, P different genomes -> each slot scored by its own weights."""
    torch.manual_seed(2)
    n = 7
    board = torch.randint(0, 11, (n, 4, 4), dtype=torch.uint8)
    policy = LinearPolicy(["value16", "empty", "corner_max"])
    weights = torch.randn(n, policy.param_size)
    scores, _ = policy.candidate_scores(board, weights)
    for i in range(n):
        solo, _ = policy.candidate_scores(board[i : i + 1], weights[i : i + 1])
        assert torch.allclose(scores[i], solo[0])


def test_ga_reproducible():
    def run():
        cfg = GAConfig(population=16, seed=123)
        ga = GeneticAlgorithm(cfg, param_size=17, device="cpu")
        fits = []
        g = torch.Generator().manual_seed(5)
        for _ in range(4):
            fitness = torch.rand(16, generator=g)
            ga.step(fitness)
            fits.append(ga.fitness.clone())
        return fits, ga.population.clone()

    f1, p1 = run()
    f2, p2 = run()
    assert all(torch.equal(a, b) for a, b in zip(f1, f2))
    assert torch.equal(p1, p2)


def test_ga_elites_survive():
    cfg = GAConfig(population=20, elite_count=2, seed=0)
    ga = GeneticAlgorithm(cfg, 8, "cpu")
    pop_before = ga.population.clone()
    fitness = torch.arange(20, dtype=torch.float32)  # genome 19 is best
    ga.step(fitness)
    best_before = pop_before[19]
    # elite #1 (index 19 in the old pop) must be present unchanged in gen 1
    assert any(torch.equal(best_before, ga.population[i]) for i in range(2))


def test_ga_selection_pressure():
    """A clearly best genome should take over a tournament population."""
    cfg = GAConfig(population=30, elite_count=1, tournament_k=5, seed=3)
    ga = GeneticAlgorithm(cfg, 4, "cpu")
    good = torch.tensor([1.0, 1.0, 1.0, 1.0])
    for _ in range(15):
        fitness = 1.0 / (1.0 + (ga.population - good).norm(dim=1))
        ga.step(fitness)
    means = ga.population.mean(dim=1)
    assert (means > 0.7).float().mean() > 0.5  # most genomes converged near `good`


def test_ga_adaptive_sigma_and_blx_and_rank():
    cfg = GAConfig(
        population=12,
        adaptive_sigma=True,
        crossover="blx",
        rank_selection=True,
        seed=9,
    )
    ga = GeneticAlgorithm(cfg, 6, "cpu")
    assert ga.genome_size == 7
    assert torch.allclose(ga.population[:, -1], torch.full((12,), torch.log(torch.tensor(0.25))))
    fitness = torch.rand(12)
    ga.step(fitness)
    assert ga.generation == 1
    assert ga.population.shape == (12, 7)


def test_ga_checkpoint_roundtrip(tmp_path):
    cfg = GAConfig(population=10, seed=4)
    ga = GeneticAlgorithm(cfg, 5, "cpu")
    ga.step(torch.rand(10))
    state = ga.state_dict()
    ga2 = GeneticAlgorithm(GAConfig(population=10, seed=999), 5, "cpu")
    ga2.load_state_dict(state)
    assert ga2.generation == ga.generation
    assert torch.equal(ga2.population, ga.population)
    fitness = torch.rand(10)  # same fitness for both -> identical next gen incl. RNG
    ga.step(fitness)
    ga2.step(fitness)
    assert torch.equal(ga2.population, ga.population)
