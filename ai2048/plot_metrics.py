"""Plot training curves from a run's metrics.jsonl into <out_dir>/curves.png.

  .venv/bin/python -m ai2048.plot_metrics runs/gpu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/gpu")
    rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    gens = [r["gen"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"2048 GA training - {run_dir} ({len(rows)} generations)")

    ax = axes[0][0]
    ax.plot(gens, [r["fitness_best"] for r in rows], label="best")
    ax.plot(gens, [r["fitness_mean"] for r in rows], label="mean")
    ax.set_title("fitness (score + maxtile bonus)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(gens, [r["score_mean"] for r in rows], label="mean score")
    ax.plot(gens, [r["score_best"] for r in rows], label="best episode score")
    ax.set_title("score per episode"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.semilogy(gens, [r["maxtile_mean"] for r in rows], label="mean max tile")
    ax.semilogy(gens, [r["maxtile_max"] for r in rows], label="max max tile")
    ax.set_title("max tile"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(gens, [r["n_2048_episodes"] for r in rows], label="episodes reaching 2048")
    ax.set_title(f"2048+ episodes per generation (of {rows[0].get('n_episodes', 'P*K')})")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
