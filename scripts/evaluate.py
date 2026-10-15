"""Compare PPO runs with the baseline policies on unseen lists (steps 4–5).

Run (PowerShell, venv active):
    python scripts\\evaluate.py                         # baselines only
    python scripts\\evaluate.py --runs runs\\pilot       # + one or more PPO runs

Every policy plays the same lists (seeds 10000, 10001, ...). The baselines'
watch threshold rho is tuned on separate seeds (20000, ...) unless --rho is given.
Per-episode results go to runs/eval/<timestamp>.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from scrolling_agent import ScrollingEnv, TrainConfig
from scrolling_agent.cli import add_feed_args, config_from_args
from scrolling_agent.evaluation import evaluate, summarize
from scrolling_agent.policies import GreedyKnown, PPOPolicy, RandomPolicy, ScrollFirst, reference_return

TUNE_SEED = 20_000
RHO_GRID = np.round(np.arange(0.1, 1.51, 0.1), 2)
COLUMNS = ["return", "watch_reward", "n_understood", "n_clicks", "n_revisits", "time_watch", "max_item_seen"]


def tune_rho(env, policy_cls, n_episodes: int) -> float:
    seeds = range(TUNE_SEED, TUNE_SEED + n_episodes)
    scores = {rho: summarize(evaluate(env, policy_cls(rho), seeds))["return"][0] for rho in RHO_GRID}
    return float(max(scores, key=scores.get))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="PPO run folders (runs/<name>)")
    parser.add_argument("--which", default="best", choices=["best", "final"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rho", type=float, default=None, help="fix rho instead of tuning")
    parser.add_argument("--tune-episodes", type=int, default=20)
    add_feed_args(parser)
    args = parser.parse_args()

    cfg = config_from_args(args)
    env = ScrollingEnv(cfg)
    start = TrainConfig().eval_seed
    seeds = list(range(start, start + args.episodes))

    policies = {"random": RandomPolicy(0)}
    for label, cls in (("scrollfirst", ScrollFirst), ("greedy", GreedyKnown)):
        rho = args.rho if args.rho is not None else tune_rho(env, cls, args.tune_episodes)
        policies[f"{label} (rho={rho:g})"] = cls(rho)
    for run in args.runs:
        policies[f"ppo:{Path(run).name}"] = PPOPolicy.load(run, args.which)

    reference = []
    for s in seeds:
        env.reset(seed=s)
        reference.append(reference_return(env.feed, cfg))
    ref_mean = float(np.mean(reference))

    rows = []
    print(f"{len(seeds)} unseen lists (seeds {seeds[0]}..{seeds[-1]}), "
          f"reference (all known, no travel) = {ref_mean:.2f}\n")
    header = f"{'policy':<28}" + "".join(f"{c:>16}" for c in COLUMNS) + f"{'% of ref':>10}"
    print(header)
    print("-" * len(header))
    for name, policy in policies.items():
        t0 = time.perf_counter()
        results = evaluate(env, policy, seeds)
        stats = summarize(results)
        cells = "".join(f"{stats[c][0]:>9.2f} +-{stats[c][1]:>5.2f}" for c in COLUMNS)
        print(f"{name:<28}{cells}{100 * stats['return'][0] / ref_mean:>9.1f}%"
              f"   ({time.perf_counter() - t0:.0f}s)")
        rows += [dict(policy=name, **r) for r in results]

    out = Path("runs") / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-episode results: {out}")


if __name__ == "__main__":
    main()
