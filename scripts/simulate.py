"""Compare model conditions with the baseline policies (and optionally PPO runs).

Run (PowerShell, venv active):
    python scripts\\simulate.py --cond base --cond crowded screen.cards_per_screen=5
    python scripts\\simulate.py --cond base --cond kappa08 feed.kappa=0.8 --tune --episodes 100

Each --cond is a name followed by zero or more section.field=value overrides
(applied on top of the shared --set / feed flags). Every condition plays the same
lists and DDM seeds (10000+), so differences come from the model, not the sample.

Policies (--policies): random, scrollfirst, greedy, greedy-dwell (greedy that
stops up to --dwell s per unread visible item), ppo:<run folder>.
By default all baselines use one fixed rho (--rho): the policy is held fixed and
only the model changes (draft 09 B). With --tune, rho is re-tuned per condition
on seeds 20000+ (each condition gets its own best baseline, draft 09 A).

Writes runs/sim/<timestamp>.csv: one row per condition x policy x episode.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from scrolling_agent import ScrollingEnv, TrainConfig
from scrolling_agent.cli import add_config_args, apply_overrides, config_from_args
from scrolling_agent.evaluation import evaluate, summarize
from scrolling_agent.policies import GreedyKnown, PPOPolicy, RandomPolicy, ScrollFirst

TUNE_SEED = 20_000
RHO_GRID = np.round(np.arange(0.2, 1.21, 0.1), 2)
BASE_COLUMNS = ["return", "n_understood", "n_clicks", "n_revisits", "n_short_bouts",
                "time_stay_list", "time_watch", "max_item_seen"]
SKIP_COLUMNS = {"seed", "invalid_actions", "watch_reward", "time_list", "time_transition", "mean_bout",
                "n_watched_items"}


def baseline(name: str, rho: float, dwell: float):
    if name == "random":
        return RandomPolicy(0)
    if name == "scrollfirst":
        return ScrollFirst(rho)
    if name == "greedy":
        return GreedyKnown(rho)
    if name == "greedy-dwell":
        return GreedyKnown(rho, dwell=dwell)
    raise SystemExit(f"unknown policy {name}")


def tuned_rho(env, name: str, dwell: float, n_episodes: int) -> float:
    seeds = range(TUNE_SEED, TUNE_SEED + n_episodes)
    score = {rho: summarize(evaluate(env, baseline(name, rho, dwell), seeds))["return"][0] for rho in RHO_GRID}
    return float(max(score, key=score.get))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cond", nargs="+", action="append", metavar=("NAME", "SECTION.FIELD=VALUE"),
                        help="a condition: name and overrides (repeat for several)")
    parser.add_argument("--policies", nargs="+", default=["scrollfirst", "greedy", "greedy-dwell"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--dwell", type=float, default=1.5)
    parser.add_argument("--tune", action="store_true", help="re-tune rho per condition and policy")
    parser.add_argument("--tune-episodes", type=int, default=10)
    add_config_args(parser)
    args = parser.parse_args()

    conditions = args.cond or [["base"]]
    start = TrainConfig().eval_seed
    seeds = list(range(start, start + args.episodes))
    print(f"{len(conditions)} condition(s) x {len(args.policies)} policies x {len(seeds)} lists "
          f"(seeds {seeds[0]}..{seeds[-1]}), rho {'tuned' if args.tune else args.rho}, dwell {args.dwell}\n")

    rows, table = [], []
    for name, *overrides in conditions:
        env = ScrollingEnv(apply_overrides(config_from_args(args), overrides))
        for pname in args.policies:
            t0 = time.perf_counter()
            if pname.startswith("ppo:"):
                label, policy = pname, PPOPolicy.load(pname[4:])
            else:
                rho = tuned_rho(env, pname, args.dwell, args.tune_episodes) if args.tune else args.rho
                label, policy = f"{pname} (rho={rho:g})", baseline(pname, rho, args.dwell)
            results = evaluate(env, policy, seeds)
            rows += [dict(condition=name, policy=label, **r) for r in results]
            table.append((name, label, summarize(results), time.perf_counter() - t0))

    extra = [k for k in table[0][2] if k not in BASE_COLUMNS and k not in SKIP_COLUMNS]
    columns = BASE_COLUMNS + extra
    width = {c: max(len(c) + 2, 10) for c in columns}
    width["return"] = 16
    header = f"{'condition':<14}{'policy':<24}" + "".join(f"{c:>{width[c]}}" for c in columns)
    print(header)
    print("-" * len(header))
    for name, label, stats, secs in table:
        ret = f"{stats['return'][0]:>9.2f} +-{stats['return'][1]:>4.1f}"
        rest = "".join(f"{stats[c][0]:>{width[c]}.2f}" if c in stats else f"{'-':>{width[c]}}"
                       for c in columns[1:])
        print(f"{name:<14}{label:<24}{ret}{rest}   ({secs:.0f}s)")

    out = Path("runs") / "sim" / f"{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-episode results: {out}")


if __name__ == "__main__":
    main()
