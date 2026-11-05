"""When does an agent stop watching? Leaving point of PPO vs GreedyKnown (P0 follow-up).

For every back from WATCH it records the known rate of the watched item (A exp(-lambda w)),
the best known rate of any other understood item, the bout length and the watch time w.
GreedyKnown(rho) leaves exactly when the rate falls below rho. It also plays a hybrid:
PPO decides in the list, the greedy rule decides while watching (stay while rate >= rho).

Run (from the repository root, PYTHONPATH pointing at src):
    python scripts/experiments/watch_leaving.py --runs runs/topk-k5-lr1m --episodes 100
Writes <out>/leaves.csv (one row per back), <out>/episodes.csv and prints a summary.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scrolling_agent import ScrollingEnv
from scrolling_agent.env import WATCH
from scrolling_agent.evaluation import run_episode
from scrolling_agent.policies import GreedyKnown, PPOPolicy, known_rates

EVAL_SEED = 10_000


class LeaveRecorder:
    """Wraps a policy; records the situation at every requested back from WATCH."""

    def __init__(self, policy):
        self.policy, self.rows = policy, []
        self._bout_start = None

    def __call__(self, env: ScrollingEnv) -> int:
        action = int(self.policy(env))
        if env.mode == WATCH:
            if self._bout_start is None:
                self._bout_start = env.step_count
            if action == env.back_action:
                rates = known_rates(env)
                i = env.current
                others = np.delete(rates, i)
                self.rows.append({
                    "step": env.step_count,
                    "item": i,
                    "rate": float(rates[i]),
                    "best_other": float(others.max()) if np.isfinite(others).any() else float("nan"),
                    "w": float(env.w[i]),
                    "bout_s": (env.step_count - self._bout_start) * env.cfg.time.dt,
                })
                self._bout_start = None
        else:
            self._bout_start = None
        return action


class GreedyWatch:
    """Hybrid: the inner policy decides in the list, the greedy rho rule decides while watching."""

    def __init__(self, inner, rho: float):
        self.inner, self.rho = inner, rho

    def __call__(self, env: ScrollingEnv) -> int:
        if env.mode == WATCH:
            return env.back_action if known_rates(env)[env.current] < self.rho else env.stay_action
        return self.inner(env)


def make_policy(spec: tuple):
    kind, arg, rho = spec
    if kind == "greedy":
        return GreedyKnown(rho)
    ppo = PPOPolicy.load(arg, "best", deterministic=(kind == "ppo-det"))
    return GreedyWatch(ppo, rho) if kind == "hybrid-sto" else ppo


def play(job: tuple) -> tuple[list[dict], list[dict]]:
    label, spec, seeds = job
    leaves, episodes = [], []
    for s in seeds:
        rec = LeaveRecorder(make_policy(spec))
        m, _ = run_episode(ScrollingEnv(_config()), rec, seed=s, reference=True)
        episodes.append({"policy": label, "seed": s, **{k: m[k] for k in (
            "pct_of_ref", "return", "n_clicks", "n_understood", "time_watch", "time_transition")}})
        leaves += [{"policy": label, "seed": s, **r} for r in rec.rows]
    return leaves, episodes


def _config():
    from scrolling_agent.cli import apply_overrides
    from scrolling_agent.config import Config

    return apply_overrides(Config(), ["obs.memory=topk"])


def summarize(leaves: list[dict], episodes: list[dict], rho: float) -> str:
    lines = [f"| policy | % of ref | watch s | backs / session | median rate at back | backs with rate >= {rho} "
             "| ... of which a better item was known | median bout s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in dict.fromkeys(e["policy"] for e in episodes):
        ep = [e for e in episodes if e["policy"] == label]
        lv = [r for r in leaves if r["policy"] == label]
        rate = np.array([r["rate"] for r in lv])
        early = [r for r in lv if r["rate"] >= rho]
        better = [r for r in early if r["best_other"] > r["rate"]]
        lines.append(
            f"| {label} | {np.mean([e['pct_of_ref'] for e in ep]):.1f} ± {np.std([e['pct_of_ref'] for e in ep]):.1f}"
            f" | {np.mean([e['time_watch'] for e in ep]):.1f} | {len(lv) / len(ep):.1f}"
            f" | {np.median(rate):.2f} | {100 * len(early) / max(len(lv), 1):.0f}%"
            f" | {100 * len(better) / max(len(early), 1):.0f}%"
            f" | {np.median([r['bout_s'] for r in lv]):.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="condition folder with s*/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--out", type=Path, default=None, help="default: <runs>/watch_leaving")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    seeds = list(range(EVAL_SEED, EVAL_SEED + args.episodes))
    chunks = [seeds[i:i + 10] for i in range(0, len(seeds), 10)]
    specs = [("greedy", ("greedy", "", args.rho))]
    for run in sorted(args.runs.glob("s*/best_model.zip")):
        for kind in ("ppo-det", "ppo-sto", "hybrid-sto"):
            specs.append((f"{kind} [{run.parent.name}]", (kind, str(run.parent), args.rho)))
    jobs = [(label, spec, c) for label, spec in specs for c in chunks]
    leaves, episodes = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for lv, ep in pool.map(play, jobs):
            leaves += lv
            episodes += ep
    for e in episodes:  # pooled rows per policy kind
        e["kind"] = e["policy"].split(" [")[0]
    out = args.out or args.runs / "watch_leaving"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("leaves.csv", leaves), ("episodes.csv", episodes)):
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    pooled_ep = [{**e, "policy": e["kind"]} for e in episodes]
    pooled_lv = [{**r, "policy": r["policy"].split(" [")[0]} for r in leaves]
    text = summarize(pooled_lv, pooled_ep, args.rho) + "\n" + summarize(leaves, episodes, args.rho)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
