"""Does the policy scroll slower after understanding a good item? (test A5)

Slow-down index = P(slow | last understood item has A >= threshold) - P(slow | A < threshold), over
0.1 s steps that really move down (blocked input at the bottom and steps before the first
understood item are left out). The context is recorded before the action.

Variants, because a held decision repeats its speed for up to 5 steps:
    all           every moving step (definition of the 11-09 report)
    decisions     only steps where the policy made a new decision
    fresh         only the first decision after a new item was understood
    within1s      moving steps within 1 s of the last understanding
    unambiguous   moving steps whose last understanding was a single item

Run (from the repository root, PYTHONPATH pointing at src):
    python scripts/experiments/speed_context.py --runs runs/speed2 runs/speed2-k0.9 --episodes 100
Writes <out>/seed_summary.csv and <out>/summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scrolling_agent import ScrollingEnv
from scrolling_agent.config import config_from_dict
from scrolling_agent.env import LIST
from scrolling_agent.evaluation import run_episode
from scrolling_agent.policies import PPOPolicy

EVAL_SEED = 10_000
VARIANTS = ["all", "decisions", "fresh", "within1s", "unambiguous"]


class SpeedRecorder:
    """Wraps a policy and counts slow / moving steps per context."""

    def __init__(self, policy, threshold: float = 0.6, lookahead: int = 3):
        self.policy, self.threshold, self.lookahead = policy, threshold, lookahead
        self.counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # key -> [slow, moving]
        self._known = None
        self._last_A = self._last_step = None
        self._ambiguous = self._fresh = False

    def _note_understood(self, env: ScrollingEnv) -> None:
        if self._known is None:
            self._known = env.understood.copy()
        new = np.flatnonzero(env.understood & ~self._known)
        if new.size:
            self._last_A = float(env.revealed_profile()[0][new.max()])  # lowest in the list if several
            self._last_step, self._ambiguous, self._fresh = env.step_count, new.size > 1, True
        self._known = env.understood.copy()

    def _is_new_decision(self) -> bool:
        hold = getattr(self.policy, "_hold", None)
        return hold is None or hold._held == 1

    def __call__(self, env: ScrollingEnv) -> int:
        self._note_understood(env)  # before the action
        action = int(self.policy(env))
        new_decision = self._is_new_decision()
        moving = (env.mode == LIST and action in env.down_actions and env.scroll < env.max_scroll
                  and bool(env.action_masks()[action]))
        if moving and self._last_A is not None:
            slow = int(action == env.down_actions[0])
            good = self._last_A >= self.threshold
            since = (env.step_count - self._last_step) * env.cfg.time.dt
            flags = {"all": True, "decisions": new_decision, "fresh": new_decision and self._fresh,
                     "within1s": since <= 1.0, "unambiguous": not self._ambiguous}
            for name, on in flags.items():
                if on:
                    cell = self.counts[(name, good)]
                    cell[0] += slow
                    cell[1] += 1
            slots = env.visible_slots()
            ahead = range(int(slots.max()) + 1, min(int(slots.max()) + 1 + self.lookahead, env.n_items))
            share = round(float(np.mean(env.feed.z[[*map(int, slots), *ahead]])), 1)  # analysis only, hidden from the agent
            cell = self.counts[("share", share)]
            cell[0] += slow
            cell[1] += 1
        if env.mode == LIST and new_decision:
            self._fresh = False
        return action


def index(counts: dict, name: str) -> float:
    """P(slow | good) - P(slow | bad); nan if a context never occurred."""
    good, bad = counts.get((name, True), [0, 0]), counts.get((name, False), [0, 0])
    if not good[1] or not bad[1]:
        return float("nan")
    return good[0] / good[1] - bad[0] / bad[1]


def play(job: tuple) -> dict:
    run, seeds = job
    stored = json.loads((Path(run) / "config.json").read_text())
    cfg = config_from_dict(stored["config"])
    total: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for s in seeds:
        rec = SpeedRecorder(PPOPolicy.load(run, "best", deterministic=False))
        run_episode(ScrollingEnv(cfg), rec, seed=s)
        for k, (slow, n) in rec.counts.items():
            total[k][0] += slow
            total[k][1] += n
    return dict(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="condition folders with s*/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("runs/speed_context"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    seeds = list(range(EVAL_SEED, EVAL_SEED + args.episodes))
    chunks = [seeds[i:i + 10] for i in range(0, len(seeds), 10)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for cond in args.runs:
            for run in sorted(cond.glob("s*/best_model.zip")):
                total: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
                for part in pool.map(play, [(str(run.parent), c) for c in chunks]):
                    for k, (slow, n) in part.items():
                        total[k][0] += slow
                        total[k][1] += n
                moving = sum(v[1] for k, v in total.items() if k[0] == "all")
                slow = sum(v[0] for k, v in total.items() if k[0] == "all")
                rows.append({"condition": cond.name, "train_seed": run.parent.name, "moving_steps": moving,
                             "fast_share": 1 - slow / moving if moving else float("nan"),
                             **{f"index_{v}": index(total, v) for v in VARIANTS},
                             "n_fresh": sum(v[1] for k, v in total.items() if k[0] == "fresh")})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "seed_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| condition | seed | fast share | " + " | ".join(VARIANTS) + " | fresh n |",
             "|---|---|---:|" + "---:|" * (len(VARIANTS) + 1)]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['train_seed']} | {r['fast_share']:.3f} | "
                     + " | ".join(f"{r['index_' + v]:.3f}" for v in VARIANTS) + f" | {r['n_fresh']} |")
    text = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
