"""Stronger rule baselines for the B/C comparisons (fair-rule check).

The pipeline's rule (GreedyKnown, rho 0.6) never stops to read and only uses the slow speed. Where PPO beat
it, the gain could come from options the rule simply lacks. This script tunes two richer rules per condition
on lists 20000+ and evaluates the best on the 100 test lists 10000+ (same lists as the PPO evaluation):

    dwell     GreedyKnown(rho, dwell): stops while a visible unread item has been on screen < dwell s
    speed     ContextSpeed(rho, dwell, a_thr): as dwell, but explores at the fast speed while the most recently
              understood item had A < a_thr (two-speed conditions only)

Run (from the repository root, PYTHONPATH pointing at src):
    python scripts/experiments/fair_rules.py --conditions a-base c9-dnu-k0 ... --out runs/fair_rules
Writes <out>/summary.md and <out>/<condition>.csv (one row per test list).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scrolling_agent import ScrollingEnv
from scrolling_agent.env import LIST
from scrolling_agent.evaluation import run_episode
from scrolling_agent.policies import GreedyKnown

ROOT = Path(__file__).resolve().parents[2]
EVAL_SEED, TUNE_SEED = 10_000, 20_000
RHOS = [0.5, 0.6, 0.7]
DWELLS = [0.0, 0.25, 0.5, 1.0]
A_THRS = [0.8, 1.0]


def _pipeline():
    spec = importlib.util.spec_from_file_location("pipeline", ROOT / "scripts" / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ContextSpeed:
    """GreedyKnown that explores fast while the newest understood item looked poor (A < a_thr)."""

    rho: float = 0.6
    dwell: float = 0.0
    a_thr: float = 1.0
    _greedy: GreedyKnown = field(init=False, repr=False)
    _known: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_A: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._greedy = GreedyKnown(self.rho, dwell=self.dwell)

    def __call__(self, env: ScrollingEnv) -> int:
        if env.step_count == 0 or self._known is None:
            self._known, self._last_A = env.understood.copy(), None
        new = np.flatnonzero(env.understood & ~self._known)
        if new.size:
            self._last_A = float(env.revealed_profile()[0][new.max()])
        self._known = env.understood.copy()
        action = self._greedy(env)
        poor = self._last_A is not None and self._last_A < self.a_thr
        if env.mode == LIST and action == env.down_actions[0] and poor and len(env.down_actions) > 1:
            return env.down_actions[1]
        return action


def make(spec: tuple):
    kind, rho, dwell, a_thr = spec
    return GreedyKnown(rho, dwell=dwell) if kind == "dwell" else ContextSpeed(rho, dwell, a_thr)


def play(job: tuple) -> list[dict]:
    name, spec, seeds, reference = job
    cfg = _pipeline().condition_config(name)
    env, policy = ScrollingEnv(cfg), make(spec)
    return [{"seed": s, **run_episode(env, policy, seed=s, reference=reference)[0]} for s in seeds]


def specs_for(n_speeds: int) -> list[tuple]:
    out = [("dwell", r, d, None) for r, d in itertools.product(RHOS, DWELLS)]
    if n_speeds > 1:
        out += [("speed", r, d, a) for r, d, a in itertools.product(RHOS, DWELLS, A_THRS)]
    return out


def ppo_by_list(name: str) -> dict[str, float]:
    rows = list(csv.DictReader((ROOT / "runs" / name / "eval.csv").open(encoding="utf-8")))
    by: dict[str, list[float]] = {}
    for r in rows:
        if r["policy"] == "ppo-sto":
            by.setdefault(r["seed"], []).append(float(r["pct_of_ref"]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--tune", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("runs/fair_rules"))
    args = parser.parse_args()
    pipeline = _pipeline()
    args.out.mkdir(parents=True, exist_ok=True)
    tune_seeds = list(range(TUNE_SEED, TUNE_SEED + args.tune))
    test_chunks = [list(range(EVAL_SEED + i, EVAL_SEED + min(i + 10, args.episodes)))
                   for i in range(0, args.episodes, 10)]
    lines = ["| condition | best rule (kind, rho, dwell, A thr) | rule % | PPO % | PPO − rule (SE) | lists PPO > rule |",
             "|---|---|---:|---:|---:|---:|"]
    with ProcessPoolExecutor(args.workers) as pool:
        for name in args.conditions:
            n_speeds = len(pipeline.condition_config(name).screen.speeds)
            kinds = {}
            for kind in ("dwell", "speed") if n_speeds > 1 else ("dwell",):
                specs = [s for s in specs_for(n_speeds) if s[0] == kind]
                tuned = pool.map(play, [(name, s, tune_seeds, False) for s in specs])
                scores = {s: float(np.mean([r["return"] for r in rows])) for s, rows in zip(specs, tuned)}
                best = max(scores, key=scores.get)
                rows = [r for part in pool.map(play, [(name, best, c, True) for c in test_chunks]) for r in part]
                kinds[kind] = (best, rows)
            ppo = ppo_by_list(name)
            for kind, (best, rows) in kinds.items():
                with (args.out / f"{name}-{kind}.csv").open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                rule = {str(r["seed"]): float(r["pct_of_ref"]) for r in rows}
                d = np.array([ppo[s] - rule[s] for s in rule])
                lines.append(f"| {name} | {kind}, {best[1]}, {best[2]}, {best[3] if best[3] is not None else '–'} "
                             f"| {np.mean(list(rule.values())):.1f} | {np.mean([ppo[s] for s in rule]):.1f} "
                             f"| {d.mean():+.1f} ({d.std(ddof=1) / np.sqrt(d.size):.1f}) | {np.mean(d > 0):.2f} |")
            print("\n".join(lines[-len(kinds):]), flush=True)
    text = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
