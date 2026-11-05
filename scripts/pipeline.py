"""Condition pipeline (P0-4): condition name -> PPO training (seeds in parallel) -> evaluation -> table.

Run (PowerShell, venv active):
    python scripts\\pipeline.py list
    python scripts\\pipeline.py train base topk --seeds 0 1 2 --timesteps 5000000
    python scripts\\pipeline.py evaluate base topk --episodes 100
    python scripts\\pipeline.py table base topk

A condition is a set of model overrides (`--set` syntax) plus training overrides
(`--train` syntax), defined in CONDITIONS below. Runs go to runs/<condition>/s<seed>/.

Evaluation plays the same unseen lists (seeds 10000, ...) with
    ppo-det / ppo-sto   every trained seed's best model, deterministic / sampled actions
    greedy-fixed        GreedyKnown with rho tuned once on the base condition
    greedy-rechosen     GreedyKnown with rho tuned on this condition (seeds 20000, ...)
and writes runs/<condition>/eval.csv (one row per policy x training seed x episode).
`table` prints mean +- sd per policy and writes runs/<condition>/summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
EVAL_SEED = 10_000
TUNE_SEED = 20_000
RHO_GRID = [round(r, 1) for r in np.arange(0.1, 1.51, 0.1)]

# name -> model overrides ("section.field=value") and training overrides ("field=value")
CONDITIONS: dict[str, dict[str, list[str]]] = {
    "base": {"set": [], "train": []},                                   # draft model, full-memory observation
    "topk": {"set": ["obs.memory=topk"], "train": []},                  # draft model, top-10 known items
    "pilot-settings": {"set": [], "train": ["ent_coef=0.01", "n_steps=1024"]},  # 10-15 pilot hyperparameters
    "topk-k5": {"set": ["obs.memory=topk"], "train": ["decision_steps=5"]},  # decide every 0.5 s (learning side)
    "topk-k5-lr1m": {"set": ["obs.memory=topk"],                          # + learning rate 3e-4 -> 0 over 1M decisions
                     "train": ["decision_steps=5", "lr_schedule='linear'", "total_timesteps=1000000"]},
    "topk-k5-lr3m": {"set": ["obs.memory=topk"],                          # + learning rate 3e-4 -> 0 over 3M decisions
                     "train": ["decision_steps=5", "lr_schedule='linear'", "total_timesteps=3000000"]},
    "kappa0.9": {"set": ["obs.memory=topk", "feed.kappa=0.9"], "train": []},     # clustered feed
    "dnu-neg": {"set": ["obs.memory=topk", "feed.d_nu=-0.5"], "train": []},      # good items are harder to read
}
TABLE_COLUMNS = ["pct_of_ref", "return", "n_understood", "n_clicks", "rewatch_clicks", "n_short_bouts",
                 "time_watch", "time_transition", "time_stay", "max_item_seen"]


def condition_config(name: str):
    from scrolling_agent.cli import apply_overrides
    from scrolling_agent.config import Config

    return apply_overrides(Config(), list(CONDITIONS[name]["set"]))


# --- train ------------------------------------------------------------------------------------


def train(names: list[str], seeds: list[int], timesteps: int | None, resume: bool = False) -> None:
    # one math thread per process: several runs x 8 env workers oversubscribe the CPU otherwise
    env = {**os.environ, **{k: os.environ.get(k, "1") for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                 "OPENBLAS_NUM_THREADS")}}
    procs = []
    for name in names:
        cond = CONDITIONS[name]
        for seed in seeds:
            run = f"{name}/s{seed}"
            cmd = [sys.executable, str(ROOT / "scripts" / "train_ppo.py"), "--name", run]
            if resume and (RUNS / run / "config.json").exists():
                if (RUNS / run / "final_model.zip").exists():
                    print(f"skip {run}: already finished")
                    continue
                cmd += ["--resume"]  # settings come from the run's config.json
            else:
                cmd += ["--seed", str(seed)]
                if timesteps:
                    cmd += ["--timesteps", str(timesteps)]
                if cond["set"]:
                    cmd += ["--set", *cond["set"]]
                if cond["train"]:
                    cmd += ["--train", *cond["train"]]
            log = RUNS / name / f"s{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = log.open("a" if resume else "w", encoding="utf-8")
            procs.append((run, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"started {run}: {' '.join(cmd[2:])}")
    failed = []
    for run, proc, fh in procs:
        proc.wait()
        fh.close()
        print(f"finished {run} (exit {proc.returncode})")
        if proc.returncode:
            failed.append(run)
    if failed:
        raise SystemExit(f"training failed: {failed}")


# --- evaluate --------------------------------------------------------------------------------


def _episodes(job: tuple) -> list[dict]:
    """Worker: play `seeds` of one condition with one policy spec."""
    name, spec, seeds = job
    from scrolling_agent import ScrollingEnv
    from scrolling_agent.evaluation import run_episode
    from scrolling_agent.policies import GreedyKnown, PPOPolicy

    cfg = condition_config(name)
    kind = spec[0]
    if kind == "greedy":
        policy = GreedyKnown(spec[1])
    else:
        policy = PPOPolicy.load(spec[1], "best", deterministic=(kind == "ppo-det"))
    env = ScrollingEnv(cfg)
    return [{"seed": s, **run_episode(env, policy, seed=s, reference=True)[0]} for s in seeds]


def tune_rho(pool: ProcessPoolExecutor, name: str, n: int) -> float:
    seeds = list(range(TUNE_SEED, TUNE_SEED + n))
    jobs = [(name, ("greedy", rho), seeds) for rho in RHO_GRID]
    scores = {rho: float(np.mean([r["return"] for r in rows])) for rho, rows in zip(RHO_GRID, pool.map(_episodes, jobs))}
    return max(scores, key=scores.get)


def evaluate(names: list[str], episodes: int, tune_episodes: int, workers: int) -> None:
    seeds = list(range(EVAL_SEED, EVAL_SEED + episodes))
    chunks = [seeds[i:i + 10] for i in range(0, len(seeds), 10)]
    with ProcessPoolExecutor(workers) as pool:
        rho_fixed = tune_rho(pool, "base", tune_episodes)
        for name in names:
            t0 = time.perf_counter()
            rho_own = rho_fixed if name == "base" else tune_rho(pool, name, tune_episodes)
            specs = [("greedy-fixed", "", ("greedy", rho_fixed)), ("greedy-rechosen", "", ("greedy", rho_own))]
            for run in sorted((RUNS / name).glob("s*/best_model.zip")):
                for kind in ("ppo-det", "ppo-sto"):
                    specs.append((kind, run.parent.name, (kind, str(run.parent))))
            jobs = [(name, spec, chunk) for _, _, spec in specs for chunk in chunks]
            results = list(pool.map(_episodes, jobs))
            rows = []
            for k, (label, train_seed, spec) in enumerate(specs):
                for part in results[k * len(chunks):(k + 1) * len(chunks)]:
                    for r in part:
                        rho = spec[1] if spec[0] == "greedy" else ""
                        rows.append({"policy": label, "train_seed": train_seed, "rho": rho, **r})
            out = RUNS / name / "eval.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                writer.writeheader()
                writer.writerows(rows)
            meta = {"episodes": episodes, "rho_fixed": rho_fixed, "rho_rechosen": rho_own,
                    "overrides": CONDITIONS[name]}
            (RUNS / name / "eval_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"{name}: {len(rows)} episodes in {time.perf_counter() - t0:.0f}s -> {out}")


# --- table -------------------------------------------------------------------------------------


def table(names: list[str]) -> None:
    for name in names:
        path = RUNS / name / "eval.csv"
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        meta = json.loads((RUNS / name / "eval_meta.json").read_text())
        lines = [f"## {name}", "",
                 f"overrides: `{' '.join(CONDITIONS[name]['set']) or '(none)'}`"
                 f" · train: `{' '.join(CONDITIONS[name]['train']) or '(defaults)'}`"
                 f" · rho fixed {meta['rho_fixed']} / re-chosen {meta['rho_rechosen']}"
                 f" · {meta['episodes']} unseen lists", "",
                 "| policy | " + " | ".join(TABLE_COLUMNS) + " |",
                 "|---|" + "---:|" * len(TABLE_COLUMNS)]
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["policy"], []).append(r)
            if r["train_seed"]:
                groups.setdefault(f"{r['policy']} [{r['train_seed']}]", []).append(r)
        for label, group in groups.items():
            cells = []
            for c in TABLE_COLUMNS:
                v = np.array([float(r[c]) for r in group])
                cells.append(f"{v.mean():.1f} ± {v.std():.1f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        text = "\n".join(lines) + "\n"
        (RUNS / name / "summary.md").write_text(text, encoding="utf-8")
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("train")
    p.add_argument("conditions", nargs="+", choices=list(CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--timesteps", type=int, default=None, help="default: TrainConfig.total_timesteps")
    p.add_argument("--resume", action="store_true",
                   help="continue interrupted runs from their last checkpoint (finished runs are skipped)")
    p = sub.add_parser("evaluate")
    p.add_argument("conditions", nargs="+", choices=list(CONDITIONS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--tune-episodes", type=int, default=30)
    p.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    p = sub.add_parser("table")
    p.add_argument("conditions", nargs="+", choices=list(CONDITIONS))
    args = parser.parse_args()

    if args.cmd == "list":
        for name, cond in CONDITIONS.items():
            print(f"{name:<16} set: {' '.join(cond['set']) or '-'}   train: {' '.join(cond['train']) or '-'}")
    elif args.cmd == "train":
        train(args.conditions, args.seeds, args.timesteps, args.resume)
    elif args.cmd == "evaluate":
        evaluate(args.conditions, args.episodes, args.tune_episodes, args.workers)
    else:
        table(args.conditions)


if __name__ == "__main__":
    main()
