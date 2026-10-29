"""Kappa (clustering) experiments and time-allocation re-run for the 2nd presentation.

Rule-based policies only (no PPO training). The package code of each branch is used
as is; this script records every step to compute the metrics below.

Run from the exp/kappa-order worktree (PowerShell, venv active):
    $W = "C:\\Workspace\\scrolling-agent\\.worktrees"
    python scripts\\experiments\\kappa_order.py k1 --code $W\\feature_ppo\\src
    python scripts\\experiments\\kappa_order.py k2 --code $W\\feature_offscreen-residual\\src
    python scripts\\experiments\\kappa_order.py t-ppo --code $W\\feature_ppo\\src
    python scripts\\experiments\\kappa_order.py t-hanul --code $W\\feature_auto-preview\\src
    python scripts\\experiments\\kappa_order.py t-junho --code $W\\feature_offscreen-residual\\src

K1 / K2: evaluation feeds = default feeds of seeds 10000-10099; the same item set is
reordered with kappa in {0, 0.5, 0.8, 0.9} ("orig" = generated order, kappa default 0).
DDM noise comes from the same seed for every kappa. rho of the "re-chosen rule" is
picked per kappa on seeds 20000-20029 (reordered the same way) from 0.1..1.5.
K2 adds ddm.tau_residual = 2.0 (feature/offscreen-residual). Output: results/kappa_order/.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "kappa_order"
EVAL_SEEDS = list(range(10_000, 10_100))
TUNE_SEEDS = list(range(20_000, 20_030))
T_SEEDS = list(range(10_000, 10_020))
KAPPAS = [("orig", None), ("0", 0.0), ("0.5", 0.5), ("0.8", 0.8), ("0.9", 0.9)]
RHO_GRID = [round(r, 1) for r in np.arange(0.1, 1.51, 0.1)]
FIXED_RHO = 0.6

_CODE: str | None = None


def _init(code: str) -> None:
    global _CODE
    _CODE = code
    if code not in sys.path:
        sys.path.insert(0, code)


# --- configuration and feeds --------------------------------------------------------


def make_config(overrides: dict):
    from scrolling_agent.config import Config

    cfg = Config()
    for key, value in overrides.items():
        section, name = key.split(".")
        setattr(cfg, section, replace(getattr(cfg, section), **{name: value}))
    return cfg


def make_feed(cfg, seed: int, kappa: float | None):
    """Default-setting feed of `seed`, reordered with `kappa` (None = generated order)."""
    from scrolling_agent import ScrollingEnv
    from scrolling_agent.feed import reorder_feed

    env = ScrollingEnv(cfg)
    env.reset(seed=seed)
    if kappa is None:
        return env.feed
    rng = np.random.default_rng([seed, int(round(kappa * 100)), 2027])
    return reorder_feed(env.feed, kappa, rng)


# --- policies -------------------------------------------------------------------------


class SplitGreedy:
    """GreedyKnown with separate thresholds (rebuilt from 하늘's slide): open the best known
    item when its current rate >= rho_open; while watching, go back when it drops below
    rho_close; otherwise like greedy (move to the best known item, explore down)."""

    def __init__(self, rho_open: float, rho_close: float, dwell: float = 0.0, speed: int = 0):
        self.rho_open, self.rho_close, self.dwell, self.speed = rho_open, rho_close, dwell, speed

    def __call__(self, env) -> int:
        from scrolling_agent.env import TRANSITION, WATCH
        from scrolling_agent.policies import known_rates

        if env.mode == TRANSITION:
            return env.stay_action
        rates = known_rates(env)
        if env.mode == WATCH:
            return env.back_action if rates[env.current] < self.rho_close else env.stay_action
        target = int(np.argmax(rates))
        visible = list(env.visible_slots())
        if rates[target] < self.rho_open:
            pending = [i for i in visible if not env.understood[i] and env.exposure[i] < self.dwell]
            at_bottom = env.scroll >= env.max_scroll
            return env.stay_action if pending or at_bottom else env.down_actions[self.speed]
        if target in visible:
            return env.slot_actions[visible.index(target)]
        return env.up_actions[self.speed] if target < visible[0] else env.down_actions[self.speed]


def make_policy(spec: tuple):
    from scrolling_agent.policies import GreedyKnown, ScrollFirst

    kind = spec[0]
    if kind == "greedy":
        return GreedyKnown(spec[1], dwell=spec[2]) if spec[2] else GreedyKnown(spec[1])
    if kind == "scrollfirst":
        return ScrollFirst(spec[1])
    if kind == "split":
        return SplitGreedy(spec[1], spec[2], spec[3])
    raise ValueError(spec)


# --- one recorded episode ---------------------------------------------------------------


def play(cfg, feed, seed: int, policy) -> dict:
    """Play one episode and compute every metric of section 4 of the instructions."""
    from scrolling_agent import ScrollingEnv
    from scrolling_agent.env import LIST, WATCH
    from scrolling_agent.policies import reference_return

    env = ScrollingEnv(cfg)
    env.reset(seed=seed, options={"feed": feed})
    n, dt = env.n_items, cfg.time.dt
    frac = cfg.screen.visible_fraction
    idx = np.arange(n)
    secs = dict.fromkeys(["down", "up", "stay", "blocked", "watch", "transition"], 0.0)
    total = 0.0
    clicks = revisit_clicks = rewatch_clicks = up_moves = 0
    opened: set[int] = set()
    passed = np.zeros(n, dtype=bool)  # seen, then left the screen through the top
    end_rates: list[float] = []
    prev_up = False
    max_scroll = 0.0
    terminated = False
    while not terminated:
        mode_b, scroll_b = env.mode, env.scroll
        a = int(policy(env))
        valid = bool(env.action_masks()[a])
        if valid and mode_b == LIST and a in env.slot_actions:
            item = int(env.visible_slots()[env.slot_actions.index(a)])
            clicks += 1
            opened.add(item)
            revisit_clicks += int(passed[item])
            rewatch_clicks += int(env.w[item] > 0)
        if valid and mode_b == WATCH and a == env.back_action:
            i = env.current
            end_rates.append(float(env.feed.A[i] * math.exp(-env.feed.lam[i] * env.w[i])))
        before = dict(env.time_by_mode)
        _, reward, terminated, _, _ = env.step(a)
        total += reward
        mode_key = next(k for k in before if env.time_by_mode[k] > before[k] + 1e-12)
        if mode_key == "LIST":
            moved = env.scroll != scroll_b
            if a in env.down_actions:
                cat = "down" if moved else "blocked"
            elif a in env.up_actions:
                cat = "up" if moved else "blocked"
            else:
                cat = "stay"
        else:
            cat = "watch" if mode_key == "WATCH" else "transition"
        secs[cat] += dt
        up_moves += int(cat == "up" and not prev_up)
        prev_up = cat == "up"
        if env.mode == LIST:
            # left through the top: the part still on screen is below the visibility rule,
            # with the same 1e-9 tolerance as env.visible_items
            above = (idx < env.scroll) & (idx + 1 - env.scroll < frac - 1e-9)
            passed |= (env.exposure > 0) & above
            max_scroll = max(max_scroll, env.scroll)

    cost = cfg.time.n_steps * cfg.time.step_cost
    ref = reference_return(env.feed, cfg)
    n_und = int(env.understood.sum())
    return {
        "return": total,
        "watch_reward": float(env.watch_reward_total),
        "fixed_cost": cost,
        "reference": ref,
        "pct_of_ref": 100.0 * total / ref,
        **{f"time_{k}": v for k, v in secs.items()},
        "time_total": sum(secs.values()),
        "n_understood": n_und,
        "max_item_seen": int(np.flatnonzero(env.exposure > 0).max(initial=-1)),
        "max_scroll": max_scroll,
        "n_clicks": clicks,
        "watch_per_click": secs["watch"] / clicks if clicks else float("nan"),
        "end_rate": float(np.mean(end_rates)) if end_rates else float("nan"),
        "n_end_by_session": int(env.mode == WATCH),
        "up_moves": up_moves,
        "revisit_clicks": revisit_clicks,
        "rewatch_clicks": rewatch_clicks,
        "known_selection_rate": len(opened) / n_und if n_und else float("nan"),
    }


def feed_check(feed) -> dict:
    from scrolling_agent.feed import feed_stats

    st = feed_stats(feed)
    return {"type_persistence": st["type_persistence"], "n_z1": int(feed.z.sum()),
            "nu_A_corr_log": st["nu_A_corr_log"], "mean_A": st["mean_A"], "mean_nu": st["mean_nu"],
            "mean_lam": st["mean_lam"]}


# --- jobs (run in worker processes) -------------------------------------------------------


def job_kappa(args: tuple) -> dict:
    overrides, seed, label, kappa, spec = args
    cfg = make_config(overrides)
    feed = make_feed(make_config({}), seed, kappa)  # the item set never depends on the condition
    row = {"seed": seed, "kappa_label": label, "kappa": "" if kappa is None else kappa}
    row.update(play(cfg, feed, seed, make_policy(spec)))
    return row


def job_feed(args: tuple) -> dict:
    seed, label, kappa = args
    feed = make_feed(make_config({}), seed, kappa)
    return {"seed": seed, "kappa_label": label, "kappa": "" if kappa is None else kappa, **feed_check(feed)}


def job_plain(args: tuple) -> dict:
    overrides, seed, spec = args
    cfg = make_config(overrides)
    feed = make_feed(cfg, seed, None)
    return {"seed": seed, **play(cfg, feed, seed, make_policy(spec))}


# --- experiments ------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def kappa_experiment(name: str, overrides: dict, pool: ProcessPoolExecutor) -> None:
    t0 = time.perf_counter()
    # 1) feed check: the same item set for every kappa
    checks = list(pool.map(job_feed, [(s, lab, k) for s in EVAL_SEEDS for lab, k in KAPPAS], chunksize=20))
    write_csv(OUT / f"{name}_feed_check.csv", checks)
    # 2) re-choose rho per kappa on separate feeds
    tune_jobs = [(overrides, s, lab, k, ("greedy", rho, 0.0)) for lab, k in KAPPAS for rho in RHO_GRID for s in TUNE_SEEDS]
    tune_rows = list(pool.map(job_kappa, tune_jobs, chunksize=10))
    for row, (_, _, _, _, spec) in zip(tune_rows, tune_jobs):
        row["rho"] = spec[1]
    summary = []
    best = {}
    for lab, _ in KAPPAS:
        for rho in RHO_GRID:
            rets = [r["return"] for r in tune_rows if r["kappa_label"] == lab and r["rho"] == rho]
            summary.append({"kappa_label": lab, "rho": rho, "mean_return": float(np.mean(rets)), "n": len(rets)})
        rows = [s for s in summary if s["kappa_label"] == lab]
        best[lab] = max(rows, key=lambda s: s["mean_return"])["rho"]
    write_csv(OUT / f"{name}_tuning.csv", summary)
    # 3) evaluation on seeds 10000-10099
    policies = [("고정 규칙 greedy ρ0.6", lambda lab: ("greedy", FIXED_RHO, 0.0)),
                ("다시 고른 규칙 greedy", lambda lab: ("greedy", best[lab], 0.0)),
                ("scrollfirst ρ0.6", lambda lab: ("scrollfirst", FIXED_RHO))]
    jobs, meta = [], []
    for pname, spec_of in policies:
        for lab, k in KAPPAS:
            spec = spec_of(lab)
            for s in EVAL_SEEDS:
                jobs.append((overrides, s, lab, k, spec))
                meta.append((pname, spec[1]))
    rows = list(pool.map(job_kappa, jobs, chunksize=10))
    for row, (pname, rho) in zip(rows, meta):
        row["policy"] = pname
        row["rho"] = rho
    write_csv(OUT / f"{name}_per_feed.csv", rows)
    print(f"{name}: best rho per kappa {best} ({time.perf_counter() - t0:.0f}s)")


T_CONDITIONS = {
    "t-ppo": [
        ("기본", "feature/ppo@39580c0", {}, [("greedy ρ0.6", ("greedy", 0.6, 0.0)),
                                            ("greedy-dwell ρ0.6 dwell0.5", ("greedy", 0.6, 0.5))]),
        ("세은 hard_good", "feature/ppo@39580c0", {"feed.d_nu": -0.8},
         [("greedy ρ0.6", ("greedy", 0.6, 0.0)), ("greedy-dwell ρ0.6 dwell0.5", ("greedy", 0.6, 0.5))]),
    ],
    "t-hanul": [
        ("하늘 A", "feature/auto-preview@d10b08d", {"time.transition_seconds": 0.2},
         [("SplitGreedy 열기0.8/종료0.6", ("split", 0.8, 0.6, 0.0))]),
        ("하늘 B", "feature/auto-preview@d10b08d", {"time.transition_seconds": 1.0},
         [("SplitGreedy 열기0.8/종료0.6", ("split", 0.8, 0.6, 0.0))]),
        ("하늘 C", "feature/auto-preview@d10b08d", {"time.transition_seconds": 1.0},
         [("SplitGreedy 열기1.2/종료0.4", ("split", 1.2, 0.4, 0.0))]),
    ],
    "t-junho": [
        ("준호 τ0", "feature/offscreen-residual@a57c720", {"ddm.tau_residual": 0.0},
         [("greedy ρ0.6", ("greedy", 0.6, 0.0))]),
        ("준호 τ2", "feature/offscreen-residual@a57c720", {"ddm.tau_residual": 2.0},
         [("greedy ρ0.6", ("greedy", 0.6, 0.0))]),
    ],
}


def t_experiment(group: str, pool: ProcessPoolExecutor) -> None:
    rows = []
    for cond, branch, overrides, policies in T_CONDITIONS[group]:
        for pname, spec in policies:
            res = list(pool.map(job_plain, [(overrides, s, spec) for s in T_SEEDS]))
            for r in res:
                r.update({"condition": cond, "branch": branch, "policy": pname,
                          "settings": ";".join(f"{k}={v}" for k, v in overrides.items()) or "기본값"})
            rows += res
            print(f"{group}: {cond:16} {pname:28} return {np.mean([r['return'] for r in res]):7.2f}")
    write_csv(OUT / f"{group}_per_feed.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["k1", "k2", "t-ppo", "t-hanul", "t-junho"])
    parser.add_argument("--code", required=True, help="src folder of the branch to run")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    code = str(Path(args.code).resolve())
    _init(code)
    import scrolling_agent

    print("code:", Path(scrolling_agent.__file__).parent)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(code,)) as pool:
        if args.experiment == "k1":
            kappa_experiment("k1", {}, pool)
        elif args.experiment == "k2":
            kappa_experiment("k2", {"ddm.tau_residual": 2.0}, pool)
        else:
            t_experiment(args.experiment, pool)


if __name__ == "__main__":
    main()
