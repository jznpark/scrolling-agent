"""Run episodes with a policy and collect metrics (and, optionally, a full trace)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import LIST, WATCH, ScrollingEnv
from .feed import Feed

SHORT_BOUT_SECONDS = 1.0  # a watch bout this short counts as "opened and left at once"


@dataclass
class EpisodeTrace:
    """Per-step record of one episode (row k = state after step k + 1)."""

    feed: Feed
    dt: float
    screen_h: float
    threshold: float
    scroll: np.ndarray        # (N,)
    mode: np.ndarray          # (N,) 0 LIST, 1 WATCH, 2 TRANSITION
    current: np.ndarray       # (N,) watched / clicked item, -1 if none
    action: np.ndarray        # (N,)
    reward: np.ndarray        # (N,)
    x: np.ndarray             # (N, n) evidence
    understood: np.ndarray    # (N, n)
    exposure: np.ndarray      # (N, n) seconds on screen
    w: np.ndarray             # (N, n) watch time
    A_known: np.ndarray       # (N, n) A as the agent knows it (0 before understanding)
    lam_known: np.ndarray     # (N, n) lambda as the agent knows it

    @property
    def t(self) -> np.ndarray:
        return (np.arange(len(self.reward)) + 1) * self.dt


def run_episode(
    env: ScrollingEnv,
    policy,
    seed: int | None = None,
    feed: Feed | None = None,
    record: bool = False,
    reference: bool = False,
) -> tuple[dict[str, Any], EpisodeTrace | None]:
    """Play one full episode; return metrics and, if `record`, the trace.

    Time is split by what each step did: time_down / time_up (scrolled), time_blocked
    (scroll pressed at the list edge), time_stay (stayed in the list), time_watch,
    time_transition. revisit_clicks counts clicks on items that had already left the
    screen through the top (went back up to them); rewatch_clicks (= n_revisits) counts
    clicks on items watched before. end_rate is the true current rate at each back.
    With `reference`, the known-information upper bound and pct_of_ref are added.
    """
    options = {"feed": feed} if feed is not None else None
    env.reset(seed=seed, options=options)
    n, dt = env.n_items, env.cfg.time.dt
    frac = env.cfg.screen.visible_fraction
    row_of = np.arange(n) // env.columns
    secs = dict.fromkeys(["down", "up", "stay", "blocked"], 0.0)
    passed = np.zeros(n, dtype=bool)  # seen, then left the screen through the top
    opened: set[int] = set()
    end_rates: list[float] = []
    revisit_clicks = up_moves = 0
    prev_up = False
    max_scroll = 0.0
    rows: dict[str, list] = {k: [] for k in ("scroll", "mode", "current", "action", "reward",
                                             "x", "understood", "exposure", "w", "A_known", "lam_known")}
    total = 0.0
    n_clicks = n_revisits = n_invalid = n_stay_list = 0
    bouts: list[int] = []  # length (steps) of each uninterrupted watch bout
    terminated = False
    while not terminated:
        mode_before, scroll_before = env.mode, env.scroll
        requested = policy(env)
        action = env.resolve_action(requested)
        valid = bool(env.action_masks()[action])
        if mode_before == LIST and action in env.slot_actions and valid:
            item = int(env.visible_slots()[env.slot_actions.index(action)])
            n_clicks += 1
            n_revisits += int(env.w[item] > 0)
            revisit_clicks += int(passed[item])
            opened.add(item)
        if mode_before == WATCH and action == env.back_action and valid:
            i = env.current
            end_rates.append(float(env.feed.A[i] * np.exp(-env.feed.lam[i] * env.w[i])))
        list_before = env.time_by_mode["LIST"]
        _, reward, terminated, _, info = env.step(requested)
        total += reward
        is_up = False
        if env.time_by_mode["LIST"] > list_before + 1e-12:  # this step was spent in the list
            moved = env.scroll != scroll_before
            if action in env.down_actions:
                secs["down" if moved else "blocked"] += dt
            elif action in env.up_actions:
                secs["up" if moved else "blocked"] += dt
                is_up = moved
            else:
                secs["stay"] += dt
        up_moves += int(is_up and not prev_up)
        prev_up = is_up
        if env.mode == LIST:
            # left through the top: the part still on screen is below the visibility rule
            above = (row_of < env.scroll) & (row_of + 1 - env.scroll < frac - 1e-9)
            passed |= (env.exposure > 0) & above
            max_scroll = max(max_scroll, env.scroll)
        n_invalid += int(info["invalid_action"])
        n_stay_list += int(mode_before == LIST and env.mode == LIST and action == env.stay_action)
        if mode_before == WATCH and env.mode == WATCH:
            bouts[-1] += 1
        elif mode_before != WATCH and env.mode == WATCH:
            bouts.append(0)
        if record:
            rows["scroll"].append(env.scroll)
            rows["mode"].append(env.mode)
            rows["current"].append(-1 if env.current is None else env.current)
            rows["action"].append(action)
            rows["reward"].append(reward)
            rows["x"].append(env.x.copy())
            rows["understood"].append(env.understood.copy())
            rows["exposure"].append(env.exposure.copy())
            rows["w"].append(env.w.copy())
            A_known, lam_known = env.revealed_profile()
            rows["A_known"].append(A_known.copy())
            rows["lam_known"].append(lam_known.copy())

    times = info["time_by_mode"]
    dt = env.cfg.time.dt
    bout_s = np.array(bouts) * dt
    metrics = {
        "return": total,
        "watch_reward": env.watch_reward_total,
        "n_understood": int(env.understood.sum()),
        "n_clicks": n_clicks,
        "n_revisits": n_revisits,
        "n_watched_items": int((env.w > 0).sum()),
        "time_list": times["LIST"],
        "time_watch": times["WATCH"],
        "time_transition": times["TRANSITION"],
        "time_stay_list": n_stay_list * dt,
        "n_short_bouts": int(np.sum(bout_s <= SHORT_BOUT_SECONDS + 1e-9)),
        "mean_bout": float(bout_s.mean()) if bout_s.size else 0.0,
        "max_item_seen": int(np.flatnonzero(env.exposure > 0).max(initial=-1)),
        "invalid_actions": n_invalid,
        **{f"time_{k}": v for k, v in secs.items()},
        "max_scroll": max_scroll,
        "watch_per_click": times["WATCH"] / n_clicks if n_clicks else float("nan"),
        "end_rate": float(np.mean(end_rates)) if end_rates else float("nan"),
        "n_end_by_session": int(env.mode == WATCH),
        "up_moves": up_moves,
        "revisit_clicks": revisit_clicks,
        "rewatch_clicks": n_revisits,
        "known_selection_rate": len(opened) / n_understood if (n_understood := int(env.understood.sum())) else float("nan"),
        **env.extra_metrics(),
    }
    if reference:
        from .policies import reference_return

        ref = reference_return(env.feed, env.cfg)
        metrics["reference"] = ref
        metrics["pct_of_ref"] = 100.0 * total / ref
    trace = None
    if record:
        trace = EpisodeTrace(
            feed=env.feed,
            dt=env.cfg.time.dt,
            screen_h=env.screen_h,
            threshold=env.cfg.ddm.threshold,
            **{k: np.array(v) for k, v in rows.items()},
        )
    return metrics, trace


def evaluate(env: ScrollingEnv, policy, seeds) -> list[dict[str, Any]]:
    """Metrics for one episode per seed."""
    return [dict(seed=s, **run_episode(env, policy, seed=s)[0]) for s in seeds]


def summarize(results: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of every numeric metric."""
    keys = [k for k in results[0] if k != "seed"]
    return {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results])))
            for k in keys}
