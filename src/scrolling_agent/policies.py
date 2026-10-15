"""Comparison policies (step 4), the PPO policy wrapper and the reference return.

A policy is a callable `policy(env) -> action`. The hand-written baselines read
only what the agent observes: visible slots, understood flags, the revealed
(A, lambda) of understood items, watch times w and the scroll position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Config
from .env import TRANSITION, WATCH, ScrollingEnv
from .feed import Feed


class Policy(Protocol):
    def __call__(self, env: ScrollingEnv) -> int: ...


def known_rates(env: ScrollingEnv) -> np.ndarray:
    """Current reward rate A exp(-lambda w) of understood items; -inf for the rest."""
    rate = env.feed.A * np.exp(-env.feed.lam * env.w)
    return np.where(env.understood, rate, -np.inf)


@dataclass
class RandomPolicy:
    """Uniformly random valid action."""

    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, env: ScrollingEnv) -> int:
        return int(self.rng.choice(np.flatnonzero(env.action_masks())))


@dataclass
class ScrollFirst:
    """Scroll down at a fixed speed; click the first understood item whose current
    rate is at least `rho` (rho = 0: the first understood item); go back when the
    rate of the watched item drops below `rho`. Never scrolls up."""

    rho: float = 0.3
    speed: int = 0

    def __call__(self, env: ScrollingEnv) -> int:
        if env.mode == TRANSITION:
            return env.stay_action
        rates = known_rates(env)
        if env.mode == WATCH:
            return env.back_action if rates[env.current] < self.rho else env.stay_action
        for j, i in enumerate(env.visible_slots()):
            if rates[i] >= self.rho:
                return env.slot_actions[j]
        return env.down_actions[self.speed]


@dataclass
class GreedyKnown:
    """Go to the understood item with the highest current rate (on or off screen,
    so it revisits) and watch it while the rate is at least `rho`; when no known
    item reaches `rho`, scroll down to explore."""

    rho: float = 0.3
    speed: int = 0

    def __call__(self, env: ScrollingEnv) -> int:
        if env.mode == TRANSITION:
            return env.stay_action
        rates = known_rates(env)
        if env.mode == WATCH:
            return env.back_action if rates[env.current] < self.rho else env.stay_action

        target = int(np.argmax(rates))
        visible = list(env.visible_slots())
        if rates[target] < self.rho:
            at_bottom = env.scroll >= env.max_scroll
            return env.stay_action if at_bottom else env.down_actions[self.speed]
        if target in visible:
            return env.slot_actions[visible.index(target)]
        return env.up_actions[self.speed] if target < visible[0] else env.down_actions[self.speed]


class PPOPolicy:
    """A trained MaskablePPO model with its VecNormalize observation statistics."""

    def __init__(self, model, vecnormalize=None, deterministic: bool = True):
        self.model = model
        self.vecnormalize = vecnormalize
        self.deterministic = deterministic

    @classmethod
    def load(cls, run_dir: str | Path, which: str = "best", deterministic: bool = True) -> PPOPolicy:
        """Load `best_model.zip` (or `final_model.zip` with which="final") from a run folder."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        run_dir = Path(run_dir)
        model = MaskablePPO.load(run_dir / f"{which}_model.zip", device="cpu")
        stats = run_dir / f"{which}_vecnormalize.pkl"
        vecnormalize = None
        if stats.exists():
            vecnormalize = VecNormalize.load(str(stats), DummyVecEnv([ScrollingEnv]))
            vecnormalize.training = False
        return cls(model, vecnormalize, deterministic)

    def __call__(self, env: ScrollingEnv) -> int:
        obs = env.observe()
        if self.vecnormalize is not None:
            obs = self.vecnormalize.normalize_obs(obs)
        action, _ = self.model.predict(obs, action_masks=env.action_masks(), deterministic=self.deterministic)
        return int(action)


def reference_return(feed: Feed, cfg: Config | None = None) -> float:
    """Return of the known-information reference: every item understood, no movement
    or transition time, and each step watches the item with the largest step reward.

    Each item's step rewards decrease with w, so this greedy choice is optimal and
    gives an upper bound for the environment's return.
    """
    cfg = cfg or Config()
    dt, n_steps = cfg.time.dt, cfg.time.n_steps
    scale = feed.A / feed.lam * (1.0 - np.exp(-feed.lam * dt))
    w = np.zeros(feed.n)
    total = 0.0
    for _ in range(n_steps):
        gain = scale * np.exp(-feed.lam * w)
        i = int(np.argmax(gain))
        total += gain[i]
        w[i] += dt
    return total - n_steps * cfg.time.step_cost
