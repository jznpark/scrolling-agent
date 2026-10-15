"""Shared helpers for environment tests."""

import numpy as np

from scrolling_agent import Feed
from scrolling_agent.env import WATCH


def constant_feed(n: int = 100, A: float = 1.0, nu: float = 1.0, lam: float = 0.3) -> Feed:
    return Feed(
        z=np.zeros(n, dtype=np.int8),
        A=np.full(n, A),
        nu=np.full(n, nu),
        lam=np.full(n, lam),
    )


def run_until(env, action, cond, max_steps=2000):
    """Repeat `action` until cond(env) holds; return the rewards collected."""
    rewards = []
    for _ in range(max_steps):
        if cond(env):
            return rewards
        rewards.append(env.step(action)[1])
    raise AssertionError("condition not reached")


def click(env, item):
    """Click a visible, understood item and wait for the transition to finish."""
    slot = list(env.visible_slots()).index(item)
    rewards = [env.step(env.slot_actions[slot])[1]]
    while env.mode != WATCH:
        rewards.append(env.step(env.stay_action)[1])
    return rewards


def random_masked_actions(env, rng: np.random.Generator, n_steps: int) -> list[int]:
    """Play `n_steps` uniformly random valid actions; return them."""
    actions = []
    for _ in range(n_steps):
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        actions.append(a)
        env.step(a)
    return actions
