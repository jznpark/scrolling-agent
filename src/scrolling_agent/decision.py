"""Decision interval for learning agents (learning side only; the model keeps dt and its actions).

A learning agent decides at most every `k` env steps instead of every step:
- scroll / stay actions are held for up to k steps (while they stay valid);
- transitions (click / back) are passed through with stay(None), the only valid action there;
- arriving in a new mode (LIST <-> WATCH) ends the hold, so the agent decides right away.

`DecisionHold` applies the rule step by step to any decision function (used by PPOPolicy at
evaluation); `DecisionInterval` is the matching Gymnasium wrapper for training. Both give the
same trajectory for the same decisions.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from .env import TRANSITION, ScrollingEnv


def hold_continues(env: ScrollingEnv, action: int, mode: int, held: int, k: int) -> bool:
    """Whether the hold started in `mode` with `action` continues for one more step."""
    if env.mode == TRANSITION:
        return True
    return env.mode == mode and held < k and bool(env.action_masks()[action])


class DecisionHold:
    """Wrap a per-step decision function so it is consulted only at decision points."""

    def __init__(self, decide: Callable[[ScrollingEnv], int], k: int):
        self.decide, self.k = decide, k
        self._env_step = -1
        self.reset()

    def reset(self) -> None:
        self._action = self._mode = None
        self._held = 0

    def __call__(self, env: ScrollingEnv) -> int:
        if env.step_count == 0 or env.step_count != self._env_step + 1:
            self.reset()  # new episode (or a jump): decide afresh
        self._env_step = env.step_count
        if self._action is not None and hold_continues(env, self._action, self._mode, self._held, self.k):
            self._held += 1
            return env.stay_action if env.mode == TRANSITION else self._action
        self._action, self._mode, self._held = int(self.decide(env)), env.mode, 1
        return self._action


class DecisionInterval(gym.Wrapper):
    """One agent step = one decision held as described above; rewards are summed."""

    def __init__(self, env: ScrollingEnv, k: int):
        super().__init__(env)
        self.k = k

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()

    def step(self, action: int):
        env: ScrollingEnv = self.env.unwrapped
        mode, held, total = env.mode, 1, 0.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        total += reward
        while not (terminated or truncated) and hold_continues(env, action, mode, held, self.k):
            held += 1
            obs, reward, terminated, truncated, info = self.env.step(
                env.stay_action if env.mode == TRANSITION else action)
            total += reward
        info["env_steps"] = held
        return obs, total, terminated, truncated, info
