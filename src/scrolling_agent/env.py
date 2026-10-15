"""Gymnasium environment (step 2, draft sections 03–05, 08, 10).

Geometry: card height = 1, screen height H = cards_per_screen (cards), scroll
position s = top edge of the screen in cards, 0 <= s <= n_items - H. Item i
covers [i, i + 1]; it is visible when at least `visible_fraction` of it is on
screen. Speeds are in screen heights / s (as v0 in g(|v|)).

Modes: LIST (move / stay, DDM of visible items runs), WATCH (watch the clicked
item), TRANSITION (click or back, `transition_steps` steps counted from the step
the action is chosen; no reward and no DDM).

Actions (flat discrete index, for MaskablePPO), with S speeds and K slots:
    0 .. S-1           down(speeds[k])
    S .. 2S-1          up(speeds[k])
    2S                 stay(None)     LIST: wait and understand · WATCH: keep watching
    2S+1 .. 2S+K       stay(slot j)   click the understood item in visible slot j (top to bottom)
    2S+K+1             back           WATCH -> LIST at the previous scroll position
An action outside `action_masks()` is replaced by stay(None).

Observation (float32 vector, physical units; hidden A, lambda of items not yet
understood, true nu and latent z are never included):
    global   mode one-hot (LIST, WATCH, TRANSITION), transition left (fraction),
             scroll position (fraction), remaining time (fraction),
             previous action one-hot, previous reward
    current  A, lambda, w, current reward rate of the item being watched (else 0)
    slots    K x [present, id / n, top offset (screen heights), x / B, exposure (s),
             understood, A, lambda, w]
    memory   n x [understood, A, lambda, w, x / B, offset of item centre from
             screen centre (screen heights)]
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import Config
from .feed import Feed, generate_feed

LIST, WATCH, TRANSITION = 0, 1, 2
MODE_NAMES = ("LIST", "WATCH", "TRANSITION")

N_SLOT_FEATURES = 9
N_MEMORY_FEATURES = 6
N_CURRENT_FEATURES = 4


def visible_items(scroll: float, n_items: int, screen_h: float, frac: float) -> np.ndarray:
    """Items with at least `frac` of their height on screen, top to bottom."""
    lo = max(int(np.floor(scroll)), 0)
    hi = min(int(np.ceil(scroll + screen_h)), n_items)
    idx = np.arange(lo, hi)
    overlap = np.minimum(idx + 1, scroll + screen_h) - np.maximum(idx, scroll)
    return idx[overlap >= frac - 1e-9]


def advance_evidence(
    x: np.ndarray, drift: np.ndarray, sigma: float, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """One Euler–Maruyama step of dx = drift dt + sigma dW."""
    return x + drift * dt + sigma * np.sqrt(dt) * rng.standard_normal(np.shape(x))


def watch_reward(A: float, lam: float, w: float, dt: float) -> float:
    """Integral of A exp(-lam t) over [w, w + dt]."""
    return (A / lam) * np.exp(-lam * w) * (1.0 - np.exp(-lam * dt))


class ScrollingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config | None = None, feed: Feed | None = None):
        self.cfg = config or Config()
        self.n_items = self.cfg.screen.n_items
        if feed is not None and feed.n != self.n_items:
            raise ValueError(f"feed has {feed.n} items, config expects {self.n_items}")
        self._fixed_feed = feed

        self.screen_h = self.cfg.screen.cards_per_screen
        self.max_scroll = max(self.n_items - self.screen_h, 0.0)
        self.n_speeds = len(self.cfg.screen.speeds)
        self.n_slots = self.cfg.screen.max_visible_slots

        S, K = self.n_speeds, self.n_slots
        self.down_actions = list(range(S))
        self.up_actions = list(range(S, 2 * S))
        self.stay_action = 2 * S
        self.slot_actions = list(range(2 * S + 1, 2 * S + 1 + K))
        self.back_action = 2 * S + K + 1
        self.n_actions = 2 * S + K + 2
        self.action_space = spaces.Discrete(self.n_actions)

        self._n_global = 3 + 3 + self.n_actions + 1
        obs_dim = (
            self._n_global
            + N_CURRENT_FEATURES
            + K * N_SLOT_FEATURES
            + self.n_items * N_MEMORY_FEATURES
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng_feed: np.random.Generator | None = None
        self._rng_ddm: np.random.Generator | None = None

    # --- episode ---------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None or self._rng_feed is None:
            feed_ss, ddm_ss = np.random.SeedSequence(seed).spawn(2)
            self._rng_feed = np.random.default_rng(feed_ss)
            self._rng_ddm = np.random.default_rng(ddm_ss)

        feed = (options or {}).get("feed") or self._fixed_feed
        if feed is None:
            feed = generate_feed(self.cfg.feed, self.n_items, self._rng_feed)
        elif feed.n != self.n_items:
            raise ValueError(f"feed has {feed.n} items, config expects {self.n_items}")
        self.feed = feed

        n = self.n_items
        self.scroll = 0.0
        self.mode = LIST
        self.current: int | None = None
        self._transition_left = 0
        self._transition_to = LIST
        self.x = np.zeros(n)
        self.understood = np.zeros(n, dtype=bool)
        self.exposure = np.zeros(n)
        self.w = np.zeros(n)
        self.step_count = 0
        self.time_by_mode = {name: 0.0 for name in MODE_NAMES}
        self.watch_reward_total = 0.0
        self._prev_action: int | None = None
        self._prev_reward = 0.0
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        invalid = not self.action_masks()[action]
        if invalid:
            action = self.stay_action
        dt = self.cfg.time.dt

        if self.mode == LIST and action in self.slot_actions:
            self.current = int(self.visible_slots()[self.slot_actions.index(action)])
            self._start_transition(WATCH)
        elif self.mode == WATCH and action == self.back_action:
            self._start_transition(LIST)

        reward_watch = 0.0
        if self.mode == TRANSITION:
            self._transition_left -= 1
            if self._transition_left == 0:
                self.mode = self._transition_to
            self.time_by_mode["TRANSITION"] += dt
        elif self.mode == LIST:
            self._advance_list(self._velocity(action), dt)
            self.time_by_mode["LIST"] += dt
        else:
            i = self.current
            reward_watch = float(watch_reward(self.feed.A[i], self.feed.lam[i], self.w[i], dt))
            self.w[i] += dt
            self.watch_reward_total += reward_watch
            self.time_by_mode["WATCH"] += dt
        if self.mode == LIST:
            self.current = None

        reward = reward_watch - self.cfg.time.step_cost
        self.step_count += 1
        self._prev_action = action
        self._prev_reward = reward
        terminated = self.step_count >= self.cfg.time.n_steps
        info = self._info()
        info["invalid_action"] = invalid
        return self._obs(), reward, terminated, False, info

    # --- dynamics --------------------------------------------------------------

    def _start_transition(self, target: int) -> None:
        self.mode = TRANSITION
        self._transition_to = target
        self._transition_left = self.cfg.time.transition_steps

    def _velocity(self, action: int) -> float:
        """Scroll velocity in cards / s (positive = down the list)."""
        speeds = self.cfg.screen.speeds
        if action in self.down_actions:
            return speeds[action] * self.screen_h
        if action in self.up_actions:
            return -speeds[action - self.n_speeds] * self.screen_h
        return 0.0

    def _advance_list(self, velocity: float, dt: float) -> None:
        """Move and run the DDM of visible items in inner steps of dt_internal."""
        ddm, scr = self.cfg.ddm, self.cfg.screen
        n_sub = max(int(round(dt / ddm.dt_internal)), 1)
        h = dt / n_sub
        for _ in range(n_sub):
            new_scroll = float(np.clip(self.scroll + velocity * h, 0.0, self.max_scroll))
            speed = abs(new_scroll - self.scroll) / h / self.screen_h  # actual, 0 when blocked
            self.scroll = new_scroll
            vis = visible_items(self.scroll, self.n_items, self.screen_h, scr.visible_fraction)
            self.exposure[vis] += h
            active = vis[~self.understood[vis]]
            if active.size == 0:
                continue
            drift = self.feed.nu[active] * ddm.speed_gain(speed)
            self.x[active] = advance_evidence(self.x[active], drift, ddm.noise, h, self._rng_ddm)
            self.understood[active[self.x[active] >= ddm.threshold]] = True

    # --- masks and observation -------------------------------------------------

    def visible_slots(self) -> np.ndarray:
        """Item ids in the visible slots, top to bottom."""
        vis = visible_items(self.scroll, self.n_items, self.screen_h, self.cfg.screen.visible_fraction)
        return vis[: self.n_slots]

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.stay_action] = True
        if self.mode == WATCH:
            mask[self.back_action] = True
        elif self.mode == LIST:
            mask[self.down_actions + self.up_actions] = True
            for j, i in enumerate(self.visible_slots()):
                mask[self.slot_actions[j]] = self.understood[i]
        return mask

    def observe(self) -> np.ndarray:
        """Current observation (the same vector reset / step return)."""
        return self._obs()

    def revealed_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """(A, lambda) as the agent knows them: revealed for understood items, 0 otherwise."""
        return np.where(self.understood, self.feed.A, 0.0), np.where(self.understood, self.feed.lam, 0.0)

    def extra_metrics(self) -> dict[str, float]:
        """Episode counters of optional model extensions (empty in the MVP)."""
        return {}

    def _obs(self) -> np.ndarray:
        n, H, B = self.n_items, self.screen_h, self.cfg.ddm.threshold
        A_known, lam_known = self.revealed_profile()
        x_rel = np.minimum(self.x / B, 1.0)

        g = np.zeros(self._n_global)
        g[self.mode] = 1.0
        g[3] = self._transition_left / max(self.cfg.time.transition_steps, 1)
        g[4] = self.scroll / self.max_scroll if self.max_scroll > 0 else 0.0
        g[5] = 1.0 - self.step_count / self.cfg.time.n_steps
        if self._prev_action is not None:
            g[6 + self._prev_action] = 1.0
        g[6 + self.n_actions] = self._prev_reward

        cur = np.zeros(N_CURRENT_FEATURES)
        if self.current is not None:
            i = self.current
            cur[:] = A_known[i], lam_known[i], self.w[i], A_known[i] * np.exp(-lam_known[i] * self.w[i])

        slots = np.zeros((self.n_slots, N_SLOT_FEATURES))
        for j, i in enumerate(self.visible_slots()):
            slots[j] = (
                1.0, i / n, (i - self.scroll) / H, x_rel[i], self.exposure[i],
                self.understood[i], A_known[i], lam_known[i], self.w[i],
            )

        centre_offset = (np.arange(n) + 0.5 - self.scroll - H / 2) / H
        memory = np.column_stack([self.understood, A_known, lam_known, self.w, x_rel, centre_offset])
        return np.concatenate([g, cur, slots.ravel(), memory.ravel()]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "mode": MODE_NAMES[self.mode],
            "time_by_mode": dict(self.time_by_mode),
            "n_understood": int(self.understood.sum()),
            "watch_reward": self.watch_reward_total,
        }
