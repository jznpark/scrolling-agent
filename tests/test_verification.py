"""Implementation checks from draft section 10 (step 3)."""

import numpy as np
import pytest
from helpers import click, constant_feed, random_masked_actions, run_until

from scrolling_agent import Config, ScrollingEnv, TimeConfig
from scrolling_agent.env import LIST, WATCH, advance_evidence, watch_reward

# --- 1. mean first-passage time ≈ B / nu ------------------------------------


@pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
def test_mean_first_passage_time(nu):
    """x(0) = 0, fixed exposure, positive drift, single upper threshold.

    Every trial runs until it hits (no early stop, no selection of trials that
    hit within an observation window). Discrete monitoring adds a small known
    overshoot (~0.58 sigma sqrt(dt) / nu), so the tolerance is 3 %.
    """
    ddm = Config().ddm
    rng = np.random.default_rng(0)
    n = 20_000
    x, t = np.zeros(n), np.zeros(n)
    alive = np.ones(n, dtype=bool)
    while alive.any():
        x[alive] = advance_evidence(x[alive], nu, ddm.noise, ddm.dt_internal, rng)
        t[alive] += ddm.dt_internal
        alive &= x < ddm.threshold
    assert t.mean() == pytest.approx(ddm.threshold / nu, rel=0.03)


def test_mean_understanding_time_in_env():
    """Same check through the environment: stay still, items 0–2 fully visible."""
    cfg = Config()
    env = ScrollingEnv(cfg, feed=constant_feed(nu=1.0))
    times = []
    for seed in range(1000):
        env.reset(seed=seed)
        first = np.full(3, np.nan)
        while np.isnan(first).any():
            env.step(env.stay_action)
            new = env.understood[:3] & np.isnan(first)
            first[new] = env.step_count * cfg.time.dt
        times.extend(first)
    # detection happens at the end of a step: expect B/nu plus about dt / 2
    expected = cfg.ddm.threshold / 1.0 + cfg.time.dt / 2
    assert np.mean(times) == pytest.approx(expected, rel=0.03)


# --- 2. time and memory ----------------------------------------------------


def test_time_sums_to_T_and_w_grows_only_while_watching():
    cfg = Config()
    env = ScrollingEnv(cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(cfg.time.n_steps):
        mode, w_before = env.mode, env.w.copy()
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, r, terminated, _, info = env.step(a)
        grew = env.w - w_before
        if mode == WATCH and a == env.stay_action:
            assert grew.sum() == pytest.approx(cfg.time.dt) and np.count_nonzero(grew) == 1
        else:
            assert not grew.any()
        if mode == LIST and env.mode == LIST:
            # understanding alone never gives reward or a click
            assert r == pytest.approx(-cfg.time.step_cost)

    assert terminated
    times = info["time_by_mode"]
    assert sum(times.values()) == pytest.approx(cfg.time.session_seconds)
    assert times["WATCH"] > 0 and env.w.sum() == pytest.approx(times["WATCH"])


def test_state_preserved_across_round_trip():
    cfg = Config()
    A, lam = 2.0, 0.4
    env = ScrollingEnv(cfg, feed=constant_feed(A=A, lam=lam))
    env.reset(seed=0)
    down, up = env.down_actions[0], env.up_actions[0]

    # understand item 0, watch it for 1 s, go back to the list
    run_until(env, env.stay_action, lambda e: e.understood[0])
    click(env, 0)
    for _ in range(10):
        env.step(env.stay_action)
    env.step(env.back_action)
    run_until(env, env.stay_action, lambda e: e.mode == LIST)
    w0 = env.w[0]

    # scroll until items 0–2 are off screen, then away and back
    run_until(env, down, lambda e: e.scroll > 2.5)
    snapshot = env.x.copy(), env.understood.copy(), env.w.copy()
    for _ in range(40):
        env.step(down)
    run_until(env, up, lambda e: e.scroll <= 2.5)
    np.testing.assert_array_equal(env.x[:3], snapshot[0][:3])
    np.testing.assert_array_equal(env.understood[:3], snapshot[1][:3])
    np.testing.assert_array_equal(env.w, snapshot[2])

    # back at the top: item 0 is still understood and watching resumes from w0
    run_until(env, up, lambda e: e.scroll == 0.0)
    assert env.understood[0] and env.w[0] == w0
    click(env, 0)
    _, r, *_ = env.step(env.stay_action)
    assert r == pytest.approx(watch_reward(A, lam, w0, cfg.time.dt) - cfg.time.step_cost)


def test_off_screen_evidence_frozen_during_random_play():
    env = ScrollingEnv()
    env.reset(seed=2)
    rng = np.random.default_rng(3)
    H = env.screen_h
    for _ in range(600):
        s0, x0 = env.scroll, env.x.copy()
        random_masked_actions(env, rng, 1)
        lo, hi = min(s0, env.scroll), max(s0, env.scroll) + H
        idx = np.arange(env.n_items)
        never_on_screen = (idx + 1 <= lo) | (idx >= hi)
        np.testing.assert_array_equal(env.x[never_on_screen], x0[never_on_screen])


# --- 3. fixed cost consistency ---------------------------------------------


def test_cost_change_shifts_return_by_N_delta_c():
    c, c2 = 0.01, 0.05
    envs = [ScrollingEnv(Config(time=TimeConfig(step_cost=cost))) for cost in (c, c2)]
    for e in envs:
        e.reset(seed=7)

    rng = np.random.default_rng(0)
    returns = [0.0, 0.0]
    for _ in range(envs[0].cfg.time.n_steps):
        masks = [e.action_masks() for e in envs]
        np.testing.assert_array_equal(*masks)  # same dynamics under both costs
        a = int(rng.choice(np.flatnonzero(masks[0])))
        for k, e in enumerate(envs):
            returns[k] += e.step(a)[1]

    N = envs[0].cfg.time.n_steps
    assert envs[0].watch_reward_total == envs[1].watch_reward_total
    assert returns[1] - returns[0] == pytest.approx(-N * (c2 - c))
