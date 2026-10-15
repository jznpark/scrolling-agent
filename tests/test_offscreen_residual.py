"""Extension: off-screen residual processing (DDMConfig.tau_residual)."""

from dataclasses import replace

import numpy as np
import pytest
from helpers import click, constant_feed, run_until

from scrolling_agent import Config, ScrollingEnv
from scrolling_agent.env import residual_weight
from scrolling_agent.evaluation import evaluate, summarize
from scrolling_agent.policies import GreedyKnown


def residual(tau=2.0, in_watch=False, **ddm) -> Config:
    cfg = Config()
    cfg.ddm = replace(cfg.ddm, tau_residual=tau, residual_in_watch=in_watch, **ddm)
    return cfg


def test_residual_weight_telescopes():
    tau, dt = 1.5, 0.1
    edges = np.arange(0, 50.0 + dt, dt)
    total = sum(residual_weight(np.array(a), np.array(b), tau) for a, b in zip(edges[:-1], edges[1:]))
    assert total == pytest.approx(tau * (1 - np.exp(-50.0 / tau)))


def test_off_by_default_matches_mvp_exactly():
    envs = [ScrollingEnv(), ScrollingEnv(residual(tau=0.0))]
    for e in envs:
        e.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        a = int(rng.choice(np.flatnonzero(envs[0].action_masks())))
        for e in envs:
            e.step(a)
    np.testing.assert_array_equal(envs[0].x, envs[1].x)
    assert envs[1].extra_metrics() == {"n_off_screen_understood": 0, "n_hidden_understood": 0,
                                       "n_late_clicks": 0}


@pytest.mark.parametrize("tau", [0.5, 2.0])
def test_extra_evidence_mean_and_variance(tau):
    """Long after leaving: E[dx] = nu g tau, Var[dx] = sigma^2 tau (threshold out of reach)."""
    cfg = residual(tau=tau, threshold=1e9)
    env = ScrollingEnv(cfg, feed=constant_feed(nu=1.2))
    gains = []
    for seed in range(60):
        env.reset(seed=seed)
        env.last_visible[:] = 0.0
        env.gain_last[:] = 0.8
        x0 = env.x.copy()
        for k in range(300):  # 30 s off screen
            env._advance_residual(k * 0.1, (k + 1) * 0.1, exclude=np.array([], dtype=int))
        gains.append(env.x - x0)
    gains = np.concatenate(gains)
    assert gains.mean() == pytest.approx(1.2 * 0.8 * tau, rel=0.05)
    assert gains.var() == pytest.approx(cfg.ddm.noise**2 * tau, rel=0.1)


def test_evidence_keeps_growing_after_leaving_then_settles():
    cfg = residual(tau=1.0)
    env = ScrollingEnv(cfg, feed=constant_feed(nu=0.6))
    env.reset(seed=2)
    for _ in range(3):  # 0.3 s on screen
        env.step(env.stay_action)
    run_until(env, env.down_actions[0], lambda e: e.scroll > 2.6)  # items 0-2 off screen
    x_left = env.x[:3].copy()
    for _ in range(40):
        env.step(env.down_actions[0])
    moved = env.x[:3] != x_left
    assert moved.all() or env.understood[:3].any()
    x_mid = env.x[:3].copy()
    for _ in range(300):  # 30 s later the weight is ~e^-30: practically frozen
        env.step(env.stay_action)
    assert np.abs(env.x[:3] - x_mid).max() < 1.0


def test_unseen_items_get_no_residual():
    env = ScrollingEnv(residual(tau=3.0))
    env.reset(seed=0)
    for _ in range(50):
        env.step(env.stay_action)
    assert not env.x[10:].any()


@pytest.mark.parametrize("in_watch", [False, True])
def test_residual_while_watching_is_optional(in_watch):
    env = ScrollingEnv(residual(tau=3.0, in_watch=in_watch), feed=constant_feed(nu=0.8))
    env.reset(seed=5)
    run_until(env, env.stay_action, lambda e: e.understood[:3].any())
    item = int(np.flatnonzero(env.understood[:3])[0])
    click(env, item)
    x0 = env.x.copy()
    for _ in range(10):
        env.step(env.stay_action)
    changed = (env.x != x0).any()
    assert changed == in_watch


def test_greedy_goes_back_to_items_understood_off_screen():
    """Case 2: realised late, the agent scrolls back (only with residual processing)."""
    seeds = range(10_000, 10_030)
    base = summarize(evaluate(ScrollingEnv(), GreedyKnown(0.6), seeds))
    late = summarize(evaluate(ScrollingEnv(residual(tau=2.0)), GreedyKnown(0.6), seeds))
    assert base["n_off_screen_understood"][0] == 0 and base["n_late_clicks"][0] == 0
    assert late["n_off_screen_understood"][0] > 1
    assert late["n_late_clicks"][0] > 0
    assert late["n_hidden_understood"][0] == 0  # residual_in_watch is off


def test_scroll_first_never_counts_late_clicks():
    """It never scrolls up, so it cannot go back to an item understood off screen."""
    from scrolling_agent.policies import ScrollFirst

    cfg = residual(tau=2.0, in_watch=True)
    stats = summarize(evaluate(ScrollingEnv(cfg), ScrollFirst(0.6), range(10_000, 10_010)))
    assert stats["n_late_clicks"][0] == 0
    assert stats["n_hidden_understood"][0] > 0
