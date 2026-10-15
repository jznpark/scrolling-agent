"""Extension: attention split across visible unread items (DDMConfig.attention)."""

import math
from dataclasses import replace

import numpy as np
import pytest
from helpers import constant_feed

from scrolling_agent import Config, DDMConfig, ScrollingEnv
from scrolling_agent.evaluation import evaluate, summarize
from scrolling_agent.policies import GreedyKnown


def attention(mode="log", beta=1.0, **ddm) -> Config:
    cfg = Config()
    cfg.ddm = replace(cfg.ddm, attention=mode, attention_beta=beta, **ddm)
    return cfg


@pytest.mark.parametrize(
    "mode, beta, n, expected",
    [("none", 1.0, 3, 1.0), ("log", 1.0, 0, 1.0), ("log", 1.0, 1, 1.0), ("log", 1.0, 3, 0.5),
     ("log", 1.0, 2, 1 / math.log2(3)), ("power", 1.0, 4, 0.25), ("power", 0.5, 4, 0.5), ("power", 2.0, 1, 1.0)],
)
def test_attention_gain(mode, beta, n, expected):
    assert DDMConfig(attention=mode, attention_beta=beta).attention_gain(n) == pytest.approx(expected)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        DDMConfig(attention="sqrt")


def test_off_by_default_matches_mvp():
    envs = [ScrollingEnv(), ScrollingEnv(attention("none"))]
    for e in envs:
        e.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(400):
        a = int(rng.choice(np.flatnonzero(envs[0].action_masks())))
        for e in envs:
            e.step(a)
    np.testing.assert_array_equal(envs[0].x, envs[1].x)


def test_drift_is_shared_and_recovers_when_items_are_understood():
    """No noise: three items on screen share attention; once one is understood the
    other two speed up (power, beta = 1)."""
    feed = constant_feed(nu=1.0)
    feed.nu[0] = 3.0
    env = ScrollingEnv(attention("power", noise=0.0), feed=feed)
    env.reset(seed=0)
    for _ in range(5):  # 0.5 s, N = 3
        env.step(env.stay_action)
    np.testing.assert_allclose(env.x[:3], np.array([3.0, 1.0, 1.0]) * 0.5 / 3, rtol=1e-9)
    for _ in range(5):  # item 0 reaches B = 1 at t = 1 s
        env.step(env.stay_action)
    assert env.understood[0]
    for _ in range(10):  # 1 s more with N = 2
        env.step(env.stay_action)
    assert env.x[1] == pytest.approx(1.0 / 3 + 0.5, abs=0.01)


def test_mean_unread_metric():
    env = ScrollingEnv(feed=constant_feed(nu=1e-6))  # nothing gets understood
    env.reset(seed=0)
    for _ in range(10):
        env.step(env.stay_action)
    assert env.extra_metrics()["mean_unread_on_screen"] == pytest.approx(3.0)


def test_crowding_slows_understanding_and_lengthens_stops():
    """With a split, a stop-and-look policy stays longer and understands fewer items."""
    seeds = range(10_000, 10_015)
    policy = GreedyKnown(0.6, dwell=1.5)
    off = summarize(evaluate(ScrollingEnv(), policy, seeds))
    on = summarize(evaluate(ScrollingEnv(attention("log")), policy, seeds))
    assert on["time_stay_list"][0] > off["time_stay_list"][0]
    assert on["n_understood"][0] < off["n_understood"][0]
    assert on["mean_unread_on_screen"][0] > off["mean_unread_on_screen"][0]
