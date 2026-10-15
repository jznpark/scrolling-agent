import math

import numpy as np
import pytest
from helpers import constant_feed, run_until

from scrolling_agent import Config, Feed, ScrollingEnv
from scrolling_agent.env import WATCH
from scrolling_agent.evaluation import evaluate, run_episode, summarize
from scrolling_agent.policies import GreedyKnown, RandomPolicy, ScrollFirst, reference_return

SEEDS = range(100, 106)


@pytest.mark.parametrize("policy", [RandomPolicy(0), ScrollFirst(0.3), GreedyKnown(0.3)], ids=str)
def test_policies_never_choose_invalid_actions(policy):
    env = ScrollingEnv()
    metrics, _ = run_episode(env, policy, seed=1)
    assert metrics["invalid_actions"] == 0


def test_baselines_beat_random():
    env = ScrollingEnv()
    rand = summarize(evaluate(env, RandomPolicy(0), SEEDS))["return"][0]
    for policy in (ScrollFirst(0.3), GreedyKnown(0.3)):
        assert summarize(evaluate(env, policy, SEEDS))["return"][0] > rand


def test_scroll_first_never_scrolls_up():
    env = ScrollingEnv()
    policy = ScrollFirst(0.3)
    env.reset(seed=0)
    for _ in range(env.cfg.time.n_steps):
        a = policy(env)
        assert a not in env.up_actions
        env.step(a)


def test_greedy_goes_back_to_a_known_item_off_screen():
    """Item 0 is understood and valuable but off screen: greedy scrolls up and watches it."""
    feed = constant_feed(A=0.2)
    feed.A[0] = 5.0
    env = ScrollingEnv(feed=feed)
    env.reset(seed=0)
    run_until(env, env.stay_action, lambda e: e.understood[0])
    for _ in range(40):
        env.step(env.down_actions[0])
    assert 0 not in env.visible_slots()

    policy = GreedyKnown(rho=0.6)
    actions = []
    while env.mode != WATCH:
        actions.append(policy(env))
        env.step(actions[-1])
    assert env.current == 0 and env.up_actions[0] in actions


def test_greedy_dwell_stops_until_items_are_understood():
    env = ScrollingEnv(feed=constant_feed(A=0.2))  # nothing worth watching
    env.reset(seed=0)
    policy = GreedyKnown(rho=0.6, dwell=1.5)
    # at the start nothing is understood: it stays instead of scrolling
    assert policy(env) == env.stay_action
    while policy(env) == env.stay_action:
        env.step(env.stay_action)
    vis = env.visible_slots()
    assert all(env.understood[i] or env.exposure[i] >= 1.5 for i in vis)
    assert GreedyKnown(rho=0.6)(env) == env.down_actions[0]


def test_reference_return_single_item_closed_form():
    cfg = Config()
    feed = Feed(z=np.zeros(1, np.int8), A=np.array([2.0]), nu=np.ones(1), lam=np.array([0.1]))
    T, c = cfg.time.session_seconds, cfg.time.step_cost
    expected = (2.0 / 0.1) * (1 - math.exp(-0.1 * T)) - cfg.time.n_steps * c
    assert reference_return(feed, cfg) == pytest.approx(expected)


def test_reference_return_bounds_policies():
    env = ScrollingEnv()
    for s in SEEDS:
        metrics, _ = run_episode(env, GreedyKnown(0.3), seed=s)
        assert metrics["return"] <= reference_return(env.feed, env.cfg)
