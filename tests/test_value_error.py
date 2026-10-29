"""Extension: value estimation error (EstimateConfig)."""

import math

import numpy as np
import pytest
from helpers import click, constant_feed, random_masked_actions, run_until

from scrolling_agent import Config, ScrollingEnv
from scrolling_agent.config import EstimateConfig
from scrolling_agent.evaluation import evaluate, summarize
from scrolling_agent.policies import GreedyKnown


def noisy(sigma=0.5, tau=0.0, noise_lambda=True) -> Config:
    return Config(estimate=EstimateConfig(sigma_est=sigma, tau_learn=tau, noise_lambda=noise_lambda))


def test_off_by_default_reveals_truth():
    env = ScrollingEnv()
    env.reset(seed=0)
    env.understood[:] = True
    A, lam = env.revealed_profile()
    np.testing.assert_array_equal(A, env.feed.A)
    np.testing.assert_array_equal(lam, env.feed.lam)


def test_feed_and_ddm_streams_unchanged():
    """The error has its own seed stream: same actions -> same evidence and flags as the MVP."""
    envs = [ScrollingEnv(), ScrollingEnv(noisy(0.8))]
    for e in envs:
        e.reset(seed=4)
    np.testing.assert_array_equal(envs[0].feed.A, envs[1].feed.A)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = int(rng.choice(np.flatnonzero(envs[0].action_masks() & envs[1].action_masks())))
        for e in envs:
            e.step(a)
    np.testing.assert_array_equal(envs[0].x, envs[1].x)
    np.testing.assert_array_equal(envs[0].understood, envs[1].understood)


@pytest.mark.parametrize("sigma", [0.3, 0.8])
def test_log_error_distribution(sigma):
    errs_A, errs_lam = [], []
    env = ScrollingEnv(noisy(sigma))
    for seed in range(40):
        env.reset(seed=seed)
        A_hat, lam_hat = env.estimated_profile()
        errs_A.append(np.log(A_hat / env.feed.A))
        errs_lam.append(np.log(lam_hat / env.feed.lam))
    errs_A, errs_lam = np.concatenate(errs_A), np.concatenate(errs_lam)
    assert errs_A.mean() == pytest.approx(0.0, abs=0.05)
    assert errs_A.std() == pytest.approx(sigma, rel=0.05)
    assert errs_lam.std() == pytest.approx(sigma, rel=0.05)
    assert abs(np.corrcoef(errs_A, errs_lam)[0, 1]) < 0.05


def test_noise_lambda_off_keeps_lambda_exact():
    env = ScrollingEnv(noisy(0.5, noise_lambda=False))
    env.reset(seed=0)
    _, lam_hat = env.estimated_profile()
    np.testing.assert_array_equal(lam_hat, env.feed.lam)


def test_unknown_items_stay_hidden():
    env = ScrollingEnv(noisy(0.5))
    env.reset(seed=0)
    A, lam = env.revealed_profile()
    assert not A[~env.understood].any() and not lam[~env.understood].any()


def watch_steps(env, k):
    return [env.step(env.stay_action)[1] for _ in range(k)]


def test_truth_known_after_first_watched_step_and_reward_is_true():
    cfg = noisy(0.8)
    A, lam = 2.0, 0.4
    env = ScrollingEnv(cfg, feed=constant_feed(A=A, lam=lam))
    env.reset(seed=1)
    run_until(env, env.stay_action, lambda e: e.understood[0])
    perceived = env.revealed_profile()[0][0]
    assert not np.isclose(perceived, A)  # noisy at understanding
    click(env, 0)
    assert np.isclose(env.revealed_profile()[0][0], perceived)  # still noisy before watching
    rewards = watch_steps(env, 10)
    assert env.revealed_profile()[0][0] == A  # exact once watched
    watched = sum(rewards) + 10 * cfg.time.step_cost
    assert watched == pytest.approx((A / lam) * (1 - math.exp(-lam * 1.0)))  # true reward


def test_error_fades_with_tau_learn():
    tau = 2.0
    env = ScrollingEnv(noisy(0.8, tau=tau), feed=constant_feed(A=2.0))
    env.reset(seed=1)
    run_until(env, env.stay_action, lambda e: e.understood[0])
    err0 = np.log(env.revealed_profile()[0][0] / 2.0)
    click(env, 0)
    watch_steps(env, 20)  # w = 2 s
    err = np.log(env.revealed_profile()[0][0] / 2.0)
    assert err == pytest.approx(err0 * math.exp(-2.0 / tau))


def test_click_metrics():
    env = ScrollingEnv(noisy(0.0))
    env.reset(seed=0)
    random_masked_actions(env, np.random.default_rng(0), 400)
    assert env.extra_metrics() == {"n_overestimated_clicks": 0, "click_log_error": 0.0}


def test_noise_makes_greedy_open_and_leave_more_often():
    """Overestimated items are opened, found worse, and left at once (case 3)."""
    seeds = range(10_000, 10_020)
    exact = summarize(evaluate(ScrollingEnv(), GreedyKnown(0.6), seeds))
    noisy_ = summarize(evaluate(ScrollingEnv(noisy(0.6)), GreedyKnown(0.6), seeds))
    assert noisy_["n_short_bouts"][0] > exact["n_short_bouts"][0]
    assert noisy_["click_log_error"][0] > 0  # clicked items looked better than they were
    assert noisy_["return"][0] < exact["return"][0]
