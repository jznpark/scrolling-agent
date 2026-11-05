"""All optional extensions switched on together still give a consistent episode."""
from dataclasses import replace

import numpy as np

from scrolling_agent import ScrollingEnv
from scrolling_agent.config import Config
from scrolling_agent.policies import GreedyKnown


def all_extensions() -> Config:
    c = Config()
    return replace(
        c,
        screen=replace(c.screen, columns=2),
        ddm=replace(c.ddm, tau_residual=1.0, attention="log"),
        estimate=replace(c.estimate, sigma_est=0.3),
        preview=replace(c.preview, enabled=True),
        ads=replace(c.ads, every=5),
    )


def test_all_extensions_run_a_full_episode_with_finite_metrics():
    cfg = all_extensions()
    env = ScrollingEnv(cfg)
    policy = GreedyKnown()
    env.reset(seed=3)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(policy(env))
        done = terminated or truncated
    assert np.isclose(sum(env.time_by_mode.values()), cfg.time.session_seconds)
    metrics = env.extra_metrics()
    assert all(np.isfinite(v) for v in metrics.values()), metrics
    assert metrics["n_ads"] > 0


def test_clicking_an_ad_does_not_break_the_value_error_metric():
    env = ScrollingEnv(all_extensions())
    env.reset(seed=0)
    ad = int(np.flatnonzero(env.feed.ads)[0])
    env._record_click(ad)
    assert env.extra_metrics()["click_log_error"] == 0.0
