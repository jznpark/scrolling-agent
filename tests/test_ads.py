"""Ad-slots extension: sponsored cards in the feed (off by default)."""

import numpy as np
import pytest

from scrolling_agent.config import AdsConfig, Config
from scrolling_agent.env import WATCH, ScrollingEnv
from scrolling_agent.feed import generate_feed, insert_ads

from helpers import click, run_until


def make_env(**ads) -> ScrollingEnv:
    cfg = Config()
    cfg.ads = AdsConfig(**ads)
    return ScrollingEnv(cfg)


# --- feed ----------------------------------------------------------------------------


def test_insert_ads_positions_and_values():
    feed = generate_feed(Config().feed, 100, np.random.default_rng(0))
    with_ads = insert_ads(feed, every=5, m_nu_ad=0.3)
    pos = np.arange(4, 100, 5)
    assert with_ads.ads.sum() == 20 and with_ads.ads[pos].all()
    assert (with_ads.A[pos] == 0.0).all() and (with_ads.z[pos] == 0).all()
    assert np.allclose(with_ads.nu[pos], np.exp(0.3))
    keep = ~with_ads.ads
    for name in ("z", "A", "nu", "lam"):
        assert (getattr(with_ads, name)[keep] == getattr(feed, name)[keep]).all()


def test_insert_ads_off_and_errors():
    feed = generate_feed(Config().feed, 50, np.random.default_rng(1))
    assert insert_ads(feed, every=0) is feed
    assert feed.is_ad is None and not feed.ads.any()
    with pytest.raises(ValueError):
        insert_ads(feed, every=-1)


def test_reindex_keeps_ads_with_their_items():
    feed = insert_ads(generate_feed(Config().feed, 30, np.random.default_rng(2)), every=3)
    order = np.random.default_rng(3).permutation(30)
    moved = feed.reindex(order)
    assert (moved.ads == feed.ads[order]).all()
    assert (moved.A[moved.ads] == 0.0).all()


# --- environment ---------------------------------------------------------------------


def test_default_off_matches_draft_feed():
    env_off, env_ads = make_env(), make_env(every=5)
    env_off.reset(seed=123)
    env_ads.reset(seed=123)
    assert env_off.feed.is_ad is None and not env_off.understood.any()
    assert env_off.extra_metrics() == {"n_ads": 0, "n_ad_clicks": 0, "time_watch_ads": 0.0}
    keep = ~env_ads.feed.ads
    for name in ("z", "A", "nu", "lam"):  # same seed: non-ad items are the draft feed
        assert (getattr(env_ads.feed, name)[keep] == getattr(env_off.feed, name)[keep]).all()


def test_disclosed_ads_are_understood_at_reset():
    env = make_env(every=5, disclosed=True)
    env.reset(seed=0)
    assert (env.understood == env.feed.ads).all()
    env = make_env(every=5, disclosed=False)
    env.reset(seed=0)
    assert not env.understood.any()


def test_watching_an_ad_gives_only_the_step_cost():
    env = make_env(every=2, disclosed=True)  # ad at index 1, visible at reset
    env.reset(seed=0)
    assert env.feed.ads[1] and 1 in env.visible_slots()
    click(env, 1)
    rewards = [env.step(env.stay_action)[1] for _ in range(5)]
    assert env.mode == WATCH and env.current == 1
    assert all(r == -env.cfg.time.step_cost for r in rewards)
    metrics = env.extra_metrics()
    assert metrics["n_ad_clicks"] == 1
    assert metrics["time_watch_ads"] == pytest.approx(5 * env.cfg.time.dt)
    assert env.watch_reward_total == 0.0


def test_time_still_sums_to_session(seed=7):
    env = make_env(every=5, disclosed=True)
    env.reset(seed=seed)
    run_until(env, env.stay_action, lambda e: e.step_count >= e.cfg.time.n_steps)
    assert sum(env.time_by_mode.values()) == pytest.approx(env.cfg.time.session_seconds)
