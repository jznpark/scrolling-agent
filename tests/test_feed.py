import math
from dataclasses import replace

import numpy as np
import pytest

from scrolling_agent import FeedConfig, feed_stats, generate_feed, reorder_feed, sample_types

N_LARGE = 200_000


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def expected_persistence(p: float, kappa: float) -> float:
    return kappa + (1 - kappa) * (p**2 + (1 - p) ** 2)


def item_set(feed) -> list[tuple]:
    return sorted(zip(feed.z.tolist(), feed.A.tolist(), feed.nu.tolist(), feed.lam.tolist()))


# --- reproducibility -------------------------------------------------------


def test_same_seed_same_feed():
    cfg = FeedConfig(kappa=0.5, d_nu=-0.3)
    f1, f2 = generate_feed(cfg, 100, rng(42)), generate_feed(cfg, 100, rng(42))
    for name in ("z", "A", "nu", "lam"):
        np.testing.assert_array_equal(getattr(f1, name), getattr(f2, name))


def test_different_seed_different_feed():
    cfg = FeedConfig()
    assert not np.array_equal(generate_feed(cfg, 100, rng(1)).A, generate_feed(cfg, 100, rng(2)).A)


def test_reorder_reproducible():
    feed = generate_feed(FeedConfig(), 100, rng(0))
    r1, r2 = reorder_feed(feed, 0.7, rng(5)), reorder_feed(feed, 0.7, rng(5))
    np.testing.assert_array_equal(r1.A, r2.A)


# --- distribution ----------------------------------------------------------


def test_values_positive_and_shapes():
    feed = generate_feed(FeedConfig(), 100, rng(0))
    assert feed.n == 100
    assert feed.z.dtype == np.int8 and set(np.unique(feed.z)) <= {0, 1}
    assert (feed.A > 0).all() and (feed.nu > 0).all() and (feed.lam > 0).all()


@pytest.mark.parametrize("kappa", [0.0, 0.8])
def test_promising_share_matches_p(kappa):
    feed = generate_feed(FeedConfig(p=0.3, kappa=kappa), N_LARGE, rng(0))
    assert feed_stats(feed)["promising_share"] == pytest.approx(0.3, abs=0.01)


def test_log_parameters_by_type():
    cfg = FeedConfig(d_nu=-0.5)
    feed = generate_feed(cfg, N_LARGE, rng(0))
    one, zero = feed.z == 1, feed.z == 0
    logA, lognu, loglam = np.log(feed.A), np.log(feed.nu), np.log(feed.lam)

    assert logA[zero].mean() == pytest.approx(cfg.a0, abs=0.01)
    assert logA[one].mean() == pytest.approx(cfg.a0 + cfg.a1, abs=0.01)
    assert logA[zero].std() == pytest.approx(cfg.sigma_A, abs=0.01)
    assert lognu[one].mean() - lognu[zero].mean() == pytest.approx(cfg.d_nu, abs=0.01)
    assert lognu[zero].std() == pytest.approx(cfg.sigma_nu, abs=0.01)
    # lambda does not depend on the type
    assert loglam[one].mean() == pytest.approx(cfg.m_lambda, abs=0.01)
    assert loglam[zero].mean() == pytest.approx(cfg.m_lambda, abs=0.01)
    assert loglam.std() == pytest.approx(cfg.sigma_lambda, abs=0.01)


@pytest.mark.parametrize("kappa", [0.0, 0.5, 0.9])
def test_type_persistence_matches_kappa(kappa):
    feed = generate_feed(FeedConfig(p=0.3, kappa=kappa), N_LARGE, rng(0))
    assert feed_stats(feed)["type_persistence"] == pytest.approx(expected_persistence(0.3, kappa), abs=0.01)


def test_nu_A_correlation_follows_d_nu():
    base = FeedConfig(p=0.3)
    neg = feed_stats(generate_feed(replace(base, d_nu=-0.5), N_LARGE, rng(0)))
    zero = feed_stats(generate_feed(replace(base, d_nu=0.0), N_LARGE, rng(0)))

    # analytic correlation of log nu and log A under the generator
    var_z = base.p * (1 - base.p)
    cov = base.a1 * -0.5 * var_z
    var_A = base.a1**2 * var_z + base.sigma_A**2
    var_nu = 0.25 * var_z + base.sigma_nu**2
    assert neg["nu_A_corr_log"] == pytest.approx(cov / math.sqrt(var_A * var_nu), abs=0.01)
    assert neg["nu_A_corr_rank"] < 0
    assert zero["nu_A_corr_log"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("p, expected", [(0.0, 0), (1.0, 1)])
def test_extreme_p(p, expected):
    feed = generate_feed(FeedConfig(p=p, kappa=0.5), 50, rng(0))
    assert (feed.z == expected).all()


@pytest.mark.parametrize("kwargs", [{"kappa": 1.0}, {"kappa": -0.1}, {"p": 1.5}, {"a1": 0.0}])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        generate_feed(FeedConfig(**kwargs), 10, rng(0))


def test_invalid_n_items_raises():
    with pytest.raises(ValueError):
        generate_feed(FeedConfig(), 0, rng(0))


def test_sample_types_single_item():
    assert sample_types(1, 1.0, 0.5, rng(0)).tolist() == [1]


# --- reordering ------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.5, 0.95])
def test_reorder_keeps_item_set(kappa):
    feed = generate_feed(FeedConfig(p=0.3), 100, rng(0))
    reordered = reorder_feed(feed, kappa, rng(1))
    assert item_set(reordered) == item_set(feed)
    assert reordered.z.sum() == feed.z.sum()


def test_reorder_kappa_controls_clustering():
    feed = generate_feed(FeedConfig(p=0.3, kappa=0.0), 2000, rng(0))
    p_hat = feed.z.mean()
    low = np.mean([feed_stats(reorder_feed(feed, 0.0, rng(s)))["type_persistence"] for s in range(5)])
    high = np.mean([feed_stats(reorder_feed(feed, 0.9, rng(s)))["type_persistence"] for s in range(5)])
    assert low == pytest.approx(expected_persistence(p_hat, 0.0), abs=0.02)
    assert high == pytest.approx(expected_persistence(p_hat, 0.9), abs=0.02)


@pytest.mark.parametrize("n_ones", [0, 100])
def test_reorder_all_same_type(n_ones):
    feed = generate_feed(FeedConfig(p=n_ones / 100), 100, rng(0))
    assert reorder_feed(feed, 0.5, rng(1)).z.sum() == n_ones


def test_reorder_invalid_kappa_raises():
    feed = generate_feed(FeedConfig(), 10, rng(0))
    with pytest.raises(ValueError):
        reorder_feed(feed, 1.0, rng(0))


# --- stats -----------------------------------------------------------------


def test_feed_stats_hand_built():
    from scrolling_agent import Feed

    feed = Feed(
        z=np.array([1, 1, 0, 0], dtype=np.int8),
        A=np.array([4.0, 3.0, 2.0, 1.0]),
        nu=np.array([1.0, 2.0, 3.0, 4.0]),
        lam=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    stats = feed_stats(feed)
    assert stats["n"] == 4
    assert stats["promising_share"] == 0.5
    assert stats["type_persistence"] == pytest.approx(2 / 3)
    assert stats["nu_A_corr_rank"] == pytest.approx(-1.0)
    assert stats["nu_A_corr_log"] < -0.9
    assert stats["mean_lam"] == pytest.approx(0.25)
