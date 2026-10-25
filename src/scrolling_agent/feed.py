"""Stochastic feed generator (step 1, draft section 06).

    z_1 ~ Bernoulli(p)  (the stationary share of the chain below)
    z_{i+1} = z_i with prob. kappa, otherwise z_{i+1} ~ Bernoulli(p)
    log A_i      = a0 + a1 * z_i + sigma_A * eps
    log nu_i     = m_nu + d_nu * z_i + sigma_nu * eps
    log lambda_i = m_lambda + sigma_lambda * eps

`reorder_feed` keeps a generated item set fixed and only changes its order
(how strongly same-type items cluster), so supply composition and ordering can
be compared separately. `feed_stats` records the realised feed, including the
nu–A relation (d_nu is not a correlation coefficient).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FeedConfig


@dataclass(frozen=True, eq=False)
class Feed:
    """A list of items in display order. `z` is hidden from the agent."""

    z: np.ndarray    # latent type, 0 or 1 (int8)
    A: np.ndarray    # initial reward rate
    nu: np.ndarray   # drift (ease of understanding the thumbnail)
    lam: np.ndarray  # reward decay rate (1/s)
    is_ad: np.ndarray | None = None  # ad-slots extension: sponsored cards (None = no ads)

    def __post_init__(self) -> None:
        n = len(self.z)
        extra = (self.is_ad,) if self.is_ad is not None else ()
        if any(len(a) != n for a in (self.A, self.nu, self.lam, *extra)):
            raise ValueError("z, A, nu and lam must have the same length")

    @property
    def n(self) -> int:
        return len(self.z)

    @property
    def ads(self) -> np.ndarray:
        """Boolean mask of ad positions (all False when the extension is off)."""
        return self.is_ad if self.is_ad is not None else np.zeros(self.n, dtype=bool)

    def reindex(self, order: np.ndarray) -> Feed:
        """The same items shown in `order` (order[k] = index of the item at position k)."""
        order = np.asarray(order)
        return Feed(z=self.z[order], A=self.A[order], nu=self.nu[order], lam=self.lam[order],
                    is_ad=None if self.is_ad is None else self.is_ad[order])


def _check_kappa(kappa: float) -> None:
    if not 0.0 <= kappa < 1.0:
        raise ValueError(f"kappa must satisfy 0 <= kappa < 1, got {kappa}")


def sample_types(n: int, p: float, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Latent types from the persistence chain; z_1 ~ Bernoulli(p)."""
    _check_kappa(kappa)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    keep = rng.random(n) < kappa
    keep[0] = False
    fresh = rng.random(n) < p
    # Each position copies the most recent fresh draw (the last index with keep == False).
    last_fresh = np.maximum.accumulate(np.where(keep, 0, np.arange(n)))
    return fresh[last_fresh].astype(np.int8)


def generate_feed(cfg: FeedConfig, n_items: int, rng: np.random.Generator) -> Feed:
    """Generate `n_items` items from the latent-type model."""
    if n_items < 1:
        raise ValueError(f"n_items must be >= 1, got {n_items}")
    if cfg.a1 <= 0:
        raise ValueError(f"a1 must be > 0, got {cfg.a1}")
    z = sample_types(n_items, cfg.p, cfg.kappa, rng)
    eps_A, eps_nu, eps_lam = rng.standard_normal((3, n_items))
    A = np.exp(cfg.a0 + cfg.a1 * z + cfg.sigma_A * eps_A)
    nu = np.exp(cfg.m_nu + cfg.d_nu * z + cfg.sigma_nu * eps_nu)
    lam = np.exp(cfg.m_lambda + cfg.sigma_lambda * eps_lam)
    return Feed(z=z, A=A, nu=nu, lam=lam)


def insert_ads(feed: Feed, every: int, m_nu_ad: float = 0.0) -> Feed:
    """Ad-slots extension: replace every `every`-th card with an ad.

    Ads give no watch reward (A = 0, z = 0) and all share the drift exp(m_nu_ad)
    (how quickly "this is an ad" is realised). Non-ad positions keep their item
    unchanged, so with `every = 0` (off) the feed is exactly the draft feed.
    No randomness is used — the same feed always gets the same ads.
    """
    if every < 0:
        raise ValueError(f"every must be >= 0, got {every}")
    if every == 0:
        return feed
    pos = np.arange(every - 1, feed.n, every)
    z, A, nu, lam = feed.z.copy(), feed.A.copy(), feed.nu.copy(), feed.lam.copy()
    is_ad = np.zeros(feed.n, dtype=bool)
    is_ad[pos] = True
    z[pos] = 0
    A[pos] = 0.0
    nu[pos] = np.exp(m_nu_ad)
    lam[pos] = 1.0  # unused while A = 0; kept positive for the reward formula
    return Feed(z=z, A=A, nu=nu, lam=lam, is_ad=is_ad)


def _sample_types_with_count(n: int, n_ones: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Persistence chain (p = n_ones / n) conditioned on exactly `n_ones` ones.

    Exact sampling: backward pass W[i, r, s] ∝ P(positions i+1.. hold r ones | z_i = s),
    then a forward pass draws each type in proportion to transition * W.
    Memory is O(n * n_ones).
    """
    p = n_ones / n
    q = np.array([1.0 - p, p])
    trans = kappa * np.eye(2) + (1.0 - kappa) * q  # trans[s, t] = P(z_{i+1} = t | z_i = s)

    W = np.zeros((n, n_ones + 1, 2))
    W[n - 1, 0, :] = 1.0
    for i in range(n - 2, -1, -1):
        nxt = W[i + 1]
        W[i, :, :] = np.outer(nxt[:, 0], trans[:, 0])
        W[i, 1:, :] += np.outer(nxt[:-1, 1], trans[:, 1])
        W[i] /= W[i].max()  # rescale per position; only ratios within a position matter

    z = np.empty(n, dtype=np.int8)
    remaining = n_ones
    w = q * np.array([W[0, remaining, 0], W[0, remaining - 1, 1] if remaining else 0.0])
    for i in range(n):
        t = int(rng.random() < w[1] / w.sum())
        z[i] = t
        remaining -= t
        if i + 1 < n:
            w = trans[t] * np.array(
                [W[i + 1, remaining, 0], W[i + 1, remaining - 1, 1] if remaining else 0.0]
            )
    return z


def reorder_feed(feed: Feed, kappa: float, rng: np.random.Generator) -> Feed:
    """Reorder the same items so that same-type items cluster with persistence `kappa`.

    The number of z = 1 items is kept; each item keeps its (z, A, nu, lam).
    kappa = 0 gives a uniformly random permutation of the type sequence.
    """
    _check_kappa(kappa)
    n_ones = int(feed.z.sum())
    slots = _sample_types_with_count(feed.n, n_ones, kappa, rng)
    order = np.empty(feed.n, dtype=np.intp)
    for t in (0, 1):
        order[slots == t] = rng.permutation(np.flatnonzero(feed.z == t))
    return feed.reindex(order)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def feed_stats(feed: Feed) -> dict[str, float]:
    """Summary of a realised feed.

    type_persistence is the share of adjacent pairs with the same type;
    its expectation under the generator is kappa + (1 - kappa) * (p^2 + (1 - p)^2).
    """
    return {
        "n": feed.n,
        "promising_share": float(feed.z.mean()),
        "nu_A_corr_log": _corr(np.log(feed.nu), np.log(feed.A)),
        "nu_A_corr_rank": _corr(_ranks(feed.nu), _ranks(feed.A)),
        "type_persistence": float(np.mean(feed.z[1:] == feed.z[:-1])) if feed.n > 1 else float("nan"),
        "mean_A": float(feed.A.mean()),
        "mean_nu": float(feed.nu.mean()),
        "mean_lam": float(feed.lam.mean()),
    }
