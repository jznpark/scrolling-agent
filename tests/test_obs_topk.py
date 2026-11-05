"""Compact observation: only the best known items (ObsConfig.memory = "topk")."""
from dataclasses import replace

import numpy as np
import pytest

from scrolling_agent import Config, ObsConfig, ScrollingEnv
from scrolling_agent.env import N_TOPK_FEATURES
from scrolling_agent.policies import GreedyKnown


def topk_config(k: int = 5) -> Config:
    return replace(Config(), obs=ObsConfig(memory="topk", top_k=k))


def test_default_observation_is_unchanged():
    env = ScrollingEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (655,)


def test_topk_observation_is_short_and_sorted_by_known_rate():
    k = 5
    env = ScrollingEnv(topk_config(k))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert obs.shape[0] == 655 - 100 * 6 + 1 + k * N_TOPK_FEATURES
    policy = GreedyKnown(0.6)
    for _ in range(300):
        obs, *_ = env.step(policy(env))
    memory = obs[-(1 + k * N_TOPK_FEATURES):]
    n_known = int(env.understood.sum())
    assert n_known > 0
    assert memory[0] == pytest.approx(n_known / env.n_items)
    rows = memory[1:].reshape(k, N_TOPK_FEATURES)
    present = rows[:, 0] > 0
    assert present.sum() == min(k, n_known)
    assert np.all(np.diff(rows[present, 1]) <= 1e-6)  # best first
    A, lam = env.revealed_profile()
    rate = np.where(env.understood, A * np.exp(-lam * env.w), -np.inf)
    assert rows[0, 1] == pytest.approx(rate.max(), rel=1e-5)


def test_topk_hides_unknown_items():
    env = ScrollingEnv(topk_config())
    obs, _ = env.reset(seed=2)
    assert not env.understood.any()
    assert np.all(obs[-(1 + 5 * N_TOPK_FEATURES):] == 0)


def test_bad_obs_settings_are_rejected():
    with pytest.raises(ValueError):
        ObsConfig(memory="all")
    with pytest.raises(ValueError):
        ObsConfig(top_k=0)
