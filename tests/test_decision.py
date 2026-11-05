"""Decision interval (learning side): hold decisions up to k steps, skip transitions."""
import numpy as np
import pytest

from scrolling_agent import ScrollingEnv
from scrolling_agent.decision import DecisionHold, DecisionInterval
from scrolling_agent.env import LIST, TRANSITION, WATCH
from scrolling_agent.evaluation import run_episode
from scrolling_agent.policies import GreedyKnown, RandomPolicy


class Counting:
    """Records the env state at every decision."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __call__(self, env):
        self.calls.append((env.step_count, env.mode))
        return self.inner(env)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_wrapper_and_hold_give_the_same_episode(k):
    seed = 10_003
    # per-step evaluation path (PPOPolicy uses DecisionHold)
    held = DecisionHold(Counting(GreedyKnown(0.6)), k)
    m_step, _ = run_episode(ScrollingEnv(), held, seed=seed)
    # training path: one wrapper step per decision
    env = DecisionInterval(ScrollingEnv(), k)
    decide = Counting(GreedyKnown(0.6))
    env.reset(seed=seed)
    total, terminated, n_env_steps = 0.0, False, 0
    while not terminated:
        _, r, terminated, _, info = env.step(decide(env.unwrapped))
        total += r
        n_env_steps += info["env_steps"]
    assert n_env_steps == env.unwrapped.cfg.time.n_steps
    assert total == pytest.approx(m_step["return"])
    assert decide.calls == held.decide.calls


def test_decisions_are_spaced_and_never_in_transition():
    k = 5
    decide = Counting(RandomPolicy(seed=1))
    env = DecisionInterval(ScrollingEnv(), k)
    env.reset(seed=7)
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step(decide(env.unwrapped))
        assert 1 <= info["env_steps"] <= k + env.unwrapped.cfg.time.transition_steps
    modes = {m for _, m in decide.calls}
    assert TRANSITION not in modes and modes <= {LIST, WATCH}
    steps = np.array([s for s, _ in decide.calls])
    assert len(steps) < env.unwrapped.cfg.time.n_steps / 2  # far fewer decisions than steps


def test_k1_hold_is_the_plain_policy():
    a, _ = run_episode(ScrollingEnv(), GreedyKnown(0.6), seed=10_004)
    b, _ = run_episode(ScrollingEnv(), DecisionHold(GreedyKnown(0.6), 1), seed=10_004)
    assert a["return"] == pytest.approx(b["return"]) and a["n_clicks"] == b["n_clicks"]
