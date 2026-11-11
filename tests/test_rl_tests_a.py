"""A tests: same-items reorder option, two-speed conditions, slow-down index."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scrolling_agent import ScrollingEnv
from scrolling_agent.cli import apply_overrides
from scrolling_agent.config import Config
from scrolling_agent.policies import GreedyKnown

ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = load_script("scripts/experiments/speed_context.py")
pipeline = load_script("scripts/pipeline.py")


def config(reorder=None):
    cfg = Config()
    cfg.feed = replace(cfg.feed, reorder_kappa=reorder)
    return cfg


def items(env):
    return sorted(zip(env.feed.A.tolist(), env.feed.nu.tolist(), env.feed.lam.tolist()))


def test_reorder_keeps_the_items_and_only_changes_the_order():
    a, b = ScrollingEnv(config(0.0)), ScrollingEnv(config(0.9))
    a.reset(seed=10_000)
    b.reset(seed=10_000)
    assert items(a) == items(b)
    assert not np.array_equal(a.feed.A, b.feed.A)
    same = lambda z: np.mean(z[1:] == z[:-1])  # noqa: E731
    assert same(b.feed.z) > same(a.feed.z)


def test_reorder_off_is_the_default_feed():
    a, b = ScrollingEnv(Config()), ScrollingEnv(config(None))
    a.reset(seed=3)
    b.reset(seed=3)
    assert np.array_equal(a.feed.A, b.feed.A)


@pytest.mark.parametrize("name", ["resid-0.5", "resid-2", "a-base", "a-base-k0.9", "speed2", "speed2-k0.9",
                                  "speed2-2m", "reorder-k0", "reorder-k0.9"])
def test_conditions_build_a_working_env(name):
    cfg = apply_overrides(Config(), list(pipeline.CONDITIONS[name]["set"]))
    env = ScrollingEnv(cfg)
    env.reset(seed=1)
    assert cfg.obs.memory == "topk"
    assert len(env.down_actions) == len(cfg.screen.speeds)
    env.step(env.down_actions[-1])


def test_index_is_the_difference_of_slow_shares():
    counts = {("all", True): [8, 10], ("all", False): [2, 10]}
    assert sc.index(counts, "all") == pytest.approx(0.6)
    assert np.isnan(sc.index(counts, "fresh"))


class FastAfterBad:
    """Slow after a good last understanding, fast otherwise, to check the recorder's sign."""

    def __init__(self):
        self.known, self.last = None, None

    def __call__(self, env):
        if self.known is None:
            self.known = env.understood.copy()
        new = np.flatnonzero(env.understood & ~self.known)
        if new.size:
            self.last = env.revealed_profile()[0][new.max()]
        self.known = env.understood.copy()
        if env.mode != 0:  # transitions and watching never occur here, but stay safe
            return env.stay_action
        return env.down_actions[0 if self.last is None or self.last >= 0.6 else 1]


def test_recorder_sign_and_exclusions():
    cfg = apply_overrides(Config(), ["screen.speeds=(0.5,1.5)"])
    env = ScrollingEnv(cfg)
    rec = sc.SpeedRecorder(FastAfterBad())
    env.reset(seed=10_000)
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(rec(env))
    assert sc.index(rec.counts, "all") == pytest.approx(1.0)  # always slow after good, always fast after bad
    assert rec.counts[("all", True)][1] + rec.counts[("all", False)][1] < cfg.time.n_steps  # bottom and pre-first steps left out


def test_rule_policy_has_no_hold_and_counts_every_step_as_a_decision():
    cfg = apply_overrides(Config(), ["screen.speeds=(0.5,1.5)"])
    rec = sc.SpeedRecorder(GreedyKnown(0.6))
    env = ScrollingEnv(cfg)
    env.reset(seed=10_000)
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(rec(env))
    assert rec.counts[("decisions", True)][1] == rec.counts[("all", True)][1]
