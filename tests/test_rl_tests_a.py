"""A tests: same-items reorder option, two-speed conditions, slow-down index."""
import importlib.util
import sys
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
    sys.modules[spec.name] = module  # dataclasses look their module up here
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


B_C_CONDITIONS = ["b6-k0.9-5m", "b6-reorder-k0.9-5m", "b7-recent-k0", "b7-recent-k0.9", "b8-v0.5-f1.5", "b8-v2-f1.5",
                  "b8-v0.5-f1", "b8-v1-f1", "b8-v2-f1", "c9-dnu-k0", "c9-dnu-k0.9", "c10-est0.6", "c10-attn-log"]


@pytest.mark.parametrize("name", ["resid-0.5", "resid-2", "a-base", "a-base-k0.9", "speed2", "speed2-k0.9",
                                  "speed2-2m", "reorder-k0", "reorder-k0.9", *B_C_CONDITIONS])
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


def test_b_c_conditions_set_what_they_say():
    get = lambda name: apply_overrides(Config(), list(pipeline.CONDITIONS[name]["set"]))  # noqa: E731
    assert get("b8-v2-f1").ddm.v0 == 2.0 and get("b8-v2-f1").screen.speeds == (0.5, 1.0)
    assert get("b7-recent-k0.9").obs.recent and get("b7-recent-k0.9").feed.reorder_kappa == 0.9
    assert get("c9-dnu-k0").feed.d_nu == -0.5
    assert get("c10-est0.6").estimate.sigma_est == 0.6
    assert get("c10-attn-log").ddm.attention == "log"
    assert "total_timesteps=5000000" in pipeline.CONDITIONS["b6-k0.9-5m"]["train"]


def test_b9_long_runs_differ_from_their_1m_condition_only_in_training():
    assert len(pipeline.B9_LONG) == 7
    for name in pipeline.B9_LONG:
        short, long = pipeline.CONDITIONS[name], pipeline.CONDITIONS[f"{name}-5m"]
        assert long["set"] == short["set"]
        assert long["train"] == pipeline.CONDITIONS["b6-reorder-k0.9-5m"]["train"]
        assert apply_overrides(Config(), list(long["set"])).feed.reorder_kappa == 0.9


def run_greedy(env, seed):
    obs, _ = env.reset(seed=seed)
    policy, frames, terminated = GreedyKnown(0.6), [obs], False
    while not terminated:
        obs, _, terminated, _, _ = env.step(policy(env))
        frames.append(obs)
    return np.array(frames)


def test_recent_context_off_leaves_the_observation_unchanged_and_on_appends_four_features():
    off, on = Config(), Config()
    on.obs = replace(on.obs, recent=True)
    a, b = run_greedy(ScrollingEnv(off), 7), run_greedy(ScrollingEnv(on), 7)
    assert b.shape[1] == a.shape[1] + 4
    assert np.array_equal(a, b[:, :-4])  # same trajectory, same base features


def test_recent_context_values():
    cfg = Config()
    cfg.obs = replace(cfg.obs, recent=True, recent_n=3)
    env = ScrollingEnv(cfg)
    obs, _ = env.reset(seed=10_000)
    assert np.all(obs[-4:] == 0)  # nothing understood yet
    order, terminated = [], False
    while not terminated:
        before = env.understood.copy()
        obs, _, terminated, _, _ = env.step(env.down_actions[0])
        new = np.flatnonzero(env.understood & ~before)
        order.extend(new.tolist())
        if new.size:
            A = env.feed.A
            assert obs[-4] == 1 and obs[-3] == pytest.approx(A[new.max()], rel=1e-6)
            assert obs[-2] == 0
            assert obs[-1] == pytest.approx(A[order[-3:]].mean(), rel=1e-6)
    assert 0 < obs[-2] <= 10.0  # seconds since the last understanding, capped


fr = load_script("scripts/experiments/fair_rules.py")


def play_actions(env, policy, seed=10_000):
    env.reset(seed=seed)
    actions, terminated = [], False
    while not terminated:
        a = policy(env)
        actions.append(a)
        _, _, terminated, _, _ = env.step(a)
    return actions


def test_context_speed_with_zero_threshold_is_the_greedy_rule():
    cfg = apply_overrides(Config(), ["screen.speeds=(0.5,1.5)"])
    assert play_actions(ScrollingEnv(cfg), fr.ContextSpeed(0.6, 0.25, a_thr=0.0)) == \
        play_actions(ScrollingEnv(cfg), GreedyKnown(0.6, dwell=0.25))


def test_context_speed_goes_fast_after_a_poor_item_and_slow_before_any():
    cfg = apply_overrides(Config(), ["screen.speeds=(0.5,1.5)"])
    env = ScrollingEnv(cfg)
    actions = play_actions(env, fr.ContextSpeed(0.6, 0.0, a_thr=1e9))  # every item counts as poor
    first_fast = actions.index(env.down_actions[1])
    assert env.down_actions[0] in actions[:first_fast]  # slow until something is understood
    assert env.down_actions[0] not in actions[first_fast:]


def test_fair_rule_specs_add_speed_rules_only_with_two_speeds():
    assert {s[0] for s in fr.specs_for(1)} == {"dwell"}
    assert {s[0] for s in fr.specs_for(2)} == {"dwell", "speed"}
