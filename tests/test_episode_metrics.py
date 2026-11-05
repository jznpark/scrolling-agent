"""Shared per-step episode metrics (P0-3) and the condition pipeline helpers (P0-4)."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scrolling_agent import Config, ScrollingEnv, TrainConfig
from scrolling_agent.cli import train_overrides
from scrolling_agent.evaluation import run_episode
from scrolling_agent.policies import GreedyKnown, ScrollFirst, reference_return

ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("policy", [GreedyKnown(0.6), ScrollFirst(0.4), GreedyKnown(0.3, dwell=0.5)])
def test_time_categories_add_up_to_the_session(policy):
    cfg = Config()
    m, _ = run_episode(ScrollingEnv(cfg), policy, seed=10_000, reference=True)
    parts = ["time_down", "time_up", "time_stay", "time_blocked", "time_watch", "time_transition"]
    assert sum(m[k] for k in parts) == pytest.approx(cfg.time.session_seconds)
    assert m["time_down"] + m["time_up"] + m["time_stay"] + m["time_blocked"] == pytest.approx(m["time_list"])
    assert m["rewatch_clicks"] == m["n_revisits"]
    assert 0 < m["known_selection_rate"] <= 1


def test_reference_and_percentage():
    env = ScrollingEnv()
    m, _ = run_episode(env, GreedyKnown(0.6), seed=10_001, reference=True)
    assert m["reference"] == pytest.approx(reference_return(env.feed, env.cfg))
    assert m["pct_of_ref"] == pytest.approx(100 * m["return"] / m["reference"])
    assert "reference" not in run_episode(env, GreedyKnown(0.6), seed=10_001)[0]


def test_going_back_up_counts_as_revisit_clicks():
    # with residual processing, items understood after leaving the screen draw the greedy agent back up
    cfg = Config()
    cfg.ddm = replace(cfg.ddm, tau_residual=2.0)
    totals = [run_episode(ScrollingEnv(cfg), GreedyKnown(0.6), seed=s)[0] for s in range(10_000, 10_005)]
    assert sum(m["revisit_clicks"] for m in totals) > 0
    assert sum(m["up_moves"] for m in totals) > 0


def test_train_overrides():
    t = train_overrides(TrainConfig(), ["ent_coef=0.01", "net_arch=[64, 64]"])
    assert t.ent_coef == 0.01 and t.net_arch == (64, 64)
    with pytest.raises(SystemExit):
        train_overrides(TrainConfig(), ["no_such_field=1"])


def test_pipeline_conditions_build_valid_configs():
    pipeline = load_script("scripts/pipeline.py")
    for name in pipeline.CONDITIONS:
        cfg = pipeline.condition_config(name)
        env = ScrollingEnv(cfg)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        train_overrides(TrainConfig(), pipeline.CONDITIONS[name]["train"])
    assert pipeline.condition_config("topk").obs.memory == "topk"
    assert pipeline.condition_config("kappa0.9").feed.kappa == 0.9


def test_kappa_script_uses_the_shared_metrics():
    ko = load_script("scripts/experiments/kappa_order.py")
    cfg = ko.make_config({})
    feed = ko.make_feed(cfg, 10_000, 0.9)
    row = ko.play(cfg, feed, 10_000, ko.make_policy(("greedy", 0.6, 0.0)))
    assert list(row) == ko.PLAY_COLUMNS
    assert row["time_total"] == pytest.approx(cfg.time.session_seconds)
    assert np.isfinite(row["pct_of_ref"])
