import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scrolling_agent import Config, ScrollingEnv  # noqa: E402
from scrolling_agent.evaluation import run_episode  # noqa: E402
from scrolling_agent.policies import GreedyKnown  # noqa: E402
from scrolling_agent.viz import animate_episode, episode_summary, plot_episode  # noqa: E402


@pytest.fixture(scope="module")
def recorded():
    env = ScrollingEnv()
    return run_episode(env, GreedyKnown(0.3), seed=3, record=True)


def test_metrics_are_consistent(recorded):
    metrics, trace = recorded
    cfg = Config()
    assert metrics["time_list"] + metrics["time_watch"] + metrics["time_transition"] == pytest.approx(
        cfg.time.session_seconds
    )
    assert metrics["return"] == pytest.approx(metrics["watch_reward"] - cfg.time.n_steps * cfg.time.step_cost)
    assert metrics["return"] == pytest.approx(trace.reward.sum())
    assert metrics["n_clicks"] >= metrics["n_watched_items"]


def test_trace_shapes(recorded):
    _, trace = recorded
    N, n = Config().time.n_steps, Config().screen.n_items
    assert trace.reward.shape == trace.scroll.shape == (N,)
    assert trace.x.shape == trace.understood.shape == trace.w.shape == (N, n)
    np.testing.assert_allclose(trace.t[[0, -1]], [0.1, 120.0])


def test_summary_matches_metrics(recorded):
    metrics, trace = recorded
    summary = episode_summary(trace)
    assert summary["return"] == pytest.approx(metrics["return"])
    assert summary["watch reward"] == pytest.approx(metrics["watch_reward"])
    assert summary["understood"] == metrics["n_understood"]


def test_plots_render(recorded, tmp_path):
    _, trace = recorded
    fig = plot_episode(trace, "test")
    fig.savefig(tmp_path / "ep.png")
    fig, anim = animate_episode(trace, every=200)
    anim._func(0)  # draw one frame
    assert (tmp_path / "ep.png").stat().st_size > 0
