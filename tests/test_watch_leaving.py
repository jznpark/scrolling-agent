"""Leaving-point analysis script: recorder and greedy-watch hybrid."""
import importlib.util
from pathlib import Path

import pytest

from scrolling_agent import ScrollingEnv
from scrolling_agent.evaluation import run_episode
from scrolling_agent.policies import GreedyKnown

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("watch_leaving", ROOT / "scripts/experiments/watch_leaving.py")
wl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wl)


def test_greedy_leaves_below_rho_and_one_row_per_back():
    rec = wl.LeaveRecorder(GreedyKnown(0.6))
    m, _ = run_episode(ScrollingEnv(), rec, seed=10_000)
    assert len(rec.rows) >= m["n_clicks"] - 1  # the last bout may end with the session
    assert all(r["rate"] < 0.6 and r["bout_s"] > 0 for r in rec.rows)


def test_hybrid_with_greedy_inside_is_greedy():
    a, _ = run_episode(ScrollingEnv(), GreedyKnown(0.6), seed=10_001)
    b, _ = run_episode(ScrollingEnv(), wl.GreedyWatch(GreedyKnown(0.6), 0.6), seed=10_001)
    assert a["return"] == pytest.approx(b["return"]) and a["n_clicks"] == b["n_clicks"]
