"""Shopping grid extension (screen.columns): cards side by side, row by row."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from scrolling_agent import ScrollingEnv
from scrolling_agent.config import Config, ScreenConfig
from scrolling_agent.env import WATCH, visible_items
from scrolling_agent.evaluation import evaluate, summarize
from scrolling_agent.policies import GreedyKnown, RandomPolicy

from helpers import click, constant_feed, run_until


def grid_config(columns: int = 2, n_items: int = 100) -> Config:
    cfg = Config()
    cfg.screen = ScreenConfig(n_items=n_items, columns=columns)
    return cfg


# --- off by default --------------------------------------------------------------------


def test_default_is_one_column():
    assert Config().screen.columns == 1
    assert Config().screen.max_visible_slots == 4


@pytest.mark.parametrize("scroll", [0.0, 0.4, 0.6, 1.5, 96.5, 97.0])
def test_one_column_visibility_unchanged(scroll):
    assert visible_items(scroll, 100, 3.0, 0.5, columns=1).tolist() == visible_items(scroll, 100, 3.0, 0.5).tolist()


def test_one_column_episode_identical_to_draft_model():
    """Fingerprints recorded with feature/web-demo before the extension existed."""
    for seed, policy, total, n_understood, obs_sum in [
        (10_000, GreedyKnown(0.6, dwell=0.5), 43.35275861039432, 50, 1046446.592880249),
        (7, RandomPolicy(3), 2.4780142394336817, 7, 1949826.7348632812),
    ]:
        env = ScrollingEnv()
        env.reset(seed=seed)
        r_sum = s_sum = 0.0
        for _ in range(env.cfg.time.n_steps):
            obs, r, *_ = env.step(policy(env))
            r_sum += r
            s_sum += float(obs.sum())
        assert r_sum == pytest.approx(total, rel=1e-9)
        assert int(env.understood.sum()) == n_understood
        assert s_sum == pytest.approx(obs_sum, rel=1e-6)


# --- geometry ------------------------------------------------------------------------------


def test_rows_and_slots():
    scr = ScreenConfig(n_items=101, columns=2)
    assert scr.n_rows == 51
    assert scr.max_visible_slots == 8
    with pytest.raises(ValueError):
        ScreenConfig(columns=0)


@pytest.mark.parametrize(
    "scroll, expected",
    [
        (0.0, [0, 1, 2, 3, 4, 5]),        # rows 0-2
        (0.4, [0, 1, 2, 3, 4, 5]),        # row 3 only 40 % on screen
        (0.6, [2, 3, 4, 5, 6, 7]),        # row 0 only 40 % on screen, row 3 60 %
        (0.5, [0, 1, 2, 3, 4, 5, 6, 7]),  # both edge rows exactly half: four rows
    ],
)
def test_visible_items_two_columns(scroll, expected):
    assert visible_items(scroll, 100, 3.0, 0.5, columns=2).tolist() == expected


def test_partial_last_row():
    # 5 items in 2 columns: rows [0, 1], [2, 3], [4]
    assert visible_items(0.0, 5, 3.0, 0.5, columns=2).tolist() == [0, 1, 2, 3, 4]


def test_max_scroll_counts_rows():
    env = ScrollingEnv(grid_config(columns=2))
    env.reset(seed=0)
    assert env.max_scroll == pytest.approx(50 - 3)
    run_until(env, env.down_actions[0], lambda e: e.scroll >= e.max_scroll)
    assert env.visible_slots().tolist() == list(range(94, 100))


def test_gymnasium_check_env_grid():
    check_env(ScrollingEnv(grid_config(columns=2)), skip_render_check=True)


# --- dynamics ------------------------------------------------------------------------------


def test_both_cards_of_a_row_are_processed_together():
    env = ScrollingEnv(grid_config(columns=2), feed=constant_feed())
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.stay_action)
    on_screen = env.visible_slots()
    assert on_screen.tolist() == [0, 1, 2, 3, 4, 5]
    assert np.allclose(env.exposure[on_screen], 0.5)
    assert np.all(env.exposure[6:] == 0)
    assert np.all(env.x[on_screen] != 0)


def test_click_right_column_opens_that_item():
    env = ScrollingEnv(grid_config(columns=2), feed=constant_feed(nu=50.0))
    env.reset(seed=0)
    run_until(env, env.stay_action, lambda e: e.understood[1])
    click(env, 1)
    assert env.mode == WATCH and env.current == 1


def test_grid_episode_time_sums_to_T():
    env = ScrollingEnv(grid_config(columns=2))
    env.reset(seed=1)
    policy = GreedyKnown(0.6)
    for _ in range(env.cfg.time.n_steps):
        *_, info = env.step(policy(env))
    assert sum(info["time_by_mode"].values()) == pytest.approx(env.cfg.time.session_seconds)
    assert env.watch_reward_total > 0


def test_two_columns_understand_more_items():
    """Same scroll speed, twice as many cards per screen: more items understood."""
    one = summarize(evaluate(ScrollingEnv(grid_config(columns=1)), GreedyKnown(0.6), range(10_000, 10_005)))
    two = summarize(evaluate(ScrollingEnv(grid_config(columns=2)), GreedyKnown(0.6), range(10_000, 10_005)))
    assert two["n_understood"][0] > one["n_understood"][0]


# --- web demo ------------------------------------------------------------------------------


def test_webdemo_state_and_scenario():
    from webdemo.server import Session, available_scenarios, settings_schema

    s = Session(10_000, {"screen.columns": 2}, {"name": "greedy", "rho": 0.6}, None)
    state = s.state([])
    assert state["columns"] == 2
    assert len(state["actions"]["slots"]) == 8
    assert state["slots"] == [0, 1, 2, 3, 4, 5]
    s.step(50)
    assert len(s.state([])["slots"]) >= 6
    assert Session(10_000, {}, {"name": "greedy"}, None).state([])["columns"] == 1
    assert "grid2" in {sc["id"] for sc in available_scenarios(settings_schema())}
