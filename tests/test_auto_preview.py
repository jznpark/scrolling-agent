"""Optional autoplay: clock, priority, reset, counting and demo integration."""

import math

import numpy as np
import pytest
from helpers import click, constant_feed, run_until

from scrolling_agent import Config, DDMConfig, PreviewConfig, ScreenConfig, ScrollingEnv, TimeConfig
from scrolling_agent.env import LIST, WATCH
from scrolling_agent.evaluation import run_episode
from webdemo.server import Session, build_config, compare_behaviour


def preview_env(enabled=True, seconds=1.0):
    cfg = Config(time=TimeConfig(session_seconds=10), screen=ScreenConfig(n_items=3),
                 ddm=DDMConfig(noise=0.0), preview=PreviewConfig(enabled, seconds))
    env = ScrollingEnv(cfg, feed=constant_feed(n=3))
    env.reset(seed=0)
    return env


def test_default_off_never_opens_a_card_while_waiting():
    assert Config().preview.enabled is False
    env = preview_env(enabled=False)
    for _ in range(80):
        env.step(env.stay_action)
    assert env.understood.all() and env.mode == LIST
    assert not env.w.any() and env.n_auto_previews == 0
    assert env.extra_metrics() == {}


def test_disabled_duration_has_no_effect_on_random_play():
    envs = [preview_env(enabled=False, seconds=s) for s in [0.25, 9.0]]
    rng = np.random.default_rng(8)
    for _ in range(100):
        np.testing.assert_array_equal(envs[0].action_masks(), envs[1].action_masks())
        action = int(rng.choice(np.flatnonzero(envs[0].action_masks())))
        a, b = [env.step(action) for env in envs]
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_ununderstood_cards_are_not_previewed():
    env = preview_env()
    for _ in range(5):
        env.step(env.stay_action)
    assert not env.understood.any()
    assert env.mode == LIST and env.n_auto_previews == 0
    assert not env.w.any()


@pytest.mark.parametrize("seconds", [0.25, 1.0])
def test_preview_duration_excludes_transitions_and_then_returns(seconds):
    env = preview_env(seconds=seconds)
    run_until(env, env.stay_action, lambda e: e.auto_preview_item is not None and e.mode == WATCH)
    item = env.auto_preview_item
    evidence = env.x.copy()
    assert env.w[item] == 0  # opening time is not watch time
    while env.auto_preview_item is not None:
        # A greedy policy may ask to leave immediately; preview still lasts its duration.
        env.step(env.back_action if env.mode == WATCH else env.stay_action)
    expected = math.ceil(seconds / env.cfg.time.dt - 1e-9) * env.cfg.time.dt
    assert env.mode == LIST and env.scroll == 0
    assert env.w[item] == pytest.approx(expected)
    assert env.time_by_mode['TRANSITION'] == pytest.approx(0.4)
    np.testing.assert_array_equal(env.x, evidence)


def test_explicit_click_is_a_normal_watch_without_auto_close():
    env = preview_env()
    run_until(env, env.stay_action, lambda e: e.understood.all())
    click(env, 1)
    for _ in range(20):
        env.step(env.stay_action)
    assert env.mode == WATCH and env.current == 1
    assert env.auto_preview_item is None and env.n_auto_previews == 0
    assert env.w[1] == pytest.approx(2.0)


def test_one_preview_per_card_and_evaluation_counts_executed_actions():
    env = preview_env()
    metrics, trace = run_episode(env, lambda e: e.stay_action, seed=0, record=True)
    assert metrics['n_auto_previews'] == metrics['n_clicks'] == 3
    assert metrics['invalid_actions'] == 0
    assert metrics['time_watch'] == pytest.approx(3.0)
    np.testing.assert_allclose(env.w, 1.0)
    assert int(np.isin(trace.action, env.slot_actions).sum()) == 3
    assert sum(env.time_by_mode.values()) == pytest.approx(10.0)


def test_reset_allows_the_same_cards_to_preview_again_reproducibly():
    env = preview_env()
    first, _ = run_episode(env, lambda e: e.stay_action, seed=0)
    second, _ = run_episode(env, lambda e: e.stay_action, seed=0)
    assert first == second and second['n_auto_previews'] == 3


@pytest.mark.parametrize('seconds', [0.0, -1.0, float('inf'), float('nan')])
def test_invalid_preview_duration_is_rejected(seconds):
    with pytest.raises(ValueError, match='preview.seconds'):
        build_config({'preview.seconds': seconds})


def test_web_session_reports_preview_and_counts_automatic_clicks():
    session = Session(10000, {'preview.enabled': True, 'ddm.noise': 0.0, 'feed.sigma_nu': 0.0},
                      {'name': 'human'}, None)
    session.step(15)
    state = session.state([])
    assert state['auto_preview'] == {'item': 0, 'seconds': 1.0}
    assert state['n_clicks'] == state['extra_metrics']['n_auto_previews'] == 1
    assert any('자동 미리보기' in e['text'] for e in state['events'])
    session.step(45)
    state = session.state([])
    assert state['auto_preview'] is None
    assert state['n_clicks'] == state['extra_metrics']['n_auto_previews'] == 3
    assert sum('미리보기 종료' in e['text'] for e in state['events']) == 3


def test_behaviour_comparison_includes_disabled_zero_and_enabled_preview_count():
    result = compare_behaviour({'preview.enabled': True}, {'name': 'greedy', 'rho': 0.6}, None, n=2)
    row = next(r for r in result['rows'] if r['key'] == 'n_auto_previews')
    assert row['base'] == 0 and row['now'] > 0
    unchanged = compare_behaviour({}, {'name': 'greedy'}, None, n=1)
    assert all(r['base'] == r['now'] for r in unchanged['rows'])
