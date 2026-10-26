"""Reminders interrupt the viewer without changing the simulated trajectory."""

import pytest

from scrolling_agent import ScrollingEnv
from scrolling_agent.policies import GreedyKnown
from webdemo.reminder import ReminderSettings
from webdemo.server import Session, build_config, compare_behaviour, settings_schema


def session(**settings):
    return Session(10_000, {f"reminder.{k}": v for k, v in settings.items()},
                   {"name": "greedy", "rho": 0.6}, None)


def test_default_off_matches_core_model_through_full_episode():
    s = session()
    env = ScrollingEnv(build_config({}))
    env.reset(seed=10_000)
    policy = GreedyKnown(0.6)
    total = 0.0
    for _ in range(env.cfg.time.n_steps):
        _, reward, *_ = env.step(policy(env))
        total += reward
    s.step(600)
    s.step(600)
    assert s.done and not s.reminder.settings.enabled
    assert s.total == total
    assert (s.env.w == env.w).all() and (s.env.x == env.x).all()
    assert not any(e["kind"] == "reminder" for e in s.events)
    field = next(f for sec in settings_schema() for f in sec["fields"] if f["key"] == "reminder.enabled")
    assert field["default"] is False


def test_disabled_threshold_has_no_effect():
    a, b = session(), session(enabled=False, after_seconds=0.1, snooze_seconds=0.1)
    assert a.step(600) == b.step(600)
    assert a.state([])["items"] == b.state([])["items"]
    assert not b.reminder.pending and b.reminder.count == 0


def test_batched_step_stops_at_threshold_and_cannot_bypass_pending():
    s = session(enabled=True)
    history = s.step(600)
    assert len(history) == 300 and s.state([])["t"] == 30.0
    assert s.reminder.pending and not s.done and s.reminder.count == 1
    frozen = s.state([])
    assert s.step(600) == []
    assert s.step(1, action=s.env.down_actions[0]) == []
    assert s.state([]) == frozen


def test_continue_snoozes_and_preserves_trajectory():
    s, reference = session(enabled=True), session()
    s.step(600)
    s.respond_reminder("continue")
    assert s.env.step_count == 300 and not s.reminder.pending
    assert s.reminder.state()["next_at"] == 45.0
    assert len(s.step(600)) == 150
    reference.step(450)
    assert s.reminder.pending and s.reminder.count == 2
    assert s.total == reference.total
    assert s.state([])["items"] == reference.state([])["items"]


def test_stop_ends_only_viewer_session_and_reset_starts_fresh():
    s = session(enabled=True)
    s.step(600)
    s.respond_reminder("stop")
    state = s.state([])
    assert state["done"] and state["end_reason"] == "reminder"
    assert state["t"] == 30.0 and state["t"] < state["T"]
    assert s.step(600) == [] and s.state([]) == state
    fresh = session(enabled=True)
    assert not fresh.done and not fresh.reminder.pending and fresh.reminder.count == 0


def test_threshold_rounds_up_to_simulation_step():
    s = session(enabled=True, after_seconds=1.05, snooze_seconds=0.15)
    s.step(600)
    assert s.state([])["t"] == 1.1
    s.respond_reminder("continue")
    s.step(600)
    assert s.state([])["t"] == 1.3


@pytest.mark.parametrize("after", [120.0, 130.0])
def test_time_limit_takes_precedence_over_reminder(after):
    s = session(enabled=True, after_seconds=after)
    s.step(600)
    s.step(600)
    assert s.done and s.state([])["end_reason"] == "time_limit"
    assert not s.reminder.pending and s.reminder.count == 0


@pytest.mark.parametrize("overrides", [
    {"reminder.enabled": "false"}, {"reminder.enabled": 1},
    {"reminder.after_seconds": 0}, {"reminder.after_seconds": -1},
    {"reminder.after_seconds": float("nan")}, {"reminder.after_seconds": float("inf")},
    {"reminder.snooze_seconds": 0}, {"reminder.snooze_seconds": True},
    {"reminder.unknown": 1},
])
def test_invalid_reminder_values_rejected(overrides):
    with pytest.raises(ValueError):
        build_config(overrides)


def test_responses_require_an_active_reminder_and_known_choice():
    s = session(enabled=True)
    with pytest.raises(ValueError):
        s.respond_reminder("continue")
    s.step(600)
    with pytest.raises(ValueError):
        s.respond_reminder("dismiss")
    assert s.reminder.pending
    s.respond_reminder("stop")
    with pytest.raises(ValueError):
        s.respond_reminder("continue")


def test_reminder_does_not_enter_model_config_or_compare_claim_effect():
    cfg = build_config({"reminder.enabled": True})
    assert not hasattr(cfg, "reminder")
    result = compare_behaviour({"reminder.enabled": True}, {"name": "greedy"}, None, n=2)
    assert result["reminder_excluded"] is True
    assert all(row["base"] == row["now"] for row in result["rows"])


def test_string_seconds_supported_but_boolean_not_coerced():
    assert ReminderSettings.from_overrides({"reminder.after_seconds": "5"}).after_seconds == 5.0
