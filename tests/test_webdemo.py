"""Local web demo: settings schema, sessions and the HTTP API."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from webdemo.server import Session, build_config, serve, settings_schema


def test_schema_lists_every_setting_with_tags():
    sections = {s["name"]: {f["name"]: f for f in s["fields"]} for s in settings_schema()}
    assert {"time", "screen", "ddm", "feed"} <= set(sections)
    assert sections["time"]["session_seconds"]["tag"] == "draft"
    assert sections["feed"]["kappa"]["tag"] == "TBD"
    assert sections["screen"]["speeds"]["kind"] == "tuple"
    assert sections["ddm"]["noise"]["default"] == 0.25


def test_build_config_types_and_errors():
    cfg = build_config({"ddm.noise": "0.3", "screen.speeds": [0.5, 1.5], "time.dt": 0.1})
    assert cfg.ddm.noise == 0.3 and cfg.screen.speeds == (0.5, 1.5)
    with pytest.raises(ValueError):
        build_config({"ddm.nope": 1})
    with pytest.raises(ValueError):
        build_config({"nosection.x": 1})


def test_sessions_are_reproducible():
    def run():
        s = Session(10_000, {}, {"name": "greedy", "rho": 0.6}, None)
        s.step(300)
        return s.state([])

    a, b = run(), run()
    assert a["return"] == b["return"] and a["items"] == b["items"] and a["step"] == 300


def test_human_steps_and_invalid_actions():
    s = Session(0, {}, {"name": "human"}, None)
    back = s.env.back_action
    history = s.step(5, action=back)  # back is not allowed in LIST
    state = s.state(history)
    assert len(history) == 1 and state["step"] == 1
    assert state["events"][-1]["kind"] == "invalid"
    s.step(1, action=s.env.down_actions[0])
    assert s.env.scroll > 0


def test_state_contents():
    s = Session(1, {"ddm.noise": 0.25}, {"name": "scrollfirst"}, None)
    history = s.step(1200)  # capped per request
    assert len(history) == 600
    history += s.step(1200)
    state = s.state(history)
    assert state["done"] and len(history) == 1200
    assert len(state["items"]) == state["n_items"] == 100
    assert abs(sum(state["time_by_mode"].values()) - state["T"]) < 1e-6
    assert any(e["kind"] == "understood" for e in state["events"])
    unknown = [it for it in state["items"] if not it["understood"]]
    assert all(it["A"] == 0 and it["lam"] == 0 for it in unknown)  # hidden values only under "hidden"


@pytest.fixture(scope="module")
def server():
    srv = serve(0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.status, res.read()


def test_http_page_schema_reset_step(server):
    status, page = call(server + "/")
    assert status == 200 and b"Scrolling Agent" in page
    status, js = call(server + "/static/app.js")
    assert status == 200 and b"api(" in js
    schema = json.loads(call(server + "/api/schema")[1])
    assert "greedy" in schema["policies"] and "human" in schema["policies"]

    state = json.loads(call(server + "/api/reset", {"session": "t", "seed": 5, "policy": {"name": "greedy"},
                                                    "overrides": {"feed.kappa": 0.5}})[1])
    assert state["step"] == 0
    state = json.loads(call(server + "/api/step", {"session": "t", "n": 50})[1])
    assert state["step"] == 50 and len(state["history"]) == 50


def test_http_errors(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/step", {"session": "missing", "n": 1})
    assert err.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/reset", {"session": "x", "overrides": {"ddm.nope": 1}})
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/static/../server.py")
    assert err.value.code == 404


def test_http_reminder_pause_continue_stop_reset(server):
    body = {"session": "break", "seed": 10000, "policy": {"name": "greedy"},
            "overrides": {"reminder.enabled": True}}
    call(server + "/api/reset", body)
    state = json.loads(call(server + "/api/step", {"session": "break", "n": 600})[1])
    assert state["t"] == 30.0 and state["reminder"]["pending"]
    state = json.loads(call(server + "/api/reminder", {"session": "break", "choice": "continue"})[1])
    assert state["t"] == 30.0 and state["reminder"]["next_at"] == 45.0
    state = json.loads(call(server + "/api/step", {"session": "break", "n": 600})[1])
    assert state["t"] == 45.0 and state["reminder"]["count"] == 2
    state = json.loads(call(server + "/api/reminder", {"session": "break", "choice": "stop"})[1])
    assert state["done"] and state["end_reason"] == "reminder"
    state = json.loads(call(server + "/api/step", {"session": "break", "n": 600})[1])
    assert state["t"] == 45.0
    state = json.loads(call(server + "/api/reset", body)[1])
    assert state["t"] == 0 and not state["done"] and state["reminder"]["count"] == 0
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/reminder", {"session": "break", "choice": "continue"})
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/reminder", {"session": "no-break-session", "choice": "stop"})
    assert err.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/reset", {"session": "invalid-reminder", "overrides": {"reminder.after_seconds": 0}})
    assert err.value.code == 400


def test_every_setting_has_a_korean_description():
    from webdemo.descriptions import DESCRIPTIONS

    missing = [f["key"] for s in settings_schema() for f in s["fields"] if f["key"] not in DESCRIPTIONS]
    assert not missing


def test_ppo_runs_are_limited_to_listed_folders(tmp_path):
    from webdemo.server import list_policies, make_policy

    (tmp_path / "pilot").mkdir()
    (tmp_path / "pilot" / "best_model.zip").write_bytes(b"")
    assert "ppo:pilot" in list_policies(tmp_path)
    for name in ("ppo:../pilot", "ppo:other", "ppo:pilot/../../x"):
        with pytest.raises(ValueError):
            make_policy({"name": name}, tmp_path)


def test_last_watched_survives_batched_steps():
    s = Session(10_000, {}, {"name": "greedy", "rho": 0.6}, None)
    s.step(600)
    state = s.state([])
    assert state["last_watched"] is not None
    assert state["items"][state["last_watched"]]["w"] > 0


def test_server_binds_requested_host():
    srv = serve(0, host="127.0.0.1")
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.server_close()


def test_every_setting_has_a_plain_label():
    from webdemo.descriptions import LABELS

    assert not [f["key"] for s in settings_schema() for f in s["fields"] if f["key"] not in LABELS]


def test_scenarios_only_use_known_settings_and_build():
    from webdemo.descriptions import SCENARIOS
    from webdemo.server import available_scenarios

    shown = available_scenarios(settings_schema())
    assert shown[0]["id"] == "base" and len(shown) >= 8
    for sc in shown:
        build_config(sc["set"])  # every shown scenario is a valid config
    # extension scenarios are hidden on a branch without those settings
    assert len(shown) <= len(SCENARIOS)


def test_compare_behaviour_same_lists():
    from webdemo.server import compare_behaviour

    res = compare_behaviour({"feed.m_lambda": 0.0}, {"name": "greedy", "rho": 0.6}, None, n=3)
    rows = {r["key"]: r for r in res["rows"]}
    assert res["n"] == 3 and {"return", "n_clicks", "mean_bout"} <= set(rows)
    assert rows["mean_bout"]["now"] < rows["mean_bout"]["base"]  # content gets boring faster
    same = compare_behaviour({}, {"name": "greedy"}, None, n=2)
    assert all(r["base"] == r["now"] for r in same["rows"])


def test_no_false_late_events_without_residual_processing():
    """In the base model nothing can be understood off screen."""
    for seed in (10_000, 10_005):
        s = Session(seed, {}, {"name": "greedy", "rho": 0.6}, None)
        s.step(600)
        assert not [e for e in s.events if e["kind"] == "late"] and not s.late_items
