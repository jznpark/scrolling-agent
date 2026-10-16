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
