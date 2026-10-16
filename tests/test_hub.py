"""Branch comparison page: one public port forwarding /b/<k>/ to local demo servers."""

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from serve_branches import friendly_name, make_hub  # noqa: E402

from webdemo.server import serve  # noqa: E402


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture(scope="module")
def demo():
    srv = serve(0)
    start(srv)
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def get(url, data=None, headers=None, follow=True):
    req = urllib.request.Request(url, data=data, headers=headers or {"Content-Type": "application/json"})
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    with opener.open(req, timeout=20) as res:
        return res.status, res.read(), res.headers


def test_friendly_names():
    assert friendly_name("feature/offscreen-residual @ abc") == "늦은 깨달음 (화면 밖 잔여 처리)"
    assert friendly_name("feature/something-else") == "feature/something-else"


def test_one_port_forwards_pages_and_api(demo):
    hub = make_hub("127.0.0.1", 0, ["feature/ppo @ abc"], [demo])
    base = start(hub)
    try:
        status, page, _ = get(base + "/")
        assert status == 200 and b'data-path="b/1/"' in page and "기본 모델".encode() in page
        with pytest.raises(urllib.error.HTTPError) as err:  # /b/1 -> /b/1/
            get(base + "/b/1", follow=False)
        assert err.value.code == 301 and err.value.headers["Location"] == "/b/1/"
        assert b'src="static/app.js"' in get(base + "/b/1/")[1]
        assert b"api(" in get(base + "/b/1/static/app.js")[1]
        schema = json.loads(get(base + "/b/1/api/schema")[1])
        assert "greedy" in schema["policies"]
        state = json.loads(get(base + "/b/1/api/reset", json.dumps({"session": "h", "seed": 3}).encode())[1])
        assert state["step"] == 0
        with pytest.raises(urllib.error.HTTPError) as err:
            get(base + "/b/9/")
        assert err.value.code == 404
    finally:
        hub.shutdown()
        hub.server_close()


def test_token_sets_cookie(demo):
    hub = make_hub("127.0.0.1", 0, ["feature/ppo @ abc"], [demo], token="pw")
    base = start(hub)
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            get(base + "/b/1/api/schema")
        assert err.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as err:
            get(base + "/?token=pw", follow=False)
        assert err.value.code == 303 and "demo_token=pw" in err.value.headers["Set-Cookie"]
        status, _, _ = get(base + "/b/1/api/schema", headers={"Cookie": "demo_token=pw"})
        assert status == 200
        with pytest.raises(urllib.error.HTTPError) as err:
            get(base + "/", headers={"Cookie": "demo_token=wrong"})
        assert err.value.code == 401
    finally:
        hub.shutdown()
        hub.server_close()


def test_late_items_are_marked():
    from webdemo.server import Session

    s = Session(10_000, {}, {"name": "greedy"}, None)
    s.late_items.add(5)
    assert s.state([])["items"][5]["late"] is True
