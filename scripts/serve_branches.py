"""Run the web demo for several branches at once and switch between them with tabs.

Run (PowerShell, venv active, from this checkout):
    python scripts\\serve_branches.py --open
    python scripts\\serve_branches.py --branches feature/value-error feature/attention-split
    python scripts\\serve_branches.py --host 0.0.0.0              # other PCs on the same network
    python scripts\\serve_branches.py --token mypass              # ask for ?token=mypass once
    cloudflared tunnel --url http://localhost:8000                # share outside (see README)

Each branch is checked out (detached) into .worktrees/<branch> and served by its own
demo server on this PC only (ports base + 1, + 2, ...). Everything is reached through
the base port: the tab page at /, branch k at /b/k/ (forwarded), so a single port is
enough for the firewall or a tunnel. The branches are not modified. Each branch must
contain the shared tooling (based on feature/ppo or later). Use origin/<branch> names
to run what is on GitHub.
"""

from __future__ import annotations

import argparse
import html
import http.client
import re
import subprocess
import sys
import time
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WORKTREES = ROOT / ".worktrees"
DEFAULT_BRANCHES = ["feature/ppo", "feature/value-error", "feature/offscreen-residual", "feature/attention-split"]
PROXY_TIMEOUT = 180  # s; a behaviour comparison can take a while


def git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def checkout(branch: str) -> Path:
    """Detached worktree of `branch` (created or moved to the branch's current commit)."""
    sha = git("rev-parse", "--verify", f"{branch}^{{commit}}")
    path = WORKTREES / branch.replace("/", "_")
    if path.exists():
        if git("status", "--porcelain", cwd=path):
            print(f"  {path} has local changes; using it as is")
        else:
            git("checkout", "--quiet", "--detach", sha, cwd=path)
    else:
        WORKTREES.mkdir(exist_ok=True)
        git("worktree", "add", "--quiet", "--detach", str(path), sha)
    return path


BRANCH_NAMES = {  # branch name fragment -> everyday name of what that branch adds
    "value-error": "썸네일 착각 (가치 추정 오차)",
    "offscreen-residual": "늦은 깨달음 (화면 밖 잔여 처리)",
    "attention-split": "화면 붐빔 (주의 분산)",
    "ppo": "기본 모델",
    "web-demo": "기본 모델",
    "main": "기본 모델",
}


def friendly_name(branch: str) -> str:
    for fragment, name in BRANCH_NAMES.items():
        if fragment in branch:
            return name
    return branch


def hub_page(labels: list[str]) -> bytes:
    """One page with a tab per branch; only the selected branch is shown."""
    tabs = "\n".join(
        f'<button class="tab" data-i="{k}" title="{html.escape(label)}">{html.escape(friendly_name(label))}'
        f'<small>{html.escape(label.split(" @ ")[0])}</small></button>'
        for k, label in enumerate(labels)
    )
    frames = "\n".join(
        f'<iframe data-i="{k}" data-path="b/{k + 1}/" title="{html.escape(label)}"></iframe>'
        for k, label in enumerate(labels)
    )
    policy_options = "".join(f'<option{" selected" if p == "greedy" else ""}>{p}</option>'
                             for p in ["greedy", "greedy-dwell", "scrollfirst", "random", "human"])
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>브랜치 비교</title>
<style>
:root {{ --bg: #f5f6f8; --panel: #fff; --text: #1f2328; --muted: #6b7280; --border: #d8dce2; --accent: #2563eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111317; --panel: #1a1d23; --text: #e6e8eb; --muted: #9aa1ab; --border: #2f343d; --accent: #6ea0ff; }} }}
* {{ box-sizing: border-box; }}
html, body {{ height: 100%; margin: 0; }}
body {{ background: var(--bg); color: var(--text); font: 14px system-ui, "Malgun Gothic", sans-serif; display: flex; flex-direction: column; }}
header {{ background: var(--panel); border-bottom: 1px solid var(--border); padding: 8px 14px 0; }}
.row {{ display: flex; gap: 8px 14px; align-items: center; flex-wrap: wrap; }}
.tabs {{ display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }}
.tab {{ font: inherit; font-size: 15px; font-weight: 600; color: var(--muted); background: transparent; border: 1px solid transparent;
  border-bottom: none; border-radius: 8px 8px 0 0; padding: 8px 16px 6px; cursor: pointer; display: flex; flex-direction: column; align-items: flex-start; }}
.tab small {{ font-size: 11px; font-weight: 400; }}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--accent); background: var(--bg); border-color: var(--border); margin-bottom: -1px; }}
.sync {{ font-size: 13px; color: var(--muted); }}
.sync input, .sync select, .sync button {{ font: inherit; color: var(--text); background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; }}
.sync input[type=number] {{ width: 90px; }}
.sync button {{ cursor: pointer; }}
main {{ position: relative; flex: 1; min-height: 0; }}
iframe {{ position: absolute; inset: 0; width: 100%; height: 100%; border: 0; visibility: hidden; }}
iframe.active {{ visibility: visible; }}
</style></head><body>
<header>
  <div class="row sync">
    <b style="color:var(--text);font-size:15px">브랜치 비교</b>
    <span>모든 탭에 같이 적용:</span>
    <label>피드 번호 <input id="seed" type="number" value="10000"></label>
    <label>누가 조작? <select id="policy">{policy_options}</select></label>
    <label>속도 <input id="speed" type="number" value="20" min="1" max="200"></label>
    <label><input id="hidden" type="checkbox"> 숨은 값</label>
    <button data-cmd="reset">같은 조건으로 다시 시작</button>
    <button data-cmd="play">▶ 모두 재생</button>
    <button data-cmd="pause">⏸ 모두 멈춤</button>
    <a id="open" target="_blank" style="margin-left:auto;color:var(--muted)">이 탭만 새 창으로 ↗</a>
  </div>
  <div class="tabs">{tabs}</div>
</header>
<main>{frames}</main>
<script>
// every branch is served by this same address under b/<k>/ (one port for firewalls and tunnels)
const frames = [...document.querySelectorAll("iframe")];
const tabs = [...document.querySelectorAll(".tab")];
frames.forEach((f) => (f.src = f.dataset.path + "?compact=1"));
function show(i) {{
  i = Math.max(0, Math.min(frames.length - 1, i));
  frames.forEach((f, k) => f.classList.toggle("active", k === i));
  tabs.forEach((t, k) => t.classList.toggle("active", k === i));
  document.getElementById("open").href = frames[i].dataset.path;
  history.replaceState(null, "", "#" + (i + 1));
  frames[i].focus();
}}
tabs.forEach((t) => t.addEventListener("click", () => show(parseInt(t.dataset.i, 10))));
show(parseInt(location.hash.slice(1) || "1", 10) - 1);
document.querySelectorAll("header button[data-cmd]").forEach((b) => b.addEventListener("click", () => {{
  const msg = {{ type: "demo", cmd: b.dataset.cmd, seed: parseInt(document.getElementById("seed").value, 10),
    policy: document.getElementById("policy").value, speed: parseInt(document.getElementById("speed").value, 10),
    hidden: document.getElementById("hidden").checked }};
  frames.forEach((f) => f.contentWindow.postMessage(msg, "*"));
}}));
</script></body></html>"""
    return page.encode()


LOGIN_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>접속 비밀번호</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font:15px system-ui,'Malgun Gothic',sans-serif;max-width:420px;margin:15vh auto;padding:0 16px">
<h2>Scrolling Agent 데모</h2><p>받은 링크에 들어 있는 비밀번호가 필요합니다.</p>
<form method="get"><input name="token" type="password" placeholder="비밀번호" autofocus
 style="font:inherit;padding:8px;width:100%"><button style="font:inherit;padding:8px 16px;margin-top:8px">들어가기</button></form>
</body></html>""".encode()


def make_hub(host: str, port: int, labels: list[str], ports: list[int], token: str | None = None) -> ThreadingHTTPServer:
    """Comparison page at /, branch k forwarded from /b/k/ to its local demo server."""
    page = hub_page(labels)

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, status: int, body: bytes, content_type: str = "text/html; charset=utf-8", extra=()):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in extra:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            """Without a token everyone may enter; with one, ?token=... once sets a cookie."""
            if not token:
                return True
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            if "demo_token" in cookie and cookie["demo_token"].value == token:
                return True
            url = urlsplit(self.path)
            query = dict(parse_qsl(url.query))
            if query.pop("token", None) == token:
                rest = urlencode(query)
                self._send(303, b"", extra=[("Location", url.path + (f"?{rest}" if rest else "")),
                                            ("Set-Cookie", f"demo_token={token}; Path=/; HttpOnly; SameSite=Lax")])
            else:
                self._send(401, LOGIN_PAGE)
            return False

        def _handle(self, method: str) -> None:
            if not self._authorized():
                return
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, page)
                return
            match = re.match(r"^/b/(\d+)(/.*)?$", path)
            if not match or not 1 <= int(match[1]) <= len(ports):
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            if match[2] is None:  # /b/2 -> /b/2/ so relative links resolve
                self._send(301, b"", extra=[("Location", f"/b/{match[1]}/")])
                return
            target = self.path[len(f"/b/{match[1]}"):]
            body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) if method == "POST" else None
            conn = http.client.HTTPConnection("127.0.0.1", ports[int(match[1]) - 1], timeout=PROXY_TIMEOUT)
            try:
                conn.request(method, target, body=body,
                             headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
                resp = conn.getresponse()
                self._send(resp.status, resp.read(), resp.getheader("Content-Type", "application/octet-stream"))
            except OSError as exc:
                self._send(502, f"branch server unavailable: {exc}".encode(), "text/plain; charset=utf-8")
            finally:
                conn.close()

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    server = ThreadingHTTPServer((host, port), Hub)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES)
    parser.add_argument("--port", type=int, default=8000, help="the one public port; branches use port+1, +2, ... locally")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 = this PC only (default, also for tunnels); 0.0.0.0 = other PCs on the network")
    parser.add_argument("--token", default=None, help="password: visitors need ?token=<this> once")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    children, labels, ports = [], [], []
    try:
        for k, branch in enumerate(args.branches, start=1):
            path = checkout(branch)
            port = args.port + k
            label = f"{branch} @ {git('rev-parse', '--short', 'HEAD', cwd=path)}"
            cmd = [sys.executable, str(ROOT / "scripts" / "serve.py"), "--port", str(port),
                   "--code", str(path / "src"), "--label", label, "--runs-dir", str(ROOT / "runs")]
            children.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
            labels.append(label)
            ports.append(port)
            print(f"  /b/{k}/  {label}")
        time.sleep(1.5)
        for proc, label in zip(children, labels):
            if proc.poll() is not None:
                raise SystemExit(f"server for {label} stopped (exit {proc.returncode})")

        hub = make_hub(args.host, args.port, labels, ports, args.token)
        suffix = f"/?token={args.token}" if args.token else ""
        url = f"http://127.0.0.1:{args.port}{suffix}"
        print(f"Comparison page: {url}  - Ctrl+C to stop all", flush=True)
        if args.host not in ("127.0.0.1", "localhost"):
            sys.path.insert(0, str(ROOT / "scripts"))
            from serve import lan_addresses

            for ip in lan_addresses():
                print(f"  same network: http://{ip}:{args.port}{suffix}  (only port {args.port} is needed)", flush=True)
        print(f"  outside: cloudflared tunnel --url http://localhost:{args.port}"
              + ("   (no --token: anyone with the link can use it)" if not args.token else ""), flush=True)
        if args.open:
            webbrowser.open(url)
        hub.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in children:
            proc.terminate()
        for proc in children:
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
