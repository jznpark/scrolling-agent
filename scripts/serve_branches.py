"""Run the web demo for several branches at once and switch between them with tabs.

Run (PowerShell, venv active, from this checkout):
    python scripts\\serve_branches.py --open
    python scripts\\serve_branches.py --branches feature/value-error feature/attention-split
    python scripts\\serve_branches.py --host 0.0.0.0     # also other PCs (open ports 8000-8004 in the firewall)

Each branch is checked out (detached) into .worktrees/<branch> and served from its
own port (base port + 1, + 2, ...); the base port shows them as tabs.
The branches are not modified. Each branch must contain the shared tooling
(revealed_profile / extra_metrics, i.e. be based on feature/ppo or later).
Use origin/<branch> names to run what is on GitHub.
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKTREES = ROOT / ".worktrees"
DEFAULT_BRANCHES = ["feature/ppo", "feature/value-error", "feature/offscreen-residual", "feature/attention-split"]


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


def hub_page(servers: list[tuple[str, int]]) -> bytes:
    """One page with a tab per branch; only the selected branch is shown."""
    tabs = "\n".join(
        f'<button class="tab" data-i="{k}" title="{html.escape(label)}">{html.escape(friendly_name(label))}'
        f'<small>{html.escape(label.split(" @ ")[0])}</small></button>'
        for k, (label, _) in enumerate(servers)
    )
    frames = "\n".join(
        f'<iframe data-i="{k}" data-port="{port}" title="{html.escape(label)}"></iframe>'
        for k, (label, port) in enumerate(servers)
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
// each branch runs on its own port of the same host this page was opened from
const base = (port) => `${{location.protocol}}//${{location.hostname}}:${{port}}`;
const frames = [...document.querySelectorAll("iframe")];
const tabs = [...document.querySelectorAll(".tab")];
frames.forEach((f) => (f.src = base(f.dataset.port) + "/?compact=1"));
function show(i) {{
  i = Math.max(0, Math.min(frames.length - 1, i));
  frames.forEach((f, k) => f.classList.toggle("active", k === i));
  tabs.forEach((t, k) => t.classList.toggle("active", k === i));
  document.getElementById("open").href = base(frames[i].dataset.port);
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES)
    parser.add_argument("--port", type=int, default=8000, help="comparison page; branches use port+1, +2, ...")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 = this PC only (default); 0.0.0.0 = also other PCs on the network")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    children, servers = [], []
    try:
        for k, branch in enumerate(args.branches, start=1):
            path = checkout(branch)
            port = args.port + k
            label = f"{branch} @ {git('rev-parse', '--short', 'HEAD', cwd=path)}"
            cmd = [sys.executable, str(ROOT / "scripts" / "serve.py"), "--port", str(port),
                   "--code", str(path / "src"), "--label", label, "--runs-dir", str(ROOT / "runs"),
                   "--host", args.host]
            children.append(subprocess.Popen(cmd))
            servers.append((label, port))
            print(f"  {label:<45} http://127.0.0.1:{port}")
        time.sleep(1.5)
        for proc, (label, _) in zip(children, servers):
            if proc.poll() is not None:
                raise SystemExit(f"server for {label} stopped (exit {proc.returncode})")

        page = hub_page(servers)

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

        hub = ThreadingHTTPServer((args.host, args.port), Hub)
        url = f"http://127.0.0.1:{args.port}"
        print(f"Comparison page: {url}  - Ctrl+C to stop all", flush=True)
        if args.host not in ("127.0.0.1", "localhost"):
            sys.path.insert(0, str(ROOT / "scripts"))
            from serve import lan_addresses

            for ip in lan_addresses():
                print(f"  other PCs: http://{ip}:{args.port}  (uses ports {args.port}-{args.port + len(servers)})",
                      flush=True)
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
