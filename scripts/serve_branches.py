"""Run the web demo for several branches at once and compare them side by side.

Run (PowerShell, venv active, from this checkout):
    python scripts\\serve_branches.py --open
    python scripts\\serve_branches.py --branches feature/value-error feature/attention-split

Each branch is checked out (detached) into .worktrees/<branch> and served from its
own port (base port + 1, + 2, ...); the base port shows all of them in one page.
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


def hub_page(servers: list[tuple[str, int]]) -> bytes:
    frames = "\n".join(
        f'<section><h2>{html.escape(name)} <a href="http://127.0.0.1:{port}" target="_blank">:{port} ↗</a></h2>'
        f'<iframe src="http://127.0.0.1:{port}/?compact=1" title="{html.escape(name)}"></iframe></section>'
        for name, port in servers
    )
    policy_options = "".join(f'<option{" selected" if p == "greedy" else ""}>{p}</option>'
                             for p in ["greedy", "greedy-dwell", "scrollfirst", "random"])
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Branch comparison</title>
<style>
:root {{ --bg: #f5f6f8; --text: #1f2328; --border: #d8dce2; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111317; --text: #e6e8eb; --border: #2f343d; }} }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px system-ui, "Malgun Gothic", sans-serif; }}
header {{ padding: 8px 16px; border-bottom: 1px solid var(--border); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
header input, header select, header button {{ font: inherit; padding: 3px 8px; }}
header input {{ width: 90px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(900px, 1fr)); gap: 10px; padding: 10px; }}
section {{ border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
h2 {{ font-size: 14px; margin: 0; padding: 6px 10px; border-bottom: 1px solid var(--border); }}
a {{ color: inherit; font-weight: normal; font-size: 12px; }}
iframe {{ width: 100%; height: 900px; border: 0; display: block; }}
</style></head><body>
<header><b>브랜치 비교</b>
  <label>seed <input id="seed" type="number" value="10000"></label>
  <label>policy <select id="policy">{policy_options}</select></label>
  <label>속도 <input id="speed" type="number" value="20" min="1" max="200"></label>
  <label><input id="hidden" type="checkbox"> 숨은 값</label>
  <button data-cmd="reset">모두 같은 조건으로 다시 시작</button>
  <button data-cmd="play">▶ 모두 재생</button>
  <button data-cmd="pause">⏸ 모두 멈춤</button>
  <button data-cmd="step">한 스텝</button>
  <span style="opacity:.7">각 화면의 설정(확장 값 등)은 화면마다 따로 바꿀 수 있습니다. 크게 보려면 ↗</span>
</header>
<main>{frames}</main>
<script>
document.querySelectorAll("header button").forEach((b) => b.addEventListener("click", () => {{
  const msg = {{ type: "demo", cmd: b.dataset.cmd, seed: parseInt(document.getElementById("seed").value, 10),
    policy: document.getElementById("policy").value, speed: parseInt(document.getElementById("speed").value, 10),
    hidden: document.getElementById("hidden").checked }};
  document.querySelectorAll("iframe").forEach((f) => f.contentWindow.postMessage(msg, "*"));
}}));
</script></body></html>"""
    return page.encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES)
    parser.add_argument("--port", type=int, default=8000, help="comparison page; branches use port+1, +2, ...")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    children, servers = [], []
    try:
        for k, branch in enumerate(args.branches, start=1):
            path = checkout(branch)
            port = args.port + k
            label = f"{branch} @ {git('rev-parse', '--short', 'HEAD', cwd=path)}"
            cmd = [sys.executable, str(ROOT / "scripts" / "serve.py"), "--port", str(port),
                   "--code", str(path / "src"), "--label", label, "--runs-dir", str(ROOT / "runs")]
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

        hub = ThreadingHTTPServer(("127.0.0.1", args.port), Hub)
        url = f"http://127.0.0.1:{args.port}"
        print(f"Comparison page: {url}  - Ctrl+C to stop all", flush=True)
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
