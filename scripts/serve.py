"""Watch and play the scrolling environment in a browser (local web demo).

Run (PowerShell, venv active):
    python scripts\\serve.py                     # http://127.0.0.1:8000, code of this checkout
    python scripts\\serve.py --port 8001 --code .worktrees\\feature_value-error\\src --label value-error
    python scripts\\serve_branches.py            # several branches side by side (see that script)

--code picks which `scrolling_agent` package to run (e.g. another branch's worktree);
the web demo itself always comes from this checkout. PPO runs with best_model.zip
under --runs-dir appear as policies.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_label(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return f"{out} @ {sha}" if out != "HEAD" else sha
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--code", type=Path, default=ROOT / "src", help="src folder with scrolling_agent")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--label", default=None, help="name shown in the page header (default: git branch)")
    parser.add_argument("--open", action="store_true", help="open the browser")
    args = parser.parse_args()

    code = args.code.resolve()
    if not (code / "scrolling_agent").is_dir():
        raise SystemExit(f"no scrolling_agent package in {code}")
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(ROOT))  # webdemo package

    from webdemo.server import serve

    label = args.label or git_label(code)
    server = serve(args.port, runs_dir=args.runs_dir, label=label)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Scrolling agent demo [{label}] at {url}  (code: {code})  - Ctrl+C to stop", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
