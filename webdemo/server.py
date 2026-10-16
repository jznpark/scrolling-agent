"""Local web demo: run the scrolling environment step by step and watch it in a browser.

Start it with `python scripts/serve.py` (see that file). The server keeps one
environment per browser session and answers small JSON requests:

    GET  /api/schema    settings (from the Config dataclasses), policies, branch label
    POST /api/reset     {session, seed, overrides: {"ddm.noise": 0.3}, policy: {...}} -> state
    POST /api/step      {session, n, action?} -> state after up to n steps
    POST /api/compare   {overrides, policy, n} -> behaviour with default vs these settings
    GET  /, /static/*   the page (webdemo/static)

The model code is whatever `scrolling_agent` package is first on sys.path, so the
same server can show any branch (scripts/serve.py --code <worktree>/src).
Only the standard library is used. It listens on 127.0.0.1 unless a host such as
0.0.0.0 is given (other PCs); there is no login, so open it only on trusted networks.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import threading
import time
from collections import deque
from dataclasses import fields, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .descriptions import BEHAVIOURS, DESCRIPTIONS, LABELS, SCENARIOS, UI_HELP

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
MAX_STEPS_PER_REQUEST = 600
MAX_SESSIONS = 32


# --- settings schema --------------------------------------------------------------


def _field_comments(module) -> dict[str, str]:
    """Trailing comments of dataclass fields in config.py ("name: type = value  # comment")."""
    comments: dict[str, str] = {}
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return comments
    for match in re.finditer(r"^\s+(\w+):\s*[^=\n]+=[^#\n]*#\s*(.+)$", source, flags=re.M):
        comments.setdefault(match.group(1), match.group(2).strip())
    return comments


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, tuple):
        return "tuple"
    return "str"


def settings_schema() -> list[dict[str, Any]]:
    """Every Config section and field with its default, kind, comment and choices."""
    from scrolling_agent import config as config_module

    comments = _field_comments(config_module)
    cfg = config_module.Config()
    sections = []
    for section in fields(cfg):
        obj = getattr(cfg, section.name)
        # a class docstring like "Numbers are [TBD] unless noted" tags untagged fields
        default_tag = re.search(r"\[(\w+)\] unless noted", type(obj).__doc__ or "")
        items = []
        for f in fields(obj):
            value = getattr(obj, f.name)
            choices = getattr(config_module, f"{f.name.upper()}_MODES", None)
            comment = comments.get(f.name, "")
            tag = re.match(r"\[(\w+)\]", comment)
            items.append({
                "name": f.name,
                "key": f"{section.name}.{f.name}",
                "kind": _kind(value),
                "default": list(value) if isinstance(value, tuple) else value,
                "choices": list(choices) if choices else None,
                "comment": comment,
                "help": DESCRIPTIONS.get(f"{section.name}.{f.name}", comment),
                "label": LABELS.get(f"{section.name}.{f.name}", f.name),
                "tag": tag[1] if tag else (default_tag[1] if default_tag else ""),
            })
        sections.append({"name": section.name, "fields": items})
    return sections


def build_config(overrides: dict[str, Any]):
    """Config with "section.field" overrides (values already typed by the page)."""
    from scrolling_agent.config import Config

    cfg = Config()
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        if not hasattr(cfg, section):
            raise ValueError(f"unknown section in {key!r}")
        current = getattr(cfg, section)
        names = {f.name: f for f in fields(current)}
        if name not in names:
            raise ValueError(f"unknown setting {key!r}")
        default = getattr(current, name)
        if isinstance(default, tuple):
            value = tuple(float(v) for v in value)
        elif isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        setattr(cfg, section, replace(current, **{name: value}))
    return cfg


def available_scenarios(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Situations whose settings all exist in the running branch."""
    keys = {f["key"] for s in sections for f in s["fields"]}
    return [sc for sc in SCENARIOS if set(sc["set"]) <= keys]


MAX_COMPARE_EPISODES = 30


def compare_behaviour(overrides: dict[str, Any], policy: dict[str, Any], runs_dir: Path | None,
                      n: int = 10, seed_start: int = 10_000) -> dict[str, Any]:
    """Average behaviour over the same n lists with default settings and with `overrides`."""
    from scrolling_agent import ScrollingEnv
    from scrolling_agent.evaluation import evaluate, summarize

    n = max(1, min(int(n), MAX_COMPARE_EPISODES))
    seeds = range(seed_start, seed_start + n)
    t0 = time.perf_counter()
    stats = {}
    for name, ov in (("base", {}), ("now", overrides)):
        env = ScrollingEnv(build_config(ov))
        stats[name] = summarize(evaluate(env, make_policy(policy, runs_dir), seeds))
    rows = [dict(b, base=round(stats["base"][b["key"]][0], 3), now=round(stats["now"][b["key"]][0], 3))
            for b in BEHAVIOURS if b["key"] in stats["base"]]
    return {"n": n, "seeds": [seeds[0], seeds[-1]], "seconds": round(time.perf_counter() - t0, 1), "rows": rows}


# --- policies ------------------------------------------------------------------------

BASELINES = ["human", "random", "scrollfirst", "greedy", "greedy-dwell"]


def list_policies(runs_dir: Path | None) -> list[str]:
    names = list(BASELINES)
    if runs_dir and runs_dir.is_dir():
        names += [f"ppo:{p.parent.name}" for p in sorted(runs_dir.glob("*/best_model.zip"))]
    return names


def make_policy(spec: dict[str, Any], runs_dir: Path | None):
    """None for human control, else a callable policy(env) -> action."""
    from scrolling_agent import policies

    name = spec.get("name", "greedy")
    rho = float(spec.get("rho", 0.6))
    dwell = float(spec.get("dwell", 0.0))
    if name == "human":
        return None
    if name == "random":
        return policies.RandomPolicy(int(spec.get("seed", 0)))
    if name == "scrollfirst":
        return policies.ScrollFirst(rho)
    if name in ("greedy", "greedy-dwell"):
        if name == "greedy-dwell" and "dwell" in inspect.signature(policies.GreedyKnown).parameters:
            return policies.GreedyKnown(rho, dwell=dwell or 0.5)
        return policies.GreedyKnown(rho)
    if name.startswith("ppo:") and runs_dir:
        # only runs that are listed (no paths from the browser: model files are pickles)
        if name not in list_policies(runs_dir):
            raise ValueError(f"unknown PPO run {name!r}")
        return policies.PPOPolicy.load(runs_dir / name[4:])
    raise ValueError(f"unknown policy {name!r}")


# --- one browser session -------------------------------------------------------------


def _revealed(env) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(env, "revealed_profile"):
        return env.revealed_profile()
    return np.where(env.understood, env.feed.A, 0.0), np.where(env.understood, env.feed.lam, 0.0)


def action_labels(env) -> list[str]:
    labels = [""] * env.n_actions
    speeds = env.cfg.screen.speeds
    for k, a in enumerate(env.down_actions):
        labels[a] = "아래로" + (f" ({speeds[k]:g})" if len(speeds) > 1 else "")
    for k, a in enumerate(env.up_actions):
        labels[a] = "위로" + (f" ({speeds[k]:g})" if len(speeds) > 1 else "")
    labels[env.stay_action] = "멈춤"
    for j, a in enumerate(env.slot_actions):
        labels[a] = f"{j + 1}번 카드 열기"
    labels[env.back_action] = "닫고 목록으로"
    return labels


class Session:
    def __init__(self, seed: int, overrides: dict[str, Any], policy: dict[str, Any], runs_dir: Path | None):
        from scrolling_agent import ScrollingEnv

        self.lock = threading.Lock()
        self.cfg = build_config(overrides)
        self.env = ScrollingEnv(self.cfg)
        self.env.reset(seed=seed)
        self.policy = make_policy(policy, runs_dir)
        self.labels = action_labels(self.env)
        self.seed = seed
        self.total = 0.0
        self.n_clicks = 0
        self.last_action: int | None = None
        self.last_watched: int | None = None  # most recent item opened (shown in the rate curve)
        self.late_items: set[int] = set()     # understood only after leaving the screen
        self.last_reward = 0.0
        self.events: deque[dict[str, Any]] = deque(maxlen=300)
        self.event_id = 0

    @property
    def done(self) -> bool:
        return self.env.step_count >= self.cfg.time.n_steps

    def _event(self, kind: str, text: str) -> None:
        self.event_id += 1
        self.events.append({"id": self.event_id, "t": round(self.env.step_count * self.cfg.time.dt, 2),
                            "kind": kind, "text": text})

    def step(self, n: int, action: int | None = None) -> list[list[float]]:
        """Advance up to n steps (one step when `action` is given); return [t, return, mode] per step."""
        from scrolling_agent.env import LIST, WATCH, visible_items

        env, history = self.env, []
        for _ in range(1 if action is not None else min(n, MAX_STEPS_PER_REQUEST)):
            if self.done:
                break
            mode_before, current_before = env.mode, env.current
            understood_before = env.understood.copy()
            a = action if action is not None else (self.policy(env) if self.policy else env.stay_action)
            if not 0 <= a < env.n_actions or not env.action_masks()[a]:
                self._event("invalid", f"지금은 '{self.labels[a] if 0 <= a < env.n_actions else a}'을(를) 할 수 없어 멈춤")
                a = env.stay_action
            if mode_before == LIST and a in env.slot_actions:
                item = int(env.visible_slots()[env.slot_actions.index(a)])
                self.n_clicks += 1
                note = " (다시 보기)" if env.w[item] > 0 else ""
                if item in self.late_items:
                    note += " ← 뒤늦게 알아챈 콘텐츠로 되돌아옴"
                self._event("click", f"#{item} 열기{note}")
            if mode_before == WATCH and a == env.back_action:
                self._event("back", f"#{current_before} 닫음 ({env.w[current_before]:.1f}초 봄)")
            _, reward, *_ = env.step(a)
            if env.current is not None:
                self.last_watched = int(env.current)
            self.total += reward
            self.last_action, self.last_reward = a, float(reward)
            on_screen = set(visible_items(env.scroll, env.n_items, env.screen_h, env.cfg.screen.visible_fraction))
            for i in np.flatnonzero(env.understood & ~understood_before):
                where = "" if i in on_screen else " — 지나간 뒤에 뒤늦게!"
                if where:
                    self.late_items.add(int(i))
                self._event("understood" if not where else "late", f"#{i} 뭔지 알아봄{where}")
            history.append([round(env.step_count * self.cfg.time.dt, 2), round(self.total, 4), int(env.mode)])
        return history

    def state(self, history: list, since_event: int = 0) -> dict[str, Any]:
        env, f = self.env, self.env.feed
        A_known, lam_known = _revealed(env)
        rate = A_known * np.exp(-lam_known * env.w)
        B = self.cfg.ddm.threshold
        items = [{
            "id": i,
            "seen": bool(env.exposure[i] > 0),
            "understood": bool(env.understood[i]),
            "late": i in self.late_items,
            "x": round(float(env.x[i] / B), 4),
            "exposure": round(float(env.exposure[i]), 2),
            "w": round(float(env.w[i]), 2),
            "A": round(float(A_known[i]), 4),
            "lam": round(float(lam_known[i]), 4),
            "rate": round(float(rate[i]), 4),
            "hidden": {"A": round(float(f.A[i]), 4), "nu": round(float(f.nu[i]), 4),
                       "lam": round(float(f.lam[i]), 4), "z": int(f.z[i]),
                       "rate": round(float(f.A[i] * np.exp(-f.lam[i] * env.w[i])), 4)},
        } for i in range(env.n_items)]
        extra = env.extra_metrics() if hasattr(env, "extra_metrics") else {}
        return {
            "seed": self.seed,
            "step": env.step_count,
            "n_steps": self.cfg.time.n_steps,
            "t": round(env.step_count * self.cfg.time.dt, 2),
            "T": self.cfg.time.session_seconds,
            "dt": self.cfg.time.dt,
            "done": self.done,
            "mode": ["LIST", "WATCH", "TRANSITION"][env.mode],
            "transition_to": ["LIST", "WATCH", "TRANSITION"][getattr(env, "_transition_to", 0)]
            if env.mode == 2 else None,
            "current": env.current,
            "last_watched": self.last_watched,
            "scroll": round(float(env.scroll), 4),
            "max_scroll": float(env.max_scroll),
            "screen_h": float(env.screen_h),
            "n_items": env.n_items,
            "slots": [int(i) for i in env.visible_slots()],
            "mask": [bool(m) for m in env.action_masks()],
            "labels": self.labels,
            "actions": {"down": env.down_actions, "up": env.up_actions, "stay": env.stay_action,
                        "slots": env.slot_actions, "back": env.back_action},
            "last_action": None if self.last_action is None else self.labels[self.last_action],
            "last_reward": round(self.last_reward, 4),
            "return": round(self.total, 4),
            "watch_reward": round(float(env.watch_reward_total), 4),
            "n_understood": int(env.understood.sum()),
            "n_clicks": self.n_clicks,
            "time_by_mode": {k: round(v, 2) for k, v in env.time_by_mode.items()},
            "extra_metrics": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in extra.items()},
            "items": items,
            "history": history,
            "events": [e for e in self.events if e["id"] > since_event],
        }


# --- HTTP ------------------------------------------------------------------------------


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind the same port silently; refuse instead.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address, runs_dir: Path | None = None, label: str = ""):
        super().__init__(address, DemoHandler)
        self.runs_dir = runs_dir
        self.label = label
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.Lock()


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/schema":
            from scrolling_agent import config as config_module

            sections = settings_schema()
            self._json({
                "label": self.server.label,
                "module": str(Path(config_module.__file__).resolve().parent),
                "sections": sections,
                "scenarios": available_scenarios(sections),
                "ui_help": UI_HELP,
                "behaviours": {b["key"]: b["label"] for b in BEHAVIOURS},
                "policies": list_policies(self.server.runs_dir),
            })
            return
        name = "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/")
        file = (STATIC / name).resolve()
        if STATIC.resolve() not in file.parents or not file.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        self._send(HTTPStatus.OK, file.read_bytes(), CONTENT_TYPES.get(file.suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return
        sid = str(body.get("session", "default"))[:64]
        try:
            if self.path == "/api/reset":
                session = Session(int(body.get("seed", 10_000)), body.get("overrides", {}),
                                  body.get("policy", {"name": "greedy"}), self.server.runs_dir)
                with self.server.sessions_lock:
                    if len(self.server.sessions) >= MAX_SESSIONS and sid not in self.server.sessions:
                        self.server.sessions.pop(next(iter(self.server.sessions)))
                    self.server.sessions[sid] = session
                with session.lock:
                    self._json(session.state([]))
            elif self.path == "/api/compare":
                self._json(compare_behaviour(body.get("overrides", {}), body.get("policy", {"name": "greedy"}),
                                             self.server.runs_dir, body.get("n", 10), int(body.get("seed", 10_000))))
            elif self.path == "/api/step":
                session = self.server.sessions.get(sid)
                if session is None:
                    self._error(HTTPStatus.NOT_FOUND, "no such session; reset first")
                    return
                action = body.get("action")
                with session.lock:
                    history = session.step(int(body.get("n", 1)), None if action is None else int(action))
                    self._json(session.state(history, int(body.get("since_event", 0))))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, TypeError, KeyError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))


def serve(port: int = 8000, runs_dir: Path | None = None, label: str = "", host: str = "127.0.0.1") -> DemoServer:
    """Create the server (call .serve_forever() on it)."""
    return DemoServer((host, port), runs_dir=runs_dir, label=label)
