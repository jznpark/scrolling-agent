"""Plots of one recorded episode (see `evaluation.run_episode(..., record=True)`).

`plot_episode`   one static figure: scroll path, understanding and watching over
                 time, cumulative reward, value vs watch time, time to understand
`animate_episode` step-by-step playback of the list as the agent sees it

Colours by latent type z and true A, nu of items not yet understood are hidden
from the agent; they are drawn only for analysis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from .env import LIST, MODE_NAMES, TRANSITION, WATCH, visible_items
from .evaluation import EpisodeTrace

Z_COLORS = ("#1f77b4", "#ff7f0e")  # z = 0, z = 1
WATCH_COLOR = "#2ca02c"
TRANSITION_COLOR = "0.6"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of consecutive True values."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _first_understood(trace: EpisodeTrace) -> tuple[np.ndarray, np.ndarray]:
    """Items understood during the episode and the step index at which each was."""
    ever = trace.understood[-1]
    step = np.argmax(trace.understood, axis=0)
    return np.flatnonzero(ever), step[ever]


def unread_on_screen(trace: EpisodeTrace, k: int) -> int:
    """Visible, not yet understood items after step k (the N of the attention split)."""
    vis = visible_items(trace.scroll[k], trace.feed.n, trace.screen_h, 0.5)
    return int(np.sum(~trace.understood[k, vis]))


def episode_summary(trace: EpisodeTrace) -> dict[str, float]:
    f, w = trace.feed, trace.w[-1]
    watch_runs = _runs(trace.mode == WATCH)
    return {
        "return": float(trace.reward.sum()),
        "watch reward": float(np.sum(f.A / f.lam * (1 - np.exp(-f.lam * w)))),
        "understood": int(trace.understood[-1].sum()),
        "watch bouts": len(watch_runs),
        "items watched": int((w > 0).sum()),
        "time LIST (s)": float(np.sum(trace.mode == 0) * trace.dt),
        "time WATCH (s)": float(np.sum(trace.mode == WATCH) * trace.dt),
        "time TRANSITION (s)": float(np.sum(trace.mode == TRANSITION) * trace.dt),
        "deepest item seen": int(np.flatnonzero(trace.exposure[-1] > 0).max(initial=-1)),
        "mean unread on screen": float(np.mean([unread_on_screen(trace, k)
                                                for k in np.flatnonzero(trace.mode == LIST)] or [0])),
    }


def plot_episode(trace: EpisodeTrace, title: str | None = None) -> plt.Figure:
    f, t, dt, H = trace.feed, trace.t, trace.dt, trace.screen_h
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    gs = fig.add_gridspec(3, 3)
    ax_path = fig.add_subplot(gs[:2, :2])
    ax_rew = fig.add_subplot(gs[2, :2], sharex=ax_path)
    ax_val = fig.add_subplot(gs[0, 2])
    ax_ddm = fig.add_subplot(gs[1, 2])
    ax_txt = fig.add_subplot(gs[2, 2])

    # scroll path: the screen window, understanding events, watch bouts
    ax_path.fill_between(t - dt, trace.scroll, trace.scroll + H, step="post", color="0.88", label="screen")
    items, steps = _first_understood(trace)
    for z in (0, 1):
        sel = f.z[items] == z
        ax_path.scatter(t[steps[sel]], items[sel] + 0.5, s=30, facecolors="none",
                        edgecolors=Z_COLORS[z], label=f"understood (z={z})")
    for a, b in _runs(trace.mode == WATCH):
        i = trace.current[a]
        ax_path.plot([t[a] - dt, t[b - 1]], [i + 0.5] * 2, lw=6, solid_capstyle="butt",
                     color=Z_COLORS[f.z[i]])
    for a, b in _runs(trace.mode == TRANSITION):
        ax_path.axvspan(t[a] - dt, t[b - 1], color=TRANSITION_COLOR, alpha=0.25, lw=0)
    ax_path.plot([], [], lw=6, color="0.3", label="watching (colour = z)")
    ax_path.set_ylim(trace.scroll.max() + H + 1, 0)
    ax_path.set_ylabel("list position (item)")
    ax_path.set_title("Scroll path · understanding · watching (grey bands = click / back)")
    ax_path.legend(loc="lower right", fontsize=8)

    # cumulative reward with watch periods shaded
    cum = np.cumsum(trace.reward)
    ax_rew.plot(t, cum, color="k", lw=1.5, label="return")
    for a, b in _runs(trace.mode == WATCH):
        ax_rew.axvspan(t[a] - dt, t[b - 1], color=WATCH_COLOR, alpha=0.15, lw=0)
    ax_rew.axhline(0, color="0.7", lw=0.8)
    ax_rew.set_xlabel("time (s)")
    ax_rew.set_ylabel("cumulative reward")
    ax_rew.set_title("Cumulative reward (green = watching)")

    # value vs watch time for items that were on screen
    seen = trace.exposure[-1] > 0
    und = trace.understood[-1]
    for z in (0, 1):
        for flag, marker in ((True, "o"), (False, "x")):
            sel = seen & und if flag else seen & ~und
            sel &= f.z == z
            ax_val.scatter(f.A[sel], trace.w[-1][sel], c=Z_COLORS[z], marker=marker, s=25,
                           label=f"z={z}, {'understood' if flag else 'not understood'}")
    ax_val.set_xlabel("true A (hidden until understood)")
    ax_val.set_ylabel("watch time w (s)")
    ax_val.set_title("Did it watch the valuable items?")
    ax_val.legend(fontsize=7)

    # time to understand vs drift
    if items.size:
        exp_at = trace.exposure[steps, items]
        ax_ddm.scatter(f.nu[items], exp_at, c=[Z_COLORS[z] for z in f.z[items]], s=25)
        nu_grid = np.linspace(f.nu[items].min(), f.nu[items].max(), 50)
        ax_ddm.plot(nu_grid, trace.threshold / nu_grid, "k--", lw=1, label="B/ν (still screen)")
        ax_ddm.legend(fontsize=8)
    ax_ddm.set_xlabel("true ν (hidden)")
    ax_ddm.set_ylabel("exposure until understood (s)")
    ax_ddm.set_title("Time to understand")

    ax_txt.axis("off")
    summary = episode_summary(trace)
    ax_txt.text(0.0, 1.0, "\n".join(f"{k:<20} {v:8.2f}" if isinstance(v, float) else f"{k:<20} {v:8d}"
                                    for k, v in summary.items()),
                family="monospace", fontsize=10, va="top")
    if title:
        fig.suptitle(title)
    return fig


def animate_episode(
    trace: EpisodeTrace, every: int = 2, interval: int = 50, show_hidden: bool = True
) -> tuple[plt.Figure, FuncAnimation]:
    """Playback: the list around the screen (left) and the cumulative reward (right)."""
    f, t, H, B = trace.feed, trace.t, trace.screen_h, trace.threshold
    cum = np.cumsum(trace.reward)
    fig, (ax_list, ax_rew) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.1, 1]},
                                          layout="constrained")
    ax_rew.plot(t, cum, color="0.85")
    (line,) = ax_rew.plot([], [], color="k")
    (dot,) = ax_rew.plot([], [], "o", color="k")
    ax_rew.set_xlabel("time (s)")
    ax_rew.set_ylabel("cumulative reward")

    def draw(k: int):
        ax_list.cla()
        s, mode, cur = trace.scroll[k], trace.mode[k], trace.current[k]
        lo, hi = max(int(np.floor(s)) - 1, 0), min(int(np.ceil(s + H)) + 1, f.n)
        for i in range(lo, hi):
            und = trace.understood[k, i]
            selected = cur == i
            ax_list.add_patch(Rectangle((0.03, i + 0.05), 0.94, 0.9, lw=3 if selected else 1,
                                        facecolor="#e3f4e3" if und else "white",
                                        edgecolor="#d62728" if selected else "0.4"))
            frac = float(np.clip(trace.x[k, i] / B, 0, 1))
            ax_list.add_patch(Rectangle((0.06, i + 0.72), 0.88 * frac, 0.14,
                                        color=WATCH_COLOR if und else "#1f77b4"))
            if und:
                w = trace.w[k, i]
                rate = f.A[i] * np.exp(-f.lam[i] * w)
                label = f"#{i}  A={f.A[i]:.2f}  λ={f.lam[i]:.2f}  w={w:.1f}s  rate={rate:.2f}"
            else:
                label = f"#{i}  evidence {trace.x[k, i] / B:.2f}  seen {trace.exposure[k, i]:.1f}s"
            ax_list.text(0.06, i + 0.3, label, fontsize=10, va="center", clip_on=True)
            if show_hidden and not und:
                ax_list.text(0.06, i + 0.55, f"hidden: A={f.A[i]:.2f} ν={f.nu[i]:.2f} z={f.z[i]}",
                             fontsize=8, color="0.55", va="center", style="italic", clip_on=True)
        ax_list.add_patch(Rectangle((0, s), 1, H, fill=False, lw=2, ls="--", edgecolor="k"))
        ax_list.set_xlim(0, 1)
        ax_list.set_ylim(s + H + 1, s - 1)
        ax_list.set_xticks([])
        ax_list.set_ylabel("list position")
        what = f" — item #{cur}" if cur >= 0 else ""
        crowd = f"   unread on screen: {unread_on_screen(trace, k)}" if mode == LIST else ""
        ax_list.set_title(f"t = {t[k]:5.1f} s   {MODE_NAMES[mode]}{what}{crowd}   (dashed box = screen)")
        line.set_data(t[: k + 1], cum[: k + 1])
        dot.set_data([t[k]], [cum[k]])
        ax_rew.set_title(f"return so far {cum[k]:.2f}")
        return []

    anim = FuncAnimation(fig, draw, frames=range(0, len(t), every), interval=interval, repeat=False)
    return fig, anim
