"""Presentation figures for the kappa-order report (Korean labels, half a 16:9 slide).

    python scripts\\experiments\\figures_kappa_order.py
Reads results/kappa_order/*.csv, writes docs/reports/images/2026-10-27-kappa-order/*.png
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "kappa_order"
OUT = ROOT / "docs" / "reports" / "images" / "2026-10-27-kappa-order"

# reference palette (dataviz skill, light mode), categorical slots in fixed order
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TIME_KEYS = ["time_watch", "time_down", "time_transition", "time_stay", "time_up", "time_blocked"]
TIME_NAME = {"time_watch": "시청", "time_down": "아래로 이동", "time_transition": "열고 닫기",
             "time_stay": "멈춰서 보기", "time_up": "위로 이동", "time_blocked": "막힌 이동"}
KAPPAS = ["0", "0.5", "0.8", "0.9"]
SIZE = (6.4, 5.4)  # inches: about half of a 13.33 x 7.5 slide

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load(name):
    with (RES / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean_se(rows, key):
    x = np.array([float(r[key]) for r in rows], dtype=float)
    x = x[~np.isnan(x)]
    return x.mean(), x.std(ddof=1) / math.sqrt(x.size)


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# --- figure 1: return and revisit clicks by kappa, K1 | K2 ------------------------------------


def figure1():
    data = {exp: load(f"{exp}_per_feed.csv") for exp in ("k1", "k2")}
    fig, axes = plt.subplots(2, 2, figsize=SIZE, sharex=True, sharey="row")
    rows_spec = [("return", "누적 보상 (점)"), ("revisit_clicks", "되돌아가 연 횟수 (회)")]
    pols = [("고정", "고정 규칙 (ρ 0.6)", SERIES[0], -0.07), ("다시", "κ마다 다시 고른 규칙", SERIES[1], 0.07)]
    titles = {"k1": "K1 기본 모델", "k2": "K2 늦은 깨달음 (τ 2초)"}
    x = np.arange(len(KAPPAS))
    for c, exp in enumerate(("k1", "k2")):
        for r, (key, ylabel) in enumerate(rows_spec):
            ax = axes[r, c]
            style(ax)
            for prefix, label, colour, dx in pols:
                g = [row for row in data[exp] if row["policy"].startswith(prefix)]
                ms = [mean_se([row for row in g if row["kappa_label"] == k], key) for k in KAPPAS]
                m = np.array([v[0] for v in ms])
                se = np.array([v[1] for v in ms])
                ax.errorbar(x + dx, m, yerr=se, color=colour, linewidth=2, marker="o", markersize=5,
                            capsize=0, elinewidth=1.2, label=label, zorder=3)
                if prefix == "다시" and r == 0:
                    for xi, k in zip(x, KAPPAS):
                        rho = next(row["rho"] for row in g if row["kappa_label"] == k)
                        if float(rho) != 0.6:
                            ax.annotate(f"ρ {float(rho):g}", (xi + dx, m[xi] + se[xi]), textcoords="offset points",
                                        xytext=(0, 5), ha="center", fontsize=8, color=INK2)
            if r == 0:
                ax.set_title(titles[exp], loc="left")
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(x, [f"κ {k}" for k in KAPPAS])
    axes[0, 0].set_ylim(48, 57)
    axes[1, 0].set_ylim(0, 4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "피드 100개 평균 ± 표준오차 · 'ρ 0.7' = 다시 고른 기준(표시 없으면 0.6, 고정 규칙과 같음) · κ가 클수록 비슷한 것끼리 모임",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "fig1-kappa-return-revisit.png", dpi=200)
    plt.close(fig)


# --- stacked time-allocation bars -----------------------------------------------------------


def stacked(ax, labels, rows_list, value_min=6.0):
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for key, colour in zip(TIME_KEYS, SERIES):
        vals = np.array([np.mean([float(r[key]) for r in rows]) for rows in rows_list])
        ax.barh(y, vals, left=left, color=colour, height=0.62, edgecolor=SURFACE, linewidth=1.2,
                label=TIME_NAME[key])
        for yi, v, l0 in zip(y, vals, left):
            if v >= value_min:
                ax.text(l0 + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if colour in (SERIES[0], SERIES[1], SERIES[5]) else INK)
        left += vals
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("120초 세션에서 쓴 시간 (초)")
    style(ax, grid_axis="x")


def figure2():
    k1, k2 = load("k1_per_feed.csv"), load("k2_per_feed.csv")
    labels, rows_list = [], []
    for name, data in (("K1 기본", k1), ("K2 늦은 깨달음", k2)):
        for k in KAPPAS:
            labels.append(f"{name} · κ {k}")
            rows_list.append([r for r in data if r["policy"].startswith("고정") and r["kappa_label"] == k])
    fig, ax = plt.subplots(figsize=SIZE)
    stacked(ax, labels, rows_list)
    ax.axhline(3.5, color=GRID, linewidth=1)
    ax.set_title("κ별 시간 배분 (고정 규칙 greedy ρ 0.6)", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.9)
    fig.text(0.5, 0.005, "피드 100개 평균 · 6초 이상인 칸에 숫자 표시 · 늦은 깨달음에서만 '위로 이동'(약 2초)이 생김",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "fig2-kappa-time.png", dpi=200)
    plt.close(fig)


def figure3():
    rows = load("t-ppo_per_feed.csv") + load("t-hanul_per_feed.csv") + load("t-junho_per_feed.csv")
    spec = [
        ("기본 · greedy", "기본", "greedy ρ0.6"),
        ("기본 · 멈춰 보기", "기본", "greedy-dwell"),
        ("세은 · greedy", "세은 hard_good", "greedy ρ0.6"),
        ("세은 · 멈춰 보기", "세은 hard_good", "greedy-dwell"),
        ("하늘 A · 전환 0.2초", "하늘 A", "SplitGreedy"),
        ("하늘 B · 전환 1.0초", "하늘 B", "SplitGreedy"),
        ("하늘 C · 1.0초+기준 변경", "하늘 C", "SplitGreedy"),
        ("준호 · 늦은 깨달음 τ2", "준호 τ2", "greedy ρ0.6"),
    ]
    labels = [s[0] for s in spec]
    rows_list = [[r for r in rows if r["condition"] == cond and r["policy"].startswith(pol)] for _, cond, pol in spec]
    fig, ax = plt.subplots(figsize=SIZE)
    stacked(ax, labels, rows_list)
    for yline in (5.5, 3.5, 0.5):
        ax.axhline(yline, color=GRID, linewidth=1)
    ax.set_title("발표 조건별 시간 배분", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=6, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.9)
    fig.text(0.5, 0.005, "피드 20개(seed 10000–10019) 평균 · 준호 τ0은 기본 · greedy와 같음 · 6초 이상인 칸에 숫자 표시",
             ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "fig3-presentation-time.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
    print("saved to", OUT)
