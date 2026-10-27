"""Summaries of results/kappa_order/*.csv: mean ± standard error per condition, and the
paired difference to kappa = 0 (same feeds and DDM seeds), as Markdown tables.

    python scripts\\experiments\\summarize_kappa_order.py > results\\kappa_order\\summary.md
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "results" / "kappa_order"
KAPPA_ORDER = ["orig", "0", "0.5", "0.8", "0.9"]
KAPPA_NAME = {"orig": "원래 순서", "0": "κ 0", "0.5": "κ 0.5", "0.8": "κ 0.8", "0.9": "κ 0.9"}
TIME_KEYS = ["time_down", "time_up", "time_stay", "time_blocked", "time_watch", "time_transition"]
TIME_NAME = {"time_down": "아래로", "time_up": "위로", "time_stay": "정지", "time_blocked": "막힌 이동",
             "time_watch": "시청", "time_transition": "전환"}


def load(name: str) -> list[dict]:
    with (RES / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def mse(x: np.ndarray, digits: int = 2) -> str:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return "–"
    se = x.std(ddof=1) / math.sqrt(x.size) if x.size > 1 else 0.0
    return f"{x.mean():.{digits}f} ± {se:.{digits}f}"


def paired(rows_a, rows_b, key):
    """Mean and SE of b - a over matching seeds."""
    a = {r["seed"]: float(r[key]) for r in rows_a}
    d = np.array([float(r[key]) - a[r["seed"]] for r in rows_b if r["seed"] in a])
    d = d[~np.isnan(d)]
    se = d.std(ddof=1) / math.sqrt(d.size)
    return d.mean(), se


def diff_text(mean: float, se: float, digits: int = 2) -> str:
    verdict = "차이 없음" if abs(mean) <= se else ("≈ 차이 없음" if abs(mean) <= 2 * se else "차이 있음")
    return f"{mean:+.{digits}f} ± {se:.{digits}f} ({verdict})"


def kappa_tables(exp: str) -> str:
    rows = load(f"{exp}_per_feed.csv")
    checks = load(f"{exp}_feed_check.csv")
    tuning = load(f"{exp}_tuning.csv")
    out = []
    # feed check
    out.append(f"#### {exp.upper()} 피드 점검 (100개 피드 평균 ± 표준오차)\n")
    out.append("| 순서 | 인접 유형 유지율 | z=1 개수 | ν–A 상관(log) | A 평균 | ν 평균 | λ 평균 |")
    out.append("|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in checks:
        by[r["kappa_label"]].append(r)
    for lab in KAPPA_ORDER:
        g = by[lab]
        out.append(f"| {KAPPA_NAME[lab]} | {mse(num(g, 'type_persistence'), 3)} | {mse(num(g, 'n_z1'), 2)} | "
                   f"{mse(num(g, 'nu_A_corr_log'), 3)} | {mse(num(g, 'mean_A'), 3)} | {mse(num(g, 'mean_nu'), 3)} | "
                   f"{mse(num(g, 'mean_lam'), 3)} |")
    # tuning
    out.append(f"\n#### {exp.upper()} 다시 고른 ρ (seed 20000–20029, 격자 0.1–1.5)\n")
    out.append("| 순서 | 고른 ρ | 그때 평균 누적 보상 | ρ 0.6일 때 |")
    out.append("|---|---|---|---|")
    for lab in KAPPA_ORDER:
        t = [r for r in tuning if r["kappa_label"] == lab]
        best = max(t, key=lambda r: float(r["mean_return"]))
        at06 = next(r for r in t if abs(float(r["rho"]) - 0.6) < 1e-9)
        out.append(f"| {KAPPA_NAME[lab]} | {best['rho']} | {float(best['mean_return']):.2f} | {float(at06['mean_return']):.2f} |")
    # re-chosen minus fixed rule on the same feeds
    fixed = [r for r in rows if r["policy"].startswith("고정")]
    rechosen = [r for r in rows if r["policy"].startswith("다시")]
    out.append(f"\n#### {exp.upper()} 다시 고른 규칙 − 고정 규칙 (같은 피드, 평균 ± 표준오차)\n")
    out.append("| 순서 | ρ | 누적 보상 차이 | 재방문 클릭 차이 | 최대 깊이 차이 | 클릭 차이 |")
    out.append("|---|---|---|---|---|---|")
    for lab in KAPPA_ORDER:
        f = [r for r in fixed if r["kappa_label"] == lab]
        g = [r for r in rechosen if r["kappa_label"] == lab]
        if g[0]["rho"] == f[0]["rho"]:
            out.append(f"| {KAPPA_NAME[lab]} | {g[0]['rho']} | 같은 규칙 | – | – | – |")
            continue
        out.append(f"| {KAPPA_NAME[lab]} | {f[0]['rho']} → {g[0]['rho']} | {diff_text(*paired(f, g, 'return'))} | "
                   f"{diff_text(*paired(f, g, 'revisit_clicks'))} | {diff_text(*paired(f, g, 'max_item_seen'), 1)} | "
                   f"{diff_text(*paired(f, g, 'n_clicks'), 1)} |")
    # per policy tables
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    for pol in policies:
        prow = [r for r in rows if r["policy"] == pol]
        grp = {lab: [r for r in prow if r["kappa_label"] == lab] for lab in KAPPA_ORDER}
        base = grp["0"]
        out.append(f"\n#### {exp.upper()} · {pol} (seed 10000–10099, 평균 ± 표준오차)\n")
        out.append("| 순서 | ρ | 누적 보상 | κ0 대비 차이 | 시청 보상 | 상한 대비 % | 이해 수 | 최대 깊이 | 클릭 | 클릭당 시청(초) | 종료 시 보상률 | 재방문 클릭 | κ0 대비 차이 | 재시청 | 위쪽 이동 횟수 | 위로(초) | 알려진 후보 선택률 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab in KAPPA_ORDER:
            g = grp[lab]
            rho = g[0]["rho"]
            if lab == "0":
                d_ret = d_rev = "기준"
            else:
                d_ret = diff_text(*paired(base, g, "return"))
                d_rev = diff_text(*paired(base, g, "revisit_clicks"))
            out.append(
                f"| {KAPPA_NAME[lab]} | {rho} | {mse(num(g, 'return'))} | {d_ret} | {mse(num(g, 'watch_reward'))} | "
                f"{mse(num(g, 'pct_of_ref'), 1)} | {mse(num(g, 'n_understood'), 1)} | {mse(num(g, 'max_item_seen'), 1)} | "
                f"{mse(num(g, 'n_clicks'), 1)} | {mse(num(g, 'watch_per_click'))} | {mse(num(g, 'end_rate'))} | "
                f"{mse(num(g, 'revisit_clicks'))} | {d_rev} | {mse(num(g, 'rewatch_clicks'))} | {mse(num(g, 'up_moves'))} | "
                f"{mse(num(g, 'time_up'))} | {mse(num(g, 'known_selection_rate'), 3)} |")
        out.append(f"\n시간 배분(초, 평균) · {pol}\n")
        out.append("| 순서 | " + " | ".join(TIME_NAME[k] for k in TIME_KEYS) + " | 합계 |")
        out.append("|---|" + "---|" * (len(TIME_KEYS) + 1))
        for lab in KAPPA_ORDER:
            g = grp[lab]
            out.append(f"| {KAPPA_NAME[lab]} | " + " | ".join(f"{num(g, k).mean():.2f}" for k in TIME_KEYS)
                       + f" | {num(g, 'time_total').mean():.2f} |")
    return "\n".join(out)


def t_tables() -> str:
    rows = load("t-ppo_per_feed.csv") + load("t-hanul_per_feed.csv") + load("t-junho_per_feed.csv")
    out = ["#### 실험 T (seed 10000–10019, 평균 ± 표준오차)\n",
           "| 조건 | 브랜치 | 설정 | 정책 | 누적 보상 | 시청 보상 | 상한 대비 % | 이해 수 | 최대 깊이 | 클릭 | 클릭당 시청(초) | 종료 시 보상률 | 재방문 클릭 | 재시청 | 위로(초) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keyed = defaultdict(list)
    for r in rows:
        keyed[(r["condition"], r["branch"], r["settings"], r["policy"])].append(r)
    for (cond, branch, settings, pol), g in keyed.items():
        out.append(f"| {cond} | {branch} | {settings} | {pol} | {mse(num(g, 'return'))} | {mse(num(g, 'watch_reward'))} | "
                   f"{mse(num(g, 'pct_of_ref'), 1)} | {mse(num(g, 'n_understood'), 1)} | {mse(num(g, 'max_item_seen'), 1)} | "
                   f"{mse(num(g, 'n_clicks'), 1)} | {mse(num(g, 'watch_per_click'))} | {mse(num(g, 'end_rate'))} | "
                   f"{mse(num(g, 'revisit_clicks'))} | {mse(num(g, 'rewatch_clicks'))} | {mse(num(g, 'time_up'))} |")
    out.append("\n시간 배분(초, 평균)\n")
    out.append("| 조건 | 정책 | " + " | ".join(TIME_NAME[k] for k in TIME_KEYS) + " | 합계 |")
    out.append("|---|---|" + "---|" * (len(TIME_KEYS) + 1))
    for (cond, _, _, pol), g in keyed.items():
        out.append(f"| {cond} | {pol} | " + " | ".join(f"{num(g, k).mean():.2f}" for k in TIME_KEYS)
                   + f" | {num(g, 'time_total').mean():.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(kappa_tables("k1"))
    print()
    print(kappa_tables("k2"))
    print()
    print(t_tables())
