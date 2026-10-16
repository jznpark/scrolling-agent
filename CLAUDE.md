# CLAUDE.md — scrolling-agent

Guidance for Claude Code in this repository. Team-shared; personal settings live in `CLAUDE.local.md` (not committed).

## Project

CSI6545 Computational Interaction Design (Yonsei, Fall 2026), Team 4.
A scrolling agent with limited session time decides how to spend time between **exploring** (understanding thumbnails),
**watching** (collecting decaying reward) and **revisiting** known items. Perception = drift-diffusion per visible item;
behaviour = RL policy (PPO). Design source: the professor's draft *Scrolling Agent — 초기 연구 설계* v0.1 (2026-10-14),
summarised in `docs/design-notes.md`. The draft separates **agreed settings (합의)** from **initial proposals (초기 제안)** —
do not change agreed settings; proposals may change but record the change in `docs/design-notes.md`.

### Model in one screen (draft sections 02–05, 08, 10)

- Item `C_i = (nu_i, A_i, lambda_i)`; watch time `w_i` grows only while watching item i (kept across revisits)
- DDM while visible in LIST mode: `dx_i = nu_i * g(|v|) dt + sigma dW_i`, `g(|v|) = 1 / (1 + (|v|/v0)^pv)`;
  first `x_i >= B` -> understood flag, reveal `(A_i, lambda_i)`; no reward, no forced click.
  Off-screen or while watching: evidence frozen and kept (no forgetting in MVP)
- Watch reward per step: `(A/lambda) * exp(-lambda*w) * (1 - exp(-lambda*dt))`, then `w += dt`
- Every step costs `c`; fixed `T` (N = T/dt steps), gamma = 1; observation includes remaining time
- Actions: `down(speed)`, `up(speed)`, `stay(None | i)`, `back`; LIST / WATCH modes; click and back each take a fixed
  transition (0.2 s) during which neither reward nor DDM advances; masks for invalid actions
- Observation never contains hidden `A, lambda` of non-understood items, true `nu`, latent type `z`, or unseen items
- Feed: latent type z (persistence kappa, share p); `log A = a0 + a1 z + sA e`, `log nu = m_nu + d_nu z + s_nu e`,
  `log lambda = m_lambda + s_lambda e`; ordering experiments reorder a fixed item set

## Layout

- `src/scrolling_agent/config.py` — all parameters; mark values `[draft]` (from the draft) or `[TBD]` (team choice)
- `src/scrolling_agent/feed.py` — stochastic feed generator (step 1)
- `src/scrolling_agent/env.py` — Gymnasium environment (step 2)
- `src/scrolling_agent/policies.py` — baselines, PPO wrapper, reference return (steps 4–5)
- `src/scrolling_agent/evaluation.py`, `viz.py` — episode metrics / trace and plots
- `scripts/train_ppo.py`, `evaluate.py`, `visualize.py` — training, comparison, watching one episode
- `webdemo/` + `scripts/serve.py`, `serve_branches.py` — local web demo (stdlib HTTP server; `--code` runs another
  branch's `src`, worktrees in `.worktrees/`); outside the package so one demo can run any branch
- `scripts/` — runnable entry points; `tests/` — pytest; `docs/` — design notes

## Commands (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1          # every new terminal
pip install -e ".[dev]"               # after changing dependencies
pytest                                # all tests
python scripts\check_setup.py         # environment smoke test
```

## Conventions

- Python 3.12, type hints, numpy; small pure functions where possible
- Reproducibility: every random draw goes through a `numpy.random.Generator` created from an explicit seed
- Every new module gets tests; implementation checks from draft section 10 become tests
  (mean first-passage time ≈ B/nu, time sums to T, state preserved across revisits, cost shift = -N(c'-c))
- Keep experiment outputs out of git (`runs/`, `models/`, `logs/` are ignored)

## Workflow

- Before a non-trivial change, state the plan briefly and wait for approval
- Work on `feature/<topic>` branches; small commits with clear messages
- Run `pytest` before saying a step is done
- **Ask before** `git push`, deleting files, or changing agreed settings
