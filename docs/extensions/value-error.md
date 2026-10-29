# Extension 1 — Value estimation error (`estimate.*`)

Draft status: extension condition ("이해 후 보상 추정 오차와 확률적인 실제 보상", section 10).
Default: off (`sigma_est = 0` reproduces the MVP exactly).

## Model

On understanding an item, the agent sees log-noisy values instead of the truth:

    log A_hat      = log A      + sigma_est * eps_A   * f(w)
    log lambda_hat = log lambda + sigma_est * eps_lam * f(w)     (only if noise_lambda)
    f(w) = exp(-w / tau_learn)        tau_learn > 0: error fades while watching
    f(w) = 1[w == 0]                  tau_learn = 0: exact after the first watched step

- `eps_A`, `eps_lam` ~ N(0, 1), one draw per item and episode, from a third seed stream
  (`SeedSequence(seed).spawn(3)[2]`); feed and DDM streams are unchanged, so with the same
  actions the evidence and understanding times are identical to the MVP.
- Rewards always use the true A, lambda. Observation and baselines use the perceived values
  (`env.revealed_profile()`); `env.estimated_profile()` gives the perceived values of all items.

| Setting | Default | Meaning |
|---|---|---|
| `estimate.sigma_est` | 0.0 | log-scale error at understanding |
| `estimate.noise_lambda` | True | also perturb lambda |
| `estimate.tau_learn` | 0.0 | watching time over which the error fades (0 = one step) |

## Metrics (`env.extra_metrics()`)

- `n_overestimated_clicks` — clicks where the perceived current rate exceeded the true one
- `click_log_error` — mean log(perceived / true) rate at click; > 0 = selection on noise
  (clicked items look better than they are)
- shared: `n_short_bouts` (watch bouts ≤ 1 s = "opened and left"), `mean_bout`

## Simulation

    python scripts\simulate.py --cond exact --cond est0.3 estimate.sigma_est=0.3 ^
        --cond est0.6 estimate.sigma_est=0.6 --cond est0.6_slow estimate.sigma_est=0.6 estimate.tau_learn=3 ^
        --policies scrollfirst greedy --episodes 50
    python scripts\visualize.py --policy greedy --rho 0.6 --set estimate.sigma_est=0.6 --animate

The animation prints the true values in red under any understood card whose perceived values
differ; the summary figure draws a grey segment from the perceived A at understanding to the
true A.

## Pilot result (50 unseen lists, rho = 0.6 fixed)

| condition | policy | return | clicks | short bouts | overestimated clicks | click log error |
|---|---|---|---|---|---|---|
| exact | greedy | 53.7 | 28.9 | 9.5 | 0 | 0.00 |
| sigma 0.3 | greedy | 53.1 | 31.4 | 12.2 | 20.3 | 0.10 |
| sigma 0.6 | greedy | 51.9 | 33.4 | 15.3 | 24.5 | 0.30 |
| sigma 0.6, tau 3 s | greedy | 48.9 | 31.0 | 10.5 | 22.9 | 0.31 |

- More error -> more clicks and more short bouts ("opened, disappointed, left": case 3).
- `click_log_error` > 0: the threshold policy selects on noise (winner's curse).
- With slow learning (tau 3 s) short bouts fall back but the return drops most: bad items are
  watched longer before the error fades, and good items that looked bad are left too early.
- The baselines do not adapt to the error (fixed rho); a PPO policy trained under the condition
  can learn to discount perceived values or to "sample" briefly — that is the experiment to run.

## Open points

- Should the error be multiplicative on A only (`noise_lambda=False`)?
- Draft also mentions stochastic *actual* rewards; not implemented here.
