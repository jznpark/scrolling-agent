# Design notes

Source: professor's draft *Scrolling Agent — 초기 연구 설계* v0.1 (2026-10-14). The draft separates
**agreed settings** (합의) from **initial proposals** (초기 제안); only the latter are open to change.

## Agreed settings (합의)

- 1-D list; actions: up / down, stay (wait or select & watch), back to list
- Each visible item runs an independent drift-diffusion process
- Reaching the threshold = "understood": reveals expected reward profile (A, λ); clicking is not forced
- Watching reward decays to 0; fixed penalty every step; maximise cumulative reward within fixed time
- Fast scrolling lowers drift; feed composition and order controlled by a few stochastic parameters

## Initial proposals (초기 제안, open)

- Exponential reward decay `A·exp(-λw)`
- Positive drift with a single threshold
- Speed attenuation `g(|v|) = 1 / (1 + (|v|/v0)^pv)`
- Memory: off-screen evidence frozen and kept; understood info and watch time kept; no forgetting

## Team decisions

| Date | Decision |
|---|---|
| 2026-10-15 | Gymnasium + Stable-Baselines3 PPO (MaskablePPO for action masking); private repo |

## Open questions for the professor

- Can "off-screen residual processing" and "post-click value uncertainty" be added as comparison conditions?
- Basis for v0, pv; scope of individual differences and human data this semester
