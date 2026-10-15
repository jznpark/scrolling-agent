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
| 2026-10-15 | Feed: first type `z_1 ~ Bernoulli(p)` (stationary share of the persistence chain, so share ≈ p for any kappa) |
| 2026-10-15 | Feed reordering: keep the item set and the number of z=1 items; draw the type sequence from the persistence chain with kappa conditioned on that count (exact DP sampling), then place items of each type in random order. kappa=0 = random shuffle. Position control (e.g. promising items first) not included yet |
| 2026-10-15 | Env geometry: card height 1, screen 3 cards, scroll s = top edge in cards, clamped to [0, n − 3]; visible slots = up to 4 (two half cards at the edges) |
| 2026-10-15 | Env boundary: moving into the edge is allowed (time and cost pass, draft wording); the DDM uses the actual speed, so a blocked move gives g = 1 |
| 2026-10-15 | Env transition: click / back count `transition_steps` (= 2) from the step the action is chosen; during it only stay(None) is valid. Invalid actions are replaced by stay(None) |
| 2026-10-15 | Env episode end: `terminated=True` after N steps (remaining time is observed, so the end is part of the task, not a time limit) |
| 2026-10-15 | Env observation: flat float32 vector in physical units (s, reward/s, 1/s), 655 values for 100 items; scaling / normalisation is left to the training stage |
| 2026-10-15 | Seeds: `reset(seed)` spawns separate feed and DDM generators; `options={"feed": ...}` fixes the list while the DDM seed varies |
| 2026-10-15 | Baselines: ScrollFirst (fixed speed down, click the first understood item with rate ≥ rho, back when rate < rho) and GreedyKnown (go to the best known item, also off screen; explore down when none reaches rho). rho tuned on seeds 20000+ |
| 2026-10-15 | Reference return: all items known, no travel / transition time, watch the item with the largest step reward each step (optimal for decreasing rates) — an upper bound, not the draft's small-environment PPO comparison yet |
| 2026-10-15 | PPO: MaskablePPO, 8 envs, MLP 256-256, VecNormalize (obs + reward), training gamma 0.999 (evaluation uses the undiscounted return, gamma = 1), target_kl 0.02. Seeds: training seed*1000 + env index, evaluation 10000+, rho tuning 20000+ |

## Open questions for the professor

- Can "off-screen residual processing" and "post-click value uncertainty" be added as comparison conditions?
- Basis for v0, pv; scope of individual differences and human data this semester
