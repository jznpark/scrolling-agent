# P0-2 PPO baseline result (2026-11-05)

Branch `feature/rl-pipeline`. Tables: `results/p0_ppo/{base,topk}_summary.md` (made with `scripts/pipeline.py`).

## Setup
- MaskablePPO, 8 envs, MLP 256-256, VecNormalize, gamma 0.999, target_kl 0.02
- Changed from the 10-15 pilot: ent_coef 0.01 -> 0.001, n_steps 1024 -> 2048, 1M -> 5M steps
- Conditions: `base` (full per-item memory, 655 obs values) and `topk` (top-10 known items, 116 values); seeds 0, 1, 2
- Interrupted by a reboot at ~1M steps and resumed from each run's best model (0.8-1.2M)
- Evaluation: 100 unseen lists (seeds 10000-10099), best model per run, deterministic and sampled actions; greedy rho tuned on seeds 20000-20029 (0.6)

## Result (% of the known-information reference, mean ± sd over lists)

| policy | base | topk |
|---|---:|---:|
| greedy (rho 0.6) | **50.8 ± 4.3** | **50.8 ± 4.3** |
| PPO deterministic (3 seeds) | 24.4 ± 15.7 | 27.8 ± 15.9 |
| PPO sampled (3 seeds) | 30.1 ± 10.9 | 34.5 ± 8.7 |
| 10-15 pilot (1M, base) | 16 / 27 (det / sampled) | – |

- Better than the pilot, and `topk` is better and much more consistent across seeds (sampled 34.2-34.9 vs 26.9-35.2), but **PPO still reaches only about 2/3 of greedy** -> P0-2 target (>= greedy) not met
- The click -> back loop remains: PPO makes 55-75 clicks per session vs 29 for greedy, 25-40 of them on already-watched items, 40-60 watch bouts <= 1 s, and 22-30 s in transitions (greedy 11.6 s). Deterministic policies vary a lot between lists (sd ~16): on some lists they loop for most of the session
- PPO explores less: understands 27-39 items and reaches item 28-42 (greedy 59 items, item 68)
- Training curves (10 evaluation lists) peak at 1-3M steps (30-38) and then fall (11-17 at 5M): the policy gets worse with more training

## Interpretation
The difficulty is credit assignment, not capacity: one decision every 0.1 s gives 1200 decisions per session, and a click or back costs 2 dead steps whose loss shows up only slowly. The cheapest local gain (open a known good item for a moment, collect the high first-step reward, leave) is learned before the longer pattern "watch until the rate drops, then explore".

## Suggested next steps (pick one)
1. Decision interval on the learning side: each PPO action lasts k steps (e.g. 0.5 s; transitions unchanged). The model (dt = 0.1 s) and the action set stay as agreed; only the agent decides less often
2. Stabilise: learning rate 3e-4 -> 1e-4 with linear decay, target_kl 0.01, evaluate more lists during training
3. Shaping only for learning (e.g. a penalty for re-opening an item within 1 s); report evaluation with the original reward
