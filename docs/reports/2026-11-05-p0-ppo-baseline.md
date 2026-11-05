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

## Follow-up: decision interval 0.5 s (`topk-k5`, same day)
Option 1 applied (`TrainConfig.decision_steps = 5`, `scrolling_agent/decision.py`): the PPO agent decides at most every 5 env steps; scroll / stay are held, transitions are passed through, a new mode ends the hold. The model (dt = 0.1 s, actions, reward) is unchanged. 3 seeds x 3M decisions (more episodes than the 5M-step runs). Table: `results/p0_ppo/topk-k5_summary.md`.

| policy (% of reference, 100 unseen lists) | base | topk | **topk-k5** |
|---|---:|---:|---:|
| greedy (rho 0.6) | 50.8 | 50.8 | 50.8 |
| PPO deterministic | 24.4 | 27.8 | **40.8 ± 13.5** (seed 2: 46.3) |
| PPO sampled | 30.1 | 34.5 | **44.0 ± 5.8** (seeds 43.7-45.3) |

- The click -> back loop is almost gone for sampled actions: 27.7 clicks (greedy 29.4), 4.9 re-opened items, 10.9 s in transitions (greedy 11.6), and exploration matches greedy (61 items understood, item 67 reached)
- Deterministic actions still loop on some lists (sd 13.5, clicks 49 ± 58)
- Still below greedy (87% of it). Training curves peak early (49 on the 10 evaluation lists at 0.4-0.5M decisions) and then fall to 19-43 at 3M -> the remaining problem is training stability / model selection on only 10 lists
- Next candidates: learning-rate decay (3e-4 -> 0) or 1e-4, select the best model on more evaluation lists (e.g. 30), 1M decisions are enough

## Follow-up: learning-rate decay 3e-4 -> 0 (`topk-k5-lr1m`, `topk-k5-lr3m`, same day)
Option 2 (part) applied (`TrainConfig.lr_schedule = "linear"`): the rate falls linearly from 3e-4 to 0 over training. Everything else as `topk-k5`. Two lengths, 1M and 3M decisions, 3 seeds each. Tables: `results/p0_ppo/topk-k5-lr{1m,3m}_summary.md`.

| policy (% of reference, 100 unseen lists) | topk-k5 (constant) | **lr1m** | **lr3m** |
|---|---:|---:|---:|
| greedy (rho 0.6) | 50.8 | 50.8 | 50.8 |
| PPO deterministic | 40.8 ± 13.5 | **47.1 ± 9.6** | **46.8 ± 8.4** |
| PPO sampled | 44.0 ± 5.8 | **47.8 ± 5.2** (seeds 45.4-49.3) | **47.3 ± 5.8** (seeds 46.2-48.1) |

- Training is now stable: the curves no longer fall (final / best on the 10 evaluation lists: lr1m 46-48 / 46-50, lr3m 50-51 / 52; constant rate 19-43 / 49)
- Clear gain over the constant rate (+4 sampled, +6 deterministic), and the deterministic policy loops less (sd 8-10 instead of 13.5), but loops on a few lists remain (clicks 31 ± 29-38)
- Still below greedy (93-94% of it). lr3m is not better than lr1m on unseen lists although it scores 52 on the evaluation lists -> the best model is chosen on only 10 lists and that score is optimistic
- Behaviour vs greedy: PPO understands more items (65-68 vs 59) and reaches further (item 73-75 vs 68) but watches less (57-60 s vs 64 s) -> it explores a little too much
- Next candidates: select the model on more evaluation lists (e.g. 30), longer watching (check the leaving point vs the rate threshold), 1M decisions with decay is enough for further comparisons

## Follow-up: leaving point and model selection (same day)
**Leaving point** (`scripts/experiments/watch_leaving.py`, lr1m models, 100 unseen lists; table `results/p0_ppo/watch_leaving/topk-k5-lr1m_summary.md`):
- PPO leaves while the rate is still >= 0.6 in about 40% of its backs, and in 93% of those a better item is already known (it switches)
- A hybrid that keeps PPO in the list but uses the greedy rule while watching scores 48.0% (PPO sampled 47.8%) -> the leaving point is not the cause of the gap
- The gap is in the list: PPO understands more items (66 vs 59) but opens a smaller share (about 30% vs 51%). It leaves about 15 understood items with A >= 0.6 unopened per session (greedy 0.1), but these are mostly low-value (median A 0.73 vs 1.57 for opened ones; median gain until the rate falls to 0.6 is 0.43)

**Model selection on 30 separate lists** (`topk-k5-lr1m-ev30`: `n_eval_episodes=30`, `eval_seed=30000`). The earlier runs selected the best model on lists 10000-10009, which are also among the 100 test lists:

| policy (% of reference, 100 unseen lists) | lr1m (10 overlapping lists) | **lr1m-ev30** |
|---|---:|---:|
| PPO deterministic | 47.1 ± 9.6 | 46.0 ± 11.9 |
| PPO sampled | 47.8 ± 5.2 | 48.3 ± 5.3 (seeds 46.9-49.3) |

- No meaningful difference: the overlap did not inflate the earlier results, and more selection lists do not close the gap to greedy (50.8%)
