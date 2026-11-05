# Extension 4 — Attention split (`ddm.attention`)

Draft status: extension condition ("망각과 주의 자원의 경쟁은 후속 실험", section 03); form
inspired by Kim 2025 (drift v0 / log2 N). Default: off (`attention = "none"` = MVP).

## Model

The drift of each visible, not yet understood item is multiplied by h(N), N = the number of
such items on screen at that inner step:

    dx_i = nu_i * g(|v|) * h(N) dt + sigma dW_i
    h(N) = 1 / log2(N + 1)      "log"   (h(1) = 1, h(2) = 0.63, h(3) = 0.50, h(4) = 0.43)
    h(N) = N ** -beta           "power" (beta = 1: attention shared equally, h(3) = 0.33)

- `log2(N + 1)` instead of Kim's `log2 N` so a single unread item keeps its full drift.
- Understood items no longer take attention, so the remaining ones speed up as the screen
  is cleared. Noise sigma is unchanged (the draft fixes sigma); only the drift is split.
- `screen.cards_per_screen` sets how crowded the screen can be. Speeds are in screen heights/s,
  so to keep the same card speed with 5 cards use `screen.speeds=(0.3,)` (0.3 x 5 = 1.5 cards/s).

| Setting | Default | Meaning |
|---|---|---|
| `ddm.attention` | "none" | "none", "log" or "power" |
| `ddm.attention_beta` | 1.0 | exponent of "power" |

## Metrics

- `mean_unread_on_screen` — time average of N over list time (`env.extra_metrics()`)
- shared: `time_stay_list` (stopping in the list), `n_understood`
- the animation title shows "unread on screen: N"; the summary lists the mean

## Simulation

    python scripts\simulate.py --cond base --cond log ddm.attention=log --cond power1 ddm.attention=power ^
        --cond crowd5 screen.cards_per_screen=5 "screen.speeds=(0.3,)" ^
        --cond crowd5_log screen.cards_per_screen=5 "screen.speeds=(0.3,)" ddm.attention=log ^
        --policies scrollfirst greedy greedy-dwell --episodes 50
    python scripts\visualize.py --policy greedy --dwell 0.5 --rho 0.6 --set ddm.attention=power --animate

## Pilot result (50 unseen lists, rho = 0.6, dwell = 1.5 s fixed)

| condition | policy | return | understood | stay in list (s) | unread on screen |
|---|---|---|---|---|---|
| base | greedy | 53.7 | 58.9 | 0 | 1.94 |
| base | greedy-dwell | 33.9 | 41.5 | 41.2 | 0.68 |
| log | greedy | 33.6 | 40.5 | 4.8* | 2.54 |
| log | greedy-dwell | 33.2 | 41.2 | 41.8 | 0.72 |
| power (beta 1) | greedy | 6.6 | 18.7 | 32.8* | 1.99 |
| power (beta 1) | greedy-dwell | 32.7 | 40.7 | 42.3 | 0.77 |
| 5 cards | greedy | 57.3 | 61.6 | 0 | 1.83 |
| 5 cards + log | greedy | 50.4 | 56.4 | 0 | 3.42 |

\* greedy reaches the end of the list (item 99) and waits there.

Best stopping time per condition (greedy, dwell swept, 30 lists):

| attention | best dwell | return at best | stay in list (s) | understood |
|---|---|---|---|---|
| none | 0 s | 53.5 | 0 | 59.8 |
| log | 0.25 s | 44.6 | 16.0 | 49.5 |
| power | 0.5 s | 39.7 | 26.5 | 44.6 |

- With attention split, scrolling at a constant speed understands far fewer items (59 -> 40
  -> 19) because the screen always holds ~2–3 unread items.
- Stopping becomes optimal: the best dwell grows from 0 to 0.25 to 0.5 s and the time spent
  standing still in the list from 0 to 16 to 26.5 s — the predicted "crowded screen -> stop and
  look" behaviour. A trained PPO policy should show the same shift without a hand-set dwell.
- 5 cards per screen helps without the split (more items processed in parallel) but the split
  removes most of that gain.

## Open points

- log vs power form, and whether noise should also scale (time-sharing) — fit to human data later.
- Should N count partially visible (< 50 %) cards?
