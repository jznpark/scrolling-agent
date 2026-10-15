# Extension 3 — Off-screen residual processing (`ddm.tau_residual`)

Draft status: team hypothesis (our proposal; the draft freezes off-screen evidence and lists
"화면 밖 잔여 처리" as a question for the professor). Default: off (`tau_residual = 0` = MVP).

## Model

After an item stops being visible, its evidence keeps accumulating with a decaying weight

    a(s) = exp(-s / tau_residual),   s = time since the item was last visible
    dx = a(s) [ nu * g_last dt + sigma dW ]   as a time change:
    over [s0, s1]:  eff = tau (e^{-s0/tau} - e^{-s1/tau}),  dx = nu g_last eff + sigma sqrt(eff) N(0, 1)

- In expectation the item gets `tau_residual` extra seconds of processing at the drift it had
  when it left (`g_last` = speed gain at the last visible inner step). Variance grows the same way.
- Applies to items that were seen and are not yet understood. Reaching B off screen reveals
  (A, lambda) exactly as on screen; nothing forces a click.
- `residual_in_watch` (default False): also continue while the list is hidden by watching or
  a transition (the draft stops the list DDM while watching, so this is off by default).
- Residual draws use the DDM stream only when `tau_residual > 0`: off = bit-identical to the MVP.
- Cost: about 3–4x slower simulation with the extension on (residual update per inner step).

| Setting | Default | Meaning |
|---|---|---|
| `ddm.tau_residual` | 0.0 | mean extra processing time after leaving the screen (s) |
| `ddm.residual_in_watch` | False | also while the list is hidden (watching / transition) |

## Metrics (`env.extra_metrics()`)

- `n_off_screen_understood` — threshold reached after the agent scrolled away ("아까 그거 괜찮았는데")
- `n_hidden_understood` — reached while the list was hidden but the item stayed under the
  screen window (only with `residual_in_watch`)
- `n_late_clicks` — clicks on off-screen-understood items = the agent went back to them (case 2)

Figures: stars on the scroll path mark off-screen understanding; the animation shows a yellow
"understood off screen: #i" note for 2 s.

## Simulation

    python scripts\simulate.py --cond frozen --cond tau0.5 ddm.tau_residual=0.5 ^
        --cond tau2 ddm.tau_residual=2.0 --cond tau2_watch ddm.tau_residual=2.0 ddm.residual_in_watch=True ^
        --policies scrollfirst greedy --episodes 50
    python scripts\visualize.py --policy greedy --rho 0.6 --set ddm.tau_residual=2.0 --animate

## Pilot result (50 unseen lists, rho = 0.6 fixed)

| condition | policy | return | understood | off-screen understood | late clicks | deepest item |
|---|---|---|---|---|---|---|
| frozen (MVP) | scrollfirst | 53.7 | 59.3 | 0 | 0 | 68.3 |
| frozen (MVP) | greedy | 53.7 | 58.9 | 0 | 0 | 67.8 |
| tau 0.5 s | scrollfirst | 53.8 | 62.9 | 4.0 | 0 | 68.0 |
| tau 0.5 s | greedy | 55.3 | 58.7 | 3.7 | 1.8 | 63.7 |
| tau 2 s | scrollfirst | 53.5 | 65.8 | 6.8 | 0 | 68.4 |
| tau 2 s | greedy | 54.5 | 59.4 | 6.2 | 3.0 | 61.6 |
| tau 2 s + in watch | greedy | 56.0 | 60.9 | 4.9 (+18.3 hidden) | 2.5 | 60.6 |

- Late realisation appears only with residual processing; going back (case 2) only with a
  policy that can scroll up (greedy). ScrollFirst understands more items but cannot use the late ones.
- Going back costs exploration: greedy's deepest item drops from 68 to 62 at tau 2 s, while
  the return rises slightly (+0.7 to +1.6): late items are good enough to justify the trip.
- The trade-off "go back vs keep exploring" is what a trained policy must learn; the greedy
  baseline always goes back when a known item clears rho.

## Open points (for the professor / team)

- Is exponential decay with one time constant enough, or should the residual stop at a fixed
  horizon / be limited to the most recent items (memory capacity, draft "기억 감소" extension)?
- Should the residual run while watching (`residual_in_watch`)? It changes the meaning of
  "watching blocks list processing" in the draft.
