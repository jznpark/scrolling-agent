# Extension — Ad slots (`ads.*`)

Draft status: not in the draft — a team extension (Task#3, 광고/추천 아이디어).
Default: off (`ads.every = 0` reproduces the draft model exactly).

## Model

Every `every`-th card of a **generated** feed is replaced by a sponsored card:

    positions   every-1, 2*every-1, ...          (deterministic, no extra randomness)
    A = 0       watching an ad earns nothing (every step still costs c)
    z = 0       ads carry no latent type
    nu = exp(m_nu_ad)   drift of the "this is an ad" realisation

- The DDM treats an ad like any other thumbnail: the agent must understand it
  before knowing it is worthless. `disclosed = True` models a visible "sponsored"
  label — ads start the episode already understood (rate 0, never clicked by the
  rate-based baselines).
- Non-ad positions keep exactly the items the draft feed generator produced for
  them (same RNG stream), so an episode with ads differs from the draft feed
  only at ad positions.
- A feed passed to the environment by the caller (`options={"feed": ...}` or the
  constructor, e.g. ordering experiments) is used as-is; use
  `feed.insert_ads(feed, every, m_nu_ad)` to add ads to a fixed feed.

| Setting | Default | Meaning |
|---|---|---|
| `ads.every` | 0 | every k-th card is an ad; 0 = off |
| `ads.disclosed` | False | ads are labelled: understood from the first sight |
| `ads.m_nu_ad` | 0.0 | log drift of the "it's an ad" realisation (lower = sneakier) |

## Metrics (`env.extra_metrics()`)

- `n_ads` — ad cards in this episode's feed
- `n_ad_clicks` — times an ad was opened
- `time_watch_ads` — seconds spent watching ads

## What changes (greedy / greedy-dwell, rho 0.6, 20 feeds, seeds 10000–10019)

- **Dilution**: with `every = 5` (20% ads) and `greedy`, return drops 52.7 → 47.2
  (−10%). Understanding in the base model is per-item and parallel, so unlabelled
  ads do not slow down understanding of real items — the loss is the missing
  content itself. Labelling ads barely helps `greedy` (47.0).
- **Wasted dwell**: a policy that pauses at unknown cards pays extra for sneaky
  ads. `greedy-dwell` with `every = 5, m_nu_ad = -1`: return 42.0 → 37.6 (−10%)
  and time stopped in the list rises 24.7 → 27.3 s. With `disclosed = True` it
  does not stop at ads: 42.0 → 39.5 (−6%), stopped time falls to 23.0 s.
- With the attention-split branch (screen crowding) unlabelled ads should also
  crowd the screen and slow real items down; not measured here.

## Demo

The web demo shows an `AD` badge on ads once they are understood (immediately
when `disclosed`), says "광고 · 봐도 재미 없음" on the card, and adds two rows to
the behaviour comparison. Situations: "피드 5개마다 몰래 광고" and
"광고라고 표시해 주는 앱" (both greedy-dwell).

## Tests

`tests/test_ads.py` — off = draft feed (and `extra_metrics` zeros), ad positions
and values, reindex keeps ads with their items, disclosed = understood at reset,
watching an ad earns only `-step_cost`, time still sums to T.
