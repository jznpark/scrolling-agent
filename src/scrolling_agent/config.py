"""Configuration for the scrolling agent.

Values marked [draft] come from the professor's design draft v0.1 (2026-10-14),
section 10 "초기 구현과 검증 기준" — they are starting values, not fixed choices.
Values marked [TBD] are not given in the draft; they are temporary team choices
and should be revisited once the environment runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass
class TimeConfig:
    session_seconds: float = 120.0      # [draft] T
    dt: float = 0.1                     # [draft] step length (s)
    step_cost: float = 0.01             # [draft] c per step
    transition_seconds: float = 0.2     # [draft] click / back each take 0.2 s (2 steps)

    @property
    def n_steps(self) -> int:
        """N = T / dt."""
        return int(round(self.session_seconds / self.dt))

    @property
    def cost_per_second(self) -> float:
        """c / dt — keep this equal when comparing different dt."""
        return self.step_cost / self.dt

    @property
    def transition_steps(self) -> int:
        """Steps taken by one click or back transition."""
        return int(round(self.transition_seconds / self.dt))


@dataclass
class ScreenConfig:
    n_items: int = 100                  # [draft] finite list length
    cards_per_screen: float = 3.0       # [draft] screen height ≈ 3 cards
    visible_fraction: float = 0.5       # [draft] DDM active when ≥ 50% of a thumbnail is visible
    speeds: tuple[float, ...] = (0.5,)  # [draft] screen heights / s; later (0.5, 1.5)

    @property
    def max_visible_slots(self) -> int:
        """Most items that can pass the visibility rule at once (two half cards at the edges)."""
        return math.ceil(self.cards_per_screen) + 1


@dataclass
class DDMConfig:
    threshold: float = 1.0              # [draft] B (single upper threshold)
    noise: float = 0.25                 # [draft] σ, evidence / sqrt(s)
    v0: float = 1.0                     # [draft] speed at which drift halves (screen heights / s)
    pv: float = 2.0                     # [draft] shape of speed attenuation
    dt_internal: float = 0.01           # [TBD] inner integration step for exposure while moving

    def speed_gain(self, speed: float) -> float:
        """g(|v|) = 1 / (1 + (|v|/v0)^pv); g(0) = 1."""
        return 1.0 / (1.0 + (abs(speed) / self.v0) ** self.pv)


@dataclass
class FeedConfig:
    """Stochastic feed generator (draft section 06). Numbers are [TBD] unless noted."""

    p: float = 0.3                      # share of promising type z=1
    kappa: float = 0.0                  # persistence of neighbouring types, 0 <= kappa < 1
    a0: float = math.log(0.5)           # log A intercept
    a1: float = math.log(3.0)           # log A boost for z=1 (must be > 0)
    sigma_A: float = 0.3
    m_nu: float = 0.0                   # log nu mean  -> nu ~ 1 evidence/s
    d_nu: float = 0.0                   # negative: promising items are harder to understand
    sigma_nu: float = 0.3
    m_lambda: float = math.log(0.3)     # log lambda mean (1/s)
    sigma_lambda: float = 0.3


@dataclass
class Config:
    time: TimeConfig = field(default_factory=TimeConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    ddm: DDMConfig = field(default_factory=DDMConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)


def config_from_dict(data: dict) -> Config:
    """Rebuild a Config from `dataclasses.asdict` output (e.g. a run's config.json)."""
    cfg = Config()
    for section, values in data.items():
        current = getattr(cfg, section)
        fixed = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        setattr(cfg, section, replace(current, **fixed))
    return cfg


@dataclass
class TrainConfig:
    """MaskablePPO training (step 5). All values are [TBD] starting points."""

    total_timesteps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 1024                 # rollout steps per env per update
    batch_size: int = 512
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.999                # training discount; evaluation uses the undiscounted return
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    target_kl: float = 0.02
    net_arch: tuple[int, ...] = (256, 256)
    eval_freq: int = 50_000             # total env steps between evaluations
    n_eval_episodes: int = 10
    eval_seed: int = 10_000             # evaluation lists / DDM seeds start here (never used in training)
