"""Command-line helpers shared by the scripts."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import Config

FEED_ARGS = {  # flag -> FeedConfig field
    "--p": "p",
    "--kappa": "kappa",
    "--d-nu": "d_nu",
    "--m-nu": "m_nu",
    "--m-lambda": "m_lambda",
}


def add_feed_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("feed condition (defaults: config.py FeedConfig)")
    for flag, name in FEED_ARGS.items():
        group.add_argument(flag, type=float, default=None, dest=name, help=f"FeedConfig.{name}")


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    overrides = {name: getattr(args, name) for name in FEED_ARGS.values() if getattr(args, name) is not None}
    cfg.feed = replace(cfg.feed, **overrides)
    return cfg


def make_policy(name: str, rho: float = 0.3, run: str | None = None, which: str = "best", seed: int = 0):
    from .policies import GreedyKnown, PPOPolicy, RandomPolicy, ScrollFirst

    if name == "random":
        return RandomPolicy(seed)
    if name == "scrollfirst":
        return ScrollFirst(rho)
    if name == "greedy":
        return GreedyKnown(rho)
    if name == "ppo":
        if run is None:
            raise SystemExit("--run is required for --policy ppo")
        return PPOPolicy.load(run, which)
    raise SystemExit(f"unknown policy {name}")
