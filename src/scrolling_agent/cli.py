"""Command-line helpers shared by the scripts."""

from __future__ import annotations

import argparse
import ast
from dataclasses import fields, replace

from .config import Config

FEED_ARGS = {  # flag -> FeedConfig field
    "--p": "p",
    "--kappa": "kappa",
    "--d-nu": "d_nu",
    "--m-nu": "m_nu",
    "--m-lambda": "m_lambda",
}


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Feed shortcuts plus `--set section.field=value` for any Config value."""
    group = parser.add_argument_group("model settings (defaults: config.py)")
    for flag, name in FEED_ARGS.items():
        group.add_argument(flag, type=float, default=None, dest=name, help=f"feed.{name}")
    group.add_argument("--set", nargs="+", default=[], metavar="SECTION.FIELD=VALUE",
                       help="override any setting, e.g. --set ddm.noise=0.3 screen.cards_per_screen=5")


def apply_overrides(cfg: Config, items: list[str]) -> Config:
    """Apply "section.field=value" strings; values are Python literals or plain strings."""
    for item in items:
        key, sep, raw = item.partition("=")
        section, _, name = key.partition(".")
        if not sep or not hasattr(cfg, section):
            raise SystemExit(f"bad override {item!r}; use section.field=value "
                             f"(sections: {', '.join(f.name for f in fields(cfg))})")
        current = getattr(cfg, section)
        if name not in {f.name for f in fields(current)}:
            raise SystemExit(f"unknown setting {key!r}; {section} has: "
                             f"{', '.join(f.name for f in fields(current))}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(cfg, section, replace(current, **{name: value}))
    return cfg


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    overrides = {name: getattr(args, name) for name in FEED_ARGS.values() if getattr(args, name) is not None}
    cfg.feed = replace(cfg.feed, **overrides)
    return apply_overrides(cfg, args.set)


def make_policy(name: str, rho: float = 0.3, run: str | None = None, which: str = "best", seed: int = 0,
                dwell: float = 0.0):
    from .policies import GreedyKnown, PPOPolicy, RandomPolicy, ScrollFirst

    if name == "random":
        return RandomPolicy(seed)
    if name == "scrollfirst":
        return ScrollFirst(rho)
    if name == "greedy":
        return GreedyKnown(rho, dwell=dwell)
    if name == "ppo":
        if run is None:
            raise SystemExit("--run is required for --policy ppo")
        return PPOPolicy.load(run, which)
    raise SystemExit(f"unknown policy {name}")
