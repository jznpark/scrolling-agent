"""Scrolling agent — DDM + PPO model of time allocation in a scrolling feed."""

from .config import Config, DDMConfig, FeedConfig, ScreenConfig, TimeConfig
from .feed import Feed, feed_stats, generate_feed, reorder_feed, sample_types

__all__ = [
    "Config",
    "DDMConfig",
    "Feed",
    "FeedConfig",
    "ScreenConfig",
    "TimeConfig",
    "feed_stats",
    "generate_feed",
    "reorder_feed",
    "sample_types",
]
__version__ = "0.1.0"
