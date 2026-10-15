import argparse
from dataclasses import asdict

import pytest

from scrolling_agent import Config
from scrolling_agent.cli import add_config_args, apply_overrides, config_from_args
from scrolling_agent.config import config_from_dict


def parse(argv):
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    return config_from_args(parser.parse_args(argv))


def test_feed_flags_and_set_overrides():
    cfg = parse(["--kappa", "0.8", "--set", "ddm.noise=0.3", "screen.speeds=(0.5, 1.5)"])
    assert cfg.feed.kappa == 0.8
    assert cfg.ddm.noise == 0.3
    assert cfg.screen.speeds == (0.5, 1.5)
    assert Config().ddm.noise == 0.25  # defaults untouched


@pytest.mark.parametrize("item", ["ddm.nope=1", "nosection.x=1", "ddm.noise"])
def test_bad_override_exits(item):
    with pytest.raises(SystemExit):
        apply_overrides(Config(), [item])


def test_config_round_trip_through_dict():
    cfg = parse(["--set", "screen.speeds=(0.5, 1.5)", "feed.p=0.4"])
    assert config_from_dict(asdict(cfg)) == cfg
