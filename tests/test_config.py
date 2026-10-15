import pytest

from scrolling_agent import Config


def test_step_count_matches_draft():
    cfg = Config()
    assert cfg.time.n_steps == 1200  # T = 120 s, dt = 0.1 s


def test_cost_per_second():
    cfg = Config()
    assert cfg.time.cost_per_second == pytest.approx(0.1)


@pytest.mark.parametrize("speed, expected", [(0.0, 1.0), (0.5, 0.8), (1.0, 0.5), (2.0, 0.2)])
def test_speed_gain(speed, expected):
    assert Config().ddm.speed_gain(speed) == pytest.approx(expected)


def test_feed_promising_type_has_higher_reward():
    assert Config().feed.a1 > 0
