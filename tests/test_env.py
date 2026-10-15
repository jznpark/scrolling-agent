import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from helpers import click, constant_feed, random_masked_actions, run_until

from scrolling_agent import Config, ScrollingEnv
from scrolling_agent.env import LIST, TRANSITION, WATCH, visible_items, watch_reward


def understood(i):
    return lambda env: env.understood[i]


# --- pure functions --------------------------------------------------------


@pytest.mark.parametrize(
    "scroll, expected",
    [(0.0, [0, 1, 2]), (0.4, [0, 1, 2]), (0.6, [1, 2, 3]), (0.5, [0, 1, 2, 3]), (97.0, [97, 98, 99])],
)
def test_visible_items(scroll, expected):
    assert visible_items(scroll, 100, 3.0, 0.5).tolist() == expected


def test_watch_reward_integrates_rate():
    # sum over consecutive steps = (A / lam) (1 - exp(-lam * total))
    total = sum(watch_reward(2.0, 0.5, k * 0.1, 0.1) for k in range(30))
    assert total == pytest.approx((2.0 / 0.5) * (1 - math.exp(-0.5 * 3.0)))


# --- API -------------------------------------------------------------------


def test_gymnasium_check_env():
    check_env(ScrollingEnv(), skip_render_check=True)


def test_same_seed_same_trajectory():
    def play(seed):
        env = ScrollingEnv()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, r, *_ = env.step(a)
            out.append(np.append(obs, r))
        return np.concatenate(out)

    np.testing.assert_array_equal(play(3), play(3))
    assert not np.array_equal(play(3), play(4))


def test_fixed_feed_option_and_size_check():
    env = ScrollingEnv()
    feed = constant_feed(A=2.5)
    env.reset(seed=0, options={"feed": feed})
    assert env.feed is feed
    with pytest.raises(ValueError):
        ScrollingEnv(feed=constant_feed(n=10))


def test_episode_length():
    env = ScrollingEnv()
    env.reset(seed=0)
    for k in range(env.cfg.time.n_steps):
        *_, terminated, truncated, _ = env.step(env.stay_action)
        assert terminated == (k == env.cfg.time.n_steps - 1) and not truncated


# --- masks -----------------------------------------------------------------


def test_masks_in_list_before_understanding():
    env = ScrollingEnv(feed=constant_feed())
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask[env.down_actions + env.up_actions + [env.stay_action]].all()
    assert not mask[env.back_action]
    assert not mask[env.slot_actions].any()  # nothing understood yet


def test_only_understood_items_clickable_and_watch_masks():
    env = ScrollingEnv(feed=constant_feed())
    env.reset(seed=0)
    run_until(env, env.stay_action, lambda e: e.understood[:3].any())
    mask = env.action_masks()
    for j, i in enumerate(env.visible_slots()):
        assert mask[env.slot_actions[j]] == env.understood[i]

    item = int(np.flatnonzero(env.understood)[0])
    slot = list(env.visible_slots()).index(item)
    env.step(env.slot_actions[slot])
    assert env.mode == TRANSITION
    assert env.action_masks().tolist() == [a == env.stay_action for a in range(env.n_actions)]

    env.step(env.stay_action)
    assert env.mode == WATCH and env.current == item
    mask = env.action_masks()
    assert mask[env.stay_action] and mask[env.back_action] and mask.sum() == 2


def test_invalid_action_is_replaced_by_stay():
    env = ScrollingEnv(feed=constant_feed())
    env.reset(seed=0)
    *_, info = env.step(env.back_action)
    assert info["invalid_action"] and env.mode == LIST


# --- transitions and reward ------------------------------------------------


def test_click_and_back_take_two_steps_without_reward():
    cfg = Config()
    env = ScrollingEnv(cfg, feed=constant_feed())
    env.reset(seed=0)
    run_until(env, env.stay_action, understood(0))
    x_before = env.x.copy()

    rewards = click(env, 0)
    assert len(rewards) == cfg.time.transition_steps == 2
    assert rewards == pytest.approx([-cfg.time.step_cost] * 2)
    np.testing.assert_array_equal(env.x, x_before)  # DDM frozen during transition

    scroll = env.scroll
    rewards = [env.step(env.back_action)[1]]
    rewards += run_until(env, env.stay_action, lambda e: e.mode == LIST)
    assert len(rewards) == 2 and rewards == pytest.approx([-cfg.time.step_cost] * 2)
    assert env.scroll == scroll and env.current is None


def test_watching_reward_matches_closed_form():
    cfg = Config()
    A, lam, k = 2.0, 0.4, 25
    env = ScrollingEnv(cfg, feed=constant_feed(A=A, lam=lam))
    env.reset(seed=0)
    run_until(env, env.stay_action, understood(0))
    click(env, 0)
    rewards = [env.step(env.stay_action)[1] for _ in range(k)]
    watched = sum(rewards) + k * cfg.time.step_cost
    assert watched == pytest.approx((A / lam) * (1 - math.exp(-lam * k * cfg.time.dt)))
    assert env.w[0] == pytest.approx(k * cfg.time.dt)


def test_scroll_stays_within_bounds():
    env = ScrollingEnv(feed=constant_feed())
    env.reset(seed=0)
    _, r, *_ = env.step(env.up_actions[0])
    assert env.scroll == 0.0 and r == pytest.approx(-env.cfg.time.step_cost)
    for _ in range(1000):
        env.step(env.down_actions[0])
    assert env.scroll == pytest.approx(env.max_scroll)


def test_scroll_speed():
    env = ScrollingEnv(feed=constant_feed())
    env.reset(seed=0)
    for _ in range(10):  # 1 s at 0.5 screen heights / s = 1.5 cards
        env.step(env.down_actions[0])
    assert env.scroll == pytest.approx(1.5)


# --- hidden information ----------------------------------------------------


def test_hidden_attributes_do_not_leak():
    """Feeds that differ only in hidden attributes give identical observations."""
    base = constant_feed()
    other = constant_feed()
    other.A[0], other.lam[0] = 9.0, 2.0  # visible item, not yet understood
    other.A[50], other.nu[50], other.lam[50] = 9.0, 5.0, 2.0  # never visible here
    other.z[:] = 1

    envs = [ScrollingEnv(feed=f) for f in (base, other)]
    obs = [e.reset(seed=1)[0] for e in envs]
    np.testing.assert_array_equal(*obs)
    while not envs[0].understood[0]:
        obs = [e.step(e.stay_action)[0] for e in envs]
        if not envs[0].understood[0]:
            np.testing.assert_array_equal(*obs)
    # once understood, item 0's profile is revealed and the observations differ
    assert not np.array_equal(*obs)


def test_observation_fits_space_over_random_play():
    env = ScrollingEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    random_masked_actions(env, np.random.default_rng(0), 300)
    assert env.observation_space.contains(env._obs())
