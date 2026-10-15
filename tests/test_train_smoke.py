"""MaskablePPO runs on the environment and the trained policy plays a valid episode."""

import pytest

pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from scrolling_agent import ScrollingEnv  # noqa: E402
from scrolling_agent.evaluation import run_episode  # noqa: E402
from scrolling_agent.policies import PPOPolicy  # noqa: E402


def test_maskable_ppo_smoke(tmp_path):
    venv = VecNormalize(DummyVecEnv([ScrollingEnv]))
    model = MaskablePPO("MlpPolicy", venv, n_steps=128, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(256)
    model.save(tmp_path / "best_model.zip")
    venv.save(str(tmp_path / "best_vecnormalize.pkl"))

    policy = PPOPolicy.load(tmp_path)
    metrics, _ = run_episode(ScrollingEnv(), policy, seed=0)
    assert metrics["invalid_actions"] == 0
