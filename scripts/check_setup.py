"""Check that the Python environment is ready.

Run:  python scripts/check_setup.py
1) imports the main libraries and prints their versions
2) trains PPO on CartPole for a few thousand steps (about 10–30 s on CPU)
"""

import platform
import sys


def main() -> None:
    print(f"Python {sys.version.split()[0]} on {platform.system()}")

    import gymnasium as gym
    import numpy as np
    import sb3_contrib
    import stable_baselines3 as sb3
    import torch

    import scrolling_agent
    from scrolling_agent import Config

    print(f"numpy {np.__version__} | gymnasium {gym.__version__} | torch {torch.__version__} "
          f"(CUDA: {torch.cuda.is_available()})")
    print(f"stable-baselines3 {sb3.__version__} | sb3-contrib {sb3_contrib.__version__} | "
          f"scrolling_agent {scrolling_agent.__version__}")

    cfg = Config()
    print(f"Config: T={cfg.time.session_seconds}s, dt={cfg.time.dt}s, N={cfg.time.n_steps} steps")

    from stable_baselines3 import PPO

    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=64, target_kl=0.02, seed=0, verbose=0)
    model.learn(total_timesteps=4096)

    obs, _ = env.reset(seed=1)
    total = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    print(f"PPO smoke test finished — CartPole episode return: {total:.0f}")
    print("Setup OK")


if __name__ == "__main__":
    main()
