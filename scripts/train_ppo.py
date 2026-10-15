"""Train MaskablePPO on the scrolling environment (step 5).

Run (PowerShell, venv active):
    python scripts\\train_ppo.py --name pilot --timesteps 1000000 --seed 0
    tensorboard --logdir runs          # learning curves at http://localhost:6006

Writes runs/<name>/:
    config.json                              environment + training settings
    best_model.zip, best_vecnormalize.pkl    best evaluation so far
    final_model.zip, final_vecnormalize.pkl  end of training
    evaluations.npz, tb/                     evaluation history, TensorBoard logs

Training lists / DDM noise come from seeds seed*1000 + env index; evaluation uses
seeds from TrainConfig.eval_seed (10000) on, so it never sees a training list.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym

from scrolling_agent import ScrollingEnv, TrainConfig
from scrolling_agent.cli import add_feed_args, config_from_args


class CycleSeeds(gym.Wrapper):
    """Reset with a fixed cycle of seeds, so every evaluation sees the same lists."""

    def __init__(self, env: gym.Env, seeds: list[int]):
        super().__init__(env)
        self._seeds = seeds
        self._k = 0

    def reset(self, *, seed=None, options=None):
        seed = self._seeds[self._k % len(self._seeds)]
        self._k += 1
        return self.env.reset(seed=seed, options=options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="run folder name under runs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None, help="default: TrainConfig.total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    add_feed_args(parser)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    cfg = config_from_args(args)
    overrides = {"total_timesteps": args.timesteps, "n_envs": args.n_envs, "gamma": args.gamma}
    train = replace(TrainConfig(), **{k: v for k, v in overrides.items() if v is not None})

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"seed": args.seed, "config": asdict(cfg), "train": asdict(train)}, indent=2)
    )

    venv = make_vec_env(ScrollingEnv, n_envs=train.n_envs, seed=args.seed * 1000,
                        env_kwargs={"config": cfg}, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=train.gamma)

    eval_seeds = list(range(train.eval_seed, train.eval_seed + train.n_eval_episodes))
    eval_env = DummyVecEnv([lambda: Monitor(CycleSeeds(ScrollingEnv(cfg), eval_seeds))])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    class SaveVecNormalize(BaseCallback):
        def _on_step(self) -> bool:
            self.model.get_vec_normalize_env().save(str(run_dir / "best_vecnormalize.pkl"))
            return True

    eval_cb = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=train.n_eval_episodes,
        eval_freq=max(train.eval_freq // train.n_envs, 1),
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        deterministic=True,
        callback_on_new_best=SaveVecNormalize(),
    )

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        n_steps=train.n_steps,
        batch_size=train.batch_size,
        n_epochs=train.n_epochs,
        learning_rate=train.learning_rate,
        gamma=train.gamma,
        gae_lambda=train.gae_lambda,
        ent_coef=train.ent_coef,
        target_kl=train.target_kl,
        policy_kwargs={"net_arch": list(train.net_arch)},
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=train.total_timesteps, callback=eval_cb, tb_log_name="ppo")
    model.save(run_dir / "final_model.zip")
    venv.save(str(run_dir / "final_vecnormalize.pkl"))
    venv.close()
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
