"""Train MaskablePPO on the scrolling environment (step 5).

Run (PowerShell, venv active):
    python scripts\\train_ppo.py --name pilot --timesteps 1000000 --seed 0
    python scripts\\train_ppo.py --name topk/s0 --set obs.memory=topk --train ent_coef=0.001
    tensorboard --logdir runs          # learning curves at http://localhost:6006

Resume an interrupted run (e.g. after a reboot) with the same --name and --resume:
    python scripts\\train_ppo.py --name topk/s0 --resume
It continues from last_model.zip (saved every TrainConfig.checkpoint_freq steps; best_model.zip
if there is none) up to the stored total, with the settings stored in config.json.

Writes runs/<name>/:
    config.json                              environment + training settings
    best_model.zip, best_vecnormalize.pkl    best evaluation so far
    last_model.zip, last_vecnormalize.pkl    latest checkpoint (for --resume)
    final_model.zip, final_vecnormalize.pkl  end of training
    evaluations.npz, tb/                     evaluation history, TensorBoard logs

Training lists / DDM noise come from seeds seed*1000 + env index (+500 after a resume);
evaluation uses seeds from TrainConfig.eval_seed (10000) on, so it never sees a training list.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from scrolling_agent import ScrollingEnv, TrainConfig
from scrolling_agent.cli import add_config_args, config_from_args, train_overrides
from scrolling_agent.config import config_from_dict
from scrolling_agent.decision import DecisionInterval


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


def replace_file(src: Path, dst: Path, attempts: int = 20) -> None:
    """os.replace with retries: on Windows a virus scanner may hold a just-written file for a moment."""
    for k in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(0.25)


def learning_rate(train: TrainConfig):
    """Constant rate, or a linear decay to 0 over training (SB3 passes progress_remaining 1 -> 0)."""
    if train.lr_schedule == "constant":
        return train.learning_rate
    if train.lr_schedule == "linear":
        lr0 = train.learning_rate
        return lambda progress_remaining: lr0 * progress_remaining
    raise SystemExit(f"unknown lr_schedule {train.lr_schedule!r}; use constant or linear")


def stored_settings(run_dir: Path) -> tuple[int, object, TrainConfig]:
    """Seed, Config and TrainConfig of an existing run (config.json)."""
    stored = json.loads((run_dir / "config.json").read_text())
    names = {f.name for f in fields(TrainConfig)}
    train = replace(TrainConfig(), **{k: tuple(v) if isinstance(v, list) else v
                                      for k, v in stored["train"].items() if k in names})
    return stored["seed"], config_from_dict(stored["config"]), train


def resume_point(run_dir: Path) -> tuple[Path, Path] | None:
    """(model, VecNormalize) files to continue from: the last checkpoint, else the best model."""
    for which in ("last", "best"):
        model, stats = run_dir / f"{which}_model.zip", run_dir / f"{which}_vecnormalize.pkl"
        if model.exists() and stats.exists():
            return model, stats
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="run folder name under runs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None, help="default: TrainConfig.total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--train", nargs="+", default=[], metavar="FIELD=VALUE",
                        help="override any TrainConfig value, e.g. --train ent_coef=0.001 n_steps=2048")
    parser.add_argument("--resume", action="store_true",
                        help="continue runs/<name> from its last checkpoint with its stored settings")
    add_config_args(parser)
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    run_dir = Path("runs") / args.name
    start = None
    if args.resume:
        if (run_dir / "final_model.zip").exists():
            print(f"{run_dir} already finished")
            return
        seed, cfg, train = stored_settings(run_dir)
        if args.timesteps:
            train = replace(train, total_timesteps=args.timesteps)
        start = resume_point(run_dir)
        if start is None:
            print(f"no checkpoint in {run_dir}: starting it again with its stored settings")
    else:
        seed = args.seed
        cfg = config_from_args(args)
        overrides = {"total_timesteps": args.timesteps, "n_envs": args.n_envs, "gamma": args.gamma}
        train = replace(TrainConfig(), **{k: v for k, v in overrides.items() if v is not None})
        train = train_overrides(train, args.train)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps({"seed": seed, "config": asdict(cfg), "train": asdict(train)}, indent=2)
        )

    # a resumed run draws fresh training lists; +500 stays clear of other seeds and of evaluation
    hold = ({"wrapper_class": DecisionInterval, "wrapper_kwargs": {"k": train.decision_steps}}
            if train.decision_steps > 1 else {})  # learning side: decide at most every k env steps
    venv = make_vec_env(ScrollingEnv, n_envs=train.n_envs, seed=seed * 1000 + (500 if start else 0),
                        env_kwargs={"config": cfg}, vec_env_cls=SubprocVecEnv, **hold)
    if start:
        venv = VecNormalize.load(str(start[1]), venv)
        venv.training, venv.norm_reward = True, True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=train.gamma)

    eval_seeds = list(range(train.eval_seed, train.eval_seed + train.n_eval_episodes))
    def make_eval_env():
        env = CycleSeeds(ScrollingEnv(cfg), eval_seeds)
        return Monitor(DecisionInterval(env, train.decision_steps) if train.decision_steps > 1 else env)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    class SaveVecNormalize(BaseCallback):
        def _on_step(self) -> bool:
            self.model.get_vec_normalize_env().save(str(run_dir / "best_vecnormalize.pkl"))
            return True

    class SaveCheckpoint(BaseCallback):
        """Overwrite last_model.zip / last_vecnormalize.pkl every checkpoint_freq steps (atomic replace)."""

        def _on_step(self) -> bool:
            if self.n_calls % max(train.checkpoint_freq // train.n_envs, 1) == 0:
                self.model.save(run_dir / "last_model_tmp.zip")
                self.model.get_vec_normalize_env().save(str(run_dir / "last_vecnormalize_tmp.pkl"))
                replace_file(run_dir / "last_model_tmp.zip", run_dir / "last_model.zip")
                replace_file(run_dir / "last_vecnormalize_tmp.pkl", run_dir / "last_vecnormalize.pkl")
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

    if start:
        model = MaskablePPO.load(start[0], env=venv, device="cpu", tensorboard_log=str(run_dir / "tb"))
        history = run_dir / "evaluations.npz"
        if history.exists():  # keep the evaluation history and the best score so far
            old = np.load(history)
            keep = old["timesteps"] <= model.num_timesteps  # resuming from best drops later evaluations
            eval_cb.evaluations_timesteps = list(old["timesteps"][keep])
            eval_cb.evaluations_results = [list(r) for r in old["results"][keep]]
            eval_cb.evaluations_length = [list(r) for r in old["ep_lengths"][keep]]
            eval_cb.best_mean_reward = float(old["results"].mean(axis=1).max())
        print(f"resuming {run_dir} from {start[0].name} at {model.num_timesteps} steps "
              f"(target {train.total_timesteps})")
        remaining = max(train.total_timesteps - model.num_timesteps, 0)
        reset = False
    else:
        model = MaskablePPO(
            "MlpPolicy",
            venv,
            n_steps=train.n_steps,
            batch_size=train.batch_size,
            n_epochs=train.n_epochs,
            learning_rate=learning_rate(train),
            gamma=train.gamma,
            gae_lambda=train.gae_lambda,
            ent_coef=train.ent_coef,
            target_kl=train.target_kl,
            policy_kwargs={"net_arch": list(train.net_arch)},
            tensorboard_log=str(run_dir / "tb"),
            seed=seed,
            device="cpu",
            verbose=1,
        )
        remaining, reset = train.total_timesteps, True
    model.learn(total_timesteps=remaining, callback=[eval_cb, SaveCheckpoint()], tb_log_name="ppo",
                reset_num_timesteps=reset)
    model.save(run_dir / "final_model.zip")
    venv.save(str(run_dir / "final_vecnormalize.pkl"))
    venv.close()
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
