"""Train SAC on Ant-v5 inside a per-run directory.

Each invocation lives under `runs/<run_name>/` so different reward-shaping
experiments don't clobber each other. The directory holds the model, the
replay buffer, the TensorBoard logs, eval videos, and a config.json
recording what produced it.

Examples:
    # Baseline (no wrapper, default reward), 750k steps:
    python train.py --run-name baseline_seed0 --seed 0

    # Foot-contact shaping experiment from scratch:
    python train.py --run-name foot_contact_v1 \\
                    --wrapper foot_contact \\
                    --wrapper-kwargs '{"penalty": 1.0, "window": 50}' \\
                    --seed 0 --steps 1_500_000

    # Resume the same run (auto-detected by re-using the same --run-name):
    python train.py --run-name foot_contact_v1 --steps 500_000

Re-invoking with the same --run-name resumes from disk; a new --run-name
starts fresh. The wrapper/seed args are only consulted on the *first*
invocation (when config.json is written); on resume they're loaded from
config.json so the run stays internally consistent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from wrappers import WRAPPERS

DEFAULT_ENV = "Ant-v5"

# --- Hyperparameters (the spec) --------------------------------------------
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
DEVICE = "cuda"


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "model": run_dir / "ant_sac.zip",
        "buffer": run_dir / "ant_buffer.pkl",
        "best_model": run_dir / "ant_sac_best.zip",
        "best_reward": run_dir / "ant_sac_best.txt",
        "tb": run_dir / "ant_tb",
        "videos": run_dir / "videos",
        "config": run_dir / "config.json",
    }


def _make_env(env_id: str, wrapper_name: str | None, wrapper_kwargs: dict[str, Any],
              seed: int | None, render_mode: str | None = None) -> gym.Env:
    """Build a MuJoCo env by id, optionally wrapped, optionally seeded."""
    env = gym.make(env_id, render_mode=render_mode)
    if wrapper_name is not None:
        if wrapper_name not in WRAPPERS:
            raise ValueError(f"Unknown wrapper {wrapper_name!r}. "
                             f"Available: {list(WRAPPERS)}")
        env = WRAPPERS[wrapper_name](env, **wrapper_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


class VideoEvalCallback(BaseCallback):
    """Every `record_every` env-steps, roll one greedy eval episode + save MP4.

    Logs three reward streams to TensorBoard:
      eval/mean_reward       -- what the policy actually optimizes (shaped)
      eval/base_reward       -- the unshaped Ant-v5 reward, for cross-run
                                comparison (computed as shaped - shaping)
      eval/mean_idle_legs    -- diagnostic: avg # of legs with no recent
                                ground contact (low = good quadruped gait)

    The last two are only meaningful when a shaping wrapper is active; with
    no wrapper they degenerate to base_reward == mean_reward and
    mean_idle_legs == 0.
    """

    def __init__(self, eval_env: gym.Env, record_every: int, video_dir: Path,
                 best_model_path: Path, best_reward_path: Path, fps: int = 30):
        super().__init__()
        self.eval_env = eval_env
        self.record_every = record_every
        self.video_dir = video_dir
        self.best_model_path = best_model_path
        self.best_reward_path = best_reward_path
        self.fps = fps
        self.best_eval_reward = float("-inf")

    def _on_training_start(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if self.best_reward_path.exists():
            self.best_eval_reward = float(self.best_reward_path.read_text().strip())
            print(f"[best] previous best eval reward: {self.best_eval_reward:.1f}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_every == 0:
            self._record_one_episode()
        return True

    def _record_one_episode(self) -> None:
        frames: list[np.ndarray] = []
        obs, _ = self.eval_env.reset()
        total_reward = 0.0
        total_shaping = 0.0
        total_idle = 0.0
        n_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            frames.append(self.eval_env.render())
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += float(reward)
            total_shaping += float(info.get("shaping/foot_contact", 0.0))
            total_idle += float(info.get("shaping/idle_legs", 0.0))
            n_steps += 1

        path = self.video_dir / f"eval_step_{self.num_timesteps:09d}.mp4"
        imageio.mimsave(path, frames, fps=self.fps, codec="libx264")

        base_reward = total_reward - total_shaping
        mean_idle = total_idle / max(n_steps, 1)

        self.logger.record("eval/mean_reward", total_reward)
        self.logger.record("eval/base_reward", base_reward)
        self.logger.record("eval/mean_idle_legs", mean_idle)
        self.logger.record("eval/episode_length", n_steps)

        if self.verbose:
            print(f"[video] step={self.num_timesteps:,}  "
                  f"shaped={total_reward:.1f}  base={base_reward:.1f}  "
                  f"idle_legs={mean_idle:.2f}  ->  {path}")

        if total_reward > self.best_eval_reward:
            self.best_eval_reward = total_reward
            self.model.save(self.best_model_path)
            self.best_reward_path.write_text(f"{total_reward:.6f}")
            if self.verbose:
                print(f"[best]  new best eval reward: {total_reward:.1f}  "
                      f"->  {self.best_model_path}")
        self.logger.record("eval/best_mean_reward", self.best_eval_reward)


def _load_or_init_config(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    """First invocation: write config.json from CLI args. Resume: read it back.

    Keeping wrapper/seed pinned in config.json (not re-read from CLI on resume)
    prevents an accidental --wrapper change mid-run from contaminating a buffer
    that was filled under different reward semantics.
    """
    if paths["config"].exists():
        config = json.loads(paths["config"].read_text())
        print(f"[config] loaded existing config from {paths['config']}")
        if (args.wrapper is not None or args.wrapper_kwargs != "{}"
                or args.seed is not None or args.env != DEFAULT_ENV):
            print("[config] note: --env/--wrapper/--seed args ignored on resume "
                  "(config.json is the source of truth)")
        return config

    wrapper_kwargs = json.loads(args.wrapper_kwargs)
    config = {
        "env_id": args.env,
        "algo": "SAC",
        "wrapper": args.wrapper,
        "wrapper_kwargs": wrapper_kwargs,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "policy": "MlpPolicy",
        },
        "env_kwargs": {},
        "seed": args.seed,
        "notes": "",
    }
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    paths["config"].write_text(json.dumps(config, indent=2) + "\n")
    print(f"[config] wrote new config to {paths['config']}")
    return config


def _build_model(env: gym.Env, paths: dict[str, Path], seed: int | None) -> tuple[SAC, bool]:
    """Resume from runs/<name>/ant_sac.zip if present, else fresh agent."""
    if paths["model"].exists():
        print(f"Resuming from {paths['model']}")
        model = SAC.load(paths["model"], env=env, device=DEVICE,
                         tensorboard_log=str(paths["tb"]))
        if paths["buffer"].exists():
            print(f"  loading replay buffer from {paths['buffer']}")
            model.load_replay_buffer(paths["buffer"])
        else:
            print("  no replay buffer file -- starting with an empty buffer")
        return model, True

    print(f"Starting fresh SAC run -> {paths['model']}")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        tensorboard_log=str(paths["tb"]),
        seed=seed,
        verbose=1,
    )
    return model, False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-name", type=str, required=True,
                   help="subdirectory under runs/ to write artifacts into")
    p.add_argument("--env", type=str, default=DEFAULT_ENV,
                   help=f"Gymnasium env id, set once at run creation "
                        f"(default: {DEFAULT_ENV}); pinned in config.json on resume")
    p.add_argument("--steps", type=int, default=750_000,
                   help="env-steps to run THIS invocation (default: 750_000)")
    p.add_argument("--video-every", type=int, default=50_000,
                   help="record an eval MP4 every N env-steps; 0 disables")
    p.add_argument("--wrapper", type=str, default=None, choices=list(WRAPPERS.keys()),
                   help="reward-shaping wrapper to apply (omit for none)")
    p.add_argument("--wrapper-kwargs", type=str, default="{}",
                   help='JSON dict of kwargs for the wrapper, e.g. \'{"penalty": 1.0}\'')
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for SAC + env (default: None = nondeterministic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = _run_paths(run_dir)

    config = _load_or_init_config(paths, args)
    env_id = config["env_id"]
    wrapper_name = config["wrapper"]
    wrapper_kwargs = config["wrapper_kwargs"]
    seed = config["seed"]

    train_env = _make_env(env_id, wrapper_name, wrapper_kwargs, seed)
    model, is_resume = _build_model(train_env, paths, seed)

    callbacks = []
    if args.video_every > 0:
        eval_env = _make_env(env_id, wrapper_name, wrapper_kwargs, seed, render_mode="rgb_array")
        callbacks.append(VideoEvalCallback(
            eval_env=eval_env,
            record_every=args.video_every,
            video_dir=paths["videos"],
            best_model_path=paths["best_model"],
            best_reward_path=paths["best_reward"],
        ))

    try:
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=not is_resume,
            callback=callbacks or None,
            progress_bar=True,
        )
    finally:
        print(f"\nSaving model to {paths['model']}")
        model.save(paths["model"])
        print(f"Saving replay buffer to {paths['buffer']}")
        model.save_replay_buffer(paths["buffer"])
        train_env.close()


if __name__ == "__main__":
    main()
