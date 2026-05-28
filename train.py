"""Train SAC on Ant-v5, with periodic eval videos.

What this script does:
  1. Builds (or resumes) a Soft Actor-Critic agent on the Ant-v5 environment.
  2. Trains it for `--steps` env-steps, logging to TensorBoard.
  3. Every `--video-every` env-steps, runs ONE greedy episode and saves an
     MP4 to ./videos/ so you can literally watch the policy improve.
  4. On exit (normal OR Ctrl-C), saves the model + replay buffer so the next
     `python train.py` continues seamlessly.

Examples:
    python train.py                          # 750k steps (default)
    python train.py --steps 1_000_000        # ~2 hours on an RTX 2080
    python train.py --steps 4_000_000        # ~overnight
    python train.py --video-every 25_000     # finer-grained video progression
    python train.py --video-every 0          # disable videos entirely
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# --- Where things live on disk ---------------------------------------------
ENV_ID = "Ant-v5"
MODEL_PATH = Path("ant_sac.zip")           # latest -- used to RESUME training
BUFFER_PATH = Path("ant_buffer.pkl")       # the 1M-transition replay buffer
BEST_MODEL_PATH = Path("ant_sac_best.zip") # best-ever eval policy (for watch.py)
BEST_REWARD_PATH = Path("ant_sac_best.txt")# sidecar: the float reward of the best
TB_LOG_DIR = "./ant_tb/"                   # TensorBoard event files
VIDEO_DIR = Path("videos")                 # one MP4 per eval snapshot

# --- Hyperparameters (the spec) --------------------------------------------
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
DEVICE = "cuda"


class VideoEvalCallback(BaseCallback):
    """Every `record_every` env-steps, run one greedy eval episode and save it.

    SB3 calls `_on_step` after every env-step during training. We use it as a
    timer: when the global step counter is a multiple of `record_every`, we
    spin up a separate eval env, roll one deterministic episode while
    collecting RGB frames, and write the frames to an MP4.

    The eval episode also gets logged to TensorBoard as `eval/mean_reward`,
    so the video filenames line up with points on the reward curve.
    """

    def __init__(self, env_id: str, record_every: int, out_dir: Path, fps: int = 30):
        super().__init__()
        self.env_id = env_id
        self.record_every = record_every
        self.out_dir = out_dir
        self.fps = fps
        # High-water mark for eval reward. Gets saved alongside the best model
        # so we don't clobber a great previous best with a worse first eval on resume.
        self.best_eval_reward = float("-inf")

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Restore the previous best (if any) so a resumed run doesn't reset it.
        if BEST_REWARD_PATH.exists():
            self.best_eval_reward = float(BEST_REWARD_PATH.read_text().strip())
            print(f"[best] previous best eval reward: {self.best_eval_reward:.1f}")

    def _on_step(self) -> bool:
        # num_timesteps is the GLOBAL counter -- it survives across resumes,
        # so the videos are tagged by absolute training progress.
        if self.num_timesteps % self.record_every == 0:
            self._record_one_episode()
        return True  # returning False would stop training early

    def _record_one_episode(self) -> None:
        # A separate env with rgb_array rendering -- we need raw pixel frames,
        # not a popped-up window. (The training env has no render_mode set.)
        env = gym.make(self.env_id, render_mode="rgb_array")
        frames: list[np.ndarray] = []
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            frames.append(env.render())
            # deterministic=True -> use the policy's mean action, no exploration noise.
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
        env.close()

        # Zero-pad the step in the filename so `ls` sorts them in order.
        path = self.out_dir / f"eval_step_{self.num_timesteps:09d}.mp4"
        imageio.mimsave(path, frames, fps=self.fps, codec="libx264")

        # Surface in TensorBoard so the curve and the videos stay in sync.
        self.logger.record("eval/mean_reward", total_reward)
        self.logger.record("eval/episode_length", len(frames))
        if self.verbose:
            print(f"[video] step={self.num_timesteps:,}  "
                  f"return={total_reward:.1f}  ->  {path}")

        # If this eval beats the all-time best, save a separate "best" snapshot.
        # ant_sac.zip keeps being the latest (for resume); ant_sac_best.zip is
        # the high-water-mark policy used by watch.py.
        if total_reward > self.best_eval_reward:
            self.best_eval_reward = total_reward
            self.model.save(BEST_MODEL_PATH)
            BEST_REWARD_PATH.write_text(f"{total_reward:.6f}")
            if self.verbose:
                print(f"[best]  new best eval reward: {total_reward:.1f}  ->  {BEST_MODEL_PATH}")
        self.logger.record("eval/best_mean_reward", self.best_eval_reward)


def build_fresh_model(env: gym.Env) -> SAC:
    """Construct a brand-new SAC agent with the spec's hyperparameters."""
    return SAC(
        policy="MlpPolicy",       # two hidden layers of 256 ReLU units (SB3 default)
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        tensorboard_log=TB_LOG_DIR,
        verbose=1,
    )


def load_or_create(env: gym.Env) -> tuple[SAC, bool]:
    """Resume from disk if a checkpoint exists, else build a fresh agent.

    Returns (model, is_resume). `is_resume=True` tells the training loop to
    pass reset_num_timesteps=False, so TensorBoard's x-axis stays continuous
    across runs instead of restarting at zero.
    """
    if MODEL_PATH.exists():
        print(f"Resuming from {MODEL_PATH} (step counter will continue)")
        model = SAC.load(MODEL_PATH, env=env, device=DEVICE, tensorboard_log=TB_LOG_DIR)
        if BUFFER_PATH.exists():
            print(f"  loading replay buffer from {BUFFER_PATH}")
            model.load_replay_buffer(BUFFER_PATH)
        else:
            print("  no replay buffer file -- starting with an empty buffer")
        return model, True

    print("Starting fresh SAC run on Ant-v5")
    return build_fresh_model(env), False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--steps", type=int, default=750_000,
        help="env-steps to run THIS invocation (default: 750_000)",
    )
    p.add_argument(
        "--video-every", type=int, default=50_000,
        help="record an eval MP4 every N env-steps; 0 disables (default: 50_000)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(ENV_ID)
    model, is_resume = load_or_create(env)

    callbacks = []
    if args.video_every > 0:
        callbacks.append(VideoEvalCallback(ENV_ID, args.video_every, VIDEO_DIR))

    try:
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=not is_resume,
            callback=callbacks or None,
            progress_bar=True,
        )
    finally:
        # Save in a `finally` block so Ctrl-C still leaves a clean checkpoint.
        print(f"\nSaving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
        print(f"Saving replay buffer to {BUFFER_PATH}")
        model.save_replay_buffer(BUFFER_PATH)
        env.close()


if __name__ == "__main__":
    main()
