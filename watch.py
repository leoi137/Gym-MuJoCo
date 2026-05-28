"""Render the trained ant in a real-time window.

Loads ant_sac.zip and runs N_EPISODES deterministic episodes. Each one pops
up a MuJoCo viewer window; close the window or wait for the episode to end
to move on.

Usage:
    python watch.py
"""
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

MODEL_PATH = Path("ant_sac.zip")           # latest checkpoint (for resume)
BEST_MODEL_PATH = Path("ant_sac_best.zip") # best-ever eval policy (for showing off)
N_EPISODES = 5


def main() -> None:
    # Prefer the best-ever model if train.py has saved one; otherwise fall back
    # to the latest checkpoint. This protects you from watching a policy that
    # happened to degrade late in training.
    if BEST_MODEL_PATH.exists():
        path = BEST_MODEL_PATH
    elif MODEL_PATH.exists():
        path = MODEL_PATH
    else:
        raise SystemExit("No model found. Train first with: python train.py")
    print(f"Loading {path}")

    # render_mode="human" opens a live MuJoCo window.
    # Use "rgb_array" instead if you want raw frames to make your own video.
    env = gym.make("Ant-v5", render_mode="human")

    # Inference is tiny -- CPU is plenty and avoids GPU startup overhead.
    model = SAC.load(path, device="cpu")

    try:
        for ep in range(N_EPISODES):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                # deterministic=True uses the policy's mean action (no
                # exploration noise) -- this is what you want for a "show".
                action, _ = model.predict(obs, deterministic=True)
                # Gymnasium step returns 5 values:
                #   obs, reward, terminated, truncated, info
                # terminated = the ant flipped over (task failure)
                # truncated  = hit the 1000-step time limit
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            print(f"Episode {ep + 1}/{N_EPISODES}: return = {total_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
