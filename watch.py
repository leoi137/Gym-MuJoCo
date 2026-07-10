"""Render a trained Ant policy in a real-time MuJoCo viewer window.

Loads the best-eval model from `runs/<run_name>/` and runs N_EPISODES of
deterministic rollout. Each episode pops up a window; close it or wait for
the episode to end to move on.

Usage:
    python watch.py --run baseline_2leg
    python watch.py --run foot_contact_v1 --episodes 3 --latest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import envs  # noqa: F401 -- registers Spyder-v0 with Gymnasium


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, required=True,
                   help="run-name under runs/ (e.g. baseline_2leg)")
    p.add_argument("--episodes", type=int, default=5,
                   help="how many deterministic episodes to watch")
    p.add_argument("--latest", action="store_true",
                   help="use ant_sac.zip (latest) instead of ant_sac_best.zip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run
    best = run_dir / "ant_sac_best.zip"
    latest = run_dir / "ant_sac.zip"

    if args.latest:
        path = latest
    else:
        # Prefer the best-eval snapshot; fall back to latest if no best yet.
        path = best if best.exists() else latest

    if not path.exists():
        raise SystemExit(f"No model found in {run_dir} (looked for "
                         f"{best.name} and {latest.name})")
    print(f"Loading {path}")

    # Read which env this run trained on from its config; fall back to Ant-v5
    # for legacy runs created before env_id was recorded. Always watch on the
    # unwrapped reward so videos from different shaping experiments are
    # visually & numerically comparable.
    config_path = run_dir / "config.json"
    if config_path.exists():
        env_id = json.loads(config_path.read_text()).get("env_id", "Ant-v5")
    else:
        env_id = "Ant-v5"
    print(f"Env: {env_id}")
    env = gym.make(env_id, render_mode="human")
    model = SAC.load(path, device="cpu")  # CPU is plenty for inference

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            print(f"Episode {ep + 1}/{args.episodes}: return = {total_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
