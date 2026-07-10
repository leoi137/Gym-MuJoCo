"""Custom environments. Importing this package registers them with Gymnasium.

train.py and watch.py do `import envs` for this side effect, after which
`gym.make("Spyder-v0")` works exactly like a built-in env id.
"""
from gymnasium.envs.registration import register, registry

if "Spyder-v0" not in registry:
    register(
        id="Spyder-v0",
        entry_point="envs.spyder_env:SpyderEnv",
        # Truncation horizon: episodes end after 1000 steps (50 seconds at
        # 20 Hz) even if the spider is still healthy — same as Ant-v5.
        max_episode_steps=1000,
    )
