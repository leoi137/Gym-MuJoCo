"""Custom environments. Importing this package registers them with Gymnasium.

train.py and watch.py do `import envs` for this side effect, after which
`gym.make("Spyder-v0")` works exactly like a built-in env id.
"""
from pathlib import Path

from gymnasium.envs.registration import register, registry

if "Spyder-v0" not in registry:
    register(
        id="Spyder-v0",
        entry_point="envs.spyder_env:SpyderEnv",
        # Truncation horizon: episodes end after 1000 steps (50 seconds at
        # 20 Hz) even if the spider is still healthy — same as Ant-v5.
        max_episode_steps=1000,
    )

if "SpyderDesert-v0" not in registry:
    register(
        id="SpyderDesert-v0",
        # Same env class, same spaces — only the world XML differs (desert
        # heightfield instead of the plane), so Spyder-v0 checkpoints load
        # here unchanged. Terrain files come from make_terrain.py.
        entry_point="envs.spyder_env:SpyderEnv",
        kwargs={
            "xml_file": str(
                Path(__file__).resolve().parent.parent
                / "assets"
                / "spyder12_desert.xml"
            ),
        },
        max_episode_steps=1000,
    )
