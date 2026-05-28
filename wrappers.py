"""Reward-shaping wrappers for Ant-v5.

Each wrapper is a `gymnasium.Wrapper` so it composes cleanly with the
standard `gym.make(...)` pipeline:

    env = gym.make("Ant-v5")
    env = FootContactRewardWrapper(env, penalty=1.0)

Wrappers should also write their per-step shaping contribution into
`info` so it shows up in TensorBoard via SB3's logger and we can confirm
the shaping is actually firing.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


def _resolve_ankle_body_ids(env: gym.Env) -> tuple[int, ...]:
    """Find Ant's 4 ankle body IDs by looking up geoms named `*_ankle_geom`.

    Hardcoding indices (4, 7, 10, 13) works for stock Ant-v5 but breaks if
    the XML is ever modified. Resolving by geom name is a one-time cost at
    wrapper init and survives XML edits.
    """
    model = env.unwrapped.model
    ankle_body_ids: list[int] = []
    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name.endswith("_ankle_geom"):
            # geom.bodyid is a 1-element ndarray in the mujoco python binding.
            ankle_body_ids.append(int(model.geom(geom_id).bodyid[0]))
    if len(ankle_body_ids) != 4:
        raise RuntimeError(
            f"Expected 4 ankle geoms in Ant model, found {len(ankle_body_ids)}: "
            f"{ankle_body_ids}"
        )
    return tuple(ankle_body_ids)


class FootContactRewardWrapper(gym.Wrapper):
    """Penalize gaits where some legs never touch the ground.

    Tracks, over a rolling window of `window` env-steps, whether each
    ankle's external contact-force magnitude exceeded `contact_threshold`
    at least once. The per-step penalty is

        shaping = -penalty * (4 - num_legs_with_recent_contact)

    A balanced quadrupedal gait has all 4 legs contacting within the
    window → 0 penalty. A two-legged drag has 2 legs that never touch
    ground → penalty of -2 * penalty per step.

    Rolling window (not instantaneous variance): legs *should* be airborne
    during their swing phase, so instantaneous imbalance is normal walking.
    A leg that hasn't contacted in N steps is the actual bug.

    The shaping value is added to `info` under `shaping/foot_contact` so
    you can verify in TensorBoard that the signal is firing as expected.
    """

    def __init__(
        self,
        env: gym.Env,
        penalty: float = 1.0,
        window: int = 50,
        contact_threshold: float = 1.0,
    ):
        super().__init__(env)
        self.penalty = penalty
        self.window = window
        self.contact_threshold = contact_threshold
        self._ankle_body_ids = _resolve_ankle_body_ids(env)
        self._contact_history: list[deque[bool]] = [
            deque(maxlen=window) for _ in self._ankle_body_ids
        ]

    def reset(self, **kwargs: Any):
        for hist in self._contact_history:
            hist.clear()
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cfrc = self.env.unwrapped.data.cfrc_ext  # (nbody, 6): 3 force + 3 torque
        legs_with_recent_contact = 0
        for leg_i, body_id in enumerate(self._ankle_body_ids):
            force_mag = float(np.linalg.norm(cfrc[body_id]))
            self._contact_history[leg_i].append(force_mag > self.contact_threshold)
            if any(self._contact_history[leg_i]):
                legs_with_recent_contact += 1

        idle_legs = len(self._ankle_body_ids) - legs_with_recent_contact
        shaping = -self.penalty * idle_legs
        info["shaping/foot_contact"] = shaping
        info["shaping/idle_legs"] = idle_legs
        return obs, reward + shaping, terminated, truncated, info


# Registry so train.py can pick a wrapper by name via a CLI flag.
WRAPPERS: dict[str, type[gym.Wrapper]] = {
    "foot_contact": FootContactRewardWrapper,
}
