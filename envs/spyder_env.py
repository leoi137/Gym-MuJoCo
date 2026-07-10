"""Spyder-v0: a 12-DoF spider locomotion env (Ant-v5 recipe, custom morphology).

This is the repo's first hand-built environment. It follows Gymnasium's
Ant-v5 exactly in *structure* (obs layout, reward terms, termination rule)
so results stay comparable to the Ant baselines, but runs the spyder12
model (assets/spyder12.xml): 4 legs x 3 joints instead of Ant's 4 x 2.

Action space -- Box(-1, 1, (12,)):
    One float per motor, in per-leg order (the XML <actuator> block is the
    contract): [hip_1, lift_1, knee_1, hip_2, ..., knee_4].
    Sign = rotation direction around that joint's axis, magnitude = fraction
    of max torque (gear=150, so action 0.4 -> 60 N*m at the joint).

Observation space -- Box(-inf, inf, (113,)):
    [ 0:17]  qpos minus x,y — torso z (1), torso orientation quaternion (4),
             12 joint angles. Global x,y are excluded on purpose: walking
             looks identical at any point on the plane, so feeding absolute
             position would only invite overfitting to "where" instead of
             "how".
    [17:35]  qvel — torso linear velocity (3), angular velocity (3),
             12 joint angular velocities.
    [35:113] cfrc_ext — external contact force/torque (6) for each of the
             13 non-world bodies. This is how the policy "feels" the ground.

Reward (Ant-v5's terms; ctrl weight retuned for this morphology, see __init__):
    reward = forward_velocity              # dx/dt of the torso, m/s
           + 1.0 * healthy                 # alive bonus, paid every step
           - 0.1 * ||action||^2            # torque cost: don't flail
           - 5e-4 * ||clip(cfrc,-1,1)||^2  # contact cost: don't slam

Reward-hacking postmortem (v0 + v1, 100K steps each): with Ant's raw
constants this robot learned suicide-by-jumping — living cost ~1/step for
a clumsy policy while its legs could launch the torso past the healthy
ceiling, so instant termination maximized return. Gymnasium's envs avoid
this in one of two ways: termination physically unreachable (Ant: max jump
0.78 < ceiling 1.0) or living net-positive for any policy (Hopper/Walker:
ctrl weight 1e-3; Humanoid: alive=5). v1 tried Ant's mechanism (gear
150 -> 40; crude pumping only reached z=0.72) and SAC beat it with a
coordinated spring-loaded launch to 2.43 m — a reachability test is a
lower bound, and the optimizer searches harder than your test. Hence v2
uses the Hopper mechanism, which is exploit-agnostic: ctrl weight 0.1
makes being alive pay ~+0.6/step even while flailing, so no reachable
termination can ever out-earn living. Gear stays 40 (calmer dynamics).

Termination: torso z leaves [0.2, 1.0] (fallen / launched) or any state
value goes non-finite. Truncation at 1000 steps is handled by the
registration in envs/__init__.py, not here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

SPYDER_XML = str(Path(__file__).resolve().parent.parent / "assets" / "spyder12.xml")

DEFAULT_CAMERA_CONFIG = {
    # Tracking camera pinned to the torso (body 1; body 0 is the world), so
    # eval videos follow the spider instead of filming it walking off-frame.
    "type": 1,  # mjCAMERA_TRACKING
    "trackbodyid": 1,
    "distance": 4.0,
}


class SpyderEnv(MujocoEnv, utils.EzPickle):
    # dt = timestep (0.01) * frame_skip (5) = 0.05s -> the policy acts at
    # 20 Hz while physics integrates at 100 Hz. render_fps must match 1/dt.
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file: str = SPYDER_XML,
        frame_skip: int = 5,
        forward_reward_weight: float = 1.0,
        # 0.1, not Ant's 0.5: with 12 motors a random policy pays
        # 0.1 * 12 * E[a^2] ~= 0.4/step, so being alive nets ~+0.6/step even
        # while flailing. That makes the strategy ordering
        # walking (+~1.9) > standing (+1.0) > dying (~0) hold from step one
        # — the Hopper/Walker mechanism (they use 1e-3), needed because this
        # morphology can reach any credible jump ceiling (measured apex
        # 2.43 m at gear 40) so Ant's closed-door mechanism can't apply.
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        # Ceiling 3.0 sits above the measured best-effort jump apex (2.43 m),
        # so the z-check only catches genuine physics anomalies. Jumping is
        # legal now — just unprofitable (costs torque, earns nothing extra).
        # The 0.2 floor is physically unreachable (torso sphere r=0.25 rests
        # at 0.25) and kept only as a NaN/penetration guard.
        healthy_z_range: tuple[float, float] = (0.2, 3.0),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # set below, once model sizes are known
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Size the observation from the loaded model instead of hardcoding
        # 113, so an XML edit (say, adding a tail) can't silently desync the
        # env from the robot. qpos minus the 2 excluded world x,y; full qvel;
        # 6 contact-force values per non-world body.
        obs_size = (
            (self.data.qpos.size - 2)
            + self.data.qvel.size
            + (self.model.nbody - 1) * 6
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    # --- Reward pieces -------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        return bool(np.isfinite(state).all() and min_z <= state[2] <= max_z)

    def control_cost(self, action: np.ndarray) -> float:
        return self._ctrl_cost_weight * float(np.sum(np.square(action)))

    @property
    def contact_cost(self) -> float:
        # cfrc_ext can spike to thousands of newtons on hard impacts; clipping
        # to [-1, 1] before squaring keeps the penalty bounded so one bad
        # landing can't dominate an episode's return.
        raw = self.data.cfrc_ext
        clipped = np.clip(raw, *self._contact_force_range)
        return self._contact_cost_weight * float(np.sum(np.square(clipped)))

    # --- Gym API --------------------------------------------------------------

    def step(self, action: np.ndarray):
        xy_before = self.data.body("torso").xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_after = self.data.body("torso").xpos[:2].copy()

        x_velocity, y_velocity = (xy_after - xy_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        terminated = self._terminate_when_unhealthy and not self.is_healthy

        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "x_position": float(self.data.body("torso").xpos[0]),
            "y_position": float(self.data.body("torso").xpos[1]),
            "x_velocity": float(x_velocity),
            "y_velocity": float(y_velocity),
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        position = self.data.qpos.flatten()[2:]  # drop world x,y (see docstring)
        velocity = self.data.qvel.flatten()
        contact_force = self.data.cfrc_ext[1:].flatten()  # [1:] skips the world body
        return np.concatenate((position, velocity, contact_force))

    def reset_model(self) -> np.ndarray:
        # Small noise around the authored stance: enough that the policy can't
        # memorize one exact trajectory, small enough that it always starts
        # standing. The joint springs (stiffness=30 in the XML) hold the arch.
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
