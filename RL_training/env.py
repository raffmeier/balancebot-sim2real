import gymnasium as gym
import mujoco
import numpy as np
import os
import math
from collections import deque
from scipy.spatial.transform import Rotation as R


MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

ENCODER_LSB = 2 * np.pi / 4096  # AS5600 wheel encoder resolution (12 bit), rad

# IMU errors that stay constant during an episode (domain randomization only)
GYRO_BIAS_STD = 0.01  # rad/s, residual bias after the boot calibration
IMU_TILT_MAX = 0.03   # rad (~1.7 deg), IMU mounting / PITCH_OFFSET error


class BalanceRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, 
                 max_steps = 1000,
                 domain_randomization = False, 
                 render_mode = None,
                 alive_bonus = 0.2,
                 v_reward_coeff = 0.8,
                 v_reward_sigma_wide = 0.2,       # m/s, tracking reward = coeff * (wide + narrow kernel) / 2:
                 v_reward_sigma_narrow = 0.05,    #   wide pulls a far off policy in, narrow rewards precision
                 yawrate_reward_coeff = 0.5,
                 yawrate_reward_sigma_wide = 1.5,     # rad/s
                 yawrate_reward_sigma_narrow = 0.3,   # rad/s
                 v_reward_sigma2 = None,          # old single kernel reward: ignored, keeps old run configs loadable
                 yawrate_reward_sigma2 = None,    # (same)
                 pitch_penalty_coeff = 0.3,
                 action_penalty_coeff = 0.05,
                 action_rate_penalty_coeff = 0.3,
                 max_action_delay = 3,
                 cmd_resample_steps = 400,
                 v_cmd_max = 0.3,
                 yawrate_cmd_max=2.0,
                 zero_cmd_prob=0.2,
                 v_cmd_acc_max=None,              # m/s^2, the reference ramps to the command at most this fast, None = step
                 yawrate_cmd_acc_max=None,        # rad/s^2, same for the yaw rate
                 pitchrate_penalty_coeff=0.02,    # damping: penalizes pitch_rate^2 (rocking at 3 rad/s costs 0.18 per step)
                 push_force_max=0.0,              # N, random pushes on the chassis during the episode, 0 = off
                 push_interval=(200, 600),        # steps between random pushes
                 push_duration=(10, 30),          # steps a random push lasts
                 push_height_max=0.06):           # m above the chassis center of mass (higher = more pitch torque)
        super().__init__()

        self.max_torque = 0.06 # Nm
        self.fall_angle = np.deg2rad(30)

        self.domain_randomization = domain_randomization
        self.render_mode = render_mode
        self.alive_bonus = alive_bonus
        self.v_reward_coeff = v_reward_coeff
        self.v_reward_sigmas = (v_reward_sigma_wide, v_reward_sigma_narrow)
        self.yawrate_reward_coeff = yawrate_reward_coeff
        self.yawrate_reward_sigmas = (yawrate_reward_sigma_wide, yawrate_reward_sigma_narrow)
        self.pitch_penalty_coeff = pitch_penalty_coeff
        self.action_penalty_coeff = action_penalty_coeff
        self.action_rate_penalty_coeff = action_rate_penalty_coeff
        self.pitchrate_penalty_coeff = pitchrate_penalty_coeff
        self.push_force_max = push_force_max
        self.push_interval = push_interval
        self.push_duration = push_duration
        self.push_height_max = push_height_max
        self._push_wrench = np.zeros(6)  # force + torque on the chassis while a push lasts
        self._push_height = 0.0          # m above the center of mass, for drawing the push
        self._push_steps_left = 0
        self._pushing = False
        self._next_push = 0
        self._prev_action = np.zeros(2, dtype=np.float32)  # last policy output: observation + smoothness penalty
        self.max_action_delay = max_action_delay           # with domain randomization: 0..max steps of actuation delay
        self.action_delay = 0
        self._action_queue = deque()                       # actions sent but not applied yet
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self._default_body_mass = self.model.body_mass.copy()
        self._default_friction = self.model.geom_friction.copy()
        self._default_body_inertia = self.model.body_inertia.copy()
        self._default_body_ipos = self.model.body_ipos.copy()
        self._default_dof_damping = self.model.dof_damping.copy()
        self._default_dof_frictionloss = self.model.dof_frictionloss.copy()

        self.sensor_imu_accel = "imu_accel"
        self.sensor_imu_gyro = "imu_gyro"
        self.sensor_left_wheel_pos = "left_wheel_pos"
        self.sensor_right_wheel_pos = "right_wheel_pos"
        self._prev_wheel_pos = np.zeros(2)  # last measured wheel angles, for the velocity estimate
        self._gyro_bias = np.zeros(3)
        self._imu_tilt = 0.0

        self._chassis_body = self.model.body("chassis").id
        self._wheel_bodies = [self.model.body(n).id for n in ("wheel_left", "wheel_right")]
        self._wheel_geoms = [self.model.geom(n).id for n in ("wheel_left_collision", "wheel_right_collision")]
        self._other_geoms = [self.model.geom(n).id for n in ("ground", "chassis_collision")]
        self._wheel_dofs = [self.model.joint(n).dofadr[0] for n in ("left_wheel_joint", "right_wheel_joint")]

        self.cmd_resample_steps = cmd_resample_steps
        self.v_cmd_max = v_cmd_max
        self.yawrate_cmd_max = yawrate_cmd_max
        self.zero_cmd_prob = zero_cmd_prob
        # The command can jump, the robot can't follow a jump (it has to lean first): observation and reward
        # use a reference that ramps to the command with limited acceleration. The firmware ramps the same way.
        self.cmd_acc_max = np.array([v_cmd_acc_max or np.inf, yawrate_cmd_acc_max or np.inf])
        self._cmd = np.zeros(2, dtype=np.float32)  # commanded [forward velocity, yaw rate]
        self._ref = np.zeros(2, dtype=np.float32)  # ramped command: observed and tracked
        self._actual = np.zeros(2)      # measured [forward velocity, yaw rate], for visualisation
        self._actual_vis = np.zeros(2)  # smoothed version of _actual, so the arrows don't jitter

        # Observation: pitch, pitchrate, yawrate, wheel_vel_left, wheel_vel_right, v_ref, yawrate_ref,
        #              prev_action_left, prev_action_right
        obs_high = np.array([np.pi, 30.0, 30.0, 60.0, 60.0, 5.0, 30.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: wheel_torque_left, wheel_torque_right
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.max_steps = max_steps
        self._step = 0

        self._pitch = 0.0
        self.torque_scale = 1.0

        self.obs_scale = np.array([0.2, 3.0, 3.0, 15.0, 15.0, 0.5, 3.0, 1.0, 1.0], dtype=np.float32)


    def _sample_command(self):
        if self.np_random.random() < self.zero_cmd_prob:
            self._cmd[:] = 0.0
        else:
            self._cmd[0] = self.np_random.uniform(-self.v_cmd_max, self.v_cmd_max)
            self._cmd[1] = self.np_random.uniform(-self.yawrate_cmd_max, self.yawrate_cmd_max)

    def set_command(self, v, yawrate, ramp=True):
        """New command (m/s, rad/s). ramp=False: the reference jumps to it (also the observation of the next step)."""
        self._cmd[:] = v, yawrate
        if not ramp:
            self._ref[:] = self._cmd

    def _ramp_reference(self):
        max_step = self.cmd_acc_max * self.model.opt.timestep
        self._ref += np.clip(self._cmd - self._ref, -max_step, max_step).astype(np.float32)

    @staticmethod
    def _tracking_reward(err, coeff, sigmas):
        wide, narrow = sigmas
        return coeff * 0.5 * (np.exp(-(err / wide)**2) + np.exp(-(err / narrow)**2))

    
    def _get_true_angles(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        return yaw, pitch, roll


    def _randomize_model(self):

        # Chassis
        b = self._chassis_body
        chassis_mass_scale = self.np_random.uniform(0.85, 1.15)
        self.model.body_mass[b] = self._default_body_mass[b] * chassis_mass_scale
        self.model.body_inertia[b] = self._default_body_inertia[b] * chassis_mass_scale

        chassis_com_x_offset = self.np_random.uniform(-0.003, 0.003)
        chassis_com_z_offset = self.np_random.uniform(-0.0075, 0.0075)
        self.model.body_ipos[b, 0] = self._default_body_ipos[b, 0] + chassis_com_x_offset
        self.model.body_ipos[b, 2] = self._default_body_ipos[b, 2] + chassis_com_z_offset # Offset z center of mass of the chassis +- 2cm

        # Wheels
        wheels_mass_scale = self.np_random.uniform(0.85, 1.15)
        for i in self._wheel_bodies:
            self.model.body_mass[i] = self._default_body_mass[i] * wheels_mass_scale
            self.model.body_inertia[i] = self._default_body_inertia[i] * wheels_mass_scale

        # Friction
        wheels_friction_scale = self.np_random.uniform(0.7, 1.3, size=3)
        for g in self._wheel_geoms:
            self.model.geom_friction[g] = self._default_friction[g] * wheels_friction_scale
        for g in self._other_geoms:
            self.model.geom_friction[g] = self._default_friction[g] * self.np_random.uniform(0.7, 1.3, size=3)

        # Damping
        d = self._wheel_dofs
        self.model.dof_damping[d] = self._default_dof_damping[d] * self.np_random.uniform(0.7, 1.4, size=2)
        self.model.dof_frictionloss[d] = self._default_dof_frictionloss[d] * self.np_random.uniform(0.5, 2, size=2)

        # Torque scale
        self.torque_scale = self.np_random.uniform(0.85, 1.15)

        # Actuation delay (sensor read + inference + motor current loop), in control steps
        self.action_delay = int(self.np_random.integers(0, self.max_action_delay + 1))

        # IMU
        self._gyro_bias = self.np_random.normal(0.0, GYRO_BIAS_STD, size=3)
        self._imu_tilt = self.np_random.uniform(-IMU_TILT_MAX, IMU_TILT_MAX)

    def _restore_default_model(self):
        """Undo any domain randomization; put the model back to the compiled XML values."""
        self.model.body_mass[:] = self._default_body_mass
        self.model.body_inertia[:] = self._default_body_inertia
        self.model.body_ipos[:] = self._default_body_ipos
        self.model.geom_friction[:] = self._default_friction
        self.model.dof_damping[:] = self._default_dof_damping
        self.model.dof_frictionloss[:] = self._default_dof_frictionloss
        self.torque_scale = 1.0
        self.action_delay = 0
        self._gyro_bias[:] = 0.0
        self._imu_tilt = 0.0
        

    def _get_obs(self) -> np.ndarray:
        # IMU readings with noise and bias
        accel_x, _, accel_z = self._read_sensor(self.sensor_imu_accel)
        _, pitch_rate, yaw_rate = self._read_sensor(self.sensor_imu_gyro) + self._gyro_bias

        accel_pitch = -math.atan2(accel_x, accel_z) + self._imu_tilt

        # Complementary filter to estimate the pitch angle
        self._pitch = 0.99 * (self._pitch + pitch_rate * self.model.opt.timestep) + 0.01 * accel_pitch

        # Wheel velocities like the firmware: quantized encoder angle, difference over one control step
        wheel_pos = self._measured_wheel_pos()
        left_vel, right_vel = (wheel_pos - self._prev_wheel_pos) / self.model.opt.timestep
        self._prev_wheel_pos = wheel_pos


        obs = np.array([self._pitch, pitch_rate, yaw_rate, left_vel, right_vel, self._ref[0], self._ref[1],
                        *self._prev_action], dtype=np.float32)
        obs_norm = obs / self.obs_scale

        return obs_norm

    def _read_sensor(self, name) -> np.ndarray:
        """Sensor value plus the white noise declared in balancebot.xml"""
        value = self.data.sensor(name).data
        return value + self.np_random.normal(0.0, self.model.sensor(name).noise[0], size=value.shape)

    def _measured_wheel_pos(self) -> np.ndarray:
        pos = np.array([self.data.sensor(self.sensor_left_wheel_pos).data[0],
                        self.data.sensor(self.sensor_right_wheel_pos).data[0]])
        return np.round(pos / ENCODER_LSB) * ENCODER_LSB

    def push(self, force, steps, direction=0.0, height=0.04):
        """Pushes the chassis horizontally (like a hand) for the next `steps` control steps.
        force in N, direction in rad relative to the robot's heading (0 = forward),
        height in m above the chassis center of mass (higher = more pitch torque)."""
        heading = self._get_true_angles()[0] + direction
        f = force * np.array([np.cos(heading), np.sin(heading), 0.0])
        self._push_wrench[:3] = f
        self._push_wrench[3:] = np.cross([0.0, 0.0, height], f)  # force applied above the center of mass
        self._push_height = height
        self._push_steps_left = steps

    def _random_push(self):
        """Starts a random push every push_interval steps (if push_force_max > 0)."""
        if self.push_force_max <= 0:
            return
        self._next_push -= 1
        if self._next_push > 0:
            return
        self._next_push = int(self.np_random.integers(*self.push_interval))
        direction = np.pi * self.np_random.integers(2) + self.np_random.uniform(-0.5, 0.5)  # mostly front / back
        self.push(self.np_random.uniform(0.0, self.push_force_max),
                  int(self.np_random.integers(*self.push_duration)),
                  direction, self.np_random.uniform(0.0, self.push_height_max))

        
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        if self.domain_randomization == True:
            self._randomize_model()
        else:
            self._restore_default_model()

        mujoco.mj_setConst(self.model, self.data)

        start_pitch = self.np_random.uniform(-0.17, 0.17)      # ~±10°
        self._pitch = start_pitch + self._imu_tilt  # settled filter, like the firmware (initialised from the accelerometer at rest)
        self.data.qpos[3:7] = [np.cos(start_pitch / 2), 0.0, np.sin(start_pitch / 2), 0.0]

        self.data.qpos[2] = 0.0425 #+ 0.09 * np.cos(start_pitch)

        self.data.qvel[4] = self.np_random.uniform(-2, 2)
 
        mujoco.mj_forward(self.model, self.data)
        self._prev_wheel_pos = self._measured_wheel_pos()
        self._step = 0
        self._prev_action[:] = 0.0
        self._push_steps_left = 0
        self._pushing = False
        if self.push_force_max > 0:
            self._next_push = int(self.np_random.integers(*self.push_interval))
        self._action_queue = deque(np.zeros(2, dtype=np.float32) for _ in range(self.action_delay))

        self._sample_command()
        # Robot at rest: with a ramp the reference starts at 0, without one it is the command right away
        self._ref[:] = np.where(np.isfinite(self.cmd_acc_max), 0.0, self._cmd)

        return self._get_obs(), {}


    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        action_change = action - self._prev_action
        self._prev_action = action  # the next observation contains the action just taken

        # The policy's action reaches the motors action_delay steps later
        self._action_queue.append(action)
        applied_action = self._action_queue.popleft()

        torque = applied_action * self.max_torque * self.torque_scale
        self.data.ctrl[:] = torque

        self._random_push()
        # Only touch the chassis force while a push lasts and once to end it: the viewer's mouse pushes use it too
        if self._push_steps_left > 0:
            self.data.xfrc_applied[self._chassis_body] = self._push_wrench
            self._push_steps_left -= 1
            self._pushing = True
        elif self._pushing:
            self.data.xfrc_applied[self._chassis_body] = 0.0
            self._pushing = False

        mujoco.mj_step(self.model, self.data)
        self._step += 1

        if self._step % self.cmd_resample_steps == 0:
            self._sample_command()
        self._ramp_reference()

        obs = self._get_obs()

        true_yaw, true_pitch, _ = self._get_true_angles()
        world_vx, world_vy = self.data.qvel[0:2]
        true_yawrate = self.data.qvel[5]

        robot_vx = world_vx * np.cos(true_yaw) + world_vy * np.sin(true_yaw)
        self._actual[:] = robot_vx, true_yawrate

        v_err = robot_vx - self._ref[0]
        yawrate_err = true_yawrate - self._ref[1]

        r_v = self._tracking_reward(v_err, self.v_reward_coeff, self.v_reward_sigmas)
        r_yawrate = self._tracking_reward(yawrate_err, self.yawrate_reward_coeff, self.yawrate_reward_sigmas)
        pitch_penalty = self.pitch_penalty_coeff * true_pitch**2
        action_penalty = self.action_penalty_coeff * np.sum(action**2)
        action_rate_penalty = self.action_rate_penalty_coeff * np.sum(action_change**2)   # prefer smooth torques
        pitchrate_penalty = self.pitchrate_penalty_coeff * self.data.qvel[4]**2             # damping: no rocking

        # reward
        reward = (
            self.alive_bonus
            + r_v
            + r_yawrate
            - pitch_penalty
            - action_penalty
            - action_rate_penalty
            - pitchrate_penalty
        )

        # print(
        #     f"r_v: {r_v:.3f} | r_yawrate: {r_yawrate:.3f} | "
        #     f"pitch_pen: {pitch_penalty:.3f} | action_pen: {action_penalty:.3f} | "
        #     f"total: {reward:.3f}"
        # )
 
        fell = abs(true_pitch) > self.fall_angle
        terminated = bool(fell)
        truncated = self._step >= self.max_steps
 
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        with self.viewer.lock():
            self._draw_velocity_arrows()
            self._draw_push_arrow()
        self.viewer.sync()

    def _draw_velocity_arrows(self):
        """Green = command, red = actual.
        Horizontal arrows: forward velocity. Vertical arrows: yaw rate (up = counter-clockwise)."""
        self._actual_vis = 0.95 * self._actual_vis + 0.05 * self._actual

        yaw = self._get_true_angles()[0]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        base = self.data.xpos[self._chassis_body] + 0.25 * up  # float above the robot
        yaw_base = base - 0.08 * forward                       # yaw arrows sit behind

        v_scale = 1.0     # metres of arrow per m/s
        yaw_scale = 0.1   # metres of arrow per rad/s

        green, red = [0.1, 0.9, 0.1, 1.0], [0.9, 0.1, 0.1, 1.0]

        self.viewer.user_scn.ngeom = 0
        self._add_arrow(base,                   self._cmd[0] * v_scale * forward,        green)
        self._add_arrow(base + 0.02 * up,       self._actual_vis[0] * v_scale * forward, red)
        self._add_arrow(yaw_base,               self._cmd[1] * yaw_scale * up,           green)
        self._add_arrow(yaw_base + 0.02 * left, self._actual_vis[1] * yaw_scale * up,    red)

    def _draw_push_arrow(self):
        """Yellow arrow into the point where a push (env.push) acts, length 5 cm per N."""
        if not self._pushing:
            return
        point = self.data.xipos[self._chassis_body] + np.array([0.0, 0.0, self._push_height])
        vec = 0.05 * self._push_wrench[:3]
        self._add_arrow(point - vec, vec, [1.0, 0.8, 0.0, 1.0])

    def _add_arrow(self, start, vec, rgba):
        """Add an arrow from start to start + vec to the viewer's custom scene."""
        scn = self.viewer.user_scn
        if np.linalg.norm(vec) < 1e-3 or scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.01, start, start + vec)
        scn.ngeom += 1
 
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
 
 
if __name__ == "__main__":
    # Quick sanity check: random actions, print reward/termination.
    # This should run without errors and the episode should end quickly
    # (random torques don't balance anything) -- that's expected.
    env = BalanceRobotEnv(domain_randomization=True)
    obs, info = env.reset(seed=0)
    print("initial obs:", obs)
 
    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"episode ended at step {step}, total_reward={total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
 
    print("sanity check complete.")


