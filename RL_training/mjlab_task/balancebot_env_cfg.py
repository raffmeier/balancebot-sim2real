"""BalanceBot velocity tracking environment for mjlab.

Port of RL_training/env.py (gymnasium + stable-baselines3) to mjlab. Same robot, same
control rate (200 Hz, one physics step per policy step), same observations (complementary
filter pitch, gyro rates, quantized encoder wheel velocities, ramped command, last
action), same rewards, terminations and domain randomization.

Observation (9, already divided by OBS_SCALE like the firmware expects):
  pitch, pitch_rate, yaw_rate, wheel_vel_left, wheel_vel_right, v_ref, yawrate_ref,
  prev_action_left, prev_action_right
Action (2): wheel torques in [-1, 1], times MAX_TORQUE Nm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  apply_body_impulse,
  dr,
  generated_commands,
  last_action,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortAction, JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs.mdp.events import randomize_terrain
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  HfDiscreteObstaclesTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# resolve(): mjlab imports this file through a symlink
_BALANCEBOT_XML: Path = Path(__file__).resolve().parent.parent / "balancebot.xml"

# Robot / sensor constants (same values as RL_training/env.py and scene.xml).
TIMESTEP = 0.005  # s, physics step = control step (200 Hz)
MAX_TORQUE = 0.06  # Nm, action 1.0
FALL_ANGLE = math.radians(30.0)
ENCODER_LSB = 2 * math.pi / 4096  # AS5600 wheel encoder resolution (12 bit), rad
GYRO_NOISE = 0.05  # rad/s, white noise (noise attribute in the XML)
ACCEL_NOISE = 0.4  # m/s^2
GYRO_BIAS_STD = 0.01  # rad/s, residual bias after the boot calibration (DR only)
IMU_TILT_MAX = 0.03  # rad, IMU mounting / PITCH_OFFSET error (DR only)
FILTER_ALPHA = 0.99  # complementary filter, same as the firmware

# Observation normalisation: obs_norm = obs / OBS_SCALE (export_weights.py uses this).
OBS_SCALE = (0.2, 3.0, 3.0, 15.0, 15.0, 0.5, 3.0, 1.0, 1.0)

ROBOT = "robot"
_WHEELS = ("left_wheel_joint", "right_wheel_joint")
_WHEEL_GEOMS = ("wheel_left_collision", "wheel_right_collision")
_ROBOT_CFG = SceneEntityCfg(ROBOT)
_WHEEL_JOINT_CFG = SceneEntityCfg(ROBOT, joint_names=_WHEELS, preserve_order=True)


##
# Entity.
##


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_BALANCEBOT_XML))


def _get_robot_cfg(domain_randomization: bool, max_action_delay: int) -> EntityCfg:
  # Actuation delay (sensor read + inference + motor current loop). One lag per
  # episode for both motors, 0..max_action_delay physics steps (= control steps).
  delay = max_action_delay if domain_randomization else 0
  actuator = XmlActuatorCfg(
    target_names_expr=_WHEELS,
    delay_min_lag=0,
    delay_max_lag=delay,
    delay_hold_prob=1.0,  # sampled on reset, then held for the whole episode
  )
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=EntityArticulationInfoCfg(actuators=(actuator,)),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0425),  # wheel axle height: wheels touch the ground
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )


##
# Helpers.
##


def _yaw_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Yaw and pitch of a wxyz quaternion (same formulas as env._get_true_angles)."""
  w, x, y, z = quat.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
  return yaw, pitch


def _true_pitch(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return _yaw_pitch(asset.data.root_link_quat_w)[1]


def _forward_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """World velocity projected on the heading (pitch does not change it)."""
  asset: Entity = env.scene[asset_cfg.name]
  yaw, _ = _yaw_pitch(asset.data.root_link_quat_w)
  vel_w = asset.data.root_link_lin_vel_w
  return vel_w[:, 0] * torch.cos(yaw) + vel_w[:, 1] * torch.sin(yaw)


def _tracking(err: torch.Tensor, sigma_wide: float, sigma_narrow: float) -> torch.Tensor:
  """0.5 * (wide + narrow gaussian kernel): wide pulls a far off policy in, narrow
  rewards precision."""
  return 0.5 * (torch.exp(-((err / sigma_wide) ** 2)) + torch.exp(-((err / sigma_narrow) ** 2)))


##
# Command: [forward velocity, yaw rate] with an acceleration limited reference.
##


class BalanceBotCommand(CommandTerm):
  """Samples [v, yaw rate] commands; observation and reward use a reference that
  ramps to the command with limited acceleration (the firmware ramps the same way)."""

  cfg: BalanceBotCommandCfg

  def __init__(self, cfg: BalanceBotCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.cmd = torch.zeros(self.num_envs, 2, device=self.device)
    self.ref = torch.zeros(self.num_envs, 2, device=self.device)
    acc = [cfg.v_acc_max or math.inf, cfg.yawrate_acc_max or math.inf]
    self.acc_max = torch.tensor(acc, device=self.device)
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["error_v"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_yawrate"] = torch.zeros(self.num_envs, device=self.device)
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    """The ramped reference, shape (num_envs, 2)."""
    return self.ref

  def _update_metrics(self) -> None:
    steps = self._env.max_episode_length
    v = _forward_velocity(self._env, SceneEntityCfg(self.cfg.entity_name))
    yawrate = self.robot.data.root_link_ang_vel_b[:, 2]
    self.metrics["error_v"] += torch.abs(v - self.ref[:, 0]) / steps
    self.metrics["error_yawrate"] += torch.abs(yawrate - self.ref[:, 1]) / steps

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = torch.empty(n, device=self.device)
    self.cmd[env_ids, 0] = r.uniform_(-self.cfg.v_max, self.cfg.v_max)
    self.cmd[env_ids, 1] = r.uniform_(-self.cfg.yawrate_max, self.cfg.yawrate_max)
    zero = torch.rand(n, device=self.device) < self.cfg.zero_prob
    self.cmd[env_ids] *= (~zero).float().unsqueeze(-1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)  # resamples cmd
    # Robot at rest: with a ramp the reference starts at 0, without one it is the
    # command right away.
    self.ref[env_ids] = torch.where(
      torch.isfinite(self.acc_max), torch.zeros_like(self.cmd[env_ids]), self.cmd[env_ids]
    )
    self._fresh[env_ids] = True
    return extras

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is not None:  # reset path (dt = 0): no ramp step
      self._fresh[env_ids] = False
      return
    max_step = self.acc_max * self._env.step_dt
    new_ref = self.ref + torch.clamp(self.cmd - self.ref, -max_step, max_step)
    # Freshly auto-reset envs keep ref from reset() for this step (like env.reset()).
    self.ref = torch.where(self._fresh.unsqueeze(-1), self.ref, new_ref)
    self._fresh[:] = False

  def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    super().compute(dt, env_ids)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.cmd[idx, i] = s.value

  # GUI (play / viser viewer): sliders for the command of the selected env.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      sliders = [
        server.gui.add_slider(
          label, min=-m, max=m, step=0.01 if label == "v" else 0.05, initial_value=0.0
        )
        for label, m in (("v", self.cfg.v_max), ("yaw_rate", self.cfg.yawrate_max))
      ]
      zero_btn = server.gui.add_button("Zero")

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Green = command, red = actual. Horizontal: forward velocity (1 m per m/s),
    vertical behind the robot: yaw rate (0.1 m per rad/s, up = counter clockwise)."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    data = self.robot.data
    yaw = _yaw_pitch(data.root_link_quat_w)[0].cpu().numpy()
    pos = data.root_link_pos_w.cpu().numpy()
    v = _forward_velocity(self._env, SceneEntityCfg(self.cfg.entity_name)).cpu().numpy()
    yawrate = data.root_link_ang_vel_b[:, 2].cpu().numpy()
    cmd = self.cmd.cpu().numpy()
    green, red = (0.1, 0.9, 0.1, 0.9), (0.9, 0.1, 0.1, 0.9)
    up = np.array([0.0, 0.0, 1.0])
    for i in env_indices:
      fwd = np.array([math.cos(yaw[i]), math.sin(yaw[i]), 0.0])
      base = pos[i] + 0.25 * up
      yaw_base = base - 0.08 * fwd
      for start, vec, color in (
        (base, cmd[i, 0] * fwd, green),
        (base + 0.02 * up, v[i] * fwd, red),
        (yaw_base, 0.1 * cmd[i, 1] * up, green),
        (yaw_base + 0.02 * np.cross(up, fwd), 0.1 * yawrate[i] * up, red),
      ):
        if np.linalg.norm(vec) > 1e-3:
          visualizer.add_arrow(start, start + vec, color=color, width=0.01)


@dataclass(kw_only=True)
class BalanceBotCommandCfg(CommandTermCfg):
  entity_name: str = ROBOT
  v_max: float = 0.3  # m/s
  yawrate_max: float = 2.0  # rad/s
  zero_prob: float = 0.2  # 1.0 = always zero command (balance only)
  v_acc_max: float | None = 0.5  # m/s^2, None = the reference jumps to the command
  yawrate_acc_max: float | None = 5.0  # rad/s^2

  def build(self, env: ManagerBasedRlEnv) -> BalanceBotCommand:
    return BalanceBotCommand(self, env)


##
# Action: wheel torques, with a per-episode torque scale (DR).
##


class WheelTorqueAction(JointEffortAction):
  cfg: WheelTorqueActionCfg

  def __init__(self, cfg: WheelTorqueActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self.torque_scale = torch.ones(self.num_envs, 1, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    self._processed_actions = self._processed_actions * self.torque_scale

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if self.cfg.torque_scale_range is not None:
      n = self.torque_scale[env_ids].shape[0]
      self.torque_scale[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
        *self.cfg.torque_scale_range
      )


@dataclass(kw_only=True)
class WheelTorqueActionCfg(JointEffortActionCfg):
  torque_scale_range: tuple[float, float] | None = None

  def build(self, env: ManagerBasedRlEnv) -> WheelTorqueAction:
    return WheelTorqueAction(self, env)


##
# Observations.
##


class imu_encoder_obs(ManagerTermBase):
  """What the firmware measures: [pitch, pitch_rate, yaw_rate, wheel_vel_l, wheel_vel_r].

  pitch: complementary filter of the noisy gyro + accelerometer (with gyro bias and
  IMU tilt error under DR), started settled at the true pitch on reset.
  wheel velocities: quantized encoder angle differenced over one control step.

  Stateful: updated once per env step, no matter how many groups read it.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    n, dev = self.num_envs, self.device
    self.randomize_imu: bool = cfg.params.get("randomize_imu", False)
    self.pitch = torch.zeros(n, device=dev)
    self.prev_wheel_pos = torch.zeros(n, 2, device=dev)
    self.gyro_bias = torch.zeros(n, 3, device=dev)
    self.imu_tilt = torch.zeros(n, device=dev)
    self.needs_init = torch.ones(n, dtype=torch.bool, device=dev)
    self.last_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    self.out = torch.zeros(n, 5, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.needs_init[env_ids] = True
    if self.randomize_imu:
      k = self.imu_tilt[env_ids].shape[0]
      self.gyro_bias[env_ids] = torch.randn(k, 3, device=self.device) * GYRO_BIAS_STD
      self.imu_tilt[env_ids] = torch.empty(k, device=self.device).uniform_(
        -IMU_TILT_MAX, IMU_TILT_MAX
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
    gyro_sensor: str,
    accel_sensor: str,
    randomize_imu: bool = False,
  ) -> torch.Tensor:
    del randomize_imu  # read in __init__
    update = self.needs_init | (self.last_step != env.common_step_counter)
    if not update.any():
      return self.out
    dt = env.step_dt

    gyro = env.scene[gyro_sensor].data
    gyro = gyro + torch.randn_like(gyro) * GYRO_NOISE + self.gyro_bias
    accel = env.scene[accel_sensor].data
    accel = accel + torch.randn_like(accel) * ACCEL_NOISE
    pitch_rate, yaw_rate = gyro[:, 1], gyro[:, 2]
    accel_pitch = -torch.atan2(accel[:, 0], accel[:, 2]) + self.imu_tilt

    # Settled filter on reset, like the firmware (initialised from the accel at rest).
    prev_pitch = torch.where(
      self.needs_init, _true_pitch(env, asset_cfg) + self.imu_tilt, self.pitch
    )
    pitch = FILTER_ALPHA * (prev_pitch + pitch_rate * dt) + (1 - FILTER_ALPHA) * accel_pitch

    asset: Entity = env.scene[wheel_cfg.name]
    wheel_pos = asset.data.joint_pos[:, wheel_cfg.joint_ids]
    wheel_pos = torch.round(wheel_pos / ENCODER_LSB) * ENCODER_LSB
    prev_wheel = torch.where(self.needs_init.unsqueeze(-1), wheel_pos, self.prev_wheel_pos)
    wheel_vel = (wheel_pos - prev_wheel) / dt

    out = torch.cat([pitch[:, None], pitch_rate[:, None], yaw_rate[:, None], wheel_vel], -1)
    self.pitch = torch.where(update, pitch, self.pitch)
    self.prev_wheel_pos = torch.where(update.unsqueeze(-1), wheel_pos, self.prev_wheel_pos)
    self.out = torch.where(update.unsqueeze(-1), out, self.out)
    self.last_step[update] = env.common_step_counter
    self.needs_init[update] = False
    return self.out


##
# Rewards.
##


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 every step, also on the step the robot falls (like env.py)."""
  return torch.ones(env.num_envs, device=env.device)


def track_forward_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_wide: float,
  sigma_narrow: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  ref = env.command_manager.get_command(command_name)
  assert ref is not None
  return _tracking(_forward_velocity(env, asset_cfg) - ref[:, 0], sigma_wide, sigma_narrow)


def track_yaw_rate(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma_wide: float,
  sigma_narrow: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  ref = env.command_manager.get_command(command_name)
  assert ref is not None
  asset: Entity = env.scene[asset_cfg.name]
  err = asset.data.root_link_ang_vel_b[:, 2] - ref[:, 1]
  return _tracking(err, sigma_wide, sigma_narrow)


def pitch_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
  return _true_pitch(env, asset_cfg) ** 2


def pitch_rate_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Damping: no rocking."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b[:, 1] ** 2


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(env.action_manager.action**2, dim=1)


##
# Terminations.
##


def fell_over(
  env: ManagerBasedRlEnv, limit: float, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  return torch.abs(_true_pitch(env, asset_cfg)) > limit


##
# Environment config.
##


##
# Terrain: gentle slopes and small bumps.
##


def rough_terrain_cfg(
  max_slope: float = 0.12,
  max_wave_height: float = 0.02,
  max_bump_height: float = 0.004,
  patch_size: float = 6.0,
  num_rows: int = 8,
  num_cols: int = 8,
) -> TerrainGeneratorCfg:
  """Grid of num_rows x num_cols patches, each one a random type with a random
  difficulty in [0, 1] (no curriculum: easy and hard patches are always mixed).

  Limits for this robot (0.57 kg, wheel radius 4.25 cm, 2 x 0.06 Nm):
    - holding still on a slope takes r * m * g * sin(slope) = 0.24 Nm * sin(slope),
      so 0.12 rise/run (~7°) uses ~25% of the torque and needs ~8° of lean.
    - rolling over a sharp step of height h takes m * g * sqrt(2 r h - h²), so
      ~0.11 Nm at 5 mm: that is about everything the motors have when standing
      still. 4 mm max, with momentum the policy can still make it.
  The robot spawns in the center of a patch and rarely drives the 3 m to its edge
  within a 10 s episode.
  """
  return TerrainGeneratorCfg(
    size=(patch_size, patch_size),
    num_rows=num_rows,
    num_cols=num_cols,
    border_width=3.0,  # flat ground around the grid, nobody drives off the world
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=0.2),
      # Hill with a 1 m flat top (the spawn point): the robot drives downhill.
      "slope_down": HfPyramidSlopedTerrainCfg(
        proportion=0.2,
        slope_range=(0.0, max_slope),
        platform_width=1.0,
        border_width=0.25,
        vertical_scale=0.001,  # 1 mm height steps, 5 mm would make stairs on gentle slopes
      ),
      # Pit with a 1 m flat bottom: the robot drives uphill.
      "slope_up": HfPyramidSlopedTerrainCfg(
        proportion=0.2,
        slope_range=(0.0, max_slope),
        platform_width=1.0,
        border_width=0.25,
        vertical_scale=0.001,
        inverted=True,
      ),
      # Smooth uneven floor: random heights every 0.5 m, spline interpolated.
      "waves": HfRandomUniformTerrainCfg(
        proportion=0.2,
        noise_range=(0.0, max_wave_height),
        noise_step=0.002,
        downsampled_scale=0.5,
        horizontal_scale=0.05,
        vertical_scale=0.001,
        border_width=0.25,
        scale_with_difficulty=True,
      ),
      # Sharp edges like cables, door sills, tile edges and small potholes: bumps and
      # pits of ±h/2 and ±h, 2 to 30 cm wide. 2 cm grid, so an edge is a ramp of one
      # 2 cm cell.
      "bumps": HfDiscreteObstaclesTerrainCfg(
        proportion=0.2,
        obstacle_width_range=(0.02, 0.3),
        obstacle_height_range=(0.001, max_bump_height),
        num_obstacles=150,
        platform_width=0.5,
        horizontal_scale=0.02,
        vertical_scale=0.001,
        border_width=0.25,
      ),
    },
  )


def balancebot_env_cfg(
  play: bool = False,
  domain_randomization: bool = False,
  rough_terrain: bool = False,
  episode_steps: int = 2000,
  # Rewards (same names / defaults as BalanceRobotEnv and train.py CONFIG).
  alive_bonus_coeff: float = 0.2,
  v_reward_coeff: float = 1.0,
  v_reward_sigma_wide: float = 0.2,
  v_reward_sigma_narrow: float = 0.05,
  yawrate_reward_coeff: float = 0.5,
  yawrate_reward_sigma_wide: float = 1.5,
  yawrate_reward_sigma_narrow: float = 0.3,
  pitch_penalty_coeff: float = 0.3,
  action_penalty_coeff: float = 0.05,
  action_rate_penalty_coeff: float = 0.3,
  pitchrate_penalty_coeff: float = 0.02,
  # Commands.
  cmd_resample_steps: int = 400,
  v_cmd_max: float = 1.0,
  yawrate_cmd_max: float = 3.0,
  zero_cmd_prob: float = 0.2,
  v_cmd_acc_max: float | None = 1.0,
  yawrate_cmd_acc_max: float | None = 10.0,
  # Disturbances / DR.
  max_action_delay: int = 3,
  push_force_max: float = 3,
  push_interval: tuple[int, int] = (200, 600),
  push_duration: tuple[int, int] = (10, 30),
  push_height: float = 0.03,
) -> ManagerBasedRlEnvCfg:
  dt = TIMESTEP

  ##
  # Observations: same for actor and critic, like the SB3 policy.
  ##
  actor_terms = {
    "imu_encoders": ObservationTermCfg(
      func=imu_encoder_obs,
      params={
        "asset_cfg": _ROBOT_CFG,
        "wheel_cfg": _WHEEL_JOINT_CFG,
        "gyro_sensor": f"{ROBOT}/imu_gyro",
        "accel_sensor": f"{ROBOT}/imu_accel",
        "randomize_imu": domain_randomization,
      },
      scale=tuple(1.0 / s for s in OBS_SCALE[0:5]),
    ),
    "command": ObservationTermCfg(
      func=generated_commands,
      params={"command_name": "velocity"},
      scale=tuple(1.0 / s for s in OBS_SCALE[5:7]),
    ),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms),
    "critic": ObservationGroupCfg(dict(actor_terms)),
  }

  ##
  # Actions.
  ##
  actions: dict[str, ActionTermCfg] = {
    "wheel_torque": WheelTorqueActionCfg(
      entity_name=ROBOT,
      actuator_names=_WHEELS,
      preserve_order=True,
      scale=MAX_TORQUE,
      torque_scale_range=(0.85, 1.15) if domain_randomization else None,
    ),
  }

  ##
  # Commands.
  ##
  commands: dict[str, CommandTermCfg] = {
    "velocity": BalanceBotCommandCfg(
      resampling_time_range=(cmd_resample_steps * dt, cmd_resample_steps * dt),
      v_max=v_cmd_max,
      yawrate_max=yawrate_cmd_max,
      zero_prob=zero_cmd_prob,
      v_acc_max=v_cmd_acc_max,
      yawrate_acc_max=yawrate_cmd_acc_max,
      debug_vis=True,
    ),
  }

  ##
  # Events.
  ##
  events: dict[str, EventTermCfg] = {}
  if rough_terrain:
    # Each episode starts in the center of a new random patch (before reset_base,
    # which places the robot relative to the env origin).
    events["randomize_terrain"] = EventTermCfg(func=randomize_terrain, mode="reset")
  events |= {
    # ~±10° start pitch about the wheel axle, ±2 rad/s pitch rate, wheels at 0.
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"pitch": (-0.17, 0.17)},
        "velocity_range": {"pitch": (-2.0, 2.0)},
        "asset_cfg": _ROBOT_CFG,
      },
    ),
    "reset_wheels": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": _WHEEL_JOINT_CFG,
      },
    ),
  }

  if domain_randomization:
    # Resampled every episode, like BalanceRobotEnv._randomize_model().
    # pseudo_inertia scales mass and inertia together by exp(2 * alpha).
    mass_alpha = (0.5 * math.log(0.85), 0.5 * math.log(1.15))
    events |= {
      "dr_chassis_mass": EventTermCfg(
        func=dr.pseudo_inertia,
        mode="reset",
        params={
          "alpha_range": mass_alpha,
          "asset_cfg": SceneEntityCfg(ROBOT, body_names=("chassis",)),
        },
      ),
      "dr_wheel_mass": EventTermCfg(
        func=dr.pseudo_inertia,
        mode="reset",
        params={
          "alpha_range": mass_alpha,
          "asset_cfg": SceneEntityCfg(ROBOT, body_names=("wheel_left", "wheel_right")),
        },
      ),
      # After pseudo_inertia: offsets are added to the default COM.
      "dr_chassis_com": EventTermCfg(
        func=dr.body_com_offset,
        mode="reset",
        params={
          "ranges": {0: (-0.003, 0.003), 2: (-0.0075, 0.0075)},
          "operation": "add",
          "asset_cfg": SceneEntityCfg(ROBOT, body_names=("chassis",)),
        },
      ),
      "dr_wheel_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
          "ranges": (0.7, 1.3),
          "operation": "scale",
          "axes": [0, 1, 2],
          "shared_random": True,  # same scale for both wheels
          "asset_cfg": SceneEntityCfg(ROBOT, geom_names=_WHEEL_GEOMS),
        },
      ),
      "dr_chassis_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
          "ranges": (0.7, 1.3),
          "operation": "scale",
          "axes": [0, 1, 2],
          "asset_cfg": SceneEntityCfg(ROBOT, geom_names=("chassis_collision",)),
        },
      ),
      "dr_ground_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
          "ranges": (0.7, 1.3),
          "operation": "scale",
          "axes": [0, 1, 2],
          "asset_cfg": SceneEntityCfg("terrain", geom_names=(".*",)),
        },
      ),
      "dr_wheel_damping": EventTermCfg(
        func=dr.joint_damping,
        mode="reset",
        params={
          "ranges": (0.7, 1.4),
          "operation": "scale",
          "asset_cfg": _WHEEL_JOINT_CFG,
        },
      ),
      "dr_wheel_frictionloss": EventTermCfg(
        func=dr.joint_friction,
        mode="reset",
        params={
          "ranges": (0.5, 2.0),
          "operation": "scale",
          "asset_cfg": _WHEEL_JOINT_CFG,
        },
      ),
    }

  if push_force_max > 0:
    # Random hand pushes on the chassis. Approximation of env.push(): each force
    # component is sampled from ±push_force_max and the height is fixed.
    events["push"] = EventTermCfg(
      func=apply_body_impulse,
      mode="step",
      params={
        "force_range": (-push_force_max, push_force_max),
        "torque_range": (0.0, 0.0),
        "duration_s": (push_duration[0] * dt, push_duration[1] * dt),
        "cooldown_s": (push_interval[0] * dt, push_interval[1] * dt),
        "body_point_offset": (0.0, 0.0, push_height),
        "asset_cfg": SceneEntityCfg(ROBOT, body_names=("chassis",)),
      },
    )

  ##
  # Rewards. Weights are the per-step values of env.py; mjlab multiplies them by dt
  # (scale_rewards_by_dt), which keeps returns ~O(1) for the critic with γ near 1
  # at 200 Hz. A uniform scale, so the optimal policy is unchanged.
  ##
  rewards = {
    "alive": RewardTermCfg(func=alive_bonus, weight=alive_bonus_coeff),
    "track_v": RewardTermCfg(
      func=track_forward_velocity,
      weight=v_reward_coeff,
      params={
        "command_name": "velocity",
        "sigma_wide": v_reward_sigma_wide,
        "sigma_narrow": v_reward_sigma_narrow,
      },
    ),
    "track_yawrate": RewardTermCfg(
      func=track_yaw_rate,
      weight=yawrate_reward_coeff,
      params={
        "command_name": "velocity",
        "sigma_wide": yawrate_reward_sigma_wide,
        "sigma_narrow": yawrate_reward_sigma_narrow,
      },
    ),
    "pitch": RewardTermCfg(func=pitch_l2, weight=-pitch_penalty_coeff),
    "action": RewardTermCfg(func=action_l2, weight=-action_penalty_coeff),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-action_rate_penalty_coeff),
    "pitch_rate": RewardTermCfg(func=pitch_rate_l2, weight=-pitchrate_penalty_coeff),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=fell_over, params={"limit": FALL_ANGLE}),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=(
        TerrainEntityCfg(terrain_type="generator", terrain_generator=rough_terrain_cfg())
        if rough_terrain
        else TerrainEntityCfg(terrain_type="plane")
      ),
      entities={ROBOT: _get_robot_cfg(domain_randomization, max_action_delay)},
      num_envs=1024,  # override with --env.scene.num-envs
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=ROBOT,
      body_name="chassis",
      distance=0.8,
      elevation=-20.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=dt, integrator="implicitfast"),
    ),
    decimation=1,
    episode_length_s=episode_steps * dt,
    scale_rewards_by_dt=True,
  )

  if play:
    cfg.episode_length_s = 1e10
    cfg.scene.num_envs = 16
  return cfg


##
# RL config: tuned for thousands of parallel envs (rsl_rl / mjlab style) instead of
# SB3's 64 envs. Actor stays tanh 128x128 without obs normalization so the firmware
# export (policy_forward in model.h) works unchanged.
#
# Per iteration: 4096 envs x 48 steps = ~197k samples (0.24 s of experience per env
# at 200 Hz), 4 mini-batches x 5 epochs = 20 gradient steps.
##


def balancebot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128),
      activation="tanh",
      obs_normalization=False,  # observations are scaled by OBS_SCALE already
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Actions are clipped to [-1, 1]: with std 1 a large share of the
        # exploration would saturate the motors, 0.5 keeps most samples inside.
        "init_std": 0.5,
        "std_type": "log",
      },
    ),
    # The critic is not deployed: elu and a bit more capacity are free.
    critic=RslRlModelCfg(
      hidden_dims=(256, 256),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      # Large batches allow a higher lr; the adaptive schedule lowers it whenever
      # an update moves the policy by more than desired_kl.
      learning_rate=1e-3,
      schedule="adaptive",
      desired_kl=0.01,
      # 1 s horizon at 200 Hz (0.99 would only be 0.5 s, too short to value
      # leaning into a command change; 0.998 = 2.5 s is noisy to bootstrap from
      # 48-step rollouts).
      gamma=0.995,
      lam=0.95,
      max_grad_norm=1.0,
    ),
    clip_actions=1.0,  # env.py clips the action to [-1, 1]; the firmware does too
    experiment_name="balancebot",
    save_interval=50,
    num_steps_per_env=48,
    max_iterations=1000,  # ~200M samples
  )
