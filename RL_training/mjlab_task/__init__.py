from mjlab.tasks.registry import register_mjlab_task

from .balancebot_env_cfg import balancebot_env_cfg, balancebot_ppo_runner_cfg

# Velocity tracking without domain randomization (like train.py v2_s1/s2).
register_mjlab_task(
  task_id="Mjlab-BalanceBot",
  env_cfg=balancebot_env_cfg(),
  play_env_cfg=balancebot_env_cfg(play=True),
  rl_cfg=balancebot_ppo_runner_cfg(),
)

# Same with domain randomization (masses, COM, friction, damping, torque scale,
# actuation delay, gyro bias, IMU tilt).
register_mjlab_task(
  task_id="Mjlab-BalanceBot-DR",
  env_cfg=balancebot_env_cfg(domain_randomization=True),
  play_env_cfg=balancebot_env_cfg(play=True, domain_randomization=True),
  rl_cfg=balancebot_ppo_runner_cfg(),
)

# DR + gentle slopes (up to ~7°), smooth uneven floor (up to 2 cm) and sharp small
# bumps / potholes (up to 4 mm). Same observations and actions as the tasks above, so
# it can resume from their checkpoints.
register_mjlab_task(
  task_id="Mjlab-BalanceBot-DR-Rough",
  env_cfg=balancebot_env_cfg(domain_randomization=True, rough_terrain=True),
  play_env_cfg=balancebot_env_cfg(play=True, domain_randomization=True, rough_terrain=True),
  rl_cfg=balancebot_ppo_runner_cfg(),
)
