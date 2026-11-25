"""Train a policy with CONFIG["algo"]: "sb3" or "my_ppo" (ppo_agent.py) on env.py, or "mjlab"
(settings in mjlab_task/balancebot_env_cfg.py, runs in <mjlab_dir>/logs/rsl_rl/balancebot/).

sb3 and my_ppo runs get their own folder:

    runs/<date>_<name>/
        config.json     this CONFIG + obs_scale / max_torque of the env
        env.py          copy of the env used for training
        tb/             TensorBoard logs (tensorboard --logdir runs)
        checkpoints/    intermediate models (sb3 only)
        model.zip       final model (model.pt for my_ppo)
        model.h         exported weights for the firmware

Edit CONFIG, then: python train.py
"""
import datetime
import json
import os
import shutil
import subprocess

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import HParam, configure
from stable_baselines3.common.vec_env import DummyVecEnv

from env import BalanceRobotEnv
from export_weights import export
from ppo_agent import Params, PPOAgent

CONFIG = dict(
    name="mjlab_velocity",
    algo="mjlab",            # "sb3" | "my_ppo" | "mjlab"
    notes="",
    total_timesteps=50_000_000,
    seed=0,
    # sb3 only
    parent=None,              # run folder to fine-tune from (e.g. "2026-10-20_s1_balance"), None = new policy
    n_envs=1,
    min_std=0.3,             # fine-tuning: re-open exploration if the action std collapsed (e.g. 0.3)
    env=dict(
        max_steps=2000,
        domain_randomization=True,
        alive_bonus=0.2,
        v_reward_coeff=1.0,
        v_reward_sigma_wide=0.2,       # m/s, tracking reward = coeff * (wide + narrow kernel) / 2
        v_reward_sigma_narrow=0.05,
        yawrate_reward_coeff=0.5,
        yawrate_reward_sigma_wide=1.5,     # rad/s
        yawrate_reward_sigma_narrow=0.3,
        pitch_penalty_coeff=0.3,
        action_penalty_coeff=0.05,
        action_rate_penalty_coeff=0.3,
        max_action_delay=3,
        cmd_resample_steps=400,
        v_cmd_max=2,
        yawrate_cmd_max=5,
        zero_cmd_prob=0.2,    # 1.0 = always zero command (balance only)
        v_cmd_acc_max=2,    # m/s^2, the observed / tracked reference ramps to the command (None = jumps)
        yawrate_cmd_acc_max=5.0,  # rad/s^2
        pitchrate_penalty_coeff=0.02,  # damping: penalizes pitch_rate^2
        push_force_max=0.0,   # N, random pushes like a hand (e.g. 3.0), 0 = off
        push_interval=(200, 600),      # steps between pushes (1-3 s)
        push_duration=(10, 30),        # steps a push lasts (50-150 ms)
        push_height_max=0.06,          # m above the center of mass
    ),
    # When fine-tuning, these override the parent's values (net_arch always comes from the parent)
    sb3=dict(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),
    ),
    # ppo_agent.Params
    my_ppo=dict(
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_clip=0.2,
        lr_schedule="adaptive",   # "fixed" | "adaptive"
        lr=3e-4,
        desired_kl=0.005,
        lr_min=1e-5,
        lr_max=1e-2,
    ),
    # runs `uv run train <task> --env.scene.num-envs <num_envs> <extra_args>` in mjlab_dir
    mjlab=dict(
        mjlab_dir=os.environ.get("MJLAB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mjlab")),  # default: next to this repo
        task="Mjlab-BalanceBot-DR",  # Mjlab-BalanceBot | Mjlab-BalanceBot-DR | Mjlab-BalanceBot-DR-Rough
        num_envs=4096,
        # e.g. ["--agent.resume", "True", "--agent.load-run", "<run>", "--agent.load-checkpoint", "model_200.pt"]
        extra_args=[],
    ),
)


class HParamCallback(BaseCallback):
    """Shows the config in TensorBoard's HPARAMS tab, next to the final reward and episode length."""

    def _on_training_start(self):
        flat = {"total_timesteps": CONFIG["total_timesteps"], "parent": str(CONFIG["parent"])}
        flat.update({f"env/{k}": v if isinstance(v, (int, float)) else str(v) for k, v in CONFIG["env"].items()})
        flat.update({f"ppo/{k}": v if isinstance(v, (int, float)) else str(v) for k, v in CONFIG["sb3"].items()})
        metrics = {"rollout/ep_rew_mean": 0.0, "rollout/ep_len_mean": 0.0}
        self.logger.record("hparams", HParam(flat, metrics), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self):
        return True


def make_run_dir():
    run_dir = os.path.join("runs", f"{datetime.date.today()}_{CONFIG['name']}")
    os.makedirs(run_dir)  # fails if the run exists, so nothing gets overwritten

    # Save what the policy was trained with; export_weights.py reads obs_scale / max_torque from here
    env = BalanceRobotEnv(**CONFIG["env"])
    record = dict(CONFIG, obs_scale=env.obs_scale.tolist(), max_torque=env.max_torque)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2)
    shutil.copy("env.py", run_dir)
    return run_dir


def train_sb3(run_dir):
    vec_env = make_vec_env(BalanceRobotEnv, n_envs=CONFIG["n_envs"], seed=CONFIG["seed"],
                           vec_env_cls=DummyVecEnv, env_kwargs=CONFIG["env"])

    if CONFIG["parent"]:
        ppo = {k: v for k, v in CONFIG["sb3"].items() if k != "policy_kwargs"}
        model = PPO.load(os.path.join("runs", CONFIG["parent"], "model.zip"),
                         env=vec_env, device="cpu", custom_objects=ppo)
    else:
        model = PPO("MlpPolicy", vec_env, device="cpu", seed=CONFIG["seed"], verbose=1, **CONFIG["sb3"])

    if CONFIG["min_std"]:
        with th.no_grad():
            model.policy.log_std.clamp_(min=float(np.log(CONFIG["min_std"])))

    model.set_logger(configure(os.path.join(run_dir, "tb"), ["stdout", "tensorboard"]))
    checkpoints = CheckpointCallback(save_freq=max(500_000 // CONFIG["n_envs"], 1),
                                     save_path=os.path.join(run_dir, "checkpoints"), name_prefix="model")

    # reset_num_timesteps=False: a fine-tuning run continues the parent's step axis in TensorBoard
    model.learn(total_timesteps=CONFIG["total_timesteps"], reset_num_timesteps=not CONFIG["parent"],
                callback=[HParamCallback(), checkpoints], progress_bar=True)

    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)
    return model_path


def train_my_ppo(run_dir):
    th.manual_seed(CONFIG["seed"])
    env = BalanceRobotEnv(**CONFIG["env"])
    env.reset(seed=CONFIG["seed"])

    params = Params(total_steps=CONFIG["total_timesteps"], **CONFIG["my_ppo"])
    agent = PPOAgent(run_dir, env, params)
    agent.learn()
    return agent.save()


def train_mjlab(cfg):
    mjlab_dir = os.path.abspath(os.path.expanduser(cfg["mjlab_dir"]))
    task_link = os.path.join(mjlab_dir, "src", "mjlab", "tasks", "balancebot")
    if not os.path.lexists(task_link):
        os.symlink(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mjlab_task"), task_link)
        print(f"Linked {task_link} -> mjlab_task")
    log_root = os.path.join(mjlab_dir, "logs", "rsl_rl", "balancebot")
    runs_before = set(os.listdir(log_root)) if os.path.isdir(log_root) else set()

    cmd = ["uv", "run", "train", cfg["task"], "--env.scene.num-envs", str(cfg["num_envs"]), *cfg["extra_args"]]
    print(f"Running in {mjlab_dir}: {' '.join(cmd)}")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # use mjlab's own venv
    subprocess.run(cmd, cwd=mjlab_dir, env=env, check=True)

    new_runs = sorted(set(os.listdir(log_root)) - runs_before)
    if not new_runs:
        raise RuntimeError(f"mjlab finished, but there is no new run in {log_root}")
    run_dir = os.path.join(log_root, new_runs[-1])
    export(run_dir)
    print(f"Done: {run_dir}")


def main():
    algo = CONFIG["algo"]
    if algo == "mjlab":
        train_mjlab(CONFIG["mjlab"])
        return
    if algo not in ("sb3", "my_ppo"):
        raise ValueError(f"Unknown algo {algo!r}, use 'sb3', 'my_ppo' or 'mjlab'")
    if algo == "my_ppo" and CONFIG["parent"]:
        raise ValueError("Fine-tuning (parent) is only supported with algo='sb3'")

    run_dir = make_run_dir()
    model_path = train_sb3(run_dir) if algo == "sb3" else train_my_ppo(run_dir)
    export(model_path)
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
