"""Evaluate policies in the sim: standing still, command tracking and push recovery.

    python evaluate.py runs/<run>                                   # one policy
    python evaluate.py runs/2026-10-18_s3_dr runs/<s4 run>          # compare, same seeds for all
    python evaluate.py runs/<run>/checkpoints/model_X_steps.zip --seeds 40 --forces 1 2 3 4
    python evaluate.py runs/<run> --no-dr                           # nominal model only
    python evaluate.py <mjlab>/logs/rsl_rl/balancebot/<run>   # works for SB3, my PPO and mjlab

    python evaluate.py runs/<run> --render                          # watch it get pushed, real time
    python evaluate.py runs/<run> --render --forces 2 --speed 0.25  # one force, slow motion
    python evaluate.py runs/<run> --watch                           # just watch it, env as in training

Domain randomization is on by default (as in training), random pushes from the config are off:
every robot sees exactly the same tests. Lower is better everywhere, except "survived".

--render: pushes the robot again and again (each force forward, then backward, 0.1 s, yellow arrow)
and prints how it recovered. Push it yourself: double-click the robot, then Ctrl + right-drag.
"""
import argparse
import os
import time

import numpy as np

from env import BalanceRobotEnv
from export_weights import env_actor, load_policy

DT = 0.005
SETTLE_S = 2.0  # after reset (random start tilt), before measuring

# Tracking test: (forward velocity m/s, yaw rate rad/s, seconds), within the training command ranges
COMMANDS = [(0.15, 0.0, 2.0), (-0.15, 0.0, 2.0), (0.0, 1.5, 2.0), (0.0, -1.5, 2.0), (0.1, 1.0, 2.0), (0.0, 0.0, 2.0)]


class Policy:
    def __init__(self, run, dr, render=False, speed=1.0, as_trained=False):
        self.path, policy = load_policy(run)
        # runs/<run>/model.zip -> <run>, checkpoints -> <run>/<checkpoint>
        rel = os.path.relpath(os.path.abspath(self.path), os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
        if rel.startswith(".."):  # mjlab
            rel = os.path.join(os.path.basename(os.path.dirname(self.path)), os.path.basename(self.path))
        self.name = rel.removesuffix("/model.zip").removesuffix("/model.pt").removesuffix(".zip").removesuffix(".pt").replace("/checkpoints/", "/")
        kwargs = dict(policy["env"], max_steps=10**6)
        if not as_trained:
            kwargs["push_force_max"] = 0.0
        if dr is not None:
            kwargs["domain_randomization"] = dr
        self.render = render
        self.speed = speed
        self._next_frame = 0.0
        self.env = BalanceRobotEnv(**kwargs, render_mode="human" if render else None)
        self.actor = env_actor(policy, self.env)

    def closed(self):
        """The viewer window was closed."""
        return self.render and self.env.viewer is not None and not self.env.viewer.is_running()

    def reset(self, seed):
        env = self.env
        obs, _ = env.reset(seed=seed)
        env.cmd_resample_steps = 10**9
        env.set_command(0.0, 0.0, ramp=False)
        obs[5:7] = 0.0  # the observation contains the random command from reset
        return obs

    def command(self, obs, v, yaw_rate):
        """New command, ramped like in training (if the run uses a ramp)."""
        self.env.set_command(v, yaw_rate)
        return obs

    def step(self, obs):
        action = self.actor(obs)
        obs, _, terminated, _, _ = self.env.step(action)
        if self.render:  # real time (or slower / faster with speed)
            self.env.render()
            self._next_frame = max(self._next_frame + DT / self.speed, time.time() - 0.1)
            time.sleep(max(self._next_frame - time.time(), 0.0))
        return obs, action, terminated

    def run(self, obs, seconds):
        """Steps for `seconds`; returns obs, actions, true pitch (deg), [forward velocity, yaw rate], fell."""
        actions, pitch, actual = [], [], []
        for _ in range(int(seconds / DT)):
            obs, action, terminated = self.step(obs)
            actions.append(action)
            pitch.append(np.rad2deg(self.env._get_true_angles()[1]))
            actual.append(self.env._actual.copy())
            if terminated or self.closed():
                break
        return obs, np.array(actions), np.array(pitch), np.array(actual), terminated


def settle_time(pitch):
    """Seconds until the tilt stays within 2 deg."""
    outside = np.flatnonzero(np.abs(pitch) > 2.0)
    return (outside[-1] + 1) * DT if len(outside) else 0.0


def standing(policy, seeds):
    fell, pitch_std, jitter, effort, drift = 0, [], [], [], []
    for seed in seeds:
        obs = policy.reset(seed)
        obs, _, _, _, terminated = policy.run(obs, SETTLE_S)
        if terminated:
            fell += 1
            continue
        obs, actions, pitch, actual, terminated = policy.run(obs, 10.0)
        fell += terminated
        pitch_std.append(pitch.std())
        jitter.append(np.diff(actions, axis=0).std())
        effort.append(np.abs(actions).mean())
        drift.append(np.abs(actual[:, 0]).mean())
    return {"survived": f"{len(seeds) - fell}/{len(seeds)}",
            "pitch std [deg]": np.mean(pitch_std),
            "jitter [action/step]": np.mean(jitter),
            "effort [|action|]": np.mean(effort),
            "drift [m/s]": np.mean(drift)}


def tracking(policy, seeds):
    fell, v_err, yaw_err = 0, [], []
    for seed in seeds:
        obs = policy.reset(seed)
        obs, _, _, _, terminated = policy.run(obs, SETTLE_S)
        for v, yaw_rate, seconds in COMMANDS:
            if terminated:
                break
            obs = policy.command(obs, v, yaw_rate)
            obs, _, _, actual, terminated = policy.run(obs, seconds)
            settled = actual[int(0.5 / DT):]  # the first 0.5 s is the step response
            if len(settled):
                v_err.append(np.sqrt(np.mean((settled[:, 0] - v) ** 2)))
                yaw_err.append(np.sqrt(np.mean((settled[:, 1] - yaw_rate) ** 2)))
        fell += terminated
    return {"survived": f"{len(seeds) - fell}/{len(seeds)}",
            "velocity RMS error [m/s]": np.mean(v_err),
            "yaw rate RMS error [rad/s]": np.mean(yaw_err)}


def pushes(policy, seeds, force):
    """Pushed forward / backward (alternating) at the top of the chassis for 0.1 s, then 3 s to recover."""
    fell, max_tilt, saturated, settle = 0, [], [], []
    for seed in seeds:
        obs = policy.reset(seed)
        obs, _, _, _, terminated = policy.run(obs, SETTLE_S)
        if terminated:
            fell += 1
            continue
        policy.env.push(force, int(0.1 / DT), direction=np.pi * (seed % 2), height=0.06)
        obs, actions, pitch, _, terminated = policy.run(obs, 3.0)
        fell += terminated
        max_tilt.append(np.abs(pitch).max())
        saturated.append(np.mean(np.abs(actions) > 0.99) * 100)
        if not terminated:
            settle.append(settle_time(pitch))
    return {"survived": f"{len(seeds) - fell}/{len(seeds)}",
            "max tilt [deg]": np.median(max_tilt),
            "full torque [%]": np.mean(saturated),
            "settle time [s]": np.median(settle) if settle else np.nan}


def watch(policy, forces):
    """Renders one policy and pushes it again and again: each force forward, then backward."""
    print(f"Watching {policy.name}: pushes of {', '.join(f'{f:g}' for f in forces)} N for 0.1 s at the top "
          "(yellow arrow). Close the viewer to stop.")
    print("Push it yourself: double-click the robot, then Ctrl + right-drag.")
    seed, k = 0, 0
    obs = policy.reset(seed)
    while not policy.closed():
        obs, _, _, _, terminated = policy.run(obs, SETTLE_S)
        if not terminated and not policy.closed():
            force, forward = forces[(k // 2) % len(forces)], k % 2 == 0
            k += 1
            policy.env.push(force, int(0.1 / DT), direction=0.0 if forward else np.pi, height=0.06)
            obs, actions, pitch, _, terminated = policy.run(obs, 3.0)
            if policy.closed():
                break
            result = "FELL" if terminated else f"settled after {settle_time(pitch):.2f} s"
            print(f"  push {force:g} N {'forward ' if forward else 'backward'}: max tilt {np.abs(pitch).max():5.1f}°, "
                  f"full torque {np.mean(np.abs(actions) > 0.99) * 100:3.0f}% of the time, {result}")
        if terminated:
            seed += 1
            obs = policy.reset(seed)
            print(f"  new robot (seed {seed})")
    policy.env.close()


def watch_as_trained(policy):
    print(f"Watching {policy.name}. Close the viewer to stop.")
    seed = 0
    obs, _ = policy.env.reset(seed=seed)
    while not policy.closed():
        obs, _, terminated = policy.step(obs)
        if terminated:
            seed += 1
            obs, _ = policy.env.reset(seed=seed)
            print(f"  fell, new robot (seed {seed})")
    policy.env.close()


def print_table(title, names, rows):
    keys = list(rows[0].keys())
    width = max(len(n) for n in names) + 2
    cols = [max(len(k), 9) + 2 for k in keys]
    print(f"\n== {title}")
    print(" " * width + "".join(f"{k:>{c}s}" for k, c in zip(keys, cols)))
    for name, row in zip(names, rows):
        cells = [f"{v:>{c}s}" if isinstance(v, str) else f"{v:>{c}.3g}" for v, c in zip(row.values(), cols)]
        print(f"{name:<{width}s}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folder or model file (SB3, my PPO, mjlab)")
    ap.add_argument("--seeds", type=int, default=20, help="robots per test")
    ap.add_argument("--forces", type=float, nargs="+", default=[1.0, 2.0, 3.0], help="push forces in N")
    ap.add_argument("--dr", action=argparse.BooleanOptionalAction, default=None,
                    help="domain randomization (default on, --watch: as in training); --no-dr = nominal model")
    ap.add_argument("--render", action="store_true", help="watch one policy get pushed instead of the tests")
    ap.add_argument("--speed", type=float, default=1.0, help="--render playback speed (0.25 = slow motion)")
    ap.add_argument("--watch", action="store_true", help="watch one policy in its training env, no tests")
    args = ap.parse_args()

    if args.watch:
        if len(args.runs) > 1:
            ap.error("--watch shows one policy")
        watch_as_trained(Policy(args.runs[0], args.dr, render=True, speed=args.speed, as_trained=True))
        return
    if args.dr is None:
        args.dr = True

    if args.render:
        if len(args.runs) > 1:
            ap.error("--render shows one policy")
        watch(Policy(args.runs[0], args.dr, render=True, speed=args.speed), args.forces)
        return

    policies = [Policy(run, args.dr) for run in args.runs]
    names = [p.name for p in policies]
    seeds = list(range(args.seeds))
    print(f"{args.seeds} robots per test, domain randomization {'on' if args.dr else 'off'}")

    print_table("Standing still (10 s)", names, [standing(p, seeds) for p in policies])
    print_table("Command tracking (2 s each: " + ", ".join(f"{v:+.2f} m/s {w:+.1f} rad/s" for v, w, _ in COMMANDS) + ")",
                names, [tracking(p, seeds) for p in policies])
    for force in args.forces:
        print_table(f"Push {force:.1f} N for 0.1 s at the top", names, [pushes(p, seeds, force) for p in policies])


if __name__ == "__main__":
    main()
