"""Compare a robot log (robot_log.py) with the sim, running the same policy.

    python compare_real_sim.py logs/<file>.npy runs/<run>             # longest balancing segment
    python compare_real_sim.py logs/<file>.npy runs/<run> --segment 2 --seeds 10 --dr
    python compare_real_sim.py logs/<file>.npy runs/<run> --video   # + mp4 of the sim run

1. Loop health: control rate / jitter, dropped packets, compute time, FOC rate, IMU errors.
2. Replay: feeds the logged network inputs to the Python policy. The outputs must match the
   firmware's, otherwise the exported weights, obs order or scaling differ from training.
3. Sim: runs the policy in BalanceRobotEnv with the robot's start pitch and the same commands,
   then plots real vs sim (time series, distributions, spectra) and prints a stats table.
   Big differences there are the sim-to-real gap (noise, delay, friction, mass, ...).
4. --video: renders the first sim rollout to <log>_seg<i>_sim.mp4 (real time, needs ffmpeg).
"""
import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import welch

from env import ENCODER_LSB, BalanceRobotEnv
from export_weights import env_actor, load_policy, policy_forward
from robot_log import FLAG_BALANCING, FLAG_IMU_OK, load

DT = 0.005  # control period, s
WHEEL_RADIUS = 0.0425  # m, balancebot.xml
TRACK_WIDTH = 0.1726   # m, distance between the wheels
REAL_COLOR = "#2a78d6"
SIM_COLOR = "#eb6834"


def real_time(log):
    """Seconds since the first packet (t_us is a wrapping uint32)."""
    dt_us = np.diff(log["t_us"]).astype(np.uint32)  # wraps correctly
    return np.concatenate([[0.0], np.cumsum(dt_us * 1e-6)])


def balancing_segments(log, min_len):
    """[(start, end)] index ranges where the robot was balancing, at least min_len packets long."""
    on = (log["flags"] & FLAG_BALANCING) != 0
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(log)]])
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if on[a] and b - a >= min_len]


def health_report(log):
    dt_ms = np.diff(log["t_us"]).astype(np.uint32) * 1e-3
    lost = int(np.sum(np.diff(log["step"].astype(np.int64)) - 1))
    compute = log["compute_us"]
    foc_hz = log["foc_loops"][1:] / (dt_ms * 1e-3)
    imu_fail = int(np.sum((log["flags"] & FLAG_IMU_OK) == 0))
    print("== Loop health")
    print(f"  packets        {len(log)} ({real_time(log)[-1]:.1f} s), lost {lost}")
    print(f"  control period {dt_ms.mean():.3f} ms mean, {dt_ms.std():.3f} std, {dt_ms.min():.2f}..{dt_ms.max():.2f} ms "
          f"(lost packets count as long periods)")
    print(f"  compute time   {np.median(compute):.0f} us median, {np.percentile(compute, 99):.0f} us p99, {compute.max()} us max")
    print(f"  FOC rate       {np.median(foc_hz):.0f} Hz median, {np.percentile(foc_hz, 1):.0f} Hz p1")
    print(f"  IMU errors     {imu_fail}")
    # Physically impossible jumps between two steps = corrupted sensor reads (e.g. I2C bus conflicts)
    consecutive = (np.diff(log["step"].astype(np.int64)) == 1)[:, None]  # not across lost packets
    angle_jumps = np.sum((np.abs(np.diff(log["wheel_angle"], axis=0)) > 0.5) & consecutive, axis=0)  # > 100 rad/s
    gyro_jumps = np.sum((np.abs(np.diff(log["gyro"], axis=0)) > 3.0) & consecutive, axis=0)          # > 600 rad/s^2
    print(f"  sensor glitches wheel angle L {angle_jumps[0]} R {angle_jumps[1]}, gyro x/y/z {gyro_jumps[0]}/{gyro_jumps[1]}/{gyro_jumps[2]}"
          f"{'  <-- corrupted reads' if angle_jumps.sum() + gyro_jumps.sum() else ''}")

    # Turning seen by the wheels must match the gyro: catches a wheel with the wrong sign or L/R swapped
    wv = log["wheel_vel"]
    yaw_wheels = WHEEL_RADIUS * (wv[:, 1] - wv[:, 0]) / TRACK_WIDTH
    turning = ((log["flags"] & FLAG_BALANCING) != 0) & (np.abs(log["gyro"][:, 2]) > 0.5)
    if turning.sum() >= 100:
        k = np.linalg.lstsq(yaw_wheels[turning, None], log["gyro"][turning, 2], rcond=None)[0][0]
        print(f"  wheels vs gyro yaw rate: gyro = {k:+.2f} x wheels  "
              f"{'OK' if k > 0.5 else '<-- a wheel velocity has the wrong sign, or left/right are swapped'}")
    else:
        print("  wheels vs gyro yaw rate: not enough turning in the log to check")


def replay_check(seg, policy):
    """Python policy on the logged inputs vs the firmware's outputs."""
    pin = seg["policy_in"]
    act_py = policy_forward(policy, pin)
    err = np.abs(act_py - seg["policy_out"])

    # The previous action in the input must be the last output (only across steps without lost packets)
    consecutive = np.diff(seg["step"].astype(np.int64)) == 1
    prev_err = np.abs(pin[1:, 7:9] - seg["policy_out"][:-1])[consecutive]

    print("== Replay (Python policy on the logged inputs)")
    if not np.any(pin):
        print("  policy_in is all zero: is the PolicyController active?")
    ok = err.max() < 1e-4
    print(f"  action error   {err.max():.2e} max, {err.mean():.2e} mean  {'OK' if ok else '<-- MISMATCH: wrong weights / obs order / scaling?'}")
    ok = prev_err.size == 0 or prev_err.max() < 1e-6
    print(f"  prev_action    {prev_err.max() if prev_err.size else 0:.2e} max error  {'OK' if ok else '<-- prev_action in obs is not the last output'}")
    return act_py


class SimVideo:
    """Offscreen render of a sim rollout to an mp4 through ffmpeg, in real time.

    Arrows above the robot, like the env's viewer: horizontal = forward velocity (1 m per m/s),
    vertical behind the robot = yaw rate (0.1 m per rad/s, up = counter clockwise).
    Green = command, orange = sim, blue = the real robot at the same time (from the log).
    Velocities come from the wheel encoders and gyro (observations) for both, smoothed alike.
    """
    FPS = 40  # every 5th control step at 200 Hz
    COMMAND, SIM, REAL = (0.1, 0.8, 0.1, 1.0), (0.92, 0.41, 0.2, 1.0), (0.16, 0.47, 0.84, 1.0)

    def __init__(self, env, path, obs_real, width=1280, height=720):
        model = env.model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.env, self.path, self.obs_real = env, path, obs_real
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = env._chassis_body
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.0, -30.0, 90.0
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(self.FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE)
        self.every = round(1.0 / (self.FPS * DT))
        self.sim_vis = np.zeros(2)
        self.real_vis = np.zeros(2)

    @staticmethod
    def _velocities(obs_raw):
        """[forward velocity, yaw rate] from raw observations (wheel encoders, gyro)."""
        return np.array([WHEEL_RADIUS * (obs_raw[3] + obs_raw[4]) / 2, obs_raw[2]])

    def _arrow(self, scn, start, vec, rgba):
        if np.linalg.norm(vec) < 1e-3 or scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, start + vec)
        scn.ngeom += 1

    def step(self, k, obs_raw):
        """Called after sim step k with the new raw observation."""
        self.sim_vis = 0.95 * self.sim_vis + 0.05 * self._velocities(obs_raw)
        real = self.obs_real[min(k + 1, len(self.obs_real) - 1)]
        self.real_vis = 0.95 * self.real_vis + 0.05 * self._velocities(real)
        if k % self.every:
            return
        env = self.env
        self.renderer.update_scene(env.data, self.cam)
        scn = self.renderer.scene
        yaw = env._get_true_angles()[0]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        base = env.data.xpos[env._chassis_body] + 0.18 * up  # below the camera's eye height
        yaw_base = base - 0.08 * forward
        for i, (v, rgba) in enumerate(((env._cmd, self.COMMAND), (self.sim_vis, self.SIM), (self.real_vis, self.REAL))):
            self._arrow(scn, base + 0.025 * i * up, v[0] * forward, rgba)
            self._arrow(scn, yaw_base + 0.025 * i * left, 0.1 * v[1] * up, rgba)
        self.ffmpeg.stdin.write(self.renderer.render().tobytes())

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        self.renderer.close()


def sim_rollout(env, actor, start_pitch, cmds, seed, video=None):
    """Runs the policy from the robot's start pitch, with the robot's commands at each step.
    Returns raw (unscaled) observations, actions and true pitch, until done or fallen.
    video: SimVideo that records the rollout."""
    env.reset(seed=seed)
    env.cmd_resample_steps = 10**9
    d = env.data
    d.qpos[3:7] = [np.cos(start_pitch / 2), 0.0, np.sin(start_pitch / 2), 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    env._pitch = start_pitch + env._imu_tilt   # settled filter, like reset()
    env._prev_wheel_pos = env._measured_wheel_pos()
    env.set_command(*cmds[0], ramp=False)  # logged = the reference the firmware's policy saw
    obs = env._get_obs()

    obs_raw, actions, true_pitch = [], [], []
    for k in range(len(cmds)):
        action = actor(obs)
        obs_raw.append(obs * env.obs_scale)
        actions.append(action)
        env.set_command(*cmds[min(k + 1, len(cmds) - 1)], ramp=False)
        obs, _, terminated, _, _ = env.step(action)
        true_pitch.append(env._get_true_angles()[1])
        if video is not None:
            video.step(k, obs * env.obs_scale)
        if terminated:
            break
    return dict(obs=np.array(obs_raw), action=np.array(actions), true_pitch=np.array(true_pitch))


def signals(obs, action):
    """The quantities compared between real and sim, in readable units."""
    return {
        "pitch [deg]": np.rad2deg(obs[:, 0]),
        "pitch rate [rad/s]": obs[:, 1],
        "yaw rate [rad/s]": obs[:, 2],
        "wheel vel L [rad/s]": obs[:, 3],
        "wheel vel R [rad/s]": obs[:, 4],
        "action L [-1, 1]": action[:, 0],
        "action R [-1, 1]": action[:, 1],
        "action change L [1/step]": np.diff(action[:, 0], prepend=action[0, 0]),
    }


def peak_freq(x):
    if len(x) < 64:
        return np.nan
    f, p = welch(x - x.mean(), fs=1 / DT, nperseg=min(512, len(x)))
    return f[1:][np.argmax(p[1:])]


def stats(sig):
    out = {}
    for name, x in sig.items():
        out[f"{name} mean"] = x.mean()
        out[f"{name} std"] = x.std()
    out["pitch rate peak freq [Hz]"] = peak_freq(sig["pitch rate [rad/s]"])
    out["action L peak freq [Hz]"] = peak_freq(sig["action L [-1, 1]"])
    return out


def print_stats(real, sims):
    print(f"== Real vs sim ({len(sims)} seed{'s' if len(sims) > 1 else ''}, sim: mean ± std over seeds)")
    print(f"  {'':34s} {'real':>9s}   {'sim':>17s}")
    for key, value in real.items():
        sim = np.array([s[key] for s in sims])
        print(f"  {key:34s} {value:9.3f}   {sim.mean():9.3f} ± {sim.std():5.3f}")


def style(ax):
    ax.grid(True, color="#d9d9d6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a96")
    ax.tick_params(colors="#55554f", labelsize=8)


def plot_time_series(t_real, real, sim_runs, replay_err, title):
    names = ["pitch [deg]", "wheel vel L [rad/s]",
             "pitch rate [rad/s]", "wheel vel R [rad/s]",
             "yaw rate [rad/s]", "action L [-1, 1]",
             None, "action R [-1, 1]"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=11)
    for ax, name in zip(axes.flat, names):
        style(ax)
        if name is None:
            ax.plot(t_real, replay_err, color=REAL_COLOR, linewidth=1.0)
            ax.set_title("replay error |python - firmware| (max of L, R)", fontsize=9, loc="left")
            continue
        for i, s in enumerate(sim_runs):
            t_sim = np.arange(len(s[name])) * DT
            ax.plot(t_sim, s[name], color=SIM_COLOR, linewidth=1.4 if i == 0 else 0.6,
                    alpha=1.0 if i == 0 else 0.35, label="sim" if i == 0 else None)
        ax.plot(t_real, real[name], color=REAL_COLOR, linewidth=1.4, label="real")
        ax.set_title(name, fontsize=9, loc="left")
        ax.legend(loc="best", fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]", fontsize=9)
    fig.tight_layout()
    return fig


def plot_distributions(real, sim_runs, title):
    names = list(real.keys())
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5))
    fig.suptitle(title + " - distributions and spectra", fontsize=11)
    sim_cat = {n: np.concatenate([s[n] for s in sim_runs]) for n in names}
    for ax, name in zip(axes.flat, names):
        style(ax)
        lo, hi = np.percentile(np.concatenate([real[name], sim_cat[name]]), [0.5, 99.5])
        if name.startswith("wheel vel"):  # quantized by the encoder: one bin per step
            step = ENCODER_LSB / DT
            bins = (np.arange(np.floor(lo / step), np.ceil(hi / step) + 2) - 0.5) * step
        else:
            bins = np.linspace(lo, hi, 50) if hi > lo else 50
        ax.hist(sim_cat[name], bins=bins, density=True, color=SIM_COLOR, alpha=0.55, label="sim")
        ax.hist(real[name], bins=bins, density=True, histtype="step", color=REAL_COLOR, linewidth=1.6, label="real")
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, frameon=False)

    for ax, name in zip(axes.flat[len(names):], ["pitch rate [rad/s]", "action L [-1, 1]", "wheel vel L [rad/s]", "yaw rate [rad/s]"]):
        style(ax)
        for x, color, label in [(sim_cat[name], SIM_COLOR, "sim"), (real[name], REAL_COLOR, "real")]:
            if len(x) >= 64:
                f, p = welch(x - x.mean(), fs=1 / DT, nperseg=min(512, len(x)))
                ax.semilogy(f, p, color=color, linewidth=1.4, label=label)
        ax.set_title(f"spectrum: {name}", fontsize=9, loc="left")
        ax.set_xlabel("frequency [Hz]", fontsize=9)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="robot log (.npy from robot_log.py)")
    ap.add_argument("run", help="run folder or model file (SB3, my PPO, mjlab): the policy flashed on the robot")
    ap.add_argument("--segment", type=int, help="balancing segment index (default: longest)")
    ap.add_argument("--seeds", type=int, default=5, help="sim rollouts")
    ap.add_argument("--dr", action=argparse.BooleanOptionalAction, default=None,
                    help="domain randomization in the sim (default: as in training)")
    ap.add_argument("--skip", type=float, default=0.5, help="seconds after the start excluded from the stats")
    ap.add_argument("--no-show", action="store_true", help="only save the plots")
    ap.add_argument("--video", action="store_true", help="render the first sim rollout to <log>_seg<i>_sim.mp4")
    args = ap.parse_args()

    log = load(args.log)
    model_path, policy = load_policy(args.run)
    health_report(log)

    segments = balancing_segments(log, min_len=int(1.0 / DT))
    if not segments:
        raise SystemExit("No balancing segment of at least 1 s in the log")
    t_all = real_time(log)
    print("== Balancing segments")
    for i, (a, b) in enumerate(segments):
        print(f"  [{i}] {t_all[a]:7.2f} s .. {t_all[b - 1]:7.2f} s  ({(b - a) * DT:.1f} s)")
    index = args.segment if args.segment is not None else max(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
    a, b = segments[index]
    seg = log[a:b]
    print(f"  using [{index}]")

    act_py = replay_check(seg, policy)
    replay_err = np.abs(act_py - seg["policy_out"]).max(axis=1)

    # Real signals, from exactly what the network saw
    obs_scale = np.array(policy["obs_scale"], dtype=np.float32)
    obs_real = seg["policy_in"] * obs_scale
    t_real = real_time(seg)
    real = signals(obs_real, seg["policy_out"])

    # Sim, same start pitch and commands
    env_kwargs = dict(policy["env"], max_steps=len(seg) + 1, push_force_max=0.0)  # no random pushes
    if args.dr is not None:
        env_kwargs["domain_randomization"] = args.dr
    env = BalanceRobotEnv(**env_kwargs)
    actor = env_actor(policy, env)
    cmds = obs_real[:, 5:7]
    sim_runs, sim_stats = [], []
    skip = int(args.skip / DT)
    base = os.path.splitext(args.log)[0]
    for seed in range(args.seeds):
        video = SimVideo(env, f"{base}_seg{index}_sim.mp4", obs_real) if args.video and seed == 0 else None
        r = sim_rollout(env, actor, float(obs_real[0, 0]), cmds, seed, video)
        if video is not None:
            video.close()
            print(f"  sim video (seed 0) saved to {video.path}: green = command, orange = sim, blue = real")
        if len(r["action"]) < len(seg):
            print(f"  sim seed {seed} fell after {len(r['action']) * DT:.2f} s")
        sim_runs.append(signals(r["obs"], r["action"]))
        if len(r["action"]) > skip + 64:
            sim_stats.append(stats({k: v[skip:] for k, v in sim_runs[-1].items()}))
    env.close()

    if sim_stats:
        print_stats(stats({k: v[skip:] for k, v in real.items()}), sim_stats)

    title = (f"{os.path.basename(args.log)} [{index}]  vs  sim {os.path.basename(os.path.dirname(model_path))}"
             f"  (DR {'on' if env.domain_randomization else 'off'}, {args.seeds} seeds)")
    figs = {
        "timeseries": plot_time_series(t_real, real, sim_runs, replay_err, title),
        "distributions": plot_distributions({k: v[skip:] for k, v in real.items()},
                                            [{k: v[skip:] for k, v in s.items()} for s in sim_runs], title),
    }
    for name, fig in figs.items():
        fig.savefig(f"{base}_seg{index}_{name}.png", dpi=120)
    print(f"Plots saved to {base}_seg{index}_*.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
