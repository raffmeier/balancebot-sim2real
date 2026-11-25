"""Stands in for the robot: answers the telemetry protocol (telemetry.h) with synthetic packets, to
work on the observer without hardware. Wobbles in place, drives with the commands it receives
(ramped and limited like the firmware), falls over when the motors are switched off and stands
up again 2 s after they are armed.

    python fake_robot.py
    python observer.py --robot 127.0.0.1
"""
import binascii
import math
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RL_training"))
from robot_log import (FLAG_BALANCING, FLAG_DISARMED, FLAG_IMU_OK, FLAG_REMOTE,  # noqa: E402
                       PACKET_DTYPE, TELEMETRY_PORT, TELEMETRY_VERSION)

DT = 0.005
BATCH = 8
R = 0.0425
HALF_TRACK = 0.08
OBS_SCALE = np.array([0.2, 3, 3, 15, 15, 0.5, 3, 1, 1])
COMMAND_TIMEOUT_S = 0.3
V_MAX, YAW_RATE_MAX = 0.8, 3.0   # the policy's training range (POLICY_V_CMD_MAX in model.h)
ACC_MAX, YAW_ACC_MAX = 1.0, 10.0  # the policy's command ramp (POLICY_V_CMD_ACC_MAX)


def ramp(value, target, step):
    return value + min(max(target - value, -step), step)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", TELEMETRY_PORT))
    sock.setblocking(False)
    print(f"Fake robot on UDP {TELEMETRY_PORT}, Ctrl+C to stop")

    remote, last_keepalive = None, 0.0
    cmd, cmd_time, armed = (0.0, 0.0), -1.0, True
    v_ref = yaw_ref = 0.0
    pitch_fallen, upright_at = 0.0, 0.0
    step, wheel = 0, np.zeros(2)
    batch = np.zeros(BATCH, dtype=PACKET_DTYPE)
    next_t = time.monotonic()
    while True:
        now = time.monotonic()
        try:
            while True:
                data, addr = sock.recvfrom(64)
                kind = data[:1]
                if kind == b"b":
                    remote, last_keepalive = addr, now
                elif kind == b"t":
                    remote = None
                elif kind == b"c" and len(data) >= 9:
                    cmd, cmd_time = struct.unpack("<ff", data[1:9]), now
                elif kind in (b"k", b"a"):
                    if kind == b"a" and not armed:
                        upright_at = now + 2.0  # someone stands it up
                    armed = kind == b"a"
        except BlockingIOError:
            pass
        if remote and now - last_keepalive > 3.0:
            remote = None

        fresh = now - cmd_time < COMMAND_TIMEOUT_S
        v_cmd = float(np.clip(cmd[0], -V_MAX, V_MAX)) if fresh else 0.0
        yaw_cmd = float(np.clip(cmd[1], -YAW_RATE_MAX, YAW_RATE_MAX)) if fresh else 0.0
        balancing = armed and now >= upright_at
        if not balancing:
            v_ref = yaw_ref = 0.0
        v_ref = ramp(v_ref, v_cmd, ACC_MAX * DT)
        yaw_ref = ramp(yaw_ref, yaw_cmd, YAW_ACC_MAX * DT)

        t = step * DT
        if balancing:
            pitch_fallen = 0.0
            lean = 0.15 * (v_cmd - v_ref) / ACC_MAX  # leans into accelerations
            pitch = lean + 0.02 * math.sin(2 * math.pi * 1.3 * t) + 0.005 * np.random.randn()
            pitch_rate = 2 * math.pi * 1.3 * 0.02 * math.cos(2 * math.pi * 1.3 * t)
            v, yaw_rate = v_ref, yaw_ref
        else:  # falls over, lies at ~45 deg
            pitch_fallen = min(0.8, pitch_fallen + 3 * DT)
            pitch, pitch_rate, v, yaw_rate = pitch_fallen, 0.0, 0.0, 0.0
        wheel_vel = np.array([v - yaw_rate * HALF_TRACK, v + yaw_rate * HALF_TRACK]) / R - pitch_rate
        wheel += wheel_vel * DT
        action = (np.clip([0.3 * math.sin(2 * math.pi * 1.3 * t + 0.5) + 2 * (v_cmd - v_ref)] * 2, -1, 1)
                  if balancing else np.zeros(2))

        p = batch[step % BATCH]
        p["sync"] = (0xAA, 0x55)
        p["version"] = TELEMETRY_VERSION
        p["flags"] = (FLAG_IMU_OK | (FLAG_BALANCING if balancing else 0) | (FLAG_REMOTE if fresh else 0)
                      | (0 if armed else FLAG_DISARMED))
        p["step"] = step
        p["t_us"] = int(t * 1e6) & 0xFFFFFFFF
        p["compute_us"] = 1400 + int(80 * np.random.rand())
        p["foc_loops"] = 28 + int(4 * np.random.rand())
        p["accel"] = (0.1 * np.random.randn(), 9.81 * math.sin(pitch), 9.81 * math.cos(pitch))
        p["gyro"] = (pitch_rate, 0.02 * np.random.randn(), yaw_rate + 0.02 * np.random.randn())
        p["accel_pitch"] = pitch + 0.02 * np.random.randn()
        p["pitch"] = pitch
        p["wheel_angle"] = wheel
        p["wheel_vel"] = wheel_vel + 0.3 * np.random.randn(2)
        p["policy_in"] = np.array([pitch, pitch_rate, yaw_rate, *wheel_vel, v_ref, yaw_ref, *action]) / OBS_SCALE
        p["policy_out"] = action
        p["current_cmd"] = action * 1.017
        p["current_est"] = action * 1.017 + 0.03 * np.random.randn(2)
        p["voltage"] = action * 4.0
        p["v_cmd"] = v_cmd
        p["yaw_rate_cmd"] = yaw_cmd
        p["crc"] = binascii.crc_hqx(p.tobytes()[:-2], 0)

        step += 1
        if step % BATCH == 0 and remote:
            sock.sendto(batch.tobytes(), remote)

        next_t += DT
        time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
