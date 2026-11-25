"""Record the firmware's telemetry (one packet per 200 Hz control step) over WiFi to a .npy file.

    python robot_log.py                            # -> logs/<date_time>.npy
    python robot_log.py --host 192.168.4.1 -o logs/carpet.npy

The PC must be on the robot's WiFi (default: its own network "BalancingBot", see wifi_config.h).
By default the start request is broadcast, so the robot's IP is not needed; pass --host if the
PC has several network interfaces and nothing arrives.
Ctrl+C stops and saves. Then compare with the sim:  python compare_real_sim.py logs/<file>.npy runs/<run>
"""
import argparse
import binascii
import datetime
import os
import socket
import sys
import time

import numpy as np

# Same layout as TelemetryPacket in the firmware (include/telemetry.h)
TELEMETRY_VERSION = 2
SYNC = b"\xAA\x55"
PACKET_DTYPE = np.dtype([
    ("sync", "u1", 2),
    ("version", "u1"),
    ("flags", "u1"),           # bit 0 balancing, bit 1 IMU read ok, bit 2 remote command, bit 3 disarmed
    ("step", "<u4"),           # control step counter, gaps = dropped packets
    ("t_us", "<u4"),           # micros() at the start of the step
    ("compute_us", "<u2"),     # time spent in the control step
    ("foc_loops", "<u2"),      # FOC iterations since the previous step
    ("accel", "<f4", 3),       # m/s^2, IMU frame (x = left, y = back, z = up)
    ("gyro", "<f4", 3),        # rad/s, IMU frame, bias removed
    ("accel_pitch", "<f4"),    # rad
    ("pitch", "<f4"),          # rad, complementary filter
    ("wheel_angle", "<f4", 2), # rad, left / right, forward frame
    ("wheel_vel", "<f4", 2),   # rad/s
    ("policy_in", "<f4", 9),   # normalized observation fed to the network
    ("policy_out", "<f4", 2),  # network output in [-1, 1]
    ("current_cmd", "<f4", 2), # A
    ("current_est", "<f4", 2), # A, SimpleFOC estimate
    ("voltage", "<f4", 2),     # V
    ("v_cmd", "<f4"),          # m/s, forward velocity command used
    ("yaw_rate_cmd", "<f4"),   # rad/s, yaw rate command used
    ("crc", "<u2"),            # CRC-16/XMODEM over everything above
])
assert PACKET_DTYPE.itemsize == 142
FLAG_BALANCING = 1 << 0
FLAG_IMU_OK = 1 << 1
FLAG_REMOTE = 1 << 2
FLAG_DISARMED = 1 << 3


class PacketParser:
    """Finds packets in a byte stream. Anything else (boot messages, text status lines) is skipped."""

    def __init__(self):
        self.buf = bytearray()
        self.crc_errors = 0

    def feed(self, data):
        """Returns the complete packets in data (plus what was buffered before), as raw bytes."""
        self.buf += data
        packets = []
        n = PACKET_DTYPE.itemsize
        while True:
            start = self.buf.find(SYNC)
            if start < 0:
                del self.buf[:-1]  # keep a trailing 0xAA, it may be the start of the next sync
                break
            del self.buf[:start]
            if len(self.buf) < n:
                break
            pkt = bytes(self.buf[:n])
            if binascii.crc_hqx(pkt[:-2], 0) != int.from_bytes(pkt[-2:], "little"):
                self.crc_errors += 1
                del self.buf[:1]   # false sync or corrupted, look further
                continue
            if pkt[2] != TELEMETRY_VERSION:
                sys.exit(f"Firmware sends telemetry version {pkt[2]}, robot_log.py expects {TELEMETRY_VERSION}")
            packets.append(pkt)
            del self.buf[:n]
        return packets


def to_array(packets):
    return np.frombuffer(b"".join(packets), dtype=PACKET_DTYPE)


def load(path):
    """A recorded log as a numpy structured array (fields as in PACKET_DTYPE)."""
    return np.load(path)


TELEMETRY_PORT = 4210  # same as in wifi_config.h
KEEPALIVE_S = 1.0      # the robot stops streaming 3 s after the last request


def record(host, out_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.05)
    robot = (host or "255.255.255.255", TELEMETRY_PORT)

    parser = PacketParser()
    packets = []
    last_status = last_request = 0.0
    n_last = 0
    print(f"Requesting telemetry from {robot[0]}:{robot[1]}, Ctrl+C to stop")
    try:
        while True:
            now = time.time()
            if now - last_request >= KEEPALIVE_S:
                sock.sendto(b"b", robot)  # start / keep streaming to this socket
                last_request = now
            try:
                data, sender = sock.recvfrom(65536)
                if host is None:
                    robot = (sender[0], TELEMETRY_PORT)  # found it, no more broadcasts
                packets += parser.feed(data)
            except socket.timeout:
                pass
            if not packets and now - last_status >= 2.0:
                print(f"waiting for packets ... (on the robot's WiFi? firewall allows UDP {TELEMETRY_PORT}?)")
                last_status = now
            if now - last_status >= 0.5 and packets:
                p = np.frombuffer(packets[-1], dtype=PACKET_DTYPE)[0]
                first_step = int.from_bytes(packets[0][4:8], "little")
                lost = int(p["step"]) - first_step + 1 - len(packets)
                rate = (len(packets) - n_last) / (now - last_status)
                print(f"\r{robot[0]} | {len(packets):7d} packets | {rate:5.0f} Hz | lost {lost} | crc errors {parser.crc_errors} | "
                      f"pitch {np.rad2deg(p['pitch']):6.2f} deg | {'ON ' if p['flags'] & FLAG_BALANCING else 'off'}",
                      end="", flush=True)
                n_last = len(packets)
                last_status = now
    except KeyboardInterrupt:
        pass
    finally:
        sock.sendto(b"t", robot)  # stop streaming
        sock.close()

    print()
    if not packets:
        sys.exit("No packets received. Is the telemetry firmware flashed and the PC on the robot's WiFi?")
    data = to_array(packets)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, data)
    duration = int(np.uint32(data["t_us"][-1] - data["t_us"][0])) * 1e-6  # t_us wraps
    print(f"Saved {len(data)} packets ({duration:.1f} s) to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="robot IP or balancingbot.local (default: broadcast)")
    ap.add_argument("-o", "--out", help="output .npy (default: logs/<date_time>.npy)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs",
                                   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".npy")
    record(args.host, out)
