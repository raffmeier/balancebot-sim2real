"""Live web observer for the robot: forwards the firmware's telemetry (200 Hz UDP) to the browser,
the browser's drive commands to the robot, records logs and serves them for replay.

    python observer.py                         # then open http://localhost:8080
    python observer.py --robot 192.168.4.1     # robot IP or balancingbot.local (default: broadcast)
    python observer.py --bind 0.0.0.0          # also reachable from other devices on the network

The PC must be on the same WiFi as the robot (see wifi_config.h). The robot streams to one PC only:
don't run robot_log.py at the same time. Recordings go to RL_training/logs, same format as robot_log.py.

Packet layout, parser and constants come from RL_training/robot_log.py. The browser gets the field
list on connect and decodes the raw packets itself, so new telemetry fields need no change here.
"""
import argparse
import asyncio
import datetime
import json
import math
import os
import re
import socket
import struct
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from aiohttp import WSMsgType, web

ROOT = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.join(ROOT, "..", "RL_training")
MODEL_XML = os.path.join(RL_DIR, "balancebot.xml")
POLICY_HEADER = os.path.join(ROOT, "..", "Firmware", "BalancingBot_Firmware", "weights", "model.h")
STATIC_DIR = os.path.join(ROOT, "static")
LOG_DIR = os.path.join(RL_DIR, "logs")

sys.path.insert(0, RL_DIR)
from robot_log import KEEPALIVE_S, PACKET_DTYPE, TELEMETRY_PORT, PacketParser, to_array  # noqa: E402

ROBOT_TIMEOUT_S = 3.0      # no packets for this long: look for the robot again (broadcast)
LINK_STATUS_PERIOD_S = 0.5
CLIENT_QUEUE_LENGTH = 100  # datagrams (8 packets each, 4 s), a slower browser drops the oldest
MOTOR_REPEATS = 3          # "k" / "a" are sent this often, a lost datagram must not keep the motors on
LOG_FIELDS = {"step", "t_us", "flags", "pitch", "wheel_angle", "wheel_vel", "gyro"}  # needed for replay


def packet_fields(dtype=PACKET_DTYPE):
    """A packet dtype as a list the browser can decode with a DataView."""
    fields = []
    for name in dtype.names:
        sub, offset = dtype.fields[name][:2]
        fields.append({
            "name": name,
            "offset": offset,
            "type": sub.base.str[1:],  # "f4", "u1", "u2", "u4"
            "count": int(sub.shape[0]) if sub.shape else 1,
        })
    return fields


def policy_obs_scale():
    """POLICY_OBS_SCALE from the exported policy on disk, to un-normalize policy_in.
    Only right if the robot runs this export."""
    try:
        with open(POLICY_HEADER) as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r"POLICY_OBS_SCALE\[\d+\]\s*=\s*\{([^}]*)\}", text)
    if not m:
        return None
    return [float(v.strip().rstrip("f")) for v in m.group(1).split(",")]


def vec(text, default):
    return [float(v) for v in text.split()] if text else default


def load_model():
    """The bodies, visual meshes and joints of balancebot.xml, for the 3D view."""
    root = ET.parse(MODEL_XML).getroot()
    meshes = {}
    for m in root.iter("mesh"):
        meshes[m.get("name")] = {
            "url": "/mesh/" + os.path.basename(m.get("file")),
            "file": os.path.join(os.path.dirname(MODEL_XML), m.get("file")),
            "scale": vec(m.get("scale"), [1, 1, 1]),
        }

    def body(el):
        joint = el.find("joint")
        return {
            "name": el.get("name"),
            "pos": vec(el.get("pos"), [0, 0, 0]),
            "free": el.find("freejoint") is not None,
            "joint": None if joint is None else {
                "name": joint.get("name"),
                "type": joint.get("type", "hinge"),
                "axis": vec(joint.get("axis"), [0, 0, 1]),
            },
            "geoms": [{
                "mesh": g.get("mesh"),
                "pos": vec(g.get("pos"), [0, 0, 0]),
                "euler": vec(g.get("euler"), [0, 0, 0]),
                "rgba": vec(g.get("rgba"), [0.5, 0.5, 0.5, 1]),
            } for g in el.findall("geom") if g.get("type") == "mesh"],
            "children": [body(c) for c in el.findall("body")],
            "radius": next((vec(g.get("size"), [0])[0] for g in el.findall("geom")
                            if g.get("type") == "cylinder"), None),
        }

    bodies = [body(b) for b in root.find("worldbody").findall("body")]
    return meshes, {
        "meshes": {k: {"url": v["url"], "scale": v["scale"]} for k, v in meshes.items()},
        "bodies": bodies,
    }


class RobotLink(asyncio.DatagramProtocol):
    """Requests the telemetry stream, validates packets and hands them to the browsers."""

    def __init__(self, host, on_packets):
        self.fixed_host = host
        self.robot = None  # IP that packets come from
        self.on_packets = on_packets
        self.parser = PacketParser()
        self.transport = None
        self.last_packet = 0.0
        self.version_error = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            packets = self.parser.feed(data)
        except SystemExit as e:  # PacketParser exits on a telemetry version mismatch
            self.version_error = str(e)
            self.parser.buf.clear()
            return
        if not packets:
            return
        self.version_error = None
        self.robot = addr[0]
        self.last_packet = time.monotonic()
        self.on_packets(b"".join(packets))

    def receiving(self):
        return time.monotonic() - self.last_packet < ROBOT_TIMEOUT_S

    def request(self, cmd, broadcast=True):
        """Sends a command datagram (protocol in telemetry.h). Without broadcast, only to a known robot."""
        if self.transport is None:
            return
        host = self.fixed_host or (self.robot if self.receiving() else None)
        if host is None and not broadcast:
            return
        try:
            self.transport.sendto(cmd, (host or "255.255.255.255", TELEMETRY_PORT))
        except OSError:  # e.g. network down, retried with the next keep-alive
            pass

    def drive(self, v, yaw_rate):
        """The browser repeats it while driving, the robot falls back to 0 when it stops."""
        if math.isfinite(v) and math.isfinite(yaw_rate):
            self.request(struct.pack("<cff", b"c", v, yaw_rate), broadcast=False)

    async def motors(self, on):
        for _ in range(MOTOR_REPEATS):
            self.request(b"a" if on else b"k")
            await asyncio.sleep(0.02)

    async def keepalive(self):
        while True:
            self.request(b"b")
            await asyncio.sleep(KEEPALIVE_S)


class Recorder:
    """Keeps the packets while recording, saves them like robot_log.py."""

    def __init__(self):
        self.packets = None  # list of raw packet blocks while recording
        self.count = 0
        self.started = None

    @property
    def active(self):
        return self.packets is not None

    def start(self):
        self.packets, self.count, self.started = [], 0, datetime.datetime.now()

    def add(self, data):
        if self.active:
            self.packets.append(data)
            self.count += len(data) // PACKET_DTYPE.itemsize

    def stop(self):
        """Saves the recording, returns the file name (None if nothing was received)."""
        packets, self.packets = self.packets, None
        if not packets:
            return None
        name = self.started.strftime("%Y-%m-%d_%H-%M-%S") + ".npy"
        os.makedirs(LOG_DIR, exist_ok=True)
        np.save(os.path.join(LOG_DIR, name), to_array(packets))
        return name


def log_path(name):
    """A log file in LOG_DIR by name, None for anything else."""
    path = os.path.join(LOG_DIR, os.path.basename(name))
    return path if name.endswith(".npy") and os.path.isfile(path) else None


def log_duration(log):
    return int(np.uint32(log["t_us"][-1] - log["t_us"][0])) * 1e-6 if len(log) else 0.0  # t_us wraps


class Observer:
    def __init__(self, robot_host):
        self.clients = {}  # WebSocketResponse -> asyncio.Queue of outgoing messages
        self.link = RobotLink(robot_host, self.on_packets)
        self.recorder = Recorder()
        self.mesh_files, self.model = load_model()
        self.obs_scale = policy_obs_scale()
        self.hello = json.dumps({
            "type": "hello",
            "packet_size": PACKET_DTYPE.itemsize,
            "fields": packet_fields(),
            "obs_scale": self.obs_scale,
        })

    def on_packets(self, data):
        self.recorder.add(data)
        self.broadcast(data)

    def broadcast(self, data):
        for queue in self.clients.values():
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(data)

    def link_status(self):
        rec = self.recorder
        return json.dumps({
            "type": "link",
            "robot": self.link.robot,
            "receiving": self.link.receiving(),
            "crc_errors": self.link.parser.crc_errors,
            "version_error": self.link.version_error,
            "recording": {"packets": rec.count, "started": rec.started.isoformat()} if rec.active else None,
        })

    def set_recording(self, on):
        if on and not self.recorder.active:
            self.recorder.start()
        elif not on and self.recorder.active:
            name = self.recorder.stop()
            self.broadcast(json.dumps({"type": "saved", "name": name}))
        self.broadcast(self.link_status())

    def handle_message(self, msg):
        """Commands from the browser (JSON)."""
        kind = msg.get("type")
        if kind == "drive":
            self.link.drive(float(msg["v"]), float(msg["yaw_rate"]))
        elif kind == "motors":
            asyncio.create_task(self.link.motors(bool(msg["on"])))
        elif kind == "record":
            self.set_recording(bool(msg["on"]))

    async def status_loop(self):
        while True:
            self.broadcast(self.link_status())
            await asyncio.sleep(LINK_STATUS_PERIOD_S)

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        queue = asyncio.Queue(CLIENT_QUEUE_LENGTH)
        queue.put_nowait(self.hello)
        queue.put_nowait(self.link_status())
        self.clients[ws] = queue

        async def sender():
            while True:
                msg = await queue.get()
                if isinstance(msg, str):
                    await ws.send_str(msg)
                else:
                    await ws.send_bytes(msg)

        send_task = asyncio.create_task(sender())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        self.handle_message(json.loads(msg.data))
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"Bad message from the browser: {e}")
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            send_task.cancel()
            del self.clients[ws]
        return ws

    async def index(self, request):
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))

    async def model_handler(self, request):
        return web.json_response(self.model)

    async def logs_handler(self, request):
        """The recorded logs, newest first."""
        logs = []
        for name in sorted(os.listdir(LOG_DIR) if os.path.isdir(LOG_DIR) else [], reverse=True):
            path = log_path(name)
            if path is None:
                continue
            try:
                log = np.load(path, mmap_mode="r")
                if log.dtype.names is None or not LOG_FIELDS <= set(log.dtype.names):
                    continue
                logs.append({"name": name, "packets": len(log), "duration": log_duration(log)})
            except (OSError, ValueError):
                continue
        return web.json_response(logs)

    async def log_handler(self, request):
        """One log: its own field list (older telemetry versions decode too), then the raw packets."""
        path = log_path(request.match_info["name"])
        if path is None:
            raise web.HTTPNotFound()
        log = np.load(path)
        if request.match_info.get("part") == "data":
            return web.Response(body=log.tobytes(), content_type="application/octet-stream")
        return web.json_response({
            "packet_size": log.dtype.itemsize,
            "fields": packet_fields(log.dtype),
            "obs_scale": self.obs_scale,
            "packets": len(log),
            "duration": log_duration(log),
        })

    async def mesh_handler(self, request):
        name = request.match_info["name"]
        for m in self.mesh_files.values():
            if os.path.basename(m["file"]) == name:
                return web.FileResponse(m["file"])
        raise web.HTTPNotFound()

    async def start(self, app):
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self.link, local_addr=("0.0.0.0", 0),
                                            family=socket.AF_INET, allow_broadcast=True)
        app["tasks"] = [asyncio.create_task(self.link.keepalive()),
                        asyncio.create_task(self.status_loop())]

    async def stop(self, app):
        for t in app["tasks"]:
            t.cancel()
        name = self.recorder.stop() if self.recorder.active else None
        if name:
            print(f"Saved the recording to {os.path.join(LOG_DIR, name)}")
        self.link.request(b"t")  # stop streaming
        for ws in list(self.clients):
            await ws.close()

    def app(self):
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/api/model", self.model_handler)
        app.router.add_get("/mesh/{name}", self.mesh_handler)
        app.router.add_get("/api/logs", self.logs_handler)
        app.router.add_get("/api/logs/{name}", self.log_handler)
        app.router.add_get("/api/logs/{name}/{part:data}", self.log_handler)
        app.router.add_static("/static", STATIC_DIR)
        app.on_startup.append(self.start)
        app.on_shutdown.append(self.stop)
        return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", help="robot IP or balancingbot.local (default: broadcast)")
    ap.add_argument("--bind", default="127.0.0.1", help="web server address (0.0.0.0 = all interfaces)")
    ap.add_argument("--port", type=int, default=8080, help="web server port")
    args = ap.parse_args()

    robot = socket.gethostbyname(args.robot) if args.robot else None
    observer = Observer(robot)
    print(f"Observer: http://{'localhost' if args.bind == '127.0.0.1' else args.bind}:{args.port}  "
          f"(telemetry from {robot or 'broadcast'}:{TELEMETRY_PORT}, Ctrl+C to stop)")
    web.run_app(observer.app(), host=args.bind, port=args.port, print=None)
