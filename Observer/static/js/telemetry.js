// Telemetry decoding and the buffer everything else reads from (live stream or a loaded log).

const DEG = 180 / Math.PI;
export const WHEEL_RADIUS = 0.0425;  // m, balancebot.xml
export const CONTROL_HZ = 200;
const SPEED_FILTER = 0.2;            // low pass on the odometry speed (per step), raw encoder differences are noisy

// Packet field readers by numpy type (all little endian)
const READERS = {
  f4: (v, o) => v.getFloat32(o, true),
  u1: (v, o) => v.getUint8(o),
  u2: (v, o) => v.getUint16(o, true),
  u4: (v, o) => v.getUint32(o, true),
};
const SIZES = { f4: 4, u1: 1, u2: 2, u4: 4 };

// Decodes packets with a field list from the bridge (generated from the packet dtype)
export class PacketDecoder {
  constructor(layout) {
    this.size = layout.packet_size;
    this.fields = layout.fields;
  }

  *decode(buffer) {
    const view = new DataView(buffer);
    for (let base = 0; base + this.size <= buffer.byteLength; base += this.size) {
      const p = {};
      for (const f of this.fields) {
        const read = READERS[f.type];
        if (f.count === 1) {
          p[f.name] = read(view, base + f.offset);
        } else {
          p[f.name] = Array.from({ length: f.count }, (_, i) => read(view, base + f.offset + i * SIZES[f.type]));
        }
      }
      yield p;
    }
  }
}

// Fields missing in older telemetry versions read as NaN
const num = (v) => (v === undefined ? NaN : v);

// Channels stored per packet: name -> value from the packet and the odometry state
const CHANNELS = {
  t: (p, s) => s.t,
  step: (p) => p.step,
  balancing: (p) => p.flags & 1,
  imu_ok: (p) => (p.flags >> 1) & 1,
  remote: (p) => (p.flags >> 2) & 1,
  disarmed: (p) => (p.flags >> 3) & 1,
  pitch: (p) => p.pitch * DEG,
  accel_pitch: (p) => p.accel_pitch * DEG,
  pitch_rad: (p) => p.pitch,
  gyro_x: (p) => p.gyro[0],
  gyro_y: (p) => p.gyro[1],
  gyro_z: (p) => p.gyro[2],
  accel_x: (p) => p.accel[0],
  accel_y: (p) => p.accel[1],
  accel_z: (p) => p.accel[2],
  wheel_angle_l: (p) => p.wheel_angle[0],
  wheel_angle_r: (p) => p.wheel_angle[1],
  wheel_vel_l: (p) => p.wheel_vel[0],
  wheel_vel_r: (p) => p.wheel_vel[1],
  current_cmd_l: (p) => p.current_cmd[0],
  current_cmd_r: (p) => p.current_cmd[1],
  current_est_l: (p) => p.current_est[0],
  current_est_r: (p) => p.current_est[1],
  voltage_l: (p) => p.voltage[0],
  voltage_r: (p) => p.voltage[1],
  policy_out_l: (p) => p.policy_out[0],
  policy_out_r: (p) => p.policy_out[1],
  ...Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`policy_in_${i}`, (p) => p.policy_in[i]])),
  // Command sent (telemetry v2) and the ramped reference the policy tracks (policy_in 5, 6, needs the scale)
  v_cmd: (p) => num(p.v_cmd),
  yaw_cmd: (p) => num(p.yaw_rate_cmd),
  v_ref: (p, s) => s.obsScale ? p.policy_in[5] * s.obsScale[5] : NaN,
  yaw_ref: (p, s) => s.obsScale ? p.policy_in[6] * s.obsScale[6] : NaN,
  speed: (p, s) => s.speed,
  compute_us: (p) => p.compute_us,
  foc_loops: (p) => p.foc_loops,
  x: (p, s) => s.x,
  y: (p, s) => s.y,
  yaw: (p, s) => s.yaw,
};
const NAMES = Object.keys(CHANNELS);

// One typed array per channel (t in double precision, the rest float32).
// Rolling: keeps the last `capacity` packets. Twice that is allocated and the newer half moved to
// the front when full, so any window is a contiguous subarray (no copies for the charts).
// Not rolling (a log): exactly `capacity` packets.
export class TelemetryStore {
  constructor(capacity, { rolling = true, obsScale = null } = {}) {
    this.capacity = capacity;
    this.rolling = rolling;
    const n = rolling ? 2 * capacity : capacity;
    this.data = Object.fromEntries(NAMES.map((c) => [c, c === "t" ? new Float64Array(n) : new Float32Array(n)]));
    this.obsScale = obsScale;
    this.len = 0;
    this.lastStep = null;
    this.lastTus = 0;
    this.lost = 0;
    this.t = 0;
    this.x = this.y = this.yaw = 0;
    this.prevWheel = null;
    this.speed = 0;
  }

  push(p) {
    let dt = 0;
    if (this.lastStep !== null) {
      const gap = p.step - this.lastStep;
      if (gap <= 0 && gap > -2 * CONTROL_HZ) return;  // duplicate or reordered datagram
      if (gap <= 0) {
        // Robot restarted (step counter back to 0): continue the time line, odometry from here
        dt = 1 / CONTROL_HZ;
        this.prevWheel = null;
      } else {
        this.lost += gap - 1;
        dt = ((p.t_us - this.lastTus) >>> 0) * 1e-6;  // t_us wraps every ~71 min
      }
    }
    this.t += dt;
    this.lastStep = p.step;
    this.lastTus = p.t_us;
    this.odometry(p, Math.min(dt, 0.1));

    if (this.len === this.data.t.length) {
      if (!this.rolling) return;
      for (const c of NAMES) this.data[c].copyWithin(0, this.capacity);
      this.len = this.capacity;
    }
    for (const c of NAMES) this.data[c][this.len] = CHANNELS[c](p, this);
    this.len++;
  }

  // Ground travel = wheel rotation relative to the chassis + chassis pitch (both about the axle).
  // Heading from the gyro: drifts slowly, good enough to watch.
  odometry(p, dt) {
    const wheel = (p.wheel_angle[0] + p.wheel_angle[1]) / 2;
    if (this.prevWheel !== null) {
      const ds = WHEEL_RADIUS * (wheel - this.prevWheel.wheel + p.pitch - this.prevWheel.pitch);
      this.yaw += p.gyro[2] * dt;
      this.x += ds * Math.cos(this.yaw);
      this.y += ds * Math.sin(this.yaw);
    }
    this.prevWheel = { wheel, pitch: p.pitch };
    const v = WHEEL_RADIUS * ((p.wheel_vel[0] + p.wheel_vel[1]) / 2 + p.gyro[0]);
    this.speed += SPEED_FILTER * (v - this.speed);
  }

  get firstT() {
    return this.len ? this.data.t[0] : null;
  }

  get latestT() {
    return this.len ? this.data.t[this.len - 1] : null;
  }

  // First index with t >= t0
  indexAt(t0) {
    const t = this.data.t;
    let lo = 0, hi = this.len;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (t[mid] < t0) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  // The `seconds` before endT of the given channels, as subarrays (valid until the next push)
  window(seconds, names, endT = this.latestT) {
    const end = this.len ? Math.min(this.indexAt(endT) + 1, this.len) : 0;
    const start = this.len ? this.indexAt(endT - seconds) : 0;
    return Object.fromEntries(names.map((n) => [n, this.data[n].subarray(start, end)]));
  }

  // Channels interpolated at time t (clamped to the buffer)
  sample(t, names) {
    if (!this.len) return null;
    const out = {};
    const i = Math.min(Math.max(this.indexAt(t), 1), this.len - 1);
    const t0 = this.data.t[i - 1], t1 = this.data.t[i];
    const a = this.len === 1 || t1 === t0 ? 1 : Math.min(Math.max((t - t0) / (t1 - t0), 0), 1);
    for (const n of names) {
      const d = this.data[n];
      out[n] = this.len === 1 ? d[0] : d[i - 1] + a * (d[i] - d[i - 1]);
    }
    return out;
  }

  // Loop rates and link quality over the `seconds` before endT
  stats(seconds, endT = this.latestT) {
    if (this.len < 2) return null;
    const d = this.data;
    const end = Math.min(this.indexAt(endT), this.len - 1);
    const start = Math.min(this.indexAt(endT - seconds), end);
    const span = d.t[end] - d.t[start];
    if (span <= 0) return null;
    let focLoops = 0, computeSum = 0, computeMax = 0;
    for (let i = start + 1; i <= end; i++) focLoops += d.foc_loops[i];
    for (let i = start; i <= end; i++) {
      computeSum += d.compute_us[i];
      computeMax = Math.max(computeMax, d.compute_us[i]);
    }
    const received = end - start + 1;
    const expected = d.step[end] - d.step[start] + 1;
    return {
      controlHz: (d.step[end] - d.step[start]) / span,
      focHz: focLoops / span,
      computeMean: computeSum / received,
      computeMax,
      loss: expected > 0 ? Math.max(0, 1 - received / expected) : 0,  // across a restart: unknown
    };
  }
}
