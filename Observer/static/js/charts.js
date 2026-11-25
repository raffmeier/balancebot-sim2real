// Rolling time series (uPlot), one card per quantity, crosshairs synced across all charts.
// Shaded bands: controller off (not balancing). Optional smoothing: centered moving average.
import { CONTROL_HZ } from "./telemetry.js";

// color: categorical slot (1..3), "ref" (reference the policy tracks, secondary ink, dashed)
// or "cmd" (command sent, muted ink, dotted)
const CHARTS = [
  { title: "Pitch", unit: "deg", minSpan: 4, series: [
    { ch: "accel_pitch", label: "accel only", color: 2 },  // noisy, drawn below the filtered pitch
    { ch: "pitch", label: "filtered", color: 1 },
  ] },
  { title: "Gyro", unit: "rad/s", minSpan: 0.5, series: [
    { ch: "gyro_x", label: "x (pitch rate)", color: 1 },
    { ch: "gyro_y", label: "y", color: 2 },
    { ch: "gyro_z", label: "z (yaw rate)", color: 3 },
  ] },
  { title: "Forward speed", unit: "m/s", minSpan: 0.1, series: [
    { ch: "speed", label: "odometry", color: 1 },
    { ch: "v_ref", label: "reference", color: "ref" },
    { ch: "v_cmd", label: "command", color: "cmd" },
  ] },
  { title: "Yaw rate", unit: "rad/s", minSpan: 0.5, series: [
    { ch: "gyro_z", label: "gyro z", color: 1 },
    { ch: "yaw_ref", label: "reference", color: "ref" },
    { ch: "yaw_cmd", label: "command", color: "cmd" },
  ] },
  { title: "Wheel velocity", unit: "rad/s", minSpan: 2, series: [
    { ch: "wheel_vel_l", label: "left", color: 1 },
    { ch: "wheel_vel_r", label: "right", color: 2 },
  ] },
  { title: "Motor current", unit: "A", minSpan: 0.2, series: [
    { ch: "current_cmd_l", label: "cmd L", color: 1 },
    { ch: "current_cmd_r", label: "cmd R", color: 2 },
    { ch: "current_est_l", label: "est L", color: 1, dash: true },
    { ch: "current_est_r", label: "est R", color: 2, dash: true },
  ] },
  { title: "Motor voltage", unit: "V", minSpan: 0.5, series: [
    { ch: "voltage_l", label: "left", color: 1 },
    { ch: "voltage_r", label: "right", color: 2 },
  ] },
  { title: "Policy output", unit: "[-1, 1]", range: [-1.1, 1.1], series: [
    { ch: "policy_out_l", label: "left", color: 1 },
    { ch: "policy_out_r", label: "right", color: 2 },
  ] },
  { title: "Accelerometer", unit: "m/s²", minSpan: 2, series: [
    { ch: "accel_x", label: "x (left)", color: 1 },
    { ch: "accel_y", label: "y (back)", color: 2 },
    { ch: "accel_z", label: "z (up)", color: 3 },
  ] },
  { title: "Control step time", unit: "µs", minSpan: 200, series: [
    { ch: "compute_us", label: "IMU read + controller", color: 1 },
  ] },
];

const CHART_HEIGHT = 170;

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function yRange(def) {
  if (def.range) return () => def.range;
  return (u, min, max) => {
    if (min == null || !isFinite(min)) return [-1, 1];
    const mid = (min + max) / 2;
    const half = Math.max((max - min) / 2 * 1.1, def.minSpan / 2);
    return [mid - half, mid + half];
  };
}

function fmt(v, digits = 3) {
  return v == null || !isFinite(v) ? "–" : v.toFixed(digits);
}

// Centered moving average over 2 * half + 1 samples (fewer at the ends), no lag. NaN is skipped.
function movingAverage(a, half) {
  const n = a.length, out = new Float32Array(n);
  let sum = 0, count = 0, lo = 0, hi = 0;  // window [lo, hi)
  for (let i = 0; i < n; i++) {
    for (const end = Math.min(n, i + half + 1); hi < end; hi++) {
      if (!isNaN(a[hi])) { sum += a[hi]; count++; }
    }
    for (const start = Math.max(0, i - half); lo < start; lo++) {
      if (!isNaN(a[lo])) { sum -= a[lo]; count--; }
    }
    out[i] = count ? sum / count : NaN;
  }
  return out;
}

export class ChartGrid {
  constructor(container, source) {
    this.container = container;
    this.source = source;
    this.windowS = 10;
    this.smoothS = 0;   // moving average width, 0 = raw
    this.frozen = false;
    this.charts = [];
    this.flags = new Float64Array(0);  // balancing flag aligned with the charts' x data
    this.sync = uPlot.sync("telemetry");
    this.build();
    new ResizeObserver(() => this.resize()).observe(container);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.rebuild());
  }

  build() {
    const theme = {
      series: [null, css("--series-1"), css("--series-2"), css("--series-3")],
      ref: css("--text-secondary"),
      cmd: css("--text-muted"),
      axis: css("--text-muted"),
      grid: css("--gridline"),
      shade: css("--off-band"),
    };
    const shadeOff = (u) => this.drawOffBands(u, theme.shade);

    for (const def of CHARTS) {
      const card = document.createElement("section");
      card.className = "card chart";
      const head = document.createElement("h3");
      head.textContent = def.title;
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = def.unit;
      head.append(" ", unit);
      const plot = document.createElement("div");
      card.append(head, plot);
      this.container.append(card);

      const axis = { stroke: theme.axis, grid: { stroke: theme.grid, width: 1 }, ticks: { stroke: theme.grid, width: 1 },
                     font: "11px system-ui, sans-serif" };
      const opts = {
        width: plot.clientWidth || 400,
        height: CHART_HEIGHT,
        cursor: { sync: { key: this.sync.key }, drag: { x: true, y: false } },
        legend: { live: true },
        scales: { x: { time: false }, y: { range: yRange(def) } },
        axes: [{ ...axis, values: (u, vals) => vals.map((v) => v.toFixed(1) + " s") }, { ...axis, size: 48 }],
        series: [
          { label: "t", value: (u, v) => fmt(v, 2) + " s" },
          ...def.series.map((s) => ({
            label: s.label,
            stroke: theme[s.color] ?? theme.series[s.color],
            width: s.color === "cmd" ? 1.5 : 2,
            dash: s.color === "cmd" ? [2, 3] : s.dash || s.color === "ref" ? [6, 4] : undefined,
            points: { show: false },
            value: (u, v) => fmt(v, def.unit === "µs" ? 0 : 3),
          })),
        ],
        hooks: { drawClear: [shadeOff] },
      };
      const u = new uPlot(opts, [[], ...def.series.map(() => [])], plot);
      this.charts.push({ def, u, plot });
    }
  }

  rebuild() {
    for (const c of this.charts) c.u.destroy();
    this.charts = [];
    this.container.replaceChildren();
    this.build();
    this.update(true);
  }

  resize() {
    for (const c of this.charts) c.u.setSize({ width: c.plot.clientWidth, height: CHART_HEIGHT });
  }

  // Shades where the controller was off
  drawOffBands(u, color) {
    const t = u.data[0], flags = this.flags;
    if (!t || t.length < 2 || flags.length !== t.length) return;
    const ctx = u.ctx, { top, height } = u.bbox;
    ctx.save();
    ctx.fillStyle = color;
    let i = 0;
    while (i < t.length) {
      if (flags[i]) { i++; continue; }
      const start = i;
      while (i < t.length && !flags[i]) i++;
      const x0 = u.valToPos(t[start], "x", true);
      const x1 = u.valToPos(t[Math.min(i, t.length - 1)], "x", true);
      ctx.fillRect(x0, top, Math.max(x1 - x0, 1), height);
    }
    ctx.restore();
  }

  // Frozen: keep a copy of the current window, so it can be zoomed (drag) and hovered
  setFrozen(frozen) {
    this.frozen = frozen;
    if (frozen) this.update(true);
  }

  // Redraws the window ending at the source's time (force: also while frozen, e.g. new settings)
  update(force = false) {
    if (this.frozen && !force) return;
    const endT = this.source.chartT;
    const names = new Set(["t", "balancing"]);
    for (const c of this.charts) for (const s of c.def.series) names.add(s.ch);
    let w = endT === null ? {} : this.source.store.window(this.windowS, [...names], endT);
    if (endT === null) for (const n of names) w[n] = new Float32Array(0);
    const half = Math.round(this.smoothS * CONTROL_HZ / 2);
    const series = {};
    for (const n of names) {
      series[n] = half > 0 && n !== "t" && n !== "balancing" ? movingAverage(w[n], half)
                : this.frozen ? w[n].slice() : w[n];
    }
    this.flags = series.balancing;
    for (const c of this.charts) c.u.setData([series.t, ...c.def.series.map((s) => series[s.ch])]);
  }
}
