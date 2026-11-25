import { CONTROL_HZ, PacketDecoder, TelemetryStore } from "./telemetry.js";
import { LiveSource, ReplaySource } from "./source.js";
import { ChartGrid } from "./charts.js";
import { Twin } from "./twin.js";
import { Drive } from "./drive.js";

const LIVE_BUFFER_S = 60;
const CHART_PERIOD_MS = 50;   // chart redraw
const STATUS_PERIOD_MS = 200; // tiles, badges, policy input
const STALE_MS = 500;         // no packet for this long: robot shown as not streaming
const RAD2DEG = 180 / Math.PI;

// Network input order (BalanceRobotEnv._get_obs, weights/model.h)
const POLICY_INPUTS = ["pitch", "pitch rate", "yaw rate", "wheel vel L", "wheel vel R",
                       "v ref", "yaw-rate ref", "prev action L", "prev action R"];
const STATUS_CHANNELS = ["pitch", "accel_pitch", "speed", "gyro_z", "balancing", "imu_ok", "remote", "disarmed",
                         "v_cmd", "yaw_cmd", ...POLICY_INPUTS.map((_, i) => `policy_in_${i}`)];

const $ = (id) => document.getElementById(id);

// --- settings remembered in this browser ---

const SETTINGS_KEY = "observer-settings";
const settings = (() => {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) ?? {}; } catch { return {}; }
})();
function remember(key, value) {
  settings[key] = value;
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* private mode */ }
}
for (const [id, key] of [["window", "window"], ["smoothing", "smoothing"], ["v-max", "vMax"], ["w-max", "wMax"]]) {
  if (settings[key] !== undefined) $(id).value = settings[key];
}
if (settings.follow !== undefined) $("follow").checked = settings.follow;

// --- data sources ---

const liveStore = new TelemetryStore(LIVE_BUFFER_S * CONTROL_HZ);
const live = new LiveSource(liveStore);
let source = live;
let liveOrigin = null;  // 3D origin of the live view while a log is shown

const charts = new ChartGrid($("charts"), source);
charts.windowS = +$("window").value;
charts.smoothS = +$("smoothing").value;
const twin = new Twin($("twin"), $("twin-off"));
twin.follow = $("follow").checked;
twin.load().catch((e) => console.error("3D model:", e));

// --- bridge connection ---

let ws = null;
let decoder = null;
let link = null;
let lastPacketMs = 0;

function send(msg) {
  if (ws?.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(msg));
  return true;
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onclose = () => {
    decoder = null;
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      if (!decoder) return;
      for (const p of decoder.decode(ev.data)) liveStore.push(p);
      lastPacketMs = performance.now();
      return;
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "hello") {
      decoder = new PacketDecoder(msg);
      liveStore.obsScale = msg.obs_scale;
    } else if (msg.type === "link") {
      link = msg;
    } else if (msg.type === "saved") {
      notice(msg.name ? `Saved RL_training/logs/${msg.name}` : "Nothing recorded (no packets)", msg.name);
      refreshLogs(msg.name);
    }
  };
}
connect();

// --- drive and motors ---

const drive = new Drive({ enable: $("drive-enable"), pad: $("pad"), knob: $("knob"),
                          vMax: $("v-max"), wMax: $("w-max") }, send);

function motors(on) {
  send({ type: "motors", on });
}
$("motors-off").onclick = () => motors(false);
$("motors-arm").onclick = () => motors(true);
for (const [id, key] of [["v-max", "vMax"], ["w-max", "wMax"]]) {
  $(id).oninput = () => { remember(key, $(id).value); updateLimits(); };
}
function updateLimits() {
  $("v-max-out").textContent = `${(+$("v-max").value).toFixed(2)} m/s`;
  $("w-max-out").textContent = `${(+$("w-max").value).toFixed(2)} rad/s`;
}
updateLimits();

// --- recording and replay ---

$("record").onclick = () => send({ type: "record", on: !link?.recording });

async function refreshLogs(selectName = null) {
  const select = $("log-select");
  const current = selectName ?? select.value;
  let logs = [];
  try { logs = await (await fetch("/api/logs")).json(); } catch { /* bridge offline */ }
  select.replaceChildren(...logs.map((l) => {
    const o = document.createElement("option");
    o.value = l.name;
    o.textContent = `${l.name.replace(".npy", "")} · ${l.duration.toFixed(0)} s`;
    return o;
  }));
  if (!logs.length) select.append(new Option("no logs yet", ""));
  if (logs.some((l) => l.name === current)) select.value = current;
}
refreshLogs();

async function openLog(name) {
  if (!name) return;
  const button = $("log-open");
  button.disabled = true;
  button.textContent = "Loading…";
  try {
    const meta = await (await fetch(`/api/logs/${encodeURIComponent(name)}`)).json();
    const data = await (await fetch(`/api/logs/${encodeURIComponent(name)}/data`)).arrayBuffer();
    const store = new TelemetryStore(meta.packets, { rolling: false, obsScale: meta.obs_scale });
    for (const p of new PacketDecoder(meta).decode(data)) store.push(p);
    if (store.len < 2) throw new Error("log has no packets");
    if (source.live) liveOrigin = twin.origin;
    setSource(new ReplaySource(store, name));
    twin.resetOrigin(store, store.firstT);
  } catch (e) {
    notice(`Could not open ${name}: ${e.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "Open";
  }
}
$("log-open").onclick = () => openLog($("log-select").value);

function backToLive() {
  setSource(live);
  if (liveOrigin) twin.origin = liveOrigin;
}
$("back-live").onclick = backToLive;

function setSource(s) {
  source = s;
  charts.source = s;
  drive.active = s.live;
  $("replay").hidden = s.live;
  $("drive").classList.toggle("disabled", !s.live);
  if (!s.live) {
    $("replay-name").textContent = s.name.replace(".npy", "");
    $("scrub").min = s.start;
    $("scrub").max = s.end;
    drawTimeline();
  }
  charts.update(true);
  updateStatus();
}

// Off bands of the whole log under the scrub bar, to find the falls
function drawTimeline() {
  if (source.live) return;
  const canvas = $("timeline-strip"), store = source.store;
  const w = canvas.clientWidth * devicePixelRatio, h = canvas.clientHeight * devicePixelRatio;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const style = getComputedStyle(document.documentElement);
  ctx.fillStyle = style.getPropertyValue("--gridline").trim();
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = style.getPropertyValue("--status-critical").trim();
  const t = store.data.t, on = store.data.balancing, span = source.end - source.start;
  for (let px = 0; px < w; px++) {
    const i0 = store.indexAt(source.start + span * px / w);
    const i1 = Math.max(store.indexAt(source.start + span * (px + 1) / w), i0 + 1);
    let off = false;
    for (let i = i0; i < Math.min(i1, store.len) && !off; i++) off = !on[i];
    if (off) ctx.fillRect(px, 0, 1, h);
  }
}
new ResizeObserver(drawTimeline).observe($("timeline-strip"));

let scrubbing = false;
$("scrub").addEventListener("pointerdown", () => { scrubbing = true; });
$("scrub").addEventListener("pointerup", () => { scrubbing = false; });
$("scrub").oninput = () => { if (!source.live) source.seek(+$("scrub").value); };
$("play").onclick = () => { if (!source.live) source.toggle(); };
$("speed").onchange = () => { if (!source.live) source.speed = +$("speed").value; };

// --- toolbar ---

const freezeButton = $("freeze");
function setFrozen(frozen) {
  charts.setFrozen(frozen);
  freezeButton.setAttribute("aria-pressed", frozen);
  freezeButton.textContent = frozen ? "Resume charts" : "Freeze charts";
}
freezeButton.onclick = () => setFrozen(!charts.frozen);
$("window").onchange = () => {
  charts.windowS = +$("window").value;
  remember("window", $("window").value);
  charts.update(true);
};
$("smoothing").onchange = () => {
  charts.smoothS = +$("smoothing").value;
  remember("smoothing", $("smoothing").value);
  charts.update(true);
};
$("follow").onchange = (e) => {
  twin.follow = e.target.checked;
  remember("follow", twin.follow);
};
$("reset-origin").onclick = () => twin.resetOrigin(source.store, source.t);

let noticeTimer = null;
function notice(text, logName = null) {
  const el = $("notice");
  el.hidden = false;
  el.querySelector(".text").textContent = text;
  const open = el.querySelector("button");
  open.hidden = !logName;
  open.onclick = () => { el.hidden = true; openLog(logName); };
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { el.hidden = true; }, 8000);
}

// A focused button or checkbox reacts to Space on key-up: never also press Arm
addEventListener("keyup", (e) => { if (e.key === " " && !e.target.closest("input[type=text]")) e.preventDefault(); });

// Space: motors off, always. F: freeze charts. Replay: K play/pause, J / L 1 s back / forward,
// comma / period one control step
addEventListener("keydown", (e) => {
  if (e.target.closest("select, input[type=text]") || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === " ") {
    e.preventDefault();  // not also a click on a focused button
    if (!e.repeat) motors(false);
    return;
  }
  if (e.repeat && !",.".includes(e.key)) return;
  const k = e.key.toLowerCase();
  if (k === "f") setFrozen(!charts.frozen);
  if (source.live) return;
  if (k === "k") source.toggle();
  if (k === "j") source.seek(source.t - 1);
  if (k === "l") source.seek(source.t + 1);
  if (k === ",") source.seek(source.t - 1 / CONTROL_HZ);
  if (k === ".") source.seek(source.t + 1 / CONTROL_HZ);
});

// --- status ---

function badge(el, state, text) {
  el.dataset.state = state;  // good / bad / idle: color plus icon, see style.css
  el.querySelector(".label").textContent = text;
}

function tile(id, value, detail = "") {
  $(id).querySelector(".value").textContent = value;
  $(id).querySelector(".detail").textContent = detail;
}

function buildPolicyInputs() {
  const list = $("policy-in");
  for (const name of POLICY_INPUTS) {
    const row = document.createElement("div");
    row.className = "meter";
    row.innerHTML = '<span class="name"></span><span class="track"><span class="bar"></span></span><span class="num"></span>';
    row.querySelector(".name").textContent = name;
    list.append(row);
  }
}
buildPolicyInputs();

// Bars from 0, the track spans [-1.5, 1.5] with ticks at ±1 (the training range of most inputs)
function updatePolicyInputs(s) {
  const rows = $("policy-in").children;
  for (let i = 0; i < rows.length; i++) {
    const v = s ? s[`policy_in_${i}`] : NaN;
    const row = rows[i], bar = row.querySelector(".bar");
    const c = Math.max(-1.5, Math.min(1.5, isFinite(v) ? v : 0)) / 1.5 * 50;
    bar.style.left = `${50 + Math.min(c, 0)}%`;
    bar.style.width = `${Math.abs(c)}%`;
    row.classList.toggle("over", Math.abs(v) > 1);
    row.querySelector(".num").textContent = isFinite(v) ? (Math.abs(v) > 1 ? "⚠ " : "") + v.toFixed(2) : "–";
  }
}

const fmtCmd = (v, w) => (isFinite(v) ? `${v.toFixed(2)} m/s · ${w.toFixed(2)} rad/s` : "–");
const fmtTime = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

function updateStatus() {
  const bridgeUp = ws?.readyState === WebSocket.OPEN;
  const streaming = performance.now() - lastPacketMs < STALE_MS;
  badge($("badge-bridge"), bridgeUp ? "good" : "bad", bridgeUp ? "Bridge" : "Bridge offline");
  if (!bridgeUp) badge($("badge-robot"), "idle", "Robot –");
  else if (streaming) badge($("badge-robot"), "good", `Robot ${link?.robot ?? ""}`);
  else badge($("badge-robot"), "bad", link?.robot ? `Robot ${link.robot}: no data` : "Searching for robot…");

  // Everything below shows the source: the live robot or the log at the replay time
  const t = source.chartT;
  const has = source.live ? streaming : true;
  const s = has && t !== null ? source.store.sample(t, STATUS_CHANNELS) : null;
  const on = (ch) => s && s[ch] > 0.5;
  badge($("badge-balancing"), !s ? "idle" : on("balancing") ? "good" : "bad",
        !s ? "Controller –" : on("balancing") ? "Balancing" : on("disarmed") ? "Motors switched off" : "Motors off");
  badge($("badge-imu"), !s ? "idle" : on("imu_ok") ? "good" : "bad", !s ? "IMU –" : on("imu_ok") ? "IMU ok" : "IMU error");

  const err = link?.version_error;
  $("version-error").hidden = !err;
  if (err) $("version-error").textContent = `${err} (flash the current firmware, or update robot_log.py)`;
  $("scale-note").hidden = source.store.obsScale !== null;

  const st = s ? source.store.stats(1.0, t) : null;
  const loss = s ? source.store.stats(5.0, t) : null;
  tile("tile-pitch", s ? s.pitch.toFixed(1) + "°" : "–", s ? `accel ${s.accel_pitch.toFixed(1)}°` : "");
  tile("tile-speed", s ? s.speed.toFixed(2) : "–", "m/s, odometry");
  tile("tile-yaw", s ? s.gyro_z.toFixed(2) : "–", "rad/s");
  tile("tile-control", st ? st.controlHz.toFixed(0) : "–", "Hz control loop");
  tile("tile-foc", st ? (st.focHz / 1000).toFixed(1) : "–", "kHz FOC loop");
  tile("tile-compute", st ? st.computeMean.toFixed(0) : "–", st ? `µs per step, max ${st.computeMax}` : "µs per step");
  tile("tile-loss", loss ? (loss.loss * 100).toFixed(1) + "%" : "–",
       `lost (5 s) · total ${source.store.lost}` + (source.live ? ` · CRC ${link?.crc_errors ?? 0}` : ""));
  updatePolicyInputs(s);

  // Drive panel: always about the live robot
  const ls = streaming ? liveStore.sample(liveStore.latestT, STATUS_CHANNELS) : null;
  const disarmed = ls && ls.disarmed > 0.5;
  badge($("motors-state"), !ls ? "idle" : disarmed ? "bad" : "good",
        !ls ? "Robot –" : disarmed ? "Switched off: Arm to re-enable" : "Armed");
  $("cmd-sent").textContent = drive.sent ? fmtCmd(drive.sent.v, drive.sent.yaw_rate) : "nothing (driving off)";
  $("cmd-robot").textContent = !ls ? "–" : ls.remote > 0.5 ? fmtCmd(ls.v_cmd, ls.yaw_cmd)
                             : isFinite(ls.v_cmd) ? "0 (no command)" : "– (firmware without commands)";

  const rec = link?.recording;
  $("record").setAttribute("aria-pressed", !!rec);
  $("record").textContent = rec
    ? `■ Stop recording ${fmtTime((Date.now() - Date.parse(rec.started)) / 1000)}`
    : "● Record";
  $("record").disabled = !bridgeUp;
}

// --- animation ---

let lastFrame = performance.now();
function frame(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  const t = source.advance(dt);
  twin.render(source.store, t);
  if (!source.live) {
    if (!scrubbing) $("scrub").value = t;
    $("replay-time").textContent = `${fmtTime(t - source.start)} / ${fmtTime(source.end - source.start)}`;
    $("play").textContent = source.playing ? "❚❚ Pause" : "▶ Play";
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

setInterval(() => charts.update(), CHART_PERIOD_MS);
setInterval(updateStatus, STATUS_PERIOD_MS);
updateStatus();
