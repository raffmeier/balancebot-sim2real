// Drive commands from the keyboard or the on-screen pad, repeated while driving is allowed.
// When they stop, the robot falls back to 0 (COMMAND_TIMEOUT_MS in telemetry.h); the policy ramps
// the command, so key presses don't jerk it.

const SEND_PERIOD_MS = 40;  // 25 Hz, well inside the robot's 300 ms timeout

// key -> [forward, turn left]
const KEYS = {
  w: [1, 0], arrowup: [1, 0], s: [-1, 0], arrowdown: [-1, 0],
  a: [0, 1], arrowleft: [0, 1], d: [0, -1], arrowright: [0, -1],
};

const clamp1 = (v) => Math.max(-1, Math.min(1, v));

export class Drive {
  // els: enable (checkbox), pad, knob, vMax, wMax (range inputs); send(msg) to the bridge
  constructor(els, send) {
    this.els = els;
    this.send = send;
    this.enabled = false;
    this.active = true;      // false while replaying a log
    this.keys = new Set();
    this.pad = null;         // [x, y] in [-1, 1] while the pad is held
    this.sending = false;
    this.sent = null;

    els.enable.onchange = () => { this.enabled = els.enable.checked; };
    addEventListener("keydown", (e) => this.onKey(e, true));
    addEventListener("keyup", (e) => this.onKey(e, false));
    addEventListener("blur", () => this.keys.clear());  // keyup is lost when the window loses focus
    this.setupPad();
    setInterval(() => this.tick(), SEND_PERIOD_MS);
  }

  get allowed() {
    return this.enabled && this.active && !document.hidden;
  }

  onKey(e, down) {
    const key = e.key.toLowerCase();
    if (!(key in KEYS) || e.target.closest("select, input[type=text]")) return;
    if (!down) {
      this.keys.delete(key);
      return;
    }
    if (!this.allowed || e.ctrlKey || e.altKey || e.metaKey) return;
    e.preventDefault();  // also keeps the arrow keys from moving a focused slider
    this.keys.add(key);
  }

  setupPad() {
    const { pad, knob } = this.els;
    const move = (e) => {
      const r = pad.getBoundingClientRect();
      let x = (e.clientX - r.left) / r.width * 2 - 1;
      let y = (e.clientY - r.top) / r.height * 2 - 1;
      const len = Math.hypot(x, y);
      if (len > 1) { x /= len; y /= len; }
      this.pad = [x, y];
      knob.style.transform = `translate(${x * 75}%, ${y * 75}%)`;  // knob center up to the pad edge
    };
    const release = () => {
      this.pad = null;
      knob.style.transform = "";
    };
    pad.addEventListener("pointerdown", (e) => {
      if (!this.allowed) return;
      pad.setPointerCapture(e.pointerId);
      move(e);
    });
    pad.addEventListener("pointermove", (e) => { if (this.pad) move(e); });
    pad.addEventListener("pointerup", release);
    pad.addEventListener("pointercancel", release);
  }

  // [forward, turn left] in [-1, 1]: the pad while held, else the keys
  input() {
    if (this.pad) return [-this.pad[1], -this.pad[0]];  // up = forward, left = turn left
    let fwd = 0, turn = 0;
    for (const k of this.keys) {
      fwd += KEYS[k][0];
      turn += KEYS[k][1];
    }
    return [clamp1(fwd), clamp1(turn)];
  }

  command() {
    const [fwd, turn] = this.input();
    return { v: fwd * +this.els.vMax.value, yaw_rate: turn * +this.els.wMax.value };
  }

  tick() {
    if (!this.allowed) {
      if (this.sending) this.send({ type: "drive", v: 0, yaw_rate: 0 });  // stop now, not after the timeout
      this.sending = false;
      this.sent = null;
      this.keys.clear();
      if (this.pad) this.els.pad.dispatchEvent(new PointerEvent("pointercancel"));
      return;
    }
    this.sent = this.command();
    this.send({ type: "drive", ...this.sent });
    this.sending = true;
  }
}
