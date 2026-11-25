// What the views show: the live stream or a log played back. Both give a store and the time to
// show, advanced once per animation frame.

const LIVE_DELAY = 0.06;  // s behind the newest packet: packets come in bursts of 8 (40 ms)

export class LiveSource {
  constructor(store) {
    this.store = store;
    this.live = true;
    this.t = null;
  }

  // Follows the newest packet with a fixed delay, so the 3D motion is smooth
  advance(dt) {
    const latest = this.store.latestT;
    if (latest === null) return (this.t = null);
    const target = latest - LIVE_DELAY;
    if (this.t === null || Math.abs(target - this.t) > 0.5) {
      this.t = target;
    } else {
      this.t += dt + 0.05 * (target - this.t);
    }
    return this.t;
  }

  // The charts end at the newest packet
  get chartT() {
    return this.store.latestT;
  }
}

export class ReplaySource {
  constructor(store, name) {
    this.store = store;
    this.name = name;
    this.live = false;
    this.t = store.firstT;
    this.playing = false;
    this.speed = 1;
  }

  get start() { return this.store.firstT; }
  get end() { return this.store.latestT; }

  advance(dt) {
    if (this.playing) {
      this.t += dt * this.speed;
      if (this.t >= this.end) {
        this.t = this.end;
        this.playing = false;
      }
    }
    return this.t;
  }

  seek(t) {
    this.t = Math.min(Math.max(t, this.start), this.end);
  }

  toggle() {
    if (!this.playing && this.t >= this.end) this.t = this.start;
    this.playing = !this.playing;
  }

  get chartT() {
    return this.t;
  }
}
