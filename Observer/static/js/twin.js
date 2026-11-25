// 3D view of the robot, built from balancebot.xml (via /api/model) and posed from telemetry.
// MuJoCo frame throughout: z up, robot forward = +x, left = +y.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { STLLoader } from "/static/vendor/STLLoader.js";

// Telemetry channel driving each joint of the model
const JOINT_CHANNELS = { left_wheel_joint: "wheel_angle_l", right_wheel_joint: "wheel_angle_r" };
const POSE_CHANNELS = ["x", "y", "yaw", "pitch_rad", "speed", "gyro_z", "v_cmd", "yaw_cmd", "v_ref", "yaw_ref",
                       "balancing", "disarmed", ...Object.values(JOINT_CHANNELS)];

const TRAIL_S = 30;           // path shown behind the robot
const TRAIL_POINTS = 2000;
const TRAIL_SPACING = 0.01;   // m between trail points
const FLOOR_SIZE = 20;        // m
const TILE = 0.25;            // m, checker square

// Arrows like BalanceRobotEnv.render: horizontal = forward speed, vertical = yaw rate
const V_SCALE = 0.5;          // m arrow per m/s
const YAW_SCALE = 0.05;       // m arrow per rad/s
const CMD_COLOR = 0x2ca02c;
const ACTUAL_COLOR = 0xd62728;

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

function srgb(r, g, b) {
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}

// MuJoCo's checker grid (scene.xml "grid" texture): two blues with light cross marks
function checkerTexture() {
  const n = 256, c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d");
  const a = "rgb(51,77,102)", b = "rgb(77,102,128)";
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    ctx.fillStyle = (i + j) % 2 ? b : a;
    ctx.fillRect(i * n / 2, j * n / 2, n / 2, n / 2);
  }
  ctx.strokeStyle = "rgba(204,204,204,0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(n / 2, 0); ctx.lineTo(n / 2, n);
  ctx.moveTo(0, n / 2); ctx.lineTo(n, n / 2);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(FLOOR_SIZE / (2 * TILE), FLOOR_SIZE / (2 * TILE));
  return tex;
}

// Skybox gradient like the model's (rgb1 top, rgb2 horizon)
function skyTexture() {
  const c = document.createElement("canvas");
  c.width = 2; c.height = 256;
  const ctx = c.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, "rgb(128,179,255)");
  g.addColorStop(1, "rgb(242,242,255)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function arrow(color) {
  const a = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.1, color, 0.025, 0.015);
  a.visible = false;
  return a;
}

// Points `a` along dir with signed length (hidden when ~0)
function setArrow(a, origin, dir, length) {
  a.visible = Math.abs(length) > 0.005;
  if (!a.visible) return;
  a.position.copy(origin);
  a.setDirection(length > 0 ? dir : dir.clone().negate());
  a.setLength(Math.abs(length), 0.025, 0.015);
}

// Command shown by the arrows: what was sent (telemetry v2), else the policy's reference
const pick = (a, b) => (isFinite(a) ? a : isFinite(b) ? b : 0);

export class Twin {
  constructor(container, overlay) {
    this.container = container;
    this.overlay = overlay;
    this.follow = true;
    this.joints = {};
    this.root = null;        // the free body (chassis)
    this.origin = { x: 0, y: 0, yaw: 0 };  // odometry pose shown at the world origin

    const renderer = this.renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.prepend(renderer.domElement);

    const scene = this.scene = new THREE.Scene();
    scene.background = skyTexture();
    scene.fog = new THREE.Fog(srgb(0.95, 0.95, 1), 3, 9);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
    this.camera.position.set(-0.45, -0.5, 0.32);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.target.set(0, 0, 0.08);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02;  // stay above the floor
    this.controls.update();

    scene.add(new THREE.HemisphereLight(srgb(0.85, 0.9, 1), srgb(0.3, 0.35, 0.4), 1.2));
    const sun = this.sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -0.6; sc.right = sc.top = 0.6; sc.near = 0.1; sc.far = 4;
    sun.shadow.bias = -0.0005;
    scene.add(sun, sun.target);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(FLOOR_SIZE, FLOOR_SIZE),
                                 new THREE.MeshStandardMaterial({ map: checkerTexture(), roughness: 0.95 }));
    floor.material.map.anisotropy = renderer.capabilities.getMaxAnisotropy();
    floor.receiveShadow = true;
    scene.add(floor);

    // Path driven (odometry)
    const trailGeom = new THREE.BufferGeometry();
    trailGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_POINTS * 3), 3));
    trailGeom.setDrawRange(0, 0);
    this.trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0xf2c94c }));
    scene.add(this.trail);

    this.arrows = { vCmd: arrow(CMD_COLOR), vActual: arrow(ACTUAL_COLOR),
                    yawCmd: arrow(CMD_COLOR), yawActual: arrow(ACTUAL_COLOR) };
    scene.add(...Object.values(this.arrows));

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  async load() {
    const model = await (await fetch("/api/model")).json();
    const loader = new STLLoader();
    const geoms = {};
    await Promise.all(Object.entries(model.meshes).map(async ([name, m]) => {
      const g = await loader.loadAsync(m.url);
      g.scale(...m.scale);
      geoms[name] = g;
    }));
    for (const body of model.bodies) this.scene.add(this.buildBody(body, geoms));
  }

  buildBody(def, geoms) {
    const group = new THREE.Group();
    group.name = def.name;
    group.position.fromArray(def.pos);
    for (const g of def.geoms) {
      const [r, gr, b, a] = g.rgba;
      const mesh = new THREE.Mesh(geoms[g.mesh], new THREE.MeshStandardMaterial({
        color: srgb(r, gr, b), roughness: 0.55, metalness: 0.05, transparent: a < 1, opacity: a }));
      mesh.position.fromArray(g.pos);
      mesh.rotation.set(...g.euler);
      mesh.castShadow = true;
      group.add(mesh);
    }
    if (def.free) {
      this.root = group;
      this.rootZ = def.pos[2];
    }
    if (def.joint && JOINT_CHANNELS[def.joint.name]) {
      this.joints[def.joint.name] = { group, axis: new THREE.Vector3(...def.joint.axis) };
    }
    for (const c of def.children) group.add(this.buildBody(c, geoms));
    return group;
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // The robot's current pose becomes the origin, heading +x
  resetOrigin(store, t) {
    const s = t === null ? null : store.sample(t, ["x", "y", "yaw"]);
    this.origin = s ?? { x: 0, y: 0, yaw: 0 };
  }

  // Odometry pose -> scene, relative to the origin
  toScene(x, y, yaw) {
    const o = this.origin, dx = x - o.x, dy = y - o.y, c = Math.cos(o.yaw), sn = Math.sin(o.yaw);
    return { x: c * dx + sn * dy, y: -sn * dx + c * dy, yaw: yaw - o.yaw };
  }

  // Draws the robot as it was at time t of the store
  render(store, t) {
    const s = t === null ? null : store.sample(t, POSE_CHANNELS);
    if (s && this.root) {
      this.pose(s);
      this.updateTrail(store, t);
    }
    this.overlay.hidden = !s || s.balancing > 0.5;
    if (s) this.overlay.textContent = s.disarmed > 0.5 ? "Motors switched off" : "Motors off";
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  pose(s) {
    const p = this.toScene(s.x, s.y, s.yaw);
    s = { ...s, ...p };
    const base = new THREE.Vector3(s.x, s.y, this.rootZ);
    const before = this.root.position.clone();
    this.root.position.copy(base);
    const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), s.yaw);
    const pitch = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), s.pitch_rad);
    this.root.quaternion.copy(yaw).multiply(pitch);
    for (const [name, j] of Object.entries(this.joints)) {
      j.group.quaternion.setFromAxisAngle(j.axis, s[JOINT_CHANNELS[name]]);
    }

    if (this.follow) {
      const delta = base.clone().sub(before);
      this.camera.position.add(delta);
      this.controls.target.add(delta);
    }
    this.sun.position.set(base.x + 0.8, base.y - 0.5, 2.0);
    this.sun.target.position.copy(base);

    const forward = new THREE.Vector3(Math.cos(s.yaw), Math.sin(s.yaw), 0);
    const up = new THREE.Vector3(0, 0, 1);
    const top = base.clone().add(new THREE.Vector3(0, 0, 0.14));
    const back = top.clone().addScaledVector(forward, -0.08);
    setArrow(this.arrows.vCmd, top, forward, pick(s.v_cmd, s.v_ref) * V_SCALE);
    setArrow(this.arrows.vActual, top.clone().add(new THREE.Vector3(0, 0, 0.02)), forward, s.speed * V_SCALE);
    setArrow(this.arrows.yawCmd, back, up, pick(s.yaw_cmd, s.yaw_ref) * YAW_SCALE);
    setArrow(this.arrows.yawActual, back.clone().addScaledVector(forward, -0.02), up, s.gyro_z * YAW_SCALE);
  }

  // Path of the last TRAIL_S seconds up to t, one point per TRAIL_SPACING
  updateTrail(store, t) {
    const w = store.window(TRAIL_S, ["x", "y"], t);
    const pos = this.trail.geometry.attributes.position.array;
    let n = 0, lastX = Infinity, lastY = Infinity;
    // Newest first, so the points kept are the recent ones when there are too many
    for (let i = w.x.length - 1; i >= 0 && n < TRAIL_POINTS; i--) {
      const p = this.toScene(w.x[i], w.y[i], 0);
      if (Math.hypot(p.x - lastX, p.y - lastY) < TRAIL_SPACING) continue;
      pos[3 * n] = lastX = p.x;
      pos[3 * n + 1] = lastY = p.y;
      pos[3 * n + 2] = 0.001;
      n++;
    }
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, n);
  }
}
