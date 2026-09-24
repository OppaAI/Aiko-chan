/**
 * vrm.js
 * Three.js VRM loader and real-time skeletal animation engine.
 * Loads Aiko.vrm, manages idle breathing/sway, gesture library (20+ animations),
 * thinking poses (chin-think, hand-near-mouth, etc.), and mouth shapes via blendshapes.
 * Expression/viseme/pose updates driven by window.aikoSetX() callbacks from WebSocket.
 *
 * Animation layers (additive — summed onto REST every frame):
 *   (a) vitality: breathing (3.5 s), micro-sway (incommensurate freqs),
 *       blink (120 ms), eye saccades — always on
 *   (b) state pose: idle | thinking | listening | speaking, eased over 0.4 s
 *   (c) event gestures: 40+ short motions; envelopes start/end at exactly 0
 *   - mouth: vowel-based visemes (aa, ih, ou, ee, oh) + RMS-driven amplitude
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// ── renderer ────────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const vrmSide = document.getElementById('vrm-side');
const fill = document.getElementById('progress-fill');
const loadMsg = document.getElementById('load-msg');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
// Transparent companion shell (Tauri, flagged by companion.js before this
// deferred module runs) needs a fully clear canvas so the desktop shows
// through; browsers keep the classic opaque backdrop.
if (window.aikoIsTauri) renderer.setClearColor(0x000000, 0);
else renderer.setClearColor(0x0a0a0f);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(8, 1, 0.1, 100);
camera.position.set(0.00, 1.36, 5.0);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0.00, 1.33, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;
controls.minDistance = 1.0;
controls.maxDistance = 3.2;
controls.update();

scene.add(new THREE.AmbientLight(0xc8b0ff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(1, 3, 2);
scene.add(dir);
const rim = new THREE.DirectionalLight(0x7b4fd4, 0.4);
rim.position.set(-2, 1, -1);
scene.add(rim);
const fillL = new THREE.DirectionalLight(0xd4b0ff, 0.3);
fillL.position.set(0, -1, 2);
scene.add(fillL);
scene.add(new THREE.GridHelper(10, 20, 0x1a0a2a, 0x100820));

function resize() {
  const w = vrmSide.clientWidth;
  const h = vrmSide.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
window.addEventListener('resize', resize);

// ── VRM state ──────────────────────────────────────────────────────────────
let vrm = null;
const clock = new THREE.Clock();

// ── expression state (cubic crossfades, 0.8 cap, 4 s auto-reset) ──────────
const exprTargets = {};
const exprCurrent = {};
const exprAnim = {}; // name -> { from, to, t, dur }
const MOUTH_KEYS = new Set(['aa', 'ih', 'ou', 'ee', 'oh']);
let explicitExpr = false; // true while aikoSetExpression holds a non-neutral emotion
let exprResetTimer = null;
const EXPR_RESET_DELAY = 4000;

// ── blink: single 120 ms sin(π·t) curve, every 2–6 s (slower while thinking) ─
let blinkWait = 2.5;
let blinkT2 = 0;
let blinking = false;
const BLINK_DUR = 0.12;

// ── eye saccades: small look-target jumps every 0.8–2.5 s, eased ─────────
let sacT = 1.2;
const sacTarget = { x: 0, y: 0 };
const sacCur = { x: 0, y: 0 };

let t = 0;
const TAU = Math.PI * 2;
const REST = {
  leftUpperArm: { x: -0.02, y: 0.00, z: 1.28 },
  rightUpperArm: { x: -0.02, y: 0.00, z: -1.28 },
  leftLowerArm: { x: 0.12, y: 0.00, z: 0.08 },
  rightLowerArm: { x: 0.12, y: 0.00, z: -0.08 },
  leftHand: { x: 0.00, y: 0.08, z: 0.00 },
  rightHand: { x: 0.00, y: -0.08, z: 0.00 },
};

const FINGER_BONES = [
  'leftThumbProximal', 'leftThumbDistal', 'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
  'leftMiddleProximal', 'leftMiddleIntermediate', 'leftRingProximal', 'leftLittleProximal',
  'rightThumbProximal', 'rightThumbDistal', 'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
  'rightMiddleProximal', 'rightMiddleIntermediate', 'rightRingProximal', 'rightLittleProximal',
];

function getBone(name) {
  return vrm?.humanoid?.getRawBoneNode(name) ?? null;
}

// ── gesture state ─────────────────────────────────────────────────────────────
let gestureState = 'none';
let gestureT = 0;
let gestureDuration = 0;
let gestureCooldown = 2.0 + Math.random() * 3.0;
let gestureTarget = {};
let gestureLookAt = 0; // lookAtHand head-tracking: -1 left, +1 right, 0 off
let lastVisemeAt = 0;

// ── state pose machine: idle | thinking | listening | speaking ─────────
let thinkingActive = false;
let listeningActive = false;
let thinkingSide = 1; // head-tilt side, re-picked on each activation
const stateBlend = { thinking: 0, listening: 0, speaking: 0 }; // eased 0.4 s

// ── layered pose offsets — rebuilt every frame, SUMMED onto REST ────────
//   (a) vitality: breathing / sway / saccades (always on)
//   (b) statePose: idle | thinking | listening | speaking
//   (c) gesturePose: short events; envelopes start AND end at exactly 0,
//       so the layer always returns to additive-zero — idle resumes seamlessly.
let vitality = {};
let statePose = {};
let gesturePose = {};
let fingerPose = {};

// Bones written by composePose (hips handled separately: rotation + translation).
const BLENDABLE_BONES = [
  'head', 'neck', 'spine', 'chest',
  'leftUpperArm', 'rightUpperArm',
  'leftLowerArm', 'rightLowerArm',
  'leftHand', 'rightHand',
  'leftUpperLeg', 'rightUpperLeg', // footTap
];

// ── easing ──────────────────────────────────────────────────────────────────
function easeInOutCubic(v) {
  v = Math.max(0, Math.min(1, v));
  return v < 0.5 ? 4 * v * v * v : 1 - Math.pow(-2 * v + 2, 3) / 2;
}
// Envelope: cubic ease in, hold, cubic ease out. Exactly 0 at both ends.
function holdCurve(p, inP = 0.3, outP = 0.3) {
  if (p <= 0 || p >= 1) return 0;
  if (p < inP) return easeInOutCubic(p / inP);
  if (p > 1 - outP) return easeInOutCubic((1 - p) / outP);
  return 1;
}
// Rhythmic pulse that starts and ends at 0 (integer cycles only).
function pulseCurve(p, cycles) {
  return Math.sin(Math.max(0, Math.min(1, p)) * Math.PI * cycles);
}
// Non-integer-cycle pulse tapered by sin(pi*p): keeps the rhythmic feel but
// still lands at exactly 0 on both ends (no snap).
function pulseTapered(p, cycles) {
  const q = Math.max(0, Math.min(1, p));
  return Math.sin(q * Math.PI * cycles) * Math.sin(q * Math.PI);
}

const GESTURES = [
  'lookAround',
  'lookAtHand',
  'hairBrush',
  'fingerPlay',
  'meetGaze',
  'curiousTilt',
  'shiftWeight',
  'stretchNeck',
  'raiseHand',
  'chinTouch',
  'shoulderRoll',
  'sway',
  'headNod',
  'wristFlick',
  'adjustSleeve',
  'handOnHip',
  'crossArms',
  'touchCollar',
  'brushShoulder',
  'stretchArm',
  'leanIn',
  'openPalm',
  'speakingNod',
  // ── new natural gestures ──
  'earTuck',           // tucks hair behind ear
  'gentleStretch',     // subtle chest-opening stretch
  'handsClasp',        // hands clasp together briefly in front
  'thoughtfulLook',    // slow gaze upward with head tilt
  // ── young-girl idle fidgets ──
  'hairTwirl',         // twirls a strand of hair around a finger
  'handsBehindBack',   // clasps hands behind back, sways
  'footTap',           // taps one foot, playful impatience
  'skirtSmooth',       // smooths skirt with both hands
  'hugSelf',           // wraps arms around herself, shy
  'happyBounce',       // little bounce on her toes
];

const SPEAKING_GESTURES = [
  'leanIn', 'openPalm', 'speakingNod', 'curiousTilt', 'wristFlick',
  'emphasizePoint',    // hand rises to emphasize
  'bothHandsExplain',  // both hands open outward
];

// Retuned 1–3.5 s: short enough to feel alive, long enough to read.
const GESTURE_DURATION = {
  lookAround: 3.0,
  lookAtHand: 3.0,
  hairBrush: 2.8,
  fingerPlay: 3.2,
  meetGaze: 2.5,
  curiousTilt: 2.4,
  shiftWeight: 3.0,
  stretchNeck: 2.6,
  raiseHand: 3.0,
  chinTouch: 3.2,
  shoulderRoll: 2.4,
  sway: 3.2,
  headNod: 2.2,
  wristFlick: 2.0,
  adjustSleeve: 2.4,
  handOnHip: 2.8,
  crossArms: 2.6,
  touchCollar: 2.6,
  brushShoulder: 2.2,
  stretchArm: 3.0,
  leanIn: 2.4,
  openPalm: 2.2,
  speakingNod: 2.0,
  earTuck: 2.8,
  gentleStretch: 3.2,
  handsClasp: 2.8,
  thoughtfulLook: 3.0,
  emphasizePoint: 2.2,
  bothHandsExplain: 2.4,
  chinThink: 3.2,
  handNearMouth: 3.0,
  armsFoldThink: 3.2,
  lookUpThink: 3.0,
  contemplativeNod: 3.4,
  tapFinger: 3.5,
  wave: 2.4,
  giggle: 2.2,
  bow: 2.4,
  clap: 2.4,
  dance: 3.5,
  hairTwirl: 3.2,
  handsBehindBack: 3.4,
  footTap: 2.6,
  skirtSmooth: 2.8,
  hugSelf: 3.2,
  happyBounce: 2.4,
};

function startGesture(name) {
  gestureState = name;
  gestureT = 0;
  gestureDuration = GESTURE_DURATION[name] ?? 3.0;
  const side = Math.random() < 0.5 ? -1 : 1;
  gestureTarget = {
    side,
    look: side * (0.28 + Math.random() * 0.22),
    tilt: side * (0.10 + Math.random() * 0.09),
    sway: side * (0.018 + Math.random() * 0.012),
  };
}

function pickGesture() {
  const pool = speakingRecently() ? SPEAKING_GESTURES : GESTURES;
  let gesture = pool[(Math.random() * pool.length) | 0];
  // While thinking, suppress idle look-arounds (she holds her gaze).
  if (thinkingActive && gesture === 'lookAround') gesture = 'curiousTilt';
  startGesture(gesture);
}

function speakingRecently() {
  return performance.now() - lastVisemeAt < 650;
}

// ── (a) vitality: breathing, micro-sway, saccades ────────────────────
// Amplitudes are ~1/3 of the old idle. Every component uses its own
// frequency AND phase — no two body parts ever share a sine.
function computeVitality() {
  const s = (f, ph = 0) => Math.sin(TAU * f * t + ph);
  const BREATH = 1 / 3.5; // 3.5 s breathing period
  // Asymmetric cascade: chest 0.8° leads, spine 0.4° follows ~80 ms later.
  vitality.chest = { x: 0.0140 * s(BREATH), y: 0, z: 0 };
  vitality.spine = { x: 0.0070 * s(BREATH, -TAU * BREATH * 0.08), y: 0, z: 0 };

  vitality.hips = {
    x: 0.0027 * s(0.31, 1.3), y: 0, z: 0.0040 * s(0.23),
    px: 0.0012 * s(0.37, 0.6), py: 0,
  };

  const hy = 0.016 * s(0.41) + 0.007 * s(0.53, 2.1);
  const hz = 0.006 * s(0.19, 1.1);
  const hx = 0.0045 * s(0.47, 0.4);
  vitality.head = { x: hx, y: hy, z: hz };
  vitality.neck = {
    x: hx * 0.4 + 0.003 * s(0.67, 1.7),
    y: hy * 0.3 + 0.004 * s(0.59),
    z: hz * 0.3,
  };

  vitality.leftUpperArm = { x: 0.008 * s(0.71, 0.5), y: 0.004 * s(0.43, 2.2), z: 0.007 * s(0.61, 1.0) };
  vitality.rightUpperArm = { x: 0.008 * s(0.73, 2.8), y: 0.004 * s(0.83, 0.9), z: 0.007 * s(0.79, 1.6) };
  vitality.leftLowerArm = { x: 0.006 * s(0.89, 0.2), y: 0.003 * s(0.97, 1.9), z: 0.005 * s(0.49, 0.9) };
  vitality.rightLowerArm = { x: 0.006 * s(1.02, 1.2), y: 0.003 * s(1.11, 2.5), z: 0.005 * s(0.51, 0.4) };
  vitality.leftHand = { x: 0.005 * s(1.03, 0.3), y: 0.007 * s(1.07, 1.5), z: 0.004 * s(1.13, 2.7) };
  vitality.rightHand = { x: 0.005 * s(1.09, 1.1), y: 0.007 * s(1.17, 0.2), z: 0.004 * s(1.19, 2.0) };
}

function updateSaccades(dt) {
  sacT -= dt;
  if (sacT <= 0) {
    // While thinking her gaze holds up/sideways — damp the saccades.
    const damp = 1 - easeInOutCubic(stateBlend.thinking) * 0.7;
    sacTarget.x = (Math.random() - 0.5) * 0.06 * damp;
    sacTarget.y = (Math.random() - 0.5) * 0.10 * damp;
    sacT = 0.8 + Math.random() * 1.7;
  }
  const k = Math.min(1, dt * 10);
  sacCur.x += (sacTarget.x - sacCur.x) * k;
  sacCur.y += (sacTarget.y - sacCur.y) * k;
}

// ── (b) state pose: idle | thinking | listening | speaking ────────────
function updateStateBlends(dt) {
  const rate = dt / 0.4; // 0.4 s eased transitions
  const targets = {
    thinking: thinkingActive ? 1 : 0,
    listening: listeningActive ? 1 : 0,
    speaking: (!thinkingActive && !listeningActive && speakingRecently()) ? 1 : 0,
  };
  for (const k of Object.keys(stateBlend)) {
    const b = stateBlend[k], tg = targets[k];
    stateBlend[k] = b < tg ? Math.min(tg, b + rate) : Math.max(tg, b - rate);
  }
}

function computeStatePose() {
  const sadd = (b, x = 0, y = 0, z = 0) => {
    let o = statePose[b];
    if (!o) o = statePose[b] = { x: 0, y: 0, z: 0 };
    o.x += x; o.y += y; o.z += z;
  };
  const th = easeInOutCubic(stateBlend.thinking);
  if (th > 0.001) {
    const S = thinkingSide; // ±10° head tilt to a random side, gaze up/sideways
    sadd('head', -0.05 * th, 0.10 * S * th, 0.17 * S * th);
    sadd('neck', -0.02 * th, 0.04 * S * th, 0.06 * S * th);
    sadd('spine', 0.020 * th, 0, 0);
    sadd('chest', 0.025 * th, 0, 0);
    // Subtle hand-to-chin drift (chinThink bone targets × 0.6).
    const UA = S > 0 ? 'rightUpperArm' : 'leftUpperArm';
    const LA = S > 0 ? 'rightLowerArm' : 'leftLowerArm';
    const HD = S > 0 ? 'rightHand' : 'leftHand';
    sadd(UA, -0.12 * th, 0, S * 0.14 * th);
    sadd(LA, -0.22 * th, 0, S * 0.024 * th);
    sadd(HD, -0.048 * th, -S * 0.048 * th, 0);
    const prefix = S > 0 ? 'right' : 'left';
    for (const name of FINGER_BONES) {
      if (name.startsWith(prefix)) fingerPose[name] = (fingerPose[name] || 0) + 0.18 * th;
    }
  }
  const li = easeInOutCubic(stateBlend.listening);
  if (li > 0.001) { // leaning in: the universal "I'm listening."
    sadd('spine', -0.025 * li, 0, 0);
    sadd('chest', -0.018 * li, 0, 0);
    sadd('head', 0.012 * li, 0, 0);
    sadd('neck', 0.008 * li, 0, 0);
  }
  const sp = easeInOutCubic(stateBlend.speaking);
  if (sp > 0.001) { // slight lift while talking
    sadd('spine', -0.012 * sp, 0, 0);
    sadd('chest', -0.008 * sp, 0, 0);
    sadd('head', -0.008 * sp, 0, 0);
  }
}

function applyIdle(dt) {
  if (!vrm?.humanoid) return;
  vitality = {};
  statePose = {};
  fingerPose = {};
  computeVitality();      // (a) vitality
  updateStateBlends(dt);
  computeStatePose();     // (b) state pose
  updateSaccades(dt);
}

function applyFingerCurl(side, intensity) {
  const prefix = side < 0 ? 'left' : 'right';
  const pulse = 0.5 + Math.sin(t * 8.0) * 0.5;
  const curl = intensity * (0.12 + pulse * 0.14);
  for (const name of FINGER_BONES) {
    if (!name.startsWith(prefix)) continue;
    fingerPose[name] = (fingerPose[name] || 0) + curl;
  }
}

// ── (c) event gestures ──────────────────────────────────────────────────────
// Every gesture writes PURE DELTAS into gesturePose. Envelopes start and end
// at exactly 0 (cubic ease / integer-cycle pulses), so there is no snap when
// a gesture begins or ends — the start pose is whatever the lower layers
// already are. Child bones lag parents by 60–110 ms for a natural cascade.
// Gestures never touch blink or mouth bones (lip-sync stays clean).
function applyGestures(dt) {
  gesturePose = {};
  gestureLookAt = 0;
  if (!vrm?.humanoid) return;

  if (gestureState === 'none') {
    if (speakingRecently()) gestureCooldown = Math.min(gestureCooldown, 1.0);
    gestureCooldown -= dt;
    if (gestureCooldown <= 0) {
      pickGesture();
      gestureCooldown = speakingRecently() ? 2.8 + Math.random() * 2.4 : 4.5 + Math.random() * 7.0;
    }
    return;
  }

  gestureT += dt;
  const p = Math.min(1, gestureT / gestureDuration);
  let held = holdCurve(p);                                    // parent bones
  let heldM = holdCurve((gestureT - 0.06) / gestureDuration); // child bones, 60 ms stagger
  let heldT = holdCurve((gestureT - 0.11) / gestureDuration); // fingertips, 110 ms stagger
  if (p >= 1) held = heldM = heldT = 0; // staggered envelopes end at exactly 0
  const side = gestureTarget.side;
  const S = side < 0 ? -1 : 1; // -1 → left arm, +1 → right arm
  const P = S < 0 ? 'left' : 'right';
  const UA = P + 'UpperArm', LA = P + 'LowerArm', HD = P + 'Hand';
  const add = (b, x = 0, y = 0, z = 0) => {
    let o = gesturePose[b];
    if (!o) o = gesturePose[b] = { x: 0, y: 0, z: 0 };
    o.x += x; o.y += y; o.z += z;
  };
  const hipsShift = (px, rz) => {
    let o = gesturePose.hips;
    if (!o) o = gesturePose.hips = { x: 0, y: 0, z: 0, px: 0, py: 0 };
    o.px += px; o.z += rz;
  };
  const hipsLift = (py) => {
    let o = gesturePose.hips;
    if (!o) o = gesturePose.hips = { x: 0, y: 0, z: 0, px: 0, py: 0 };
    o.py += py;
  };

  switch (gestureState) {

    case 'lookAround': {
      const lk = gestureTarget.look * 0.5 * held;
      add('head', Math.sin(p * Math.PI) * 0.015 * held, lk, 0);
      add('neck', 0, gestureTarget.look * 0.18 * held, gestureTarget.tilt * 0.12 * held);
      add('spine', 0, gestureTarget.look * 0.05 * held, 0);
      break;
    }

    case 'lookAtHand': {
      const k = Math.sin(p * Math.PI);
      add(UA, -0.16 * k, 0, S * 0.24 * k);
      add(LA, -0.32 * k, 0, S * 0.05 * k);
      add(HD, 0.08 * k, S * -0.15 * k, 0);
      gestureLookAt = S; // true head tracking, resolved in composePose
      break;
    }

    case 'hairBrush': {
      const k = Math.sin(p * Math.PI);
      add(UA, -0.15 * k, 0, S * 0.30 * k);
      add(LA, -0.27 * k, 0, S * 0.11 * k);
      add(HD, -0.05 * k, S * -0.13 * k, S * 0.07 * k);
      add('head', 0, S * 0.02 * k, -S * 0.025 * k);
      break;
    }

    case 'fingerPlay': {
      const k = Math.sin(p * Math.PI);
      add('head', 0.028 * k, S * 0.05 * k, 0);
      add(LA, -0.13 * k, 0, 0);
      add(HD, 0, S * -0.09 * k, S * Math.sin(gestureT * 5.0) * 0.035 * k);
      applyFingerCurl(side, k);
      break;
    }

    case 'meetGaze': {
      // Settle the head toward center — counter-pose the vitality sway.
      const v = vitality.head || { y: 0, z: 0 };
      add('head', 0.012 * held, -(v.y || 0) * 0.7 * held, -(v.z || 0) * 0.7 * held);
      add('neck', 0.006 * held, 0, 0);
      break;
    }

    case 'curiousTilt': {
      const ti = gestureTarget.tilt * 0.5 * held;
      add('head', -0.010 * held, S * 0.025 * held, ti);
      add('neck', 0, S * 0.015 * held, ti * 0.5);
      break;
    }

    case 'shiftWeight': {
      const sw = pulseCurve(p, 1);
      hipsShift(S * 0.008 * sw, S * 0.009 * sw);
      add('spine', 0, 0, -S * 0.007 * sw);
      add('chest', 0, 0, -S * 0.005 * sw);
      add('head', 0, 0, S * 0.006 * sw);
      break;
    }

    case 'stretchNeck':
      add('neck', -0.035 * held, 0, gestureTarget.tilt * 0.2 * held);
      add('head', -0.028 * held, 0, gestureTarget.tilt * 0.12 * held);
      add('chest', -0.015 * held, 0, 0);
      break;

    case 'raiseHand':
      add(UA, -0.24 * held, 0, S * 0.14 * held);
      add(LA, -0.30 * heldM, 0, S * 0.05 * heldM);
      add(HD, 0.04 * heldT, 0, 0);
      add('head', -0.012 * held, S * 0.035 * held, 0);
      break;

    case 'chinTouch':
      add(UA, -0.20 * held, 0, S * 0.20 * held);
      add(LA, -0.35 * heldM, 0, S * 0.035 * heldM);
      add(HD, -0.06 * heldT, S * -0.07 * heldT, 0);
      add('head', 0.020 * held, 0, S * 0.018 * held);
      add('neck', 0.015 * held, 0, 0);
      break;

    case 'shoulderRoll': {
      const roll = pulseTapered(p, 1.5);
      add(UA, -roll * 0.10, -S * roll * 0.035, -S * roll * 0.06);
      add('head', 0, 0, S * roll * 0.02);
      add('neck', 0, 0, S * roll * 0.013);
      break;
    }

    case 'sway': {
      const sw = pulseCurve(p, 2);
      hipsShift(sw * 0.007, sw * 0.008);
      add('spine', 0, 0, -sw * 0.005);
      add('chest', 0, 0, -sw * 0.004);
      add('head', 0, 0, -sw * 0.006);
      add('neck', 0, 0, -sw * 0.004);
      add('leftUpperArm', 0, 0, sw * 0.02);
      add('rightUpperArm', 0, 0, sw * 0.02);
      break;
    }

    case 'headNod': {
      const nod = pulseCurve(p, 3) * 0.05;
      add('head', nod, 0, 0);
      add('neck', nod * 0.5, 0, 0);
      break;
    }

    case 'wristFlick': {
      const k = Math.sin(p * Math.PI);
      add(LA, 0, S * -0.18 * k, 0);
      add(HD, 0, S * -0.10 * k, S * -0.17 * k);
      add('head', 0, S * 0.03 * k, 0);
      break;
    }

    case 'adjustSleeve': {
      // Hand reaches to the OPPOSITE forearm, tugs at the sleeve.
      const k = Math.sin(p * Math.PI);
      const Q = S < 0 ? 'right' : 'left';
      const QS = -S;
      add(Q + 'UpperArm', -0.12 * k, 0, QS * 0.09 * k);
      add(Q + 'LowerArm', -0.18 * k, QS * -0.14 * k, 0);
      add(Q + 'Hand', -0.07 * k, 0, QS * -0.05 * k);
      add('head', 0.018 * k, 0, 0);
      break;
    }

    case 'handOnHip': {
      const k = holdCurve(p);
      add(UA, -0.07 * k, 0, S * 0.19 * k);
      add(LA, -0.05 * k, 0, S * 0.12 * k);
      add(HD, 0, S * -0.035 * k, S * 0.09 * k);
      add('head', 0, -S * 0.02 * k, 0);
      add('spine', 0, 0, S * 0.005 * k);
      break;
    }

    case 'crossArms': {
      const k = holdCurve(p);
      add('leftUpperArm', -0.03 * k, 0, -0.05 * k);
      add('rightUpperArm', -0.03 * k, 0, 0.05 * k);
      add('leftLowerArm', -0.04 * k, 0, -0.04 * k);
      add('rightLowerArm', -0.04 * k, 0, 0.04 * k);
      add('head', 0.015 * k, 0, 0);
      break;
    }

    case 'touchCollar':
      add(UA, -0.17 * held, 0, S * 0.15 * held);
      add(LA, -0.29 * heldM, 0, S * 0.07 * heldM);
      add(HD, 0.05 * heldT, S * -0.09 * heldT, 0);
      add('head', -0.015 * held, S * 0.03 * held, 0);
      break;

    case 'brushShoulder': {
      // Brushes the OPPOSITE shoulder, like dusting something off.
      const br = pulseTapered(p, 2.5);
      const Q = S < 0 ? 'right' : 'left';
      const QS = -S;
      add(Q + 'UpperArm', -br * 0.10, 0, QS * br * 0.07);
      add(Q + 'LowerArm', -br * 0.15, 0, 0);
      add(Q + 'Hand', 0, 0, -QS * br * 0.10);
      add('head', 0, -S * br * 0.035, 0);
      break;
    }

    case 'stretchArm': {
      const st = pulseCurve(p, 1);
      add(UA, -st * 0.27, 0, S * st * 0.10);
      add(LA, -st * 0.14, 0, 0);
      add(HD, st * 0.07, 0, 0);
      add('head', 0, S * st * 0.04, 0);
      add('chest', st * 0.01, 0, 0);
      break;
    }

    case 'leanIn': {
      const lean = pulseCurve(p, 1) * 0.02;
      add('spine', -lean, 0, 0);
      add('chest', -lean * 0.72, 0, 0);
      add('head', lean * 0.42, S * lean * 0.55, 0);
      add('neck', lean * 0.25, 0, 0);
      break;
    }

    case 'openPalm': {
      const talk = pulseCurve(p, 2);
      add(UA, -talk * 0.10, 0, S * talk * 0.14);
      add(LA, -talk * 0.16, 0, 0);
      add(HD, talk * 0.06, S * -talk * 0.10, 0);
      applyFingerCurl(side, Math.abs(talk) * 0.7);
      break;
    }

    case 'speakingNod': {
      const nod = pulseCurve(p, 2) * 0.028;
      add('head', nod, S * 0.014 * Math.sin(p * Math.PI), 0);
      add('neck', nod * 0.42, 0, 0);
      add('spine', nod * 0.16, 0, 0);
      break;
    }

    case 'earTuck': {
      const swp = holdCurve(p, 0.32, 0.28);
      const stroke = pulseTapered(p, 1.5) * 0.03;
      add(UA, -0.18 * swp, 0, S * 0.26 * swp);
      add(LA, -0.31 * swp, 0, S * 0.06 * swp);
      add(HD, -0.05 * swp + stroke, S * -0.11 * swp, 0);
      add('head', 0, S * 0.02 * swp, -S * 0.03 * swp);
      add('neck', 0, 0, -S * 0.015 * swp);
      break;
    }

    case 'gentleStretch': {
      const op = pulseCurve(p, 1);
      add('leftUpperArm', -0.035 * op, 0, -0.06 * op);
      add('rightUpperArm', -0.035 * op, 0, 0.06 * op);
      add('leftLowerArm', -0.03 * op, 0, 0);
      add('rightLowerArm', -0.03 * op, 0, 0);
      add('chest', -0.015 * op, 0, 0);
      add('spine', -0.008 * op, 0, 0);
      add('head', -0.018 * op, 0, S * 0.008 * op);
      break;
    }

    case 'handsClasp': {
      const cl = holdCurve(p, 0.30, 0.32);
      add('leftUpperArm', -0.08 * cl, 0, -0.13 * cl);
      add('rightUpperArm', -0.08 * cl, 0, 0.13 * cl);
      add('leftLowerArm', -0.18 * cl, 0, -0.08 * cl);
      add('rightLowerArm', -0.18 * cl, 0, 0.08 * cl);
      add('leftHand', 0, 0.07 * cl, -0.035 * cl);
      add('rightHand', 0, -0.07 * cl, 0.035 * cl);
      add('head', 0.012 * cl, 0, 0);
      break;
    }

    case 'thoughtfulLook': {
      const lk = holdCurve(p, 0.35, 0.30);
      const drift = Math.sin(p * Math.PI * 0.8) * lk;
      add('head', -drift * 0.045, S * drift * 0.05, S * drift * 0.02);
      add('neck', -drift * 0.02, S * drift * 0.03, 0);
      add('spine', -drift * 0.008, 0, 0);
      break;
    }

    case 'emphasizePoint': {
      const em = pulseTapered(p, 1.5);
      add(UA, -em * 0.16, 0, S * em * 0.12);
      add(LA, -em * 0.21, 0, 0);
      add(HD, em * 0.05, S * -em * 0.08, 0);
      add('head', -em * 0.012, S * em * 0.02, 0);
      applyFingerCurl(side, Math.abs(em) * 0.5);
      break;
    }

    case 'bothHandsExplain': {
      const ex = pulseTapered(p, 1.8);
      add('leftUpperArm', -ex * 0.10, 0, -ex * 0.11);
      add('rightUpperArm', -ex * 0.10, 0, ex * 0.11);
      add('leftLowerArm', -ex * 0.14, 0, -ex * 0.035);
      add('rightLowerArm', -ex * 0.14, 0, ex * 0.035);
      add('leftHand', ex * 0.06, ex * 0.09, 0);
      add('rightHand', ex * 0.06, -ex * 0.09, 0);
      add('head', -Math.abs(ex) * 0.008, S * Math.abs(ex) * 0.015, 0);
      add('spine', -Math.abs(ex) * 0.006, 0, 0);
      applyFingerCurl(-1, Math.abs(ex) * 0.4);
      applyFingerCurl(1, Math.abs(ex) * 0.4);
      break;
    }

    case 'wave': {
      // Right arm raised OUTWARD-up beside the head, hand swaying.
      const wv = pulseCurve(p, 5) * 0.14;
      const lift = holdCurve(p);
      add('rightUpperArm', -0.12 * lift, 0, -0.75 * lift);
      add('rightLowerArm', -0.09 * lift, 0, 0);
      add('rightHand', 0, 0, wv * lift);
      add('head', -0.02 * lift, 0.045 * lift, 0);
      add('spine', -0.012 * lift, 0, 0);
      applyFingerCurl(1, 0.12 * lift);
      break;
    }

    case 'giggle': {
      const g = Math.abs(pulseCurve(p, 4)) * 0.04;
      const lift = holdCurve(p);
      add('leftUpperArm', -g * 1.2, 0, 0);
      add('rightUpperArm', -g * 1.2, 0, 0);
      add('head', 0.05 * lift + g * 0.5, 0, S * 0.035 * lift);
      add('chest', g * 0.8, 0, 0);
      add('spine', g * 0.4, 0, 0);
      break;
    }

    case 'bow': {
      const b = pulseCurve(p, 1) * 0.20;
      add('spine', b, 0, 0);
      add('chest', b * 0.7, 0, 0);
      add('head', b * 0.45, 0, 0);
      add('neck', b * 0.3, 0, 0);
      add('leftUpperArm', b * 0.08, 0, 0);
      add('rightUpperArm', b * 0.08, 0, 0);
      break;
    }

    case 'clap': {
      const c = pulseCurve(p, 3);
      const lift = holdCurve(p);
      const spread = (0.22 + c * 0.15) * lift;
      add('leftUpperArm', -0.30 * lift, 0, -spread * 0.55);
      add('rightUpperArm', -0.30 * lift, 0, spread * 0.55);
      add('leftLowerArm', -0.19 * lift, 0, 0);
      add('rightLowerArm', -0.19 * lift, 0, 0);
      add('head', -0.03 * lift, c * 0.02 * lift, 0);
      add('spine', -0.018 * lift + Math.abs(c) * 0.008, 0, 0);
      applyFingerCurl(-1, 0.25 * lift);
      applyFingerCurl(1, 0.25 * lift);
      break;
    }

    case 'dance': {
      const beat = p * Math.PI * 4;
      const bounce = Math.abs(Math.sin(beat)) * 0.035;
      const swd = Math.sin(beat * 0.5) * 0.06;
      const armL = Math.max(0, Math.sin(beat)) * 0.45;
      const armR = Math.max(0, Math.sin(beat + Math.PI)) * 0.45;
      const lift = holdCurve(p, 0.12, 0.18);
      hipsLift(bounce);
      add('spine', -bounce * 0.6, 0, swd * 0.5);
      add('chest', 0, 0, swd * 0.7);
      add('head', -0.025 * lift, 0, swd * 0.9);
      add('leftUpperArm', -0.10 * lift, 0, -armL * lift);
      add('rightUpperArm', -0.10 * lift, 0, armR * lift);
      add('leftLowerArm', 0, 0, -armL * 0.35 * lift);
      add('rightLowerArm', 0, 0, armR * 0.35 * lift);
      applyFingerCurl(-1, 0.1 * lift);
      applyFingerCurl(1, 0.1 * lift);
      break;
    }

    case 'hairTwirl': {
      const k = Math.sin(p * Math.PI);
      const twirl = Math.sin(gestureT * 7.0) * 0.09 * k;
      add('head', 0.018 * k, 0, S * 0.035 * k);
      add(UA, -0.17 * k, 0, S * 0.32 * k);
      add(LA, -0.29 * k, 0, S * 0.10 * k);
      add(HD, 0, S * -0.12 * k, S * twirl);
      applyFingerCurl(side, 0.55 * k);
      break;
    }

    case 'handsBehindBack': {
      const k = Math.sin(p * Math.PI);
      const swb = Math.sin(gestureT * 1.8) * 0.02 * k;
      hipsShift(swb * 0.6, swb);
      add('spine', 0, 0, -swb * 0.5);
      add('head', 0.02 * k, 0, swb * 0.7);
      add('leftUpperArm', 0.19 * k, 0, -0.06 * k);
      add('rightUpperArm', 0.19 * k, 0, 0.06 * k);
      add('leftLowerArm', 0.12 * k, 0, 0);
      add('rightLowerArm', 0.12 * k, 0, 0);
      applyFingerCurl(-1, 0.25 * k);
      applyFingerCurl(1, 0.25 * k);
      break;
    }

    case 'footTap': {
      const k = Math.sin(p * Math.PI);
      const tap = Math.abs(Math.sin(gestureT * 9.0)) * 0.12 * k;
      add((S < 0 ? 'left' : 'right') + 'UpperLeg', -tap, 0, 0);
      hipsShift(-S * 0.006 * k, -S * 0.008 * k);
      add('head', 0, 0, S * 0.02 * k);
      add('leftUpperArm', 0, 0, -0.035 * k);
      add('rightUpperArm', 0, 0, 0.035 * k);
      break;
    }

    case 'skirtSmooth': {
      const glide = holdCurve(p, 0.25, 0.35);
      const sm = Math.sin(glide * Math.PI) * 0.13;
      add('spine', sm * 0.35, 0, 0);
      add('head', sm * 0.30, 0, 0);
      add('leftUpperArm', -0.19 * glide - sm * 0.15, 0, -0.08 * glide);
      add('rightUpperArm', -0.19 * glide - sm * 0.15, 0, 0.08 * glide);
      add('leftLowerArm', -0.09 * glide, 0, 0);
      add('rightLowerArm', -0.09 * glide, 0, 0);
      applyFingerCurl(-1, 0.15 * glide);
      applyFingerCurl(1, 0.15 * glide);
      break;
    }

    case 'hugSelf': {
      const sq = holdCurve(p, 0.30, 0.35);
      add('spine', sq * 0.03, 0, 0);
      add('head', sq * 0.035, 0, S * 0.018 * sq);
      add('leftUpperArm', -sq * 0.12, -sq * 0.30, 0);
      add('rightUpperArm', -sq * 0.12, sq * 0.30, 0);
      add('leftLowerArm', -sq * 0.19, 0, 0);
      add('rightLowerArm', -sq * 0.19, 0, 0);
      applyFingerCurl(-1, 0.45 * sq);
      applyFingerCurl(1, 0.45 * sq);
      break;
    }

    case 'happyBounce': {
      const bnc = Math.abs(Math.sin(gestureT * 6.0)) * 0.022 * Math.sin(p * Math.PI);
      const lift = holdCurve(p, 0.20, 0.30);
      hipsLift(bnc);
      add('spine', -bnc * 0.8, 0, 0);
      add('head', -0.03 * lift, 0, 0);
      add('leftUpperArm', -0.11 * lift, 0, -0.20 * lift);
      add('rightUpperArm', -0.11 * lift, 0, 0.20 * lift);
      add('leftLowerArm', 0, 0, -0.09 * lift);
      add('rightLowerArm', 0, 0, 0.09 * lift);
      applyFingerCurl(-1, 0.12 * lift);
      applyFingerCurl(1, 0.12 * lift);
      break;
    }

    // ── thinking poses, now playable as ordinary gestures ───────────────

    case 'chinThink':
      add('head', 0.060 * held, -0.050 * held, -0.025 * held);
      add('neck', 0.035 * held, -0.030 * held, 0);
      add('rightUpperArm', -0.20 * held, 0, 0.23 * held);
      add('rightLowerArm', -0.36 * heldM, 0, 0.04 * heldM);
      add('rightHand', -0.08 * heldT, -0.08 * heldT, 0);
      applyFingerCurl(1, 0.30 * held);
      break;

    case 'handNearMouth':
      add('head', 0.035 * held, 0.070 * held, 0);
      add('neck', 0, 0.040 * held, 0);
      add('leftUpperArm', -0.21 * held, 0, -0.24 * held);
      add('leftLowerArm', -0.33 * heldM, 0, -0.06 * heldM);
      add('leftHand', -0.04 * heldT, 0.10 * heldT, 0);
      applyFingerCurl(-1, 0.25 * held);
      break;

    case 'armsFoldThink':
      add('head', 0.050 * held, 0, 0.030 * held);
      add('neck', 0, 0, 0.018 * held);
      add('leftUpperArm', -0.08 * held, 0, -0.21 * held);
      add('rightUpperArm', -0.08 * held, 0, 0.21 * held);
      add('leftLowerArm', -0.21 * heldM, 0, -0.16 * heldM);
      add('rightLowerArm', -0.21 * heldM, 0, 0.16 * heldM);
      break;

    case 'lookUpThink':
      add('head', -0.070 * held, 0.120 * held, -0.018 * held);
      add('neck', -0.040 * held, 0.050 * held, 0);
      add('rightUpperArm', -0.12 * held, 0, 0.14 * held);
      add('rightLowerArm', -0.24 * heldM, 0, 0);
      add('rightHand', 0, 0, 0.06 * heldT);
      break;

    case 'contemplativeNod': {
      const nod = pulseCurve(p, 3) * 0.035;
      add('head', 0.05 * held + nod, -0.08 * held, -0.02 * held);
      add('neck', 0.025 * held + nod * 0.5, -0.04 * held, 0);
      add('rightUpperArm', -0.09 * held, 0, 0.10 * held);
      add('rightLowerArm', -0.15 * heldM, 0, 0);
      add('rightHand', -0.035 * heldT, 0, 0);
      break;
    }

    case 'tapFinger': {
      const tap = Math.max(0, Math.sin(gestureT * 5.5)) * Math.sin(p * Math.PI);
      add('head', 0.055 * held, 0.040 * held, 0.020 * held);
      add('neck', 0.028 * held, 0.020 * held, 0);
      add('rightUpperArm', -0.18 * held, 0, 0.21 * held);
      add('rightLowerArm', -0.34 * heldM, 0, 0.035 * heldM);
      add('rightHand', -0.06 * heldT - tap * 0.03, -0.07 * heldT, 0);
      applyFingerCurl(1, 0.35 * held);
      break;
    }
  }

  // Envelope already at exactly 0 — no snap, no blend-out needed.
  if (p >= 1) gestureState = 'none';
}

// Former thinking-cycle poses — now ordinary playable gestures (see switch).
const THINKING_POSES = [
  'chinThink', 'handNearMouth', 'armsFoldThink', 'lookUpThink',
  'contemplativeNod', 'tapFinger',
];

// ── pose composition: REST + vitality + state + gesture ──────────────
const _Z3 = { x: 0, y: 0, z: 0 };
function composePose() {
  if (!vrm?.humanoid) return;
  const h = vrm.humanoid;
  for (const name of BLENDABLE_BONES) {
    const bone = h.getRawBoneNode(name);
    if (!bone) continue;
    const v = vitality[name] || _Z3;
    const s = statePose[name] || _Z3;
    const g = gesturePose[name] || _Z3;
    const r = REST[name] || _Z3;
    bone.rotation.set(
      r.x + v.x + s.x + g.x,
      r.y + v.y + s.y + g.y,
      r.z + v.z + s.z + g.z
    );
  }
  // Hips: rotation + translation (weight shifts, bounce). Rest height is
  // restored every frame; gestures only add offsets on top.
  const hips = h.getRawBoneNode('hips');
  if (hips) {
    if (hips.userData.restY === undefined) hips.userData.restY = hips.position.y;
    const v = vitality.hips || _Z3, s = statePose.hips || _Z3, g = gesturePose.hips || _Z3;
    hips.rotation.set(v.x + s.x + g.x, v.y + s.y + g.y, v.z + s.z + g.z);
    hips.position.x = (v.px || 0) + (s.px || 0) + (g.px || 0);
    hips.position.y = hips.userData.restY + (g.py || 0);
  }
  // Fingers: curl offsets accumulate in fingerPose, rest is 0.
  for (const name of FINGER_BONES) {
    const bone = h.getRawBoneNode(name);
    if (bone) bone.rotation.z = fingerPose[name] || 0;
  }
  // Eye saccades: offsets applied on top of the model's rest eye rotation
  // (no-op on models without eye bones).
  const setEye = (e) => {
    if (!e) return;
    if (!e.userData.restRot) e.userData.restRot = { x: e.rotation.x, y: e.rotation.y };
    e.rotation.x = e.userData.restRot.x + sacCur.x;
    e.rotation.y = e.userData.restRot.y + sacCur.y;
  };
  setEye(h.getRawBoneNode('leftEye'));
  setEye(h.getRawBoneNode('rightEye'));
  // lookAtHand: true head tracking, resolved after the arm pose is composed.
  if (gestureLookAt !== 0) {
    const P = gestureLookAt < 0 ? 'left' : 'right';
    const handBone = h.getRawBoneNode(P + 'Hand');
    const headBone = h.getRawBoneNode('head');
    const neckBone = h.getRawBoneNode('neck');
    if (handBone && headBone) {
      const handPos = new THREE.Vector3();
      const headPos = new THREE.Vector3();
      handBone.getWorldPosition(handPos);
      headBone.getWorldPosition(headPos);
      const dir = handPos.sub(headPos).normalize();
      const yaw = Math.atan2(dir.x, dir.z);
      const pitch = Math.atan2(-dir.y, Math.hypot(dir.x, dir.z));
      const k = Math.sin(Math.PI * Math.min(1, gestureT / gestureDuration));
      if (neckBone) { neckBone.rotation.y += yaw * 0.25 * k; neckBone.rotation.x += pitch * 0.25 * k; }
      headBone.rotation.y += yaw * 0.5 * k;
      headBone.rotation.x += pitch * 0.5 * k;
    }
  }
}

function updateExpressions(dt) {
  const em = vrm?.expressionManager;
  if (!em) return;
  for (const name of Object.keys(exprTargets)) {
    const target = exprTargets[name];
    const cur = exprCurrent[name] ?? 0;
    let next;
    if (MOUTH_KEYS.has(name)) {
      next = cur + (target - cur) * (1 - Math.exp(-dt / 0.04));
    } else {
      let a = exprAnim[name];
      if (!a || Math.abs(a.to - target) > 1e-4) {
        // Capture the CURRENT weight as the lerp start — no popping.
        a = exprAnim[name] = { from: cur, to: target, t: 0, dur: 0.35 };
      }
      a.t += dt;
      const e = easeInOutCubic(a.t / a.dur);
      next = a.from + (a.to - a.from) * e;
    }
    exprCurrent[name] = next;
    if (name === 'blink') continue; // driven by applyBlink
    // Cap expression weights at 0.8; mouth visemes keep full range for lip-sync.
    const w = MOUTH_KEYS.has(name) ? next : Math.min(next, 0.8);
    try { em.setValue(name, w); } catch (_) { }
  }
  // Thinking overlay: relaxed gaze while no explicit expression is held.
  const th = easeInOutCubic(stateBlend.thinking);
  if (th > 0.003 && !explicitExpr) {
    const rx = ('relaxed' in exprTargets) ? 'relaxed' : (('fun' in exprTargets) ? 'fun' : null);
    if (rx) { try { em.setValue(rx, 0.5 * th); } catch (_) { } }
  }
}

function applyBlink(dt) {
  if (!vrm?.expressionManager) return;
  const em = vrm.expressionManager;
  if (!blinking) {
    blinkWait -= dt;
    if (blinkWait <= 0) { blinking = true; blinkT2 = 0; }
    return;
  }
  blinkT2 += dt;
  const w = Math.sin(Math.PI * Math.min(1, blinkT2 / BLINK_DUR));
  try { em.setValue('blink', w); } catch (_) { }
  if (blinkT2 >= BLINK_DUR) {
    blinking = false;
    try { em.setValue('blink', 0); } catch (_) { }
    // Every 2–6 s; slower while thinking.
    blinkWait = thinkingActive ? 3.5 + Math.random() * 4.5 : 2.0 + Math.random() * 4.0;
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  t += dt;
  controls.update();
  if (vrm) {
    updateExpressions(dt); // expression weights (cubic crossfades, 0.8 cap)
    applyIdle(dt);         // (a) vitality + (b) state pose → offset layers
    applyGestures(dt);     // (c) event gestures → offset layer
    composePose();         // REST + vitality + state + gesture → bones
    applyBlink(dt);        // expressionManager blink (gestures never touch it)
    vrm.update(dt);        // humanoid + spring bones LAST, on this frame's pose
  }
  renderer.render(scene, camera);
}
animate();

// ── VRM load ─────────────────────────────────────────────────────────────────
const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));

const VRM_URL = './assets/Aiko.vrm';

if (window.location.protocol === 'file:') {
  loadMsg.textContent = 'error: open Aiko via `python main.py` or http://localhost:8787/ — browsers block VRM fetches from file://';
  throw new Error('Aiko WebUI must be served over HTTP so assets/Aiko.vrm can be fetched.');
}

loadMsg.textContent = 'loading Aiko.vrm…';
fill.style.width = '5%';

loader.load(VRM_URL,
  (gltf) => {
    fill.style.width = '95%';
    loadMsg.textContent = 'building model…';
    vrm = gltf.userData.vrm;
    window._vrm = vrm;
    window._REST = REST;

    // three-vrm v3's humanoid.update() rewrites EVERY raw humanoid bone
    // from its internal normalized rig each frame, wiping the direct
    // bone.rotation writes composePose() makes below (model froze in
    // T-pose). We drive the bones ourselves, so disable the auto rig.
    // (Pinned CDN @pixiv/three-vrm@3 now resolves to 3.5.x, where the
    // wiper runs; older 3.x left raw bones alone.)
    if (vrm.humanoid) vrm.humanoid.autoUpdateHumanBones = false;

    VRMUtils.removeUnnecessaryVertices(vrm.scene);
    vrm.scene.traverse(o => { if (o.frustumCulled) o.frustumCulled = false; });
    scene.add(vrm.scene);
    vrm.scene.rotation.y = Math.PI;

    if (vrm.expressionManager) {
      vrm.expressionManager.expressions.forEach(ex => {
        exprTargets[ex.expressionName] = 0;
        exprCurrent[ex.expressionName] = 0;
      });
    }

    fill.style.width = '100%';
    setTimeout(() => {
      const ov = document.getElementById('load-overlay');
      ov.classList.add('fade');
      setTimeout(() => ov.style.display = 'none', 800);
    }, 300);
  },
  (prog) => {
    const p = prog.total ? prog.loaded / prog.total : 0;
    fill.style.width = (5 + p * 88) + '%';
  },
  (err) => {
    const detail = err?.message || String(err);
    loadMsg.textContent = `error loading ${VRM_URL}: ${detail}. Start with python main.py and open http://localhost:8787/ instead of file://.`;
    console.error('[aiko-vrm]', err);
  }
);

// ── expression / viseme API (called by WS handler) ───────────────────────────
const VISEME = { A: 'aa', I: 'ih', U: 'ou', E: 'ee', O: 'oh' };
let activeViseme = 'aa';
let mouthOpen = 0;

function applyMouthShape(weight = mouthOpen) {
  const w = Math.max(0, Math.min(1, weight));
  ['aa', 'ih', 'ou', 'ee', 'oh'].forEach(k => exprTargets[k] = 0);
  exprTargets[activeViseme] = w;
  mouthOpen = w;
  if (w > 0.03) lastVisemeAt = performance.now();
}

const VRM_EMOJI_EXPRESSIONS = {
  '😊': 'happy', '😄': 'happy', '😁': 'happy', '😆': 'happy', '🥰': 'happy', '😍': 'happy', '🙂': 'happy', '😋': 'happy', '🌸': 'happy', '✨': 'happy', '❤️': 'happy', '💖': 'happy', '☺️': 'happy',
  '😒': 'angry', '😡': 'angry', '😠': 'angry', '😤': 'angry', '🤬': 'angry', '💢': 'angry',
  '😭': 'sorrow', '😢': 'sorrow', '🥺': 'sorrow', '☹️': 'sorrow', '🙁': 'sorrow', '😔': 'sorrow', '😞': 'sorrow', '💧': 'sorrow',
  '😮': 'surprised', '😯': 'surprised', '😲': 'surprised', '😳': 'surprised', '🤯': 'surprised', '😱': 'surprised', '⁉️': 'surprised', '❓': 'surprised',
  '😜': 'fun', '🤪': 'fun', '😏': 'fun', '😈': 'fun', '🙃': 'fun', '😉': 'fun',
  '😐': 'neutral', '😑': 'neutral', '😶': 'neutral', '🤖': 'neutral', '😴': 'neutral', '🤔': 'neutral', '💭': 'neutral'
};

const VRM_EXPR_ALIASES = {
  'sad': 'sorrow',
  'relaxed': 'fun',
  'joy': 'happy',
  'annoyed': 'angry',
  'shy': 'happy',
  'thinking': 'neutral'
};

window.aikoSetExpression = (name, intensity = 1.0) => {
  const mapped = VRM_EMOJI_EXPRESSIONS[name] || VRM_EXPR_ALIASES[name] || name;
  for (const k of Object.keys(exprTargets)) if (k !== 'blink') exprTargets[k] = 0;
  explicitExpr = Boolean(mapped && mapped !== 'neutral');
  if (mapped && mapped !== 'neutral') {
    exprTargets[mapped] = intensity;
    if (mapped === 'sorrow') exprTargets['sad'] = intensity;
    if (mapped === 'fun') exprTargets['relaxed'] = intensity;
  }

  const el = document.getElementById('vrm-emotion');
  if (el) {
    el.textContent = name ? `${name} · ${Math.round(intensity * 100)}%` : '—';
    el.className = (name && name !== 'neutral') ? 'active' : '';
  }

  clearTimeout(exprResetTimer);
  if (mapped && mapped !== 'neutral') {
    exprResetTimer = setTimeout(() => window.aikoSetExpression('neutral'), EXPR_RESET_DELAY);
  }
};

window.aikoSetViseme = (viseme, weight = 1.0) => {
  activeViseme = VISEME[viseme] ?? viseme;
  applyMouthShape(mouthOpen);
  const dot = document.getElementById('speak-dot');
  const lbl = document.getElementById('speak-label');
  const speaking = mouthOpen > 0.03;
  dot.className = speaking ? 'dot speak' : 'dot';
  lbl.textContent = speaking ? activeViseme : 'idle';
};

window.aikoSetMouthOpen = (weight = 0) => {
  applyMouthShape(weight);
  const dot = document.getElementById('speak-dot');
  const lbl = document.getElementById('speak-label');
  if (mouthOpen > 0.03) {
    dot.className = 'dot speak';
    lbl.textContent = activeViseme;
  } else {
    dot.className = 'dot';
    lbl.textContent = 'idle';
  }
};

// ── state pose API ────────────────────────────────────────────────
window.aikoSetThinking = (active = true) => {
  const on = Boolean(active);
  if (on && !thinkingActive) thinkingSide = Math.random() < 0.5 ? -1 : 1;
  thinkingActive = on;
};
window.aikoSetListening = (active = true) => {
  listeningActive = Boolean(active);
};
// Back-compat alias: aikoSetPose('thinking', b) → aikoSetThinking(b)
window.aikoSetPose = (name, active = true) => {
  if (name === 'thinking') window.aikoSetThinking(active);
};

// ── Reactive gesture playback (presence upgrade) ───────────────────────
// Called from the backend motion director via {"type":"gesture","name"}.
// Plays a single gesture immediately, blending out whatever was running.
const KNOWN_GESTURES = new Set([
  ...GESTURES,
  ...SPEAKING_GESTURES,
  ...THINKING_POSES,
  // Reactive-only: deliberately excluded from the idle pools.
  'dance', 'wave', 'giggle', 'bow', 'clap',
]);

window.aikoPlayGesture = (name) => {
  if (typeof name !== 'string' || !KNOWN_GESTURES.has(name)) return;
  // Envelopes start at exactly 0, so no snapshot/blend is needed — the new
  // gesture simply takes over from the current additive pose.
  startGesture(name);
};
