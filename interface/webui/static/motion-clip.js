// Data-only motion clips. Captured rotations use VRM normalized humanoid bones.
import * as THREE from 'three';

export function validateMotionClip(value) {
  if (!value || value.format !== 'aiko-vrm-motion' || value.version !== 1 || value.rig !== 'normalized') {
    throw new Error('Choose an Aiko VRM motion JSON exported from the video capture tool.');
  }
  const frames = value.frames;
  if (!Array.isArray(frames) || frames.length < 2 || frames.length > 36000) throw new Error('A clip needs 2–36,000 frames.');
  const vector = (v, n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
  if (!vector(value.restHips, 3) || value.restHips[1] <= 0) throw new Error('Invalid rest hips.');
  let previous = -1;
  const firstBones = Object.keys(frames[0].bones || {});
  if (!firstBones.length || firstBones.length > 64) throw new Error('Invalid bone tracks.');
  for (const frame of frames) {
    if (!Number.isFinite(frame.time) || frame.time < 0 || frame.time <= previous || frame.time > 1200) throw new Error('Frame times must increase within a 20-minute clip.');
    if (!vector(frame.hipsPos, 3) || !frame.bones || Object.keys(frame.bones).length !== firstBones.length) throw new Error('Invalid frame.');
    for (const name of firstBones) {
      const q = frame.bones[name];
      if (!/^[a-zA-Z][a-zA-Z0-9]{0,63}$/.test(name) || !vector(q, 4) || Math.hypot(...q) < 0.00001) throw new Error('Invalid bone rotation.');
    }
    previous = frame.time;
  }
  if (frames[0].time !== 0) throw new Error('The first frame must start at zero.');
  return value;
}

// A binary search handles long recordings without scanning every frame on render.
export function sampleMotionClip(clip, seconds, loop = false) {
  const frames = clip.frames;
  const duration = frames.at(-1).time;
  const time = loop ? ((seconds % duration) + duration) % duration : Math.max(0, Math.min(duration, seconds));
  let lo = 0, hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].time <= time) lo = mid; else hi = mid;
  }
  const a = frames[lo], b = frames[hi];
  return { a, b, mix: (time - a.time) / (b.time - a.time) };
}

export class MotionClipPlayer {
  constructor() { this.active = null; }
  play(vrm, data, { loop = false, speed = 1 } = {}) {
    validateMotionClip(data);
    if (!vrm?.humanoid) throw new Error('Wait for the avatar to load.');
    if (!Number.isFinite(speed) || speed < 0.25 || speed > 2) throw new Error('Speed must be 0.25–2.');
    this.stop();
    const h = vrm.humanoid;
    const hips = h.getNormalizedBoneNode('hips');
    if (!hips) throw new Error('This avatar has no humanoid hips.');
    this.active = { vrm, data, loop, speed, started: performance.now(),
      rawPose: h.getRawPose(), normalizedPose: h.getNormalizedPose(),
      restHips: hips.position.clone(), autoUpdate: h.autoUpdateHumanBones };
    // Playback owns the humanoid until stop; the normal WebUI poses use raw bones.
    h.autoUpdateHumanBones = true;
  }
  stop() {
    const a = this.active;
    if (!a) return;
    a.vrm.humanoid.setNormalizedPose(a.normalizedPose);
    a.vrm.humanoid.setRawPose(a.rawPose);
    a.vrm.humanoid.autoUpdateHumanBones = a.autoUpdate;
    this.active = null;
  }
  update(now = performance.now()) {
    const a = this.active;
    if (!a) return false;
    const seconds = (now - a.started) / 1000 * a.speed;
    if (!a.loop && seconds > a.data.frames.at(-1).time) { this.stop(); return false; }
    const { a: f0, b: f1, mix } = sampleMotionClip(a.data, seconds, a.loop);
    const h = a.vrm.humanoid;
    const q0 = new THREE.Quaternion(), q1 = new THREE.Quaternion();
    for (const name of Object.keys(f0.bones)) {
      const bone = h.getNormalizedBoneNode(name);
      if (bone) bone.quaternion.copy(q0.fromArray(f0.bones[name]).normalize().slerp(q1.fromArray(f1.bones[name]).normalize(), mix));
    }
    const hips = h.getNormalizedBoneNode('hips');
    const scale = a.restHips.y / a.data.restHips[1];
    hips.position.copy(a.restHips);
    for (let i = 0; i < 3; i++) {
      hips.position.setComponent(i, a.restHips.getComponent(i) +
        (THREE.MathUtils.lerp(f0.hipsPos[i], f1.hipsPos[i], mix) - a.data.restHips[i]) * scale);
    }
    // Mouth/face remain driven by WebUI speech and expressions.
    return true;
  }
}
