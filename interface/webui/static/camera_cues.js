/* Optional browser-side perception. Frames/templates never leave this module. */
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const FACE = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15';
let motionPromise, facePromise;
export async function loadMotion() {
  if (!motionPromise) motionPromise = (async () => {
    const {FilesetResolver, GestureRecognizer, PoseLandmarker} = await import(`${MP}/vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
    let hands;
    try {
      hands = await GestureRecognizer.createFromOptions(files, {baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task', delegate:'CPU'}, runningMode:'VIDEO', numHands:2});
      const pose = await PoseLandmarker.createFromOptions(files, {baseOptions: {
        modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', delegate:'CPU'}, runningMode:'VIDEO', numPoses:1});
      return {hands, pose};
    } catch (error) { hands?.close(); throw error; }
  })().catch(error => { motionPromise = null; throw error; });
  return motionPromise;
}
export function postureLabel(points) {
  if (!points?.length) return 'not_visible';
  const visible = i => points[i] && (points[i].visibility ?? 0) > .6;
  if (!visible(11) || !visible(12)) return 'not_visible';
  if ((visible(15) && points[15].y < points[11].y - .12) ||
      (visible(16) && points[16].y < points[12].y - .12)) return 'arms_raised';
  if (!visible(23) || !visible(24)) return 'not_visible';
  const dx = (points[11].x + points[12].x - points[23].x - points[24].x) / 2;
  if (dx < -.12) return 'leaning_left';
  if (dx > .12) return 'leaning_right';
  return 'upright';
}
export async function motionCues(video) {
  const tracker = await loadMotion();
  const now = performance.now();
  const hands = tracker.hands.recognizeForVideo(video, now);
  const pose = tracker.pose.detectForVideo(video, now);
  return {gestures:(hands.gestures || []).map(items => items[0]).filter(item => item && item.score >= .7 && item.categoryName !== 'None').map(item => item.categoryName).slice(0,2),
    posture:postureLabel(pose.landmarks?.[0])};
}
async function loadFace() {
  if (!facePromise) facePromise = (async () => {
    if (!window.faceapi) await new Promise((resolve,reject) => {
      const script = document.createElement('script'); script.src = `${FACE}/dist/face-api.js`;
      script.onload = resolve; script.onerror = () => { script.remove(); reject(new Error('Face model download failed')); };
      document.head.append(script);
    });
    const api = window.faceapi;
    await Promise.all([api.nets.tinyFaceDetector.loadFromUri(`${FACE}/model`),
      api.nets.faceLandmark68TinyNet.loadFromUri(`${FACE}/model`), api.nets.faceRecognitionNet.loadFromUri(`${FACE}/model`)]);
    return api;
  })().catch(error => { facePromise = null; throw error; });
  return facePromise;
}
function key(uid) {
  if (!uid || uid === 'guest') throw new Error('Log in before enrolling a face.');
  return `aiko-face-template-v1:${uid}`;
}
export function descriptorValid(value) {
  return Array.isArray(value) && value.length === 128 && value.every(number => typeof number === 'number' && Number.isFinite(number) && Math.abs(number) < 10);
}
async function descriptors(video) {
  const api = await loadFace();
  return api.detectAllFaces(video, new api.TinyFaceDetectorOptions({inputSize:224, scoreThreshold:.7}))
    .withFaceLandmarks(true).withFaceDescriptors();
}
export async function enrollFace(video, uid, active) {
  const storageKey = key(uid), samples = [];
  for (let i=0; i<5; i++) {
    if (!active()) throw new Error('Camera stopped; face was not enrolled.');
    const faces = await descriptors(video);
    if (!active() || faces.length !== 1) throw new Error('Enrollment needs exactly one clearly visible face.');
    const vector = Array.from(faces[0].descriptor);
    if (!descriptorValid(vector)) throw new Error('Invalid face template.');
    if (samples.length && distance(samples[0], vector) > .45) throw new Error('Face changed during enrollment. Please try again.');
    samples.push(vector);
    await new Promise(resolve => setTimeout(resolve,150));
  }
  if (!active()) throw new Error('Camera stopped; face was not enrolled.');
  const mean = samples[0].map((_,i) => samples.reduce((sum,row) => sum+row[i],0)/samples.length);
  localStorage.setItem(storageKey, JSON.stringify({descriptor:mean, model:'face-api-1.7.15', enrolledAt:Date.now()}));
}
export function forgetFace(uid) { localStorage.removeItem(key(uid)); }
export function distance(a,b) { return Math.sqrt(a.reduce((sum,value,i) => sum+(value-b[i])**2,0)); }
export async function faceCue(video, uid) {
  let saved;
  try { saved = JSON.parse(localStorage.getItem(key(uid)) || 'null'); } catch (_) { return 'not_visible'; }
  if (!descriptorValid(saved?.descriptor)) throw new Error('Enroll your face on this browser first.');
  const faces = await descriptors(video);
  if (faces.length !== 1) return 'not_visible';
  return distance(Array.from(faces[0].descriptor), saved.descriptor) < .5 ? 'enrolled_match' : 'no_match';
}
