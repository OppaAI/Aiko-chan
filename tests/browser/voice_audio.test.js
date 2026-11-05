/* Deterministic WebAudio/VAD regressions. Run: node tests/browser/voice_audio.test.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const vad = fs.readFileSync(path.join(root, 'interface/webui/static/vad.js'), 'utf8');
const script = fs.readFileSync(path.join(root, 'interface/webui/static/script.js'), 'utf8');
function vadHarness(gate, barge, aec) {
  let now = 1000;
  const sent = [];
  const window = {aikoIsSpeaking: true, AIKO_BARGE_IN_ENABLED: barge, AIKO_MIC_ECHO_CANCELLED: aec,
    AIKO_TTS_STARTED_AT: 1000, AIKO_ECHO_TAIL_MS: 400};
  window.stopTtsPlayback = () => { window.aikoIsSpeaking = false; window.AIKO_TTS_STOPPED_AT = now; };
  const ws = {readyState: 1, send: data => sent.push(data)};
  const ctx = vm.createContext({window, performance: {now: () => now}, WebSocket: {OPEN: 1},
    Float32Array, ArrayBuffer, console: {log() {}}, setTimeout, clearTimeout});
  vm.runInContext(vad, ctx);
  const frame = (value) => ctx.processEnergyVADFrame(new Float32Array(512).fill(value), ws, 0, gate);
  return {window, sent, ctx, frame, time: t => { now = t; }};
}
for (const gate of [true, false]) {
  const h = vadHarness(gate, false, true);
  for (let i=0; i<30; i++) h.frame(.8);
  assert.equal(h.sent.length, 0, 'TTS audio leaked into ASR');
  h.window.stopTtsPlayback();
  h.time(1200); h.frame(.8);
  assert.equal(h.sent.length, 0, 'acoustic tail leaked into ASR');
  h.time(1500); h.frame(.1); h.frame(.1);
  assert.ok(h.sent.some(data => typeof data !== 'string'), 'mic did not recover');
  for (const data of h.sent.filter(data => typeof data !== 'string')) {
    assert.equal(new Float32Array(data)[0], Math.fround(.1), 'old TTS pre-buffer leaked');
  }
}
{
  const h = vadHarness(true, true, false);
  h.time(2000);
  for (let i=0; i<20; i++) h.frame(.8);
  assert.equal(h.sent.length, 0, 'un-cancelled local speaker echo triggered a barge');
}
{
  const h = vadHarness(true, true, true);
  for (let i=0; i<10; i++) h.frame(.4);
  assert.equal(h.sent.length, 0, 'barge fired inside onset echo guard');
  h.time(1600);
  for (let i=0; i<4; i++) h.frame(.4);
  assert.deepEqual(h.sent.map(JSON.parse), [{type:'barge_in'}]);
  h.frame(.4); h.frame(.4);
  assert.ok(h.sent.some(data => typeof data !== 'string'), 'confirmed AEC barge did not capture speech');
}
async function playbackTest() {
  let resolve;
  let played = 0;
  const context = {state: 'running', decodeAudioData: () => new Promise(r => { resolve = r; }),
    destination: {}, createAnalyser: () => ({connect(){}}),
    createBufferSource: () => ({connect(){}, disconnect(){}, start(){played++;}, stop(){}})};
  const ctx = vm.createContext({window: {}, AudioContext: function(){return context;},
    performance: {now: () => 1000}, Float32Array, Uint8Array, ArrayBuffer,
    console, setTimeout, clearTimeout, document: {getElementById(){return null;}}, captionSpeaking(){}, setPresence(){}, typingIndicator: false, currentPresence: 'speaking'});
  const section = script.slice(script.indexOf('let ttsContext = null;'), script.indexOf('// ── mic capture'));
  vm.runInContext(section, ctx);
  vm.runInContext('startMouthAnalyserLoop = () => {}; enqueueTtsAudio(new ArrayBuffer(8));', ctx);
  assert.equal(typeof resolve, 'function');
  ctx.stopTtsPlayback();
  resolve({});
  await new Promise(r => setImmediate(r));
  assert.equal(played, 0, 'audio resumed after stop during decodeAudioData');
  assert.equal(ctx.window.aikoIsSpeaking, false);
  ctx.window.AIKO_MIC_ECHO_CANCELLED = true;
  ctx.window.AIKO_BARGE_SENT_AT = 1000;
  ctx.stopRemoteTtsPlayback();
  assert.equal(ctx.window.AIKO_TTS_STOPPED_AT, 0, 'barge stop acknowledgment discarded human onset');
  ctx.stopRemoteTtsPlayback();
  assert.equal(ctx.window.AIKO_TTS_STOPPED_AT, 0, 'duplicate stop acknowledgment discarded human onset');
}
async function blockedPlaybackTest() {
  const context = {state: 'suspended', resume: () => new Promise(() => {})};
  const ctx = vm.createContext({window: {}, AudioContext: function(){return context;},
    performance: {now: () => 1000}, Float32Array, Uint8Array, ArrayBuffer,
    document: {getElementById(){return null;}}, console: {error() {}},
    setTimeout: fn => { queueMicrotask(fn); return 1; }, clearTimeout(){},
    captionSpeaking(){}, setPresence(){}, typingIndicator: false, currentPresence: 'speaking'});
  const section = script.slice(script.indexOf('let ttsContext = null;'), script.indexOf('// ── mic capture'));
  vm.runInContext(section, ctx);
  vm.runInContext('enqueueTtsAudio(new ArrayBuffer(8));', ctx);
  await new Promise(r => setImmediate(r));
  assert.equal(ctx.window.aikoIsSpeaking, false, 'blocked autoplay left mic suppressed forever');
}
playbackTest().then(blockedPlaybackTest).then(() => console.log('Browser voice tests passed: echo/tail/pre-buffer gates, AEC barge-in, cancelled decoding and blocked autoplay.'))
  .catch(error => {console.error(error); process.exitCode=1;});
