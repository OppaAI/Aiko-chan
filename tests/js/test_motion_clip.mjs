import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as THREE from '../../node_modules/three/build/three.module.js';
const source = (await readFile(new URL('../../interface/webui/static/motion-clip.js', import.meta.url), 'utf8')).replace("from 'three'", `from '${new URL('../../node_modules/three/build/three.module.js', import.meta.url).href}'`);
const { validateMotionClip, sampleMotionClip, MotionClipPlayer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const makeClip = () => ({ format:'aiko-vrm-motion', version:1, rig:'normalized', restHips:[0,1,0], frames:[
 {time:0, bones:{head:[0,0,0,1]}, hipsPos:[0,1,0]},
 {time:1, bones:{head:[0,0,0,-1]}, hipsPos:[1,1.1,0]}
] });
const clip = makeClip();
assert.equal(validateMotionClip(clip), clip);
for (const corrupt of [null, {}, {...clip, version:2}, {...clip, restHips:[0,0,0]}, {...clip, frames:[clip.frames[0],clip.frames[0]]}, {...clip, frames:[clip.frames[0],{...clip.frames[1],bones:{head:[0,0,0,0]}}]}]) assert.throws(() => validateMotionClip(corrupt));
assert.equal(sampleMotionClip(clip, 0.5).mix, 0.5);
assert.equal(sampleMotionClip(clip, 2).mix, 1);
assert.equal(sampleMotionClip(clip, 1.5, true).mix, 0.5);
const head = new THREE.Object3D();
const hips = new THREE.Object3D(); hips.position.set(0,2,0);
let rawRestored = false, normalizedRestored = false;
const rig = { autoUpdateHumanBones:false,
 getNormalizedBoneNode: name => ({head,hips}[name]),
 getRawPose: () => ({saved:true}), getNormalizedPose: () => ({saved:true}),
 setRawPose: () => {rawRestored=true;}, setNormalizedPose: () => {normalizedRestored=true;},
};
const player = new MotionClipPlayer();
player.play({humanoid:rig}, clip);
assert.equal(rig.autoUpdateHumanBones, true);
assert.equal(player.update(player.active.started+500), true);
assert(Math.abs(head.quaternion.w) > 0.999, 'Opposite signs must represent the same orientation');
assert(Math.abs(hips.position.x-1)<0.0001, 'Root displacement must scale to the target avatar');
assert.equal(player.update(player.active.started+1100), false);
assert.equal(player.active, null);
assert.equal(rig.autoUpdateHumanBones,false);
assert(rawRestored && normalizedRestored);
player.play({humanoid:rig}, clip, {loop:true});
assert(player.update(player.active.started+2500));
player.stop();
assert.throws(() => player.play(null, clip));
assert.throws(() => player.play({humanoid:rig}, clip, {speed:NaN}));
console.log('Motion validation, timeline sampling, quaternion interpolation, avatar scaling, looping and cleanup passed.');
