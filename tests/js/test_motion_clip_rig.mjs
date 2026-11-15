import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import * as THREE from '../../node_modules/three/build/three.module.js';
import { VRMHumanoid } from '../../node_modules/@pixiv/three-vrm/lib/three-vrm.module.js';
const source = (await readFile(new URL('../../interface/webui/static/motion-clip.js', import.meta.url), 'utf8')).replace("from 'three'", `from '${new URL('../../node_modules/three/build/three.module.js', import.meta.url).href}'`);
const { MotionClipPlayer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const bones = {};
function bone(name, parent, xyz) {
  const node = new THREE.Object3D(); node.position.fromArray(xyz);
  if (parent) bones[parent].node.add(node);
  bones[name] = {node};
}
bone('hips', null, [0,1,0]);
bone('spine','hips',[0,0.2,0]);
bone('head','spine',[0,0.3,0]);
for(const side of ['left','right']) {
  const x=side==='left'?0.1:-0.1;
  bone(`${side}UpperLeg`,'hips',[x,-0.1,0]);
  bone(`${side}LowerLeg`,`${side}UpperLeg`,[0,-0.4,0]);
  bone(`${side}Foot`,`${side}LowerLeg`,[0,-0.4,0.1]);
  bone(`${side}UpperArm`,'spine',[x*2,0.2,0]);
  bone(`${side}LowerArm`,`${side}UpperArm`,[x*2,0,0]);
  bone(`${side}Hand`,`${side}LowerArm`,[x*2,0,0]);
}
const scene = new THREE.Scene();
scene.add(bones.hips.node);
scene.updateMatrixWorld(true);
const humanoid = new VRMHumanoid(bones, {autoUpdateHumanBones:false});
const before = humanoid.getRawPose();
const hips = humanoid.getNormalizedBoneNode('hips').position.toArray();
const clip={format:'aiko-vrm-motion',version:1,rig:'normalized',restHips:hips,frames:[
  {time:0,bones:{head:[0,0,0,1]},hipsPos:hips},
  {time:1,bones:{head:new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),Math.PI/2).toArray()},hipsPos:hips}
]};
const player=new MotionClipPlayer();
player.play({humanoid},clip);
player.update(player.active.started+500);
humanoid.update();
assert(Math.abs(humanoid.getRawBoneNode('head').quaternion.y)>0.2,'Normalized playback must reach raw VRM bones');
player.stop();
assert.equal(humanoid.autoUpdateHumanBones,false);
assert.deepEqual(humanoid.getRawPose(),before,'Restore original raw rig on stop');
console.log('Real three-vrm humanoid transfer and raw-pose restoration passed.');
