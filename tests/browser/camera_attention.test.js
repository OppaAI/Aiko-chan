const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
(async () => {
  const source = fs.readFileSync(path.join(root,'interface/webui/static/camera_attention.js'),'utf8');
  const {AttentionGate,lookingTowardCamera,raisedOpenPalm} = await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const gate = new AttentionGate();
  assert.equal(gate.update('camera_gaze',0),null);
  assert.equal(gate.update('camera_gaze',2000),'camera_gaze');
  assert.equal(gate.update('camera_gaze',100000),null); // Holding gaze never repeats.
  gate.update(null,100001); gate.update(null,102001);
  assert.equal(gate.update('open_palm',104001),null);
  assert.equal(gate.update('open_palm',106001),'open_palm');
  gate.update(null,107000); gate.update(null,109000);
  gate.update('open_palm',110000);
  assert.equal(gate.update('open_palm',112000),null); // Cooldown despite rearming.
  assert.equal(lookingTowardCamera([]),false);
  const points = Array.from({length:478},()=>({x:.5,y:.5}));
  for (const [a,b,top,bottom,iris,x] of [[33,133,159,145,468,.3],[362,263,386,374,473,.7]]) {
    points[a]={x:x-.06,y:.4}; points[b]={x:x+.06,y:.4};
    points[top]={x,y:.38}; points[bottom]={x,y:.42}; points[iris]={x,y:.4};
  }
  assert.equal(lookingTowardCamera(points),true);
  points[468].x=.25; assert.equal(lookingTowardCamera(points),false);
  points[468].x=.3; points[145].y=.38; assert.equal(lookingTowardCamera(points),false);
  const pose = Array.from({length:33},()=>({x:.5,y:.6,visibility:1}));
  const hand = {gestures:[[{categoryName:'Open_Palm',score:.9}]],landmarks:[[{x:.5,y:.3}]]};
  assert.equal(raisedOpenPalm(hand,pose),true);
  hand.gestures[0][0].categoryName='Closed_Fist'; assert.equal(raisedOpenPalm(hand,pose),false);
  hand.gestures[0][0].categoryName='Open_Palm'; hand.landmarks[0][0].y=.8;
  assert.equal(raisedOpenPalm(hand,pose),false);

  // Exercise the real camera controller with model loading and hardware mocked.
  let now=0, gaze=true, timers=[], sent=[];
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id,{checked:['camera-motion','camera-gaze'].includes(id),classList:{add(){},remove(){}},
      addEventListener(){},play:async()=>{},pause(){},videoWidth:640,videoHeight:480});
    return nodes.get(id);
  };
  const api={motionCues:async()=>({gestures:[],posture:'upright',raised_open_palm:false}),gazeCue:async()=>gaze,faceCue:async()=>{throw new Error('No enrolled face');}};
  const context = vm.createContext({window:{currentUserId:'alice',dispatchEvent(){},aikoSendWS:msg=>{sent.push(msg);return true;}},
    document:{hidden:false,getElementById:node,addEventListener(){},createElement:()=>({getContext:()=>({drawImage(){}}),toDataURL:()=> 'image'})},
    navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop(){},addEventListener(){}}]})}},
    crypto:{randomUUID:()=> 'watch-1'},CustomEvent:class {},performance:{now:()=>now},api,InjectedAttentionGate:AttentionGate,
    setTimeout:(fn,ms)=>{if(ms===200){fn();return 0;} const timer={fn,ms};timers.push(timer);return timer;},
    clearTimeout:timer=>{timers=timers.filter(t=>t!==timer);}});
  context.window.addEventListener=()=>{};
  const controller=fs.readFileSync(path.join(root,'interface/webui/static/camera.js'),'utf8')
    .replace("import('./camera_cues.js')",'Promise.resolve(api)')
    .replace("import('./camera_attention.js')",'Promise.resolve({AttentionGate:InjectedAttentionGate})');
  vm.runInContext(controller,context);
  const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
  const tick=async time=>{now=time;const timer=timers.find(t=>t.ms===2000);assert.ok(timer);timers=timers.filter(t=>t!==timer);await timer.fn();await flush();};
  const frames=()=>sent.filter(item=>item.type==='image_input');
  await context.window.AikoCamera.start(); await flush();
  assert.equal(frames().length,0); // Opening Watch does not call the VLM.
  await tick(2000); assert.equal(frames().length,1);
  assert.equal(frames()[0].attention_trigger,'camera_gaze'); assert.equal(frames()[0].observe_only,true);
  context.window.AikoCamera.visionEvent({watch_id:'watch-1',status:'done'});
  await tick(100000); assert.equal(frames().length,1); // No scheduled frame refresh.
  gaze=false; await tick(102000); await tick(104000);
  gaze=true; await tick(106000); await tick(108000);
  assert.equal(frames().length,2);
  context.window.AikoCamera.visionEvent({watch_id:'watch-1',status:'done'});
  node('camera-face').checked=true;
  await tick(110000); await tick(112000); await tick(210000);
  assert.equal(frames().length,2); // Face failures suppress rather than bypass matching.
  assert.equal(node('camera-face').checked,true);
  context.window.AikoCamera.configure({watch_id:'watch-1',gaze:false,palm:false,cooldown_ms:30000,dwell_ms:4000});
  assert.equal(node('camera-gaze').checked,false); assert.equal(node('camera-gaze').disabled,true);
  assert.equal(node('camera-motion').checked,false);
  await tick(212000); assert.equal(frames().length,2);
  context.window.AikoCamera.stop(); assert.equal(timers.length,0);
  console.log('Camera attention and controller regressions passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
