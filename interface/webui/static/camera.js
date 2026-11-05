/* Visible, user-requested camera sessions. One frame in flight; no recordings. */
(() => {
  const $ = id => document.getElementById(id);
  let stream = null, video = null, session = null, sequence = 0, cuesTimer = 0, deadline = 0;
  let inFlight = false, cueBusy = false, enrolling = false, currentPrompt = '', cuesModule;
  let attentionGate, faceFailed = false;
  const panel = $('camera-panel'), status = $('camera-status');
  const say = text => { if (status) status.textContent = text; };
  const send = payload => window.aikoSendWS?.(payload) === true;
  const uid = () => window.currentUserId || '';
  const valid = token => session === token && !!stream && !document.hidden;
  const module = () => cuesModule ||= import('./camera_cues.js');
  function stop(reason='Camera stopped.') {
    sequence++;
    const hadSession = session;
    session = null;
    clearTimeout(cuesTimer); clearTimeout(deadline);
    cuesTimer = deadline = 0;
    stream?.getTracks().forEach(track => track.stop()); stream = null;
    if (video) { video.pause(); video.srcObject = null; }
    if (hadSession) send({type:'camera_state', active:false, watch_id:hadSession});
    inFlight = false; cueBusy = false;
    if (panel) panel.hidden = true;
    $('watch-btn')?.classList.remove('on');
    say(reason);
  }
  async function openVideo(token) {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera requires browser permission and a secure HTTPS or localhost page.');
    const candidate = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
    if (sequence !== token || document.hidden) { candidate.getTracks().forEach(track => track.stop()); return false; }
    stream = candidate;
    video = $('camera-preview'); video.srcObject = stream; video.muted = true; video.playsInline = true;
    await video.play();
    await new Promise(resolve => setTimeout(resolve,200));
    if (sequence !== token || !stream || document.hidden) return false;
    for (const track of stream.getTracks()) track.addEventListener('ended', () => stop('Camera disconnected.'), {once:true});
    return true;
  }
  function frame() {
    if (!video?.videoWidth) throw new Error('Camera has no frame yet.');
    const canvas = document.createElement('canvas');
    const scale = Math.min(1,640/video.videoWidth,480/video.videoHeight);
    canvas.width = Math.max(1,Math.round(video.videoWidth*scale)); canvas.height = Math.max(1,Math.round(video.videoHeight*scale));
    canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
    return canvas.toDataURL('image/jpeg',.78);
  }
  function submitFrame(token, respond=false, attention=null) {
    if (!valid(token) || inFlight) return;
    inFlight = true;
    try {
      if (!send({type:'image_input', image:frame(), text:currentPrompt, observe_only:!respond, watch_id:token, attention_trigger:attention})) throw new Error('Connection lost.');
      say('Camera on · observing the current scene…');
      // No acknowledgment means no new request can accumulate.
      clearTimeout(deadline); deadline = setTimeout(() => stop('Vision response timed out.'),125000);
    } catch (error) { stop(error.message); window.hideTypingIndicator?.(); }
  }
  async function cues(token) {
    if (!valid(token)) return;
    if (!cueBusy && !enrolling) {
      cueBusy = true;
      const data = {};
      try {
        const api = await module();
        if (!valid(token)) return;
        if ($('camera-motion').checked) {
          try { Object.assign(data, await api.motionCues(video)); }
          catch (error) { $('camera-motion').checked = false; say(`Hand detection unavailable: ${error.message}`); }
        }
        if (!valid(token)) return;
        if ($('camera-gaze').checked) {
          try { data.camera_gaze = await api.gazeCue(video); }
          catch (error) { $('camera-gaze').checked = false; say(`Gaze detection unavailable: ${error.message}`); }
        }
        if (!valid(token)) return;
        if (!$('camera-face').checked) faceFailed = false;
        if ($('camera-face').checked && !faceFailed) {
          try { data.face = await api.faceCue(video, uid()); }
          catch (error) { faceFailed = true; say(`Face recognition unavailable: ${error.message}. Re-enable it to retry.`); }
        }
        if (valid(token)) {
          send({type:'visual_cues',watch_id:token,cues:data});
          const recognized = !$('camera-face').checked || data.face === 'enrolled_match';
          const reason = recognized && (data.raised_open_palm ? 'open_palm' : data.camera_gaze ? 'camera_gaze' : null);
          const trigger = attentionGate.update(reason, performance.now());
          if (trigger && !inFlight) submitFrame(token, false, trigger);
        }
      } catch (error) {
        if (valid(token)) {
          say(`Camera on · ${error.message}`);
        }
      } finally { cueBusy = false; }
    }
    if (valid(token)) cuesTimer = setTimeout(() => cues(token),2000);
  }
  async function start(prompt='Describe the person and scene briefly, including visible gestures.', respond=false) {
    stop();
    const generation = sequence;
    window.dispatchEvent(new CustomEvent('aiko:camera-starting'));
    if (!send({type:'camera_state',active:false})) { say('Connect to Aiko first.'); return; }
    try {
      panel.hidden = false; say('Opening camera…');
      if (!await openVideo(generation)) return;
      session = crypto.randomUUID(); currentPrompt = prompt;
      const token = session;
      const {AttentionGate} = await import('./camera_attention.js');
      if (!valid(token) || generation !== sequence) return;
      attentionGate = new AttentionGate(); faceFailed = false;
      $('watch-btn')?.classList.add('on');
      send({type:'camera_state', active:true, watch_id:token});
      say('Camera on · look toward the camera or raise an open palm for a few seconds');
      if (respond) submitFrame(token,true);
      cues(token);
    } catch (error) { stop(error.message); window.addMessage?.('sys', error.message); window.hideTypingIndicator?.(); }
  }
  function visionEvent(event) {
    if (!session || event.watch_id !== session) return;
    if (event.status === 'done') {
      inFlight = false; clearTimeout(deadline);
      say('Camera on · waiting for eye contact or a raised open palm');
    } else if (event.status === 'error' || event.status === 'busy') {
      stop(event.message || 'Vision is unavailable.'); window.hideTypingIndicator?.();
    }
  }
  $('watch-btn')?.addEventListener('click', () => session ? stop() : start());
  $('camera-stop')?.addEventListener('click', () => stop());
  $('camera-face')?.addEventListener('change', () => { faceFailed = false; });
  $('camera-enroll')?.addEventListener('click', async () => {
    if (!session || enrolling) return;
    const token = session; enrolling = true;
    try {
      say('Look at the camera with only your face visible…');
      // Avoid running descriptor inference concurrently with the cue tick.
      if (cueBusy) { say('Wait for the current gesture/face check, then try again.'); return; }
      await (await module()).enrollFace(video, uid(), () => valid(token));
      if (valid(token)) { $('camera-face').checked = true; faceFailed = false; say('Face enrolled on this browser · triggers now require a match.'); }
    } catch (error) { if (valid(token)) say(error.message); }
    finally { enrolling = false; }
  });
  $('camera-forget')?.addEventListener('click', async () => {
    try { (await module()).forgetFace(uid()); $('camera-face').checked = false; say('Saved face template removed from this browser.'); }
    catch (error) { say(error.message); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', () => stop());
  window.AikoCamera = {start,stop,visionEvent, configure(message) {
    if (!valid(message.watch_id)) return;
    $('camera-gaze').checked = message.gaze === true;
    $('camera-gaze').disabled = message.gaze !== true;
    $('camera-motion').checked = message.palm === true;
    $('camera-motion').disabled = message.palm !== true;
    if (Number.isFinite(message.cooldown_ms)) attentionGate.cooldown = Math.max(15000,Math.min(3600000,message.cooldown_ms));
    if (Number.isFinite(message.dwell_ms)) attentionGate.dwell = Math.max(1000,Math.min(10000,message.dwell_ms));
  }, control(message) {
    if (message.action === 'stop') stop();
    else if (message.action === 'watch') start(message.prompt);
    else if (message.action === 'capture') { stop(); window.captureImage?.(message.prompt); }
  }, get active() { return !!session; }};
})();
