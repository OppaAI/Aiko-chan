/* Visible, user-requested camera sessions. One frame in flight; no recordings. */
(() => {
  const $ = id => document.getElementById(id);
  let stream = null, video = null, session = null, sequence = 0, frameTimer = 0, cuesTimer = 0, deadline = 0;
  let inFlight = false, cueBusy = false, enrolling = false, currentPrompt = '', cuesModule;
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
    clearTimeout(frameTimer); clearTimeout(cuesTimer); clearTimeout(deadline);
    frameTimer = cuesTimer = deadline = 0;
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
  function submitFrame(token, respond=false) {
    if (!valid(token) || inFlight) return;
    inFlight = true;
    try {
      if (!send({type:'image_input', image:frame(), text:currentPrompt, observe_only:!respond, watch_id:token})) throw new Error('Connection lost.');
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
        if ($('camera-motion').checked) Object.assign(data, await api.motionCues(video));
        if (!valid(token)) return;
        if ($('camera-face').checked) data.face = await api.faceCue(video, uid());
        if (valid(token)) send({type:'visual_cues',watch_id:token,cues:data});
      } catch (error) {
        if (valid(token)) {
          say(`Camera on · ${error.message}`);
          // A missing optional model must not start repeated downloads.
          $('camera-motion').checked = false; $('camera-face').checked = false;
        }
      } finally { cueBusy = false; }
    }
    if (valid(token)) cuesTimer = setTimeout(() => cues(token),2000);
  }
  async function start(prompt='What do you notice about what I am doing?', respond=true) {
    stop();
    const generation = sequence;
    window.dispatchEvent(new CustomEvent('aiko:camera-starting'));
    if (!send({type:'camera_state',active:false})) { say('Connect to Aiko first.'); return; }
    try {
      panel.hidden = false; say('Opening camera…');
      if (!await openVideo(generation)) return;
      session = crypto.randomUUID(); currentPrompt = prompt;
      $('watch-btn')?.classList.add('on');
      send({type:'camera_state', active:true, watch_id:session});
      submitFrame(session,respond); cues(session);
    } catch (error) { stop(error.message); window.addMessage?.('sys', error.message); window.hideTypingIndicator?.(); }
  }
  function visionEvent(event) {
    if (!session || event.watch_id !== session) return;
    if (event.status === 'done') {
      inFlight = false; clearTimeout(deadline);
      say('Camera on · scene refresh every 15 seconds');
      const token = session;
      clearTimeout(frameTimer); frameTimer = setTimeout(() => submitFrame(token),15000);
    } else if (event.status === 'error' || event.status === 'busy') {
      stop(event.message || 'Vision is unavailable.'); window.hideTypingIndicator?.();
    }
  }
  $('watch-btn')?.addEventListener('click', () => session ? stop() : start());
  $('camera-stop')?.addEventListener('click', () => stop());
  $('camera-enroll')?.addEventListener('click', async () => {
    if (!session || enrolling) return;
    const token = session; enrolling = true;
    try {
      say('Look at the camera with only your face visible…');
      // Avoid running descriptor inference concurrently with the cue tick.
      if (cueBusy) { say('Wait for the current gesture/face check, then try again.'); return; }
      await (await module()).enrollFace(video, uid(), () => valid(token));
      if (valid(token)) { $('camera-face').checked = true; say('Face remembered on this browser.'); }
    } catch (error) { if (valid(token)) say(error.message); }
    finally { enrolling = false; }
  });
  $('camera-forget')?.addEventListener('click', async () => {
    try { (await module()).forgetFace(uid()); $('camera-face').checked = false; say('Saved face template removed from this browser.'); }
    catch (error) { say(error.message); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', () => stop());
  window.AikoCamera = {start,stop,visionEvent, control(message) {
    if (message.action === 'stop') stop();
    else if (message.action === 'watch') start(message.prompt);
    else if (message.action === 'capture') { stop(); window.captureImage?.(message.prompt); }
  }, get active() { return !!session; }};
})();
