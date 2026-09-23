/* Consent-first desktop companion behaviour. Screen frames are captured only
   after a user gesture and sent once; no stream is retained or auto-started. */
(function () {
  const $ = id => document.getElementById(id);
  // Tauri shell detection runs first: the deferred vrm.js module reads
  // window.aikoIsTauri to switch the 3D canvas to alpha mode, and the
  // body.tauri-companion class below strips opaque page backgrounds so the
  // native transparent window actually shows the desktop through.
  const isTauri = !!(window.__TAURI_INTERNALS__ || window.__TAURI__);
  if (isTauri) {
    window.aikoIsTauri = true;
    document.body.classList.add('tauri-companion');
  }
  const screenBtn = $('screen-btn'), dialog = $('screen-consent-dialog'), consent = $('screen-consent-check');
  const confirmBtn = $('screen-consent-confirm'), indicator = $('screen-live-indicator'), stop = $('stop-screen-share');
  const focusButton = $('focus-toggle'), focusState = $('focus-state'), focusTime = $('focus-time');
  const compactButton = $('companion-toggle'), status = $('companion-status');
  const routine = $('companion-routine'), workingOn = $('companion-working-on'), taskToggle = $('task-toggle');
  const taskPanel = $('companion-task-panel'), taskForm = $('companion-task-form'), taskInput = $('companion-task-input'), taskList = $('companion-task-list');
  const clearData = $('clear-companion-data'), webcamToggle = $('webcam-toggle'), summaryList = $('companion-summary-list');
  const STORAGE_KEY = 'aiko-companion-state';
  let focusStartedAt = 0, focusTicker = 0, screenStream = null, webcamStream = null, webcamTimer = 0;
  let state = { routine: 'coding', workingOn: '', tasks: [], summaries: [] };

  function send(payload) { return window.aikoSendWS?.(payload); }
  function format(seconds) { return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }
  function updateFocus() { const elapsed = Math.floor((Date.now() - focusStartedAt) / 1000); focusTime.textContent = format(elapsed); }
  function setAvatarMood(name) { if (window.aikoSetExpression) window.aikoSetExpression(name, .65); }
  function loadState() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      if (raw && typeof raw === 'object') {
        if (typeof raw.routine === 'string' && routineCopy[raw.routine]) state.routine = raw.routine;
        if (typeof raw.workingOn === 'string') state.workingOn = raw.workingOn.slice(0, 120);
        if (Array.isArray(raw.tasks)) {
          state.tasks = raw.tasks
            .filter(t => t && typeof t.title === 'string' && typeof t.id === 'string')
            .slice(0, 200)
            .map(t => ({ id: t.id.slice(0, 64), title: t.title.slice(0, 140), done: t.done === true }));
        }
        if (Array.isArray(raw.summaries)) {
          state.summaries = raw.summaries
            .filter(s => s && Number.isFinite(s.seconds))
            .slice(0, 30)
            .map(s => ({
              endedAt: typeof s.endedAt === 'string' ? s.endedAt : new Date().toISOString(),
              seconds: Math.max(1, Math.min(86400, Math.floor(s.seconds))),
              routine: typeof s.routine === 'string' && routineCopy[s.routine] ? s.routine : state.routine,
              workingOn: typeof s.workingOn === 'string' ? s.workingOn.slice(0, 120) : '',
            }));
        }
      }
    } catch (_) { /* start clean */ }
    if (routine) routine.value = state.routine;
    if (workingOn) workingOn.value = state.workingOn;
    renderTasks(); renderSummaries();
  }
  function saveState() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) { /* optional local-only state */ } }
  function renderTasks() { if (!taskList || !Array.isArray(state.tasks)) return; taskList.replaceChildren(...state.tasks.map(task => { const li = document.createElement('li'); li.className = task.done ? 'done' : ''; const check = document.createElement('input'); check.type = 'checkbox'; check.checked = task.done === true; check.setAttribute('aria-label', `Mark “${task.title}” done`); check.addEventListener('change', () => { task.done = check.checked; if (task.done) celebrateTask(task.title); saveState(); renderTasks(); }); const text = document.createElement('span'); text.textContent = task.title; const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'companion-task-delete'; remove.textContent = '×'; remove.setAttribute('aria-label', `Delete ${task.title}`); remove.addEventListener('click', () => { state.tasks = state.tasks.filter(item => item.id !== task.id); saveState(); renderTasks(); }); li.append(check, text, remove); return li; })); }
  function renderSummaries() { if (!summaryList || !Array.isArray(state.summaries)) return; summaryList.replaceChildren(...state.summaries.map(summary => { const li = document.createElement('li'); li.textContent = `${format(summary.seconds)} · ${summary.routine}${summary.workingOn ? ` · ${summary.workingOn}` : ''}`; return li; })); }
  function notify(title, body) { if ('Notification' in window && Notification.permission === 'granted') new Notification(title, { body, tag: 'aiko-companion' }); }
  function celebrateTask(title) { if (status) status.textContent = `nice work — “${title}” is complete`; window.dispatchEvent(new CustomEvent('aiko:task-done', { detail: { title } })); setAvatarMood('happy'); window.aikoSetPose?.('raiseHand', true); setTimeout(() => window.aikoSetPose?.('raiseHand', false), 1500); notify('Aiko', `Completed: ${title}`); }
  const routineCopy = { coding: 'I’ll keep the noise down while you build.', study: 'Study mode: steady pace, gentle reminders.', writing: 'Writing mode: keep the thread, protect the flow.', gaming: 'Game mode: I’m here for the win.', quiet: 'Quiet company mode: I’ll stay subtle.' };
  function toggleFocus() {
    if (focusTicker) {
      const seconds = Math.max(1, Math.floor((Date.now() - focusStartedAt) / 1000));
      clearInterval(focusTicker); focusTicker = 0; focusState.textContent = 'READY'; focusButton.textContent = 'Start focus';
      state.summaries = [{ endedAt: new Date().toISOString(), seconds, routine: state.routine, workingOn: state.workingOn }, ...state.summaries].slice(0, 30); saveState(); renderSummaries();
      status.textContent = `focus session saved locally (${format(seconds)})`; setAvatarMood('neutral');
      window.dispatchEvent(new CustomEvent('aiko:focus-stop', { detail: { seconds, routine: state.routine } }));
      return;
    }
    focusStartedAt = Date.now(); updateFocus(); focusTicker = setInterval(updateFocus, 1000); focusState.textContent = 'FOCUS'; focusButton.textContent = 'End focus'; status.textContent = routineCopy[state.routine] || routineCopy.coding; setAvatarMood('thinking'); if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission();
    window.dispatchEvent(new CustomEvent('aiko:focus-start', { detail: { routine: state.routine } }));
  }
  focusButton?.addEventListener('click', toggleFocus);
  routine?.addEventListener('change', () => { if (!routineCopy[routine.value]) return; state.routine = routine.value; saveState(); status.textContent = routineCopy[state.routine]; setAvatarMood(state.routine === 'gaming' ? 'happy' : 'thinking'); });
  workingOn?.addEventListener('change', () => { state.workingOn = workingOn.value.trim(); saveState(); status.textContent = state.workingOn ? `with you on: ${state.workingOn}` : 'settling in'; });
  taskToggle?.addEventListener('click', () => { if (!taskPanel) return; taskPanel.hidden = !taskPanel.hidden; taskToggle.textContent = taskPanel.hidden ? 'Tasks' : 'Hide tasks'; });
  taskForm?.addEventListener('submit', event => { event.preventDefault(); const title = taskInput.value.trim().slice(0, 140); if (!title) return; state.tasks.push({ id: (window.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`), title, done: false }); taskInput.value = ''; saveState(); renderTasks(); });
  clearData?.addEventListener('click', () => { if (!window.confirm('Delete Aiko companion tasks, focus summaries, and current-work state from this browser?')) return; state = { routine: 'coding', workingOn: '', tasks: [], summaries: [] }; saveState(); loadState(); if (status) status.textContent = 'local companion history deleted'; });
  compactButton?.addEventListener('click', () => { document.body.classList.toggle('companion-compact'); compactButton.textContent = document.body.classList.contains('companion-compact') ? 'Expand chat' : 'Compact'; });
  // Ctrl/Cmd+Shift+A is deliberately local to the page. Native wrappers can map
  // the same shortcut to show/hide their window without changing this UI.
  window.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'a') {
      event.preventDefault(); compactButton?.click();
    }
  });
  consent?.addEventListener('change', () => { if (confirmBtn) confirmBtn.disabled = !consent.checked; });
  screenBtn?.addEventListener('click', () => { if (!navigator.mediaDevices?.getDisplayMedia) { window.addMessage?.('sys', 'Screen sharing is not supported in this browser.'); return; } if (dialog && typeof dialog.showModal === 'function' && !dialog.open) dialog.showModal(); });
  dialog?.addEventListener('close', () => { if (dialog.returnValue === 'share' && consent?.checked) captureScreen(); if (consent) consent.checked = false; if (confirmBtn) confirmBtn.disabled = true; });
  let screenEndedHandler = null;
  async function captureScreen() {
    try {
      // Browser picker is intentionally invoked per capture: there is no hidden
      // capture loop and the chosen window/display is always user-selected.
      screenStream = await navigator.mediaDevices.getDisplayMedia({ video: { width: { max: 1440 }, height: { max: 900 }, frameRate: { max: 2 } }, audio: false });
      const track = screenStream.getVideoTracks()[0];
      if (!track) throw new Error('no-video-track');
      // Fires only if the user revokes the share via browser UI before we grab
      // the single frame. Detached again before our own one-shot stop below so
      // the confirmation indicator stays until the user dismisses it.
      screenEndedHandler = () => stopScreenShare();
      track.addEventListener('ended', screenEndedHandler, { once: true });
      if (indicator) indicator.hidden = false;
      if (status) status.textContent = 'screen shared for one private analysis';
      const video = document.createElement('video'); video.srcObject = screenStream; video.playsInline = true; video.muted = true; await video.play();
      if (!video.videoWidth || !video.videoHeight) throw new Error('no-frame');
      const canvas = document.createElement('canvas'); const scale = Math.min(1, 1280 / video.videoWidth); canvas.width = Math.max(1, Math.round(video.videoWidth * scale)); canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
      canvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
      const question = $('user-input')?.value.trim() || 'Describe the visible application and help me with the next useful step.';
      video.srcObject = null;
      send({ type: 'screen_input', image: canvas.toDataURL('image/jpeg', .82), text: question });
      if ($('user-input')) $('user-input').value = ''; window.addMessage?.('user', `▣ Shared screen — ${question}`);
      // Stop immediately after the single frame; the indicator remains until the user dismisses it.
      try { track.removeEventListener('ended', screenEndedHandler); } catch (_) { /* listener already fired */ }
      screenEndedHandler = null;
      screenStream.getTracks().forEach(t => { try { t.stop(); } catch (_) { /* already stopped */ } });
      screenStream = null;
    } catch (error) { if (error?.name !== 'NotAllowedError' && error?.message !== 'no-frame' && error?.message !== 'no-video-track') window.addMessage?.('sys', 'Screen sharing could not start. Please try again.'); if (error?.message === 'no-frame' || error?.message === 'no-video-track') window.addMessage?.('sys', 'Could not grab a screen frame. Please try again.'); stopScreenShare(); }
  }
  function stopScreenShare() { if (screenEndedHandler && screenStream) { try { screenStream.getVideoTracks()[0]?.removeEventListener('ended', screenEndedHandler); } catch (_) { /* already fired */ } } screenEndedHandler = null; if (screenStream) { screenStream.getTracks().forEach(track => { try { track.stop(); } catch (_) { /* already stopped */ } }); } screenStream = null; if (indicator) indicator.hidden = true; if (status) status.textContent = focusTicker ? 'focus session active' : 'settling in'; }
  stop?.addEventListener('click', stopScreenShare);
  // ── T4/T5 motion reflex arc (JS mirror of cognition/flysense/motion.py) ──
  // LPTC pooling gains baked from MaleCNS v1.0 (258,046 T4/T5->HS/VS synapses:
  // HS<-T4a/T5a front-to-back, VS<-T4d/T5d downward): GH=0.34, GV=0.66.
  // Runs entirely in-browser on 48px thumbnails; only a throttled presence
  // LEVEL (0..1) ever leaves the page, and only on bin change. Like the fly's
  // LPTC->motor reflex, reactions stay local; the server just gets arousal.
  const MOTION_W = 48, MOTION_H = 36, MOTION_THR = 0.02, MOTION_CHANGE_THR = 0.10;
  const MOTION_GH = 0.34, MOTION_GV = 0.66;
  let motionPrev = null, motionLevel = 0, motionBin = -1, lastPresenceSent = 0;
  window.aikoMotionEnergy = function (prev, curr) {
    let hOn = 0, hOff = 0, vOn = 0, vOff = 0, change = 0;
    const n = prev.length;
    const W = MOTION_W;
    for (let i = 0; i < n; i++) {
      const c = curr[i], p = prev[i];
      const onC = c > 0 ? c : 0, offC = c < 0 ? -c : 0;
      const x = i % W;
      const pR = x > 0 ? prev[i - 1] : p;
      const pL = x < W - 1 ? prev[i + 1] : p;
      const pU = i >= W ? prev[i - W] : p;
      const pD = i + W < n ? prev[i + W] : p;
      hOn += onC * pR; hOff += offC * pR;
      hOn -= onC * pL; hOff -= offC * pL;
      vOn += onC * pU - onC * pD; vOff += offC * pU - offC * pD;
      change += Math.abs(c - p);
    }
    const h = (Math.abs(hOn) + Math.abs(hOff)) / n * MOTION_GH;
    const v = (Math.abs(vOn) + Math.abs(vOff)) / n * MOTION_GV;
    return { h, v, total: h + v, change: change / n };
  };
  function sampleMotionGray(video, canvas) {
    canvas.width = MOTION_W; canvas.height = MOTION_H;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0, MOTION_W, MOTION_H);
    const d = ctx.getImageData(0, 0, MOTION_W, MOTION_H).data;
    const g = new Float32Array(MOTION_W * MOTION_H);
    let mean = 0;
    for (let i = 0; i < g.length; i++) {
      g[i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 765; // 0..1
      mean += g[i];
    }
    mean /= g.length;
    let peak = 0;
    for (let i = 0; i < g.length; i++) { g[i] -= mean; const a = Math.abs(g[i]); if (a > peak) peak = a; }
    if (peak > 0) for (let i = 0; i < g.length; i++) g[i] /= peak;
    return g;
  }
  function presenceBin(level) { return level < 0.15 ? 0 : level < 0.5 ? 1 : 2; }
  function reportPresence(level, force) {
    const now = Date.now();
    const bin = presenceBin(level);
    if (!force && (bin === motionBin || now - lastPresenceSent < 10000)) return;
    motionBin = bin; lastPresenceSent = now;
    send({ type: 'presence', level: Math.round(level * 100) / 100 });
  }
  async function toggleWebcam() {
    if (webcamStream) { stopWebcam(); return; }
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }, audio: false });
      if (webcamToggle) webcamToggle.textContent = 'Webcam on';
      if (status) status.textContent = 'webcam presence is local and can be stopped any time';
      const video = document.createElement('video'); video.srcObject = webcamStream; video.muted = true; video.playsInline = true; await video.play();
      // FaceDetector is optional and remains entirely in-browser. Aiko only uses
      // its local face position to choose a gentle look-around reaction.
      let detector = null;
      try { detector = 'FaceDetector' in window ? new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 }) : null; } catch (_) { detector = null; }
      let detecting = false;
      const motionCanvas = document.createElement('canvas');
      webcamTimer = window.setInterval(async () => {
        if (!webcamStream) return;
        // T4/T5 reflex arc: motion energy from 48px thumbnails (local only).
        try {
          const gray = sampleMotionGray(video, motionCanvas);
          if (motionPrev) {
            const e = window.aikoMotionEnergy(motionPrev, gray);
            const moved = e.total >= MOTION_THR || e.change >= MOTION_CHANGE_THR;
            motionLevel = motionLevel * 0.7 + Math.min(1, e.total * 8 + e.change) * 0.3;
            if (moved && status) status.textContent = 'movement nearby…';
            window.aikoSetPose?.('lookAround', moved);
            reportPresence(motionLevel, false);
          }
          motionPrev = gray;
        } catch (_) { /* motion is best effort; FaceDetector path below is independent */ }
        if (!detector || detecting) return; detecting = true; try { const faces = await detector.detect(video); if (faces.length > 0) window.aikoSetPose?.('lookAround', true); } catch (_) { /* detection is best effort */ } finally { detecting = false; } }, 1200);
    } catch (_) { window.addMessage?.('sys', 'Webcam access was unavailable. Aiko will keep using her idle animations.'); stopWebcam(); }
  }
  function stopWebcam() { if (webcamTimer) clearInterval(webcamTimer); webcamTimer = 0; if (webcamStream) webcamStream.getTracks().forEach(track => { try { track.stop(); } catch (_) { /* already stopped */ } }); webcamStream = null; motionPrev = null; motionLevel = 0; motionBin = -1; reportPresence(0, true); if (webcamToggle) webcamToggle.textContent = 'Webcam off'; window.aikoSetPose?.('lookAround', false); }
  webcamToggle?.addEventListener('click', toggleWebcam);
  window.addEventListener('pagehide', () => { try { stopWebcam(); } catch (_) { /* teardown only */ } }, { once: true });
  loadState();
  window.aikoCompanionVision = function (event) { if (!status || !event) return; if (event.status === 'working') status.textContent = 'looking at the screen…'; if (event.status === 'done') status.textContent = 'screen analysis complete'; if (event.status === 'error') status.textContent = 'screen analysis unavailable'; };
})();
