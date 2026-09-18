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
  function celebrateTask(title) { if (status) status.textContent = `nice work — “${title}” is complete`; setAvatarMood('happy'); window.aikoSetPose?.('raiseHand', true); setTimeout(() => window.aikoSetPose?.('raiseHand', false), 1500); notify('Aiko', `Completed: ${title}`); }
  const routineCopy = { coding: 'I’ll keep the noise down while you build.', study: 'Study mode: steady pace, gentle reminders.', writing: 'Writing mode: keep the thread, protect the flow.', gaming: 'Game mode: I’m here for the win.', quiet: 'Quiet company mode: I’ll stay subtle.' };
  function toggleFocus() {
    if (focusTicker) {
      const seconds = Math.max(1, Math.floor((Date.now() - focusStartedAt) / 1000));
      clearInterval(focusTicker); focusTicker = 0; focusState.textContent = 'READY'; focusButton.textContent = 'Start focus';
      state.summaries = [{ endedAt: new Date().toISOString(), seconds, routine: state.routine, workingOn: state.workingOn }, ...state.summaries].slice(0, 30); saveState(); renderSummaries();
      status.textContent = `focus session saved locally (${format(seconds)})`; setAvatarMood('neutral'); return;
    }
    focusStartedAt = Date.now(); updateFocus(); focusTicker = setInterval(updateFocus, 1000); focusState.textContent = 'FOCUS'; focusButton.textContent = 'End focus'; status.textContent = routineCopy[state.routine] || routineCopy.coding; setAvatarMood('thinking'); if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission();
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
      webcamTimer = window.setInterval(async () => { if (!detector || !webcamStream || detecting) return; detecting = true; try { const faces = await detector.detect(video); window.aikoSetPose?.('lookAround', faces.length > 0); } catch (_) { /* detection is best effort */ } finally { detecting = false; } }, 1200);
    } catch (_) { window.addMessage?.('sys', 'Webcam access was unavailable. Aiko will keep using her idle animations.'); stopWebcam(); }
  }
  function stopWebcam() { if (webcamTimer) clearInterval(webcamTimer); webcamTimer = 0; if (webcamStream) webcamStream.getTracks().forEach(track => { try { track.stop(); } catch (_) { /* already stopped */ } }); webcamStream = null; if (webcamToggle) webcamToggle.textContent = 'Webcam off'; window.aikoSetPose?.('lookAround', false); }
  webcamToggle?.addEventListener('click', toggleWebcam);
  window.addEventListener('pagehide', () => { try { stopWebcam(); } catch (_) { /* teardown only */ } }, { once: true });
  loadState();
  window.aikoCompanionVision = function (event) { if (!status || !event) return; if (event.status === 'working') status.textContent = 'looking at the screen…'; if (event.status === 'done') status.textContent = 'screen analysis complete'; if (event.status === 'error') status.textContent = 'screen analysis unavailable'; };
})();
