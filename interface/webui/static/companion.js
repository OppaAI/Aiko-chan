/* Consent-first desktop companion behaviour. Screen frames are captured only
   after a user gesture and sent once; no stream is retained or auto-started. */
(function () {
  const $ = id => document.getElementById(id);
  const screenBtn = $('screen-btn'), dialog = $('screen-consent-dialog'), consent = $('screen-consent-check');
  const confirm = $('screen-consent-confirm'), indicator = $('screen-live-indicator'), stop = $('stop-screen-share');
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
  function loadState() { try { state = { ...state, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') }; } catch (_) { /* start clean */ } routine.value = state.routine; workingOn.value = state.workingOn; renderTasks(); renderSummaries(); }
  function saveState() { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) { /* optional local-only state */ } }
  function renderTasks() { taskList.replaceChildren(...state.tasks.map(task => { const li = document.createElement('li'); li.className = task.done ? 'done' : ''; const check = document.createElement('input'); check.type = 'checkbox'; check.checked = task.done; check.addEventListener('change', () => { task.done = check.checked; if (task.done) celebrateTask(task.title); saveState(); renderTasks(); }); const text = document.createElement('span'); text.textContent = task.title; const remove = document.createElement('button'); remove.className = 'companion-task-delete'; remove.textContent = '×'; remove.setAttribute('aria-label', `Delete ${task.title}`); remove.addEventListener('click', () => { state.tasks = state.tasks.filter(item => item.id !== task.id); saveState(); renderTasks(); }); li.append(check, text, remove); return li; })); }
  function renderSummaries() { summaryList.replaceChildren(...state.summaries.map(summary => { const li = document.createElement('li'); li.textContent = `${format(summary.seconds)} · ${summary.routine}${summary.workingOn ? ` · ${summary.workingOn}` : ''}`; return li; })); }
  function notify(title, body) { if ('Notification' in window && Notification.permission === 'granted') new Notification(title, { body, tag: 'aiko-companion' }); }
  function celebrateTask(title) { status.textContent = `nice work — “${title}” is complete`; setAvatarMood('happy'); window.aikoSetPose?.('raiseHand', true); setTimeout(() => window.aikoSetPose?.('raiseHand', false), 1500); notify('Aiko', `Completed: ${title}`); }
  const routineCopy = { coding: 'I’ll keep the noise down while you build.', study: 'Study mode: steady pace, gentle reminders.', writing: 'Writing mode: keep the thread, protect the flow.', gaming: 'Game mode: I’m here for the win.', quiet: 'Quiet company mode: I’ll stay subtle.' };
  function toggleFocus() {
    if (focusTicker) {
      const seconds = Math.max(1, Math.floor((Date.now() - focusStartedAt) / 1000));
      clearInterval(focusTicker); focusTicker = 0; focusState.textContent = 'READY'; focusButton.textContent = 'Start focus';
      state.summaries = [{ endedAt: new Date().toISOString(), seconds, routine: state.routine, workingOn: state.workingOn }, ...state.summaries].slice(0, 30); saveState(); renderSummaries();
      status.textContent = `focus session saved locally (${format(seconds)})`; setAvatarMood('neutral'); return;
    }
    focusStartedAt = Date.now(); updateFocus(); focusTicker = setInterval(updateFocus, 1000); focusState.textContent = 'FOCUS'; focusButton.textContent = 'End focus'; status.textContent = routineCopy[state.routine]; setAvatarMood('thinking'); if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission();
  }
  focusButton?.addEventListener('click', toggleFocus);
  routine?.addEventListener('change', () => { state.routine = routine.value; saveState(); status.textContent = routineCopy[state.routine]; setAvatarMood(state.routine === 'gaming' ? 'happy' : 'thinking'); });
  workingOn?.addEventListener('change', () => { state.workingOn = workingOn.value.trim(); saveState(); status.textContent = state.workingOn ? `with you on: ${state.workingOn}` : 'settling in'; });
  taskToggle?.addEventListener('click', () => { taskPanel.hidden = !taskPanel.hidden; taskToggle.textContent = taskPanel.hidden ? 'Tasks' : 'Hide tasks'; });
  taskForm?.addEventListener('submit', event => { event.preventDefault(); const title = taskInput.value.trim(); if (!title) return; state.tasks.push({ id: crypto.randomUUID?.() || String(Date.now()), title, done: false }); taskInput.value = ''; saveState(); renderTasks(); });
  clearData?.addEventListener('click', () => { if (!confirm('Delete Aiko companion tasks, focus summaries, and current-work state from this browser?')) return; state = { routine: 'coding', workingOn: '', tasks: [], summaries: [] }; saveState(); loadState(); status.textContent = 'local companion history deleted'; });
  compactButton?.addEventListener('click', () => { document.body.classList.toggle('companion-compact'); compactButton.textContent = document.body.classList.contains('companion-compact') ? 'Expand chat' : 'Compact'; });
  // Ctrl/Cmd+Shift+A is deliberately local to the page. Native wrappers can map
  // the same shortcut to show/hide their window without changing this UI.
  window.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'a') {
      event.preventDefault(); compactButton?.click();
    }
  });
  consent?.addEventListener('change', () => { confirm.disabled = !consent.checked; });
  screenBtn?.addEventListener('click', () => { if (!navigator.mediaDevices?.getDisplayMedia) { window.addMessage?.('sys', 'Screen sharing is not supported in this browser.'); return; } dialog.showModal(); });
  dialog?.addEventListener('close', () => { if (dialog.returnValue === 'share' && consent.checked) captureScreen(); consent.checked = false; confirm.disabled = true; });
  async function captureScreen() {
    try {
      // Browser picker is intentionally invoked per capture: there is no hidden
      // capture loop and the chosen window/display is always user-selected.
      screenStream = await navigator.mediaDevices.getDisplayMedia({ video: { width: { max: 1440 }, height: { max: 900 }, frameRate: { max: 2 } }, audio: false });
      indicator.hidden = false; status.textContent = 'screen shared for one private analysis';
      const track = screenStream.getVideoTracks()[0]; track.addEventListener('ended', stopScreenShare, { once: true });
      const video = document.createElement('video'); video.srcObject = screenStream; video.playsInline = true; await video.play();
      const canvas = document.createElement('canvas'); const scale = Math.min(1, 1280 / video.videoWidth); canvas.width = Math.max(1, Math.round(video.videoWidth * scale)); canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
      canvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
      const question = $('user-input')?.value.trim() || 'Describe the visible application and help me with the next useful step.';
      send({ type: 'screen_input', image: canvas.toDataURL('image/jpeg', .82), text: question });
      if ($('user-input')) $('user-input').value = ''; window.addMessage?.('user', `▣ Shared screen — ${question}`);
      // Stop immediately after the single frame; the indicator remains until the user dismisses it.
      screenStream.getTracks().forEach(track => track.stop());
    } catch (error) { if (error.name !== 'NotAllowedError') window.addMessage?.('sys', 'Screen sharing could not start. Please try again.'); stopScreenShare(); }
  }
  function stopScreenShare() { if (screenStream) screenStream.getTracks().forEach(track => track.stop()); screenStream = null; indicator.hidden = true; status.textContent = focusTicker ? 'focus session active' : 'settling in'; }
  stop?.addEventListener('click', stopScreenShare);
  async function toggleWebcam() {
    if (webcamStream) { stopWebcam(); return; }
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }, audio: false });
      webcamToggle.textContent = 'Webcam on'; status.textContent = 'webcam presence is local and can be stopped any time';
      const video = document.createElement('video'); video.srcObject = webcamStream; video.muted = true; video.playsInline = true; await video.play();
      // FaceDetector is optional and remains entirely in-browser. Aiko only uses
      // its local face position to choose a gentle look-around reaction.
      const detector = 'FaceDetector' in window ? new FaceDetector({ fastMode: true, maxDetectedFaces: 1 }) : null;
      webcamTimer = window.setInterval(async () => { if (!detector || !webcamStream) return; try { const faces = await detector.detect(video); window.aikoSetPose?.('lookAround', faces.length > 0); } catch (_) { /* detection is best effort */ } }, 1200);
    } catch (_) { window.addMessage?.('sys', 'Webcam access was unavailable. Aiko will keep using her idle animations.'); stopWebcam(); }
  }
  function stopWebcam() { if (webcamTimer) clearInterval(webcamTimer); webcamTimer = 0; if (webcamStream) webcamStream.getTracks().forEach(track => track.stop()); webcamStream = null; webcamToggle.textContent = 'Webcam off'; window.aikoSetPose?.('lookAround', false); }
  webcamToggle?.addEventListener('click', toggleWebcam);
  loadState();
  window.aikoCompanionVision = function (event) { if (event.status === 'working') status.textContent = 'looking at the screen…'; if (event.status === 'done') status.textContent = 'screen analysis complete'; if (event.status === 'error') status.textContent = 'screen analysis unavailable'; };
})();
