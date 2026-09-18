/* Consent-first desktop companion behaviour. Screen frames are captured only
   after a user gesture and sent once; no stream is retained or auto-started. */
(function () {
  const $ = id => document.getElementById(id);
  const screenBtn = $('screen-btn'), dialog = $('screen-consent-dialog'), consent = $('screen-consent-check');
  const confirm = $('screen-consent-confirm'), indicator = $('screen-live-indicator'), stop = $('stop-screen-share');
  const focusButton = $('focus-toggle'), focusState = $('focus-state'), focusTime = $('focus-time');
  const compactButton = $('companion-toggle'), status = $('companion-status');
  let focusStartedAt = 0, focusTicker = 0, screenStream = null;

  function send(payload) { return window.aikoSendWS?.(payload); }
  function format(seconds) { return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`; }
  function updateFocus() { const elapsed = Math.floor((Date.now() - focusStartedAt) / 1000); focusTime.textContent = format(elapsed); }
  function setAvatarMood(name) { if (window.aikoSetExpression) window.aikoSetExpression(name, .65); }
  function toggleFocus() {
    if (focusTicker) {
      const seconds = Math.max(1, Math.floor((Date.now() - focusStartedAt) / 1000));
      clearInterval(focusTicker); focusTicker = 0; focusState.textContent = 'READY'; focusButton.textContent = 'Start focus';
      try { localStorage.setItem('aiko-last-focus-session', JSON.stringify({ endedAt: new Date().toISOString(), seconds })); } catch (_) { /* optional local-only history */ }
      status.textContent = `focus session saved locally (${format(seconds)})`; setAvatarMood('neutral'); return;
    }
    focusStartedAt = Date.now(); updateFocus(); focusTicker = setInterval(updateFocus, 1000); focusState.textContent = 'FOCUS'; focusButton.textContent = 'End focus'; status.textContent = 'I’ll keep you company—one thing at a time.'; setAvatarMood('thinking');
  }
  focusButton?.addEventListener('click', toggleFocus);
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
  window.aikoCompanionVision = function (event) { if (event.status === 'working') status.textContent = 'looking at the screen…'; if (event.status === 'done') status.textContent = 'screen analysis complete'; if (event.status === 'error') status.textContent = 'screen analysis unavailable'; };
})();
