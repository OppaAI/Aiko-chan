/**
 * script.js
 * Real-time chat UI with WebSocket bridge to Aiko backend.
 * Handles voice I/O (mic capture via pcm-worklet + VAD, TTS playback with mouth sync),
 * WebSocket message routing (chat, token streaming, vitals, expressions, visemes),
 * initialization status tracking (step progress), and mic/text input modes.
 *
 * Core flows:
 *   - mic capture: AudioWorklet → VAD frame → server (if speech detected)
 *   - TTS playback: binary WAV frames → decode → analyser RMS → lip-sync blendshapes
 *   - chat: text input or voice transcription → user_input message → token streaming
 *   - gestures: server sends expression/viseme/pose → window.aikoSetX() → vrm.js
 *
 *      sets AIKO_TTS_STARTED_AT on TTS start + AIKO_BARGE_ECHO_GUARD_MS on mic start
 *      so vad.js can ignore self-echo barge for BARGE_IN_ECHO_GUARD_MS after TTS begins.
 *
 * UI extras:
 *   - theme switch (style.css light ⇄ style-dark.css dark), persisted in localStorage
 *   - shoujo-mode flourishes: chat bubbles, typing indicator, emotion badge,
 *     floating emotion particles (hidden by style-dark.css in dark mode)
 */

// ── DOM refs ──────────────────────────────────────────────────────────────
const initPanel = document.getElementById('init-panel');
const chatPanel = document.getElementById('chat-panel');
const toolStatus = document.getElementById('tool-status');
const content = document.getElementById('content');
const allOnline = document.getElementById('all-online');
const input = document.getElementById('user-input');
const cameraBtn = document.getElementById('camera-btn');
const screenBtn = document.getElementById('screen-btn');
const micBtn = document.getElementById('mic-btn');
const sendBtn = document.getElementById('send-btn');
const voiceSt = document.getElementById('voice-status');
const clock = document.getElementById('panel-clock');
const wsDot = document.getElementById('ws-dot');
const wsLabel = document.getElementById('ws-label');
const vadDot = document.getElementById('vad-dot');
const vadStatus = document.getElementById('vad-status');

const emotionBadge = document.getElementById('emotion-badge');
const emotionEmoji = document.getElementById('emotion-emoji');
const emotionText = document.getElementById('emotion-text');

const bootProgressFill = document.getElementById('boot-progress-fill');
const bootProgressMsg = document.getElementById('boot-progress-msg');

const vTok = document.getElementById('v-tok');
const vToks = document.getElementById('v-toks');
const vRam = document.getElementById('v-ram');
const vUp = document.getElementById('v-up');
const vMode = document.getElementById('v-mode');

const AUTO_MIC = false;
let autoListenRequested = false;

// Default barge-in off until server mic.start sets it (S0).
window.AIKO_BARGE_IN_ENABLED = false;

// ── theme switch (style.css light ⇄ style-dark.css dark) ──────────────────
// index.html's inline <head> script has already applied the saved choice to
// the <link disabled> flags to avoid a flash; here we just keep the toggle
// button icon in sync and handle clicks.
const THEME_KEY = 'aiko-theme';
const themeToggleBtn = document.getElementById('theme-toggle');

function applyTheme(theme) {
  const dark = theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const lightLink = document.getElementById('theme-style-light');
  const darkLink = document.getElementById('theme-style-dark');
  if (lightLink && darkLink) {
    lightLink.disabled = dark;
    darkLink.disabled = !dark;
  }
  if (themeToggleBtn) {
    const sunIcon = themeToggleBtn.querySelector('.icon-sun');
    const moonIcon = themeToggleBtn.querySelector('.icon-moon');
    if (sunIcon) sunIcon.style.display = dark ? '' : 'none';
    if (moonIcon) moonIcon.style.display = dark ? 'none' : '';
    themeToggleBtn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
  }
}

// ── right sidebar collapse ─────────────────────────────────────────────
(function initPanelCollapse() {
  const btn = document.getElementById('panel-collapse');
  if (!btn) return;
  const apply = (collapsed) => {
    document.body.classList.toggle('panel-collapsed', collapsed);
    btn.innerHTML = collapsed ? '&#10217;' : '&#10218;';
    btn.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    btn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
  };
  btn.addEventListener('click', () => {
    const next = !document.body.classList.contains('panel-collapsed');
    apply(next);
    try { localStorage.setItem('aiko-panel-collapsed', next ? '1' : '0'); } catch (_) {}
  });
  try { if (localStorage.getItem('aiko-panel-collapsed') === '1') apply(true); } catch (_) {}
})();

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (_) { /* storage blocked */ }
  applyTheme(saved === 'dark' ? 'dark' : 'light');
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* storage blocked */ }
  applyTheme(next);
}

initTheme();
if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);

// ── viewport height fix (mobile browser toolbar collapse/expand) ─────────
function setAppHeight() {
  const h = window.visualViewport ? window.visualViewport.height : window.innerHeight;
  document.documentElement.style.setProperty('--app-height', `${h}px`);
}
setAppHeight();
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', setAppHeight);
  window.visualViewport.addEventListener('scroll', setAppHeight);
} else {
  window.addEventListener('resize', setAppHeight);
}
window.addEventListener('orientationchange', () => setTimeout(setAppHeight, 100));

// ── clock ─────────────────────────────────────────────────────────────────
const stageDate = document.getElementById('stage-date');
const stageTime = document.getElementById('stage-time');
function tickClock() {
  const now = new Date();
  clock.textContent = now.toLocaleString('en-GB', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  });
  if (stageDate) stageDate.textContent = now.toLocaleString('en-US', { weekday: 'short', month: 'long', day: 'numeric' });
  if (stageTime) stageTime.textContent = now.toLocaleString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
}
tickClock();
setInterval(tickClock, 1000);

// ── VAD init ──────────────────────────────────────────────────────────────
initVAD().then((status) => {
  vadDot.className = 'dot on';
  vadStatus.textContent = 'vad ready';
  vadStatus.className = 'ready';
}).catch(err => {
  vadStatus.textContent = 'vad failed';
  console.error('[vad] init error:', err);
});

// ── step / init tracking ──────────────────────────────────────────────────
let bootDone = 0, bootTotal = 0;
let bootKeys = {};

function handleStep(msg) {
  const key = msg.key, state = msg.state;
  const label = msg.label || key;
  if (state === 'loading') {
    bootProgressMsg.textContent = label;
  } else if (['done', 'skip', 'error'].includes(state)) {
    if (!bootKeys[key]) {
      bootKeys[key] = true;
      bootTotal++;
    }
    bootDone++;
  }
  const total = bootTotal || 1;
  const pct = Math.min(100, Math.round(100 * bootDone / total));
  bootProgressFill.style.width = pct + '%';
  if (bootTotal > 0 && bootDone >= bootTotal) allOnline.classList.add('show');
}

// ── phase switch ──────────────────────────────────────────────────────────
let chatPhaseActive = false;

function switchToChat() {
  chatPhaseActive = true;
  initPanel.classList.add('hidden');
  chatPanel.classList.add('show');
  input.focus();
}

// ── chat rendering ────────────────────────────────────────────────────────
let streamActive = false;
let streamRawText = '';
let streamExprApplied = null;  // expression name applied once per stream turn
let sourcesRow = null;
let filesRow = null;
let typingIndicator = null;

// ── Emotion Particles (shoujo mode; hidden by style-dark.css) ────────────
const EMOJI_PARTICLES = {
  happy:    ['🌸', '✨', '💗', '💖', '🌟'],
  angry:    ['💢', '🔥', '⚡', '💥'],
  sorrow:   ['💧', '😢', '💔', '🌧️'],
  surprised:['❗', '✨', '💫', '🌟'],
  fun:      ['🎉', '🎈', '✨', '🌈', '🎀'],
  neutral:  ['💭', '✦', '·'],
};

function spawnEmotionParticles(exprName) {
  const particles = EMOJI_PARTICLES[exprName] || EMOJI_PARTICLES.neutral;
  const count = Math.min(6, particles.length);
  for (let i = 0; i < count; i++) {
    const el = document.createElement('div');
    el.textContent = particles[i % particles.length];
    el.style.cssText = `
      position: fixed;
      pointer-events: none;
      font-size: ${14 + Math.random() * 10}px;
      z-index: 100;
      left: ${20 + Math.random() * 60}%;
      top: ${20 + Math.random() * 40}%;
      opacity: 0;
      animation: emotionParticle 1.5s ease-out forwards;
      animation-delay: ${i * 0.1}s;
    `;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2000);
  }
}

// Inject keyframes if not present
if (!document.getElementById('shoujo-animations')) {
  const style = document.createElement('style');
  style.id = 'shoujo-animations';
  style.textContent = `
    @keyframes emotionParticle {
      0% { opacity: 0; transform: translateY(0) scale(0.5); }
      30% { opacity: 0.8; transform: translateY(-20px) scale(1.1); }
      100% { opacity: 0; transform: translateY(-60px) scale(0.8); }
    }
  `;
  document.head.appendChild(style);
}

// ── karaoke caption (game dialogue box) ─────────────────────────────────
// Aiko's words appear as a game-style caption with karaoke word reveal,
// paced by the backend typewriter sync. The chat panel stays a slim log;
// her full history lives in the transcript (toggle in the header).
const captionBox = document.getElementById('caption-box');
const captionEmotionEl = document.getElementById('caption-emotion');
const captionTextEl = document.getElementById('caption-text');
const presenceDot = document.getElementById('presence-dot');
const presenceText = document.getElementById('presence-text');
const statusEmotion = document.getElementById('status-emotion');
let captionHideTimer = null;
let captionLastDialogue = null;
let currentPresence = 'idle';
const transcriptLog = [];
let transcriptVisible = false;

function captionShow() {
  if (!captionBox) return;
  captionBox.classList.remove('caption-hidden');
  clearTimeout(captionHideTimer);
  captionHideTimer = null;
}
function captionRender(dialogue, emoji) {
  if (!captionTextEl) return;
  if (dialogue === captionLastDialogue) { captionShow(); return; }
  captionLastDialogue = dialogue;
  if (emoji && captionEmotionEl) captionEmotionEl.textContent = emoji;
  captionTextEl.innerHTML = '';
  const frag = document.createDocumentFragment();
  const wordSpans = [];
  for (const p of String(dialogue).split(/(\s+)/)) {
    if (!p) continue;
    const sp = document.createElement('span');
    const isWord = /\S/.test(p);
    sp.className = isWord ? 'w' : 'wsp';
    sp.textContent = p;
    frag.appendChild(sp);
    if (isWord) wordSpans.push(sp);
  }
  // karaoke: the two freshest words glow as if being spoken
  wordSpans.slice(-2).forEach(sp => sp.classList.add('live'));
  captionTextEl.appendChild(frag);
  captionTextEl.scrollTop = captionTextEl.scrollHeight;
  captionShow();
}
function captionCommit(dialogue, emoji, holdMs = 14000) {
  captionRender(dialogue, emoji);
  clearTimeout(captionHideTimer);
  captionHideTimer = setTimeout(() => {
    if (captionBox) captionBox.classList.add('caption-hidden');
  }, holdMs);
}
function captionSpeaking(on) {
  if (captionBox) captionBox.classList.toggle('speaking', !!on);
}

// ── presence ──
const stageDot = document.getElementById('stage-dot');
const stageStatusText = document.getElementById('stage-status-text');
function setPresence(state) {
  currentPresence = state;
  const label = state.toUpperCase();
  const cls = state === 'idle' ? '' : state;
  if (presenceText) presenceText.textContent = label;
  if (presenceDot) presenceDot.className = cls;
  if (stageStatusText) stageStatusText.textContent = label;
  if (stageDot) stageDot.className = cls;
}
window.setPresence = setPresence; // growth.js signals focus start/stop through this
function setThinkingMotion(on) {
  if (window.aikoSetThinking) window.aikoSetThinking(on);
  else if (window.aikoSetPose) window.aikoSetPose('thinking', on);
}

// her words land here: caption + transcript + body language
function applyAikoEmotion(emoji) {
  if (!window.aikoSetExpression) return;
  const exprName = EMOJI_EXPRESSIONS[emoji] || emoji;
  window.aikoSetExpression(exprName, 1.0);
  spawnEmotionParticles(exprName);
  updateEmotionBadge(exprName);
  if (statusEmotion) statusEmotion.textContent = exprName;
}
let lastAikoCommit = { text: null, t: 0 };
function aikoCommitFresh(text) {
  // Guard against backend double-sends (e.g. stream commit + complete
  // chat message for the same turn): same text within 3s commits once.
  const now = Date.now();
  if (lastAikoCommit.text === text && now - lastAikoCommit.t < 3000) return false;
  lastAikoCommit = { text, t: now };
  return true;
}
function deliverAikoMessage(parsed) {
  hideTypingIndicator();
  if (!aikoCommitFresh(parsed.dialogueText)) return;
  if (parsed.emoji) applyAikoEmotion(parsed.emoji);
  captionCommit(parsed.dialogueText, parsed.emoji);
  transcriptLog.push({ t: Date.now(), emoji: parsed.emoji, text: parsed.dialogueText,
                       action: parsed.action, nonVerbal: parsed.nonVerbalText });
  if (transcriptVisible) renderTranscript();
  postResponseBehavior(parsed);
}
function addSysLine(text) {
  const div = document.createElement('div');
  div.className = 'msg msg-sys';
  div.textContent = `  ◈  ${text}`;
  chatPanel.insertBefore(div, toolStatus);
  scrollBottom();
}

// emoji + ACTION: + *stage directions* drive her body after she speaks
const ACTION_GESTURE_MAP = [
  [/\bwav(?:e|es|ing)\b/i, 'wave'],
  [/\bgiggl\w*|\blaugh\w*|\bchuckl\w*/i, 'giggle'],
  [/\bnod(?:s|ded)?\b/i, 'headNod'],
  [/\bbow(?:s|ed|ing)?\b/i, 'bow'],
  [/\bclap(?:s|ped|ping)?\b/i, 'clap'],
  [/\bshrug(?:s|ged|ging)?\b/i, 'shoulderRoll'],
  [/\bthink(?:s|ing)?\b|\bponder\w*|\bcontemplat\w*/i, 'chinThink'],
  [/\bglanc\w*|\blooks? (?:up|away|around)\b/i, 'lookAround'],
  [/\btilts? (?:her |his |their |the )?head\b|\bhead tilt\b/i, 'curiousTilt'],
  [/\bbounc\w*|\bexcited\w*|\bjump\w* with joy\b/i, 'happyBounce'],
  [/\bhug\w* (?:herself|himself|herself)/i, 'hugSelf'],
  [/\bsigh\w*|\bstretch\w*/i, 'gentleStretch'],
];
const EMOJI_GESTURE = {
  '😂': 'giggle', '🤣': 'giggle', '👋': 'wave', '🙏': 'bow',
  '👏': 'clap', '😭': 'hugSelf', '🥺': 'hugSelf', '🤔': 'chinThink',
};
function postResponseBehavior(parsed) {
  const hay = [parsed.action || '', parsed.nonVerbalText || ''].join(' ');
  let played = false;
  if (hay.trim() && window.aikoPlayGesture) {
    for (const [re, gesture] of ACTION_GESTURE_MAP) {
      if (re.test(hay)) { window.aikoPlayGesture(gesture); played = true; break; }
    }
  }
  if (!played && parsed.emoji && EMOJI_GESTURE[parsed.emoji] && window.aikoPlayGesture) {
    window.aikoPlayGesture(EMOJI_GESTURE[parsed.emoji]);
  }
  const bits = [];
  if (parsed.action) bits.push(`*${parsed.action}*`);
  if (parsed.nonVerbalText) bits.push(parsed.nonVerbalText);
  if (bits.length) addSysLine('Aiko ' + bits.join(' '));
}

// transcript (her full history, on demand)
function renderTranscript() {
  document.querySelectorAll('#chat-panel .transcript-row').forEach(el => el.remove());
  if (!transcriptVisible) return;
  for (const entry of transcriptLog) {
    const { row, bubble } = createMessageRow('aiko');
    row.classList.add('transcript-row');
    const head = document.createElement('div');
    head.className = 'msg-prefix';
    head.textContent = new Date(entry.t).toLocaleTimeString() + (entry.emoji ? ' ' + entry.emoji : '');
    bubble.appendChild(head);
    const body = document.createElement('div');
    body.innerHTML = esc(entry.text).replace(/\n/g, '<br>');
    bubble.appendChild(body);
    chatPanel.insertBefore(row, toolStatus);
  }
  scrollBottom();
}

function updateEmotionBadge(exprName) {
  if (!emotionBadge) return;
  const emojiMap = {
    happy: '🌸', angry: '💢', sorrow: '💧', surprised: '✨', fun: '🎀', neutral: '💭'
  };
  emotionBadge.style.display = 'flex';
  emotionEmoji.textContent = emojiMap[exprName] || '💗';
  emotionText.textContent = exprName;
}

// ── Typing Indicator ──────────────────────────────────────────────────────
function showTypingIndicator() {
  if (typingIndicator) return;
  const row = document.createElement('div');
  row.className = 'typing-row';
  row.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar aiko';
  avatar.textContent = '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'typing-bubble';
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = 'typing-dot';
    bubble.appendChild(dot);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatPanel.insertBefore(row, toolStatus);
  typingIndicator = row;
  setPresence('thinking');
  setThinkingMotion(true);
  scrollBottom();
}

function hideTypingIndicator() {
  if (typingIndicator) {
    typingIndicator.remove();
    typingIndicator = null;
  }
  setThinkingMotion(false);
  if (currentPresence === 'thinking') setPresence('idle');
}

function ensureAuxRow(kind) {
  let row = kind === 'sources' ? sourcesRow : filesRow;
  if (row && row.parentNode) return row;
  row = document.createElement('div');
  row.className = kind === 'sources' ? 'sources-row' : 'files-row';
  chatPanel.insertBefore(row, toolStatus);
  if (kind === 'sources') sourcesRow = row;
  else filesRow = row;
  return row;
}

function clearAuxRows() {
  if (sourcesRow) { sourcesRow.remove(); sourcesRow = null; }
  if (filesRow) { filesRow.remove(); filesRow = null; }
}

function faviconUrl(domain) {
  if (!domain) return '';
  return 'https://www.google.com/s2/favicons?domain=' + encodeURIComponent(domain) + '&sz=32';
}

function renderSources(items) {
  if (!Array.isArray(items) || !items.length) return;
  const row = ensureAuxRow('sources');
  row.innerHTML = '';
  const label = document.createElement('span');
  label.className = 'sources-label';
  label.textContent = 'sources';
  row.appendChild(label);
  for (const it of items.slice(0, 8)) {
    const a = document.createElement('a');
    a.className = 'source-chip';
    let href = it.url || '#';
    try {
      const u = new URL(href);
      if (u.protocol !== 'http:' && u.protocol !== 'https:') {
        href = '#';
      }
    } catch {
      href = '#';
    }
    a.href = href;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.title = it.title || it.url || '';
    const img = document.createElement('img');
    img.className = 'source-favicon';
    img.alt = '';
    img.src = faviconUrl(it.domain || '');
    img.referrerPolicy = 'no-referrer';
    const name = document.createElement('span');
    name.textContent = it.domain || (it.title || 'link').slice(0, 24);
    a.appendChild(img);
    a.appendChild(name);
    row.appendChild(a);
  }
  scrollBottom();
}

function renderFiles(items) {
  if (!Array.isArray(items) || !items.length) return;
  const row = ensureAuxRow('files');
  row.innerHTML = '';
  const label = document.createElement('span');
  label.className = 'files-label';
  label.textContent = 'files';
  row.appendChild(label);
  for (const it of items) {
    const chip = document.createElement('button');
    chip.className = 'file-chip';
    const path = it.path || '';
    const name = it.label || path || 'file';
    chip.title = path || name;
    chip.textContent = name;
    chip.disabled = !path;
    chip.addEventListener('click', function () {
      if (navigator.clipboard && path) {
        navigator.clipboard.writeText(path).then(function () {
          chip.classList.add('copied');
          setTimeout(function () { chip.classList.remove('copied'); }, 900);
        }).catch(function () {});
      }
    });
    row.appendChild(chip);
  }
  scrollBottom();
}


const EMOJI_EXPRESSIONS = {
  '😊': 'happy', '😄': 'happy', '😁': 'happy', '😆': 'happy', '🥰': 'happy', '😍': 'happy', '🙂': 'happy', '😋': 'happy', '🌸': 'happy', '✨': 'happy', '❤️': 'happy', '💖': 'happy',
  '😒': 'angry', '😡': 'angry', '😠': 'angry', '😤': 'angry', '🤬': 'angry', '💢': 'angry',
  '😭': 'sorrow', '😢': 'sorrow', '🥺': 'sorrow', '☹️': 'sorrow', '🙁': 'sorrow', '😔': 'sorrow', '😞': 'sorrow', '💧': 'sorrow',
  '😮': 'surprised', '😯': 'surprised', '😲': 'surprised', '😳': 'surprised', '🤯': 'surprised', '😱': 'surprised', '⁉️': 'surprised', '❓': 'surprised',
  '😜': 'fun', '🤪': 'fun', '😏': 'fun', '😈': 'fun', '🙃': 'fun', '😉': 'fun',
  '😐': 'neutral', '😑': 'neutral', '😶': 'neutral', '🤖': 'neutral', '😴': 'neutral', '🤔': 'neutral', '💭': 'neutral'
};

function esc(s) {
  // Previous implementation replaced &, <, >, and " with themselves
  // (identity regex substitutions — no-ops), leaving only the apostrophe
  // actually escaped. Since this feeds innerHTML in addMessage() and
  // renderAikoContent() for both the user's own echoed text and Aiko's
  // responses, any "<", ">", "&", or '"' in chat text — including an ASR
  // transcript, tool output, or a search-result summary — rendered as
  // live HTML. Using textContent to build the escaped string sidesteps
  // hand-writing an entity map entirely and can't get this wrong again.
  const div = document.createElement('div');
  div.textContent = s || '';
  return div.innerHTML;
}

function parseMarkdown(text) {
  if (!text) return '';
  let safe = esc(text);
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  safe = safe.replace(/_([^_]+)_/g, '<em>$1</em>');
  safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  safe = safe.replace(/\n/g, '<br>');
  return safe;
}

function isControlTokenChunk(s) {
  // Server status/search/source control lines must not enter the bubble.
  return /^\s*__(?:STATUS|SEARCHING|RETRYING|SOURCES|REPLACE)(?:__:|\b)/.test(s || '');
}

function stripControlLines(text) {
  return (text || '')
    .split('\n')
    .filter((line) => !isControlTokenChunk(line) && !/^\s*__\w+__/.test(line))
    .join('\n');
}

/**
 * Parse Aiko message structure.
 * soft=true (while streaming): keep body as dialogue only — no *()[] / ACTION
 * split that causes half-sentence / colour-box flicker mid-token.
 * soft=false (commit / final): full parse for non-verbal + action boxes.
 */
function parseAikoMessage(rawText, soft = false) {
  let text = stripControlLines(rawText || '');
  let emoji = null;
  let action = null;

  // Remove --- separators first
  text = text.replace(/^---+\s*$/gm, '');

  // 1. Parse EMOTION: prefix if present (legacy format) or leading emoji only (new format)
  const legacyEmotionMatch = text.match(/^\s*EMOTION:\s*([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF]|[a-zA-Z_-]+)\s*/u);
  if (legacyEmotionMatch) {
    emoji = legacyEmotionMatch[1];
    text = text.replace(/^\s*EMOTION:\s*[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF]|[a-zA-Z_-]+\s*/u, '');
  } else {
    const directEmojiRegex = /^\s*([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF])\s*:?\s*/u;
    const directEmojiMatch = text.match(directEmojiRegex);
    if (directEmojiMatch) {
      emoji = directEmojiMatch[1];
      text = text.replace(directEmojiRegex, '');
    }
  }

  // 2. Parse ACTION: only when complete (not soft / mid-stream)
  if (!soft) {
    const actionMatch = text.match(/(?:^|\n|\s*)ACTION:\s*([^\n]+)/i);
    if (actionMatch) {
      action = actionMatch[1].trim();
      text = text.replace(/(?:^|\n|\s*)ACTION:\s*[^\n]+\n?/i, '\n');
    }
  }

  // 3. Check for emoji header format (e.g., "😊: hello") if emoji not set yet
  const emojiHeaderRegex = /^\s*(?:([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF])|:([a-zA-Z0-9_-]+):)?\s*:\s*/u;
  const match = text.match(emojiHeaderRegex);
  if (match && !emoji) {
    emoji = match[1] || match[2] || null;
    text = text.replace(emojiHeaderRegex, '');
  }

  const nonVerbalParts = [];

  if (!soft) {
    text = text.replace(/\*([^*]+)\*/g, (m, p1) => {
      nonVerbalParts.push(`*${p1.trim()}*`);
      return '';
    });

    text = text.replace(/\(([^)]+)\)/g, (m, p1) => {
      nonVerbalParts.push(`(${p1.trim()})`);
      return '';
    });

    text = text.replace(/\[([^\]]+)\]/g, (m, p1) => {
      // keep markdown links [text](url) — only strip bare [stage directions]
      return m;
    });
    // Bare [direction] without following (
    text = text.replace(/\[([^\]\n]+)\](?!\()/g, (m, p1) => {
      nonVerbalParts.push(`[${p1.trim()}]`);
      return '';
    });
  }

  // Ensure all --- lines and multiple blank lines are removed
  text = text.replace(/^---+\s*$/gm, '');
  text = text.replace(/\n{3,}/g, '\n\n').trim();

  const nonVerbalText = soft ? '' : nonVerbalParts.join(' ').trim();
  const dialogueText = soft
    ? text.replace(/\s{2,}/g, ' ').trim()
    : text.replace(/\s{2,}/g, ' ').trim();

  return { emoji, action, nonVerbalText, dialogueText };
}

function renderAikoContent(container, parsed, showCursor = false) {
  container.replaceChildren();

  const prefixSpan = document.createElement('span');
  prefixSpan.className = 'msg-prefix';
  prefixSpan.textContent = 'Aiko: ';
  container.appendChild(prefixSpan);

  if (parsed.nonVerbalText) {
    const nvDiv = document.createElement('div');
    nvDiv.className = 'msg-non-verbal';
    nvDiv.innerHTML = parseMarkdown(parsed.nonVerbalText);
    container.appendChild(nvDiv);
  }

  if (parsed.action && parsed.action.toLowerCase() !== 'none') {
    const actionDiv = document.createElement('div');
    actionDiv.className = 'msg-action';
    const cleanAction = parsed.action.replace(/^ACTION:\s*/i, '').trim();
    actionDiv.innerHTML = parseMarkdown(cleanAction);
    container.appendChild(actionDiv);
  }

  const dialSpan = document.createElement('span');
  dialSpan.className = 'msg-dialogue';
  dialSpan.innerHTML = parseMarkdown(parsed.dialogueText || '');

  if (showCursor) {
    const cursorSpan = document.createElement('span');
    cursorSpan.className = 'cursor';
    dialSpan.appendChild(cursorSpan);
  }

  container.appendChild(dialSpan);
}

// ── Bubble Message Rendering ──────────────────────────────────────────────
function createMessageRow(sender) {
  const row = document.createElement('div');
  row.className = 'msg-row' + (sender === 'you' ? ' user' : '');

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar ' + (sender === 'aiko' ? 'aiko' : 'user');
  avatar.textContent = sender === 'aiko' ? '🤖' : '🌸';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble ' + (sender === 'aiko' ? 'aiko' : 'user');

  row.appendChild(avatar);
  row.appendChild(bubble);
  return { row, bubble };
}

function addMessage(sender, text) {
  flushStream();
  hideTypingIndicator();
  let insertEl;
  if (sender === 'you') {
    const { row, bubble } = createMessageRow('you');
    const prefix = document.createElement('span');
    prefix.className = 'msg-prefix';
    prefix.textContent = window.currentUsername || 'You';
    bubble.appendChild(prefix);
    const body = document.createElement('div');
    body.innerHTML = esc(text).replace(/\n/g, '<br>');
    bubble.appendChild(body);
    insertEl = row;
  } else if (sender === 'aiko') {
    deliverAikoMessage(parseAikoMessage(text));
    return;   // her words live in the caption; the chat stays a slim log
  } else {
    const div = document.createElement('div');
    div.className = 'msg msg-sys';
    div.textContent = `  ◈  ${text}`;
    insertEl = div;
    const errLine = document.getElementById('companion-error');
    if (errLine) { errLine.textContent = text; errLine.hidden = false; }
    const errLog = document.getElementById('error-log');
    if (errLog) {
      const line = document.createElement('div');
      line.className = 'elog-line';
      const t = new Date();
      const ts = [t.getHours(), t.getMinutes(), t.getSeconds()].map(n => String(n).padStart(2, '0')).join(':');
      line.textContent = ts + '  ' + text;
      errLog.appendChild(line);
      while (errLog.children.length > 40) errLog.removeChild(errLog.firstChild);
      errLog.scrollTop = errLog.scrollHeight;   // newest at bottom, older scroll up
    }
  }
  chatPanel.insertBefore(insertEl, toolStatus);
  scrollBottom();
}

function appendToken(text) {
  if (text == null || text === '') return;
  // Drop pure control chunks (status/search) so they never typewrite into the caption.
  if (isControlTokenChunk(text) && !streamRawText) return;
  if (!streamActive) {
    hideTypingIndicator();
    streamActive = true;
    streamRawText = '';
    streamExprApplied = null;
    captionTextEl.innerHTML = '';
    captionLastDialogue = '';
    if (captionEmotionEl) captionEmotionEl.textContent = '';
    captionShow();
  }
  streamRawText += text;
  // Soft parse while streaming: dialogue words reveal karaoke-style, paced by TTS.
  const parsed = parseAikoMessage(streamRawText, true);
  if (parsed.emoji && parsed.emoji !== streamExprApplied) {
    streamExprApplied = parsed.emoji;
    applyAikoEmotion(parsed.emoji);
  }
  captionRender(parsed.dialogueText, parsed.emoji);
}

function flushStream() {
  if (streamActive) {
    // Full parse only when the turn is complete.
    const parsed = parseAikoMessage(streamRawText, false);
    if (!aikoCommitFresh(parsed.dialogueText)) { streamActive = false; streamRawText = ''; streamExprApplied = null; toolStatus.textContent = ''; return; }
    if (parsed.emoji && parsed.emoji !== streamExprApplied) applyAikoEmotion(parsed.emoji);
    captionCommit(parsed.dialogueText, parsed.emoji);
    transcriptLog.push({ t: Date.now(), emoji: parsed.emoji, text: parsed.dialogueText,
                         action: parsed.action, nonVerbal: parsed.nonVerbalText });
    if (transcriptVisible) renderTranscript();
    postResponseBehavior(parsed);
    streamActive = false;
    streamRawText = '';
    streamExprApplied = null;
  }
  toolStatus.textContent = '';
}

function scrollBottom() { content.scrollTop = content.scrollHeight; }

const transcriptToggle = document.getElementById('transcript-toggle');
const transcriptStore = document.getElementById('transcript-store');
function setTranscriptVisible(v) {
  transcriptVisible = v;
  if (transcriptToggle) transcriptToggle.classList.toggle('on', v);
  if (transcriptStore) transcriptStore.hidden = !v;
  renderTranscript();
}
if (transcriptToggle) transcriptToggle.addEventListener('click', () => setTranscriptVisible(!transcriptVisible));
const tsClose = document.getElementById('ts-close');
if (tsClose) tsClose.addEventListener('click', () => setTranscriptVisible(false));

// ── vitals ────────────────────────────────────────────────────────────────
function applyVitals(v) {
  asrOn = !!v.asr;
  vTok.textContent = `${(v.tokens || 0).toLocaleString()} tok`;
  vToks.textContent = v.tok_s > 0 ? `${v.tok_s} t/s` : '— t/s';
  vRam.textContent = `RAM ${v.ram || '—'}`;
  vUp.textContent = `↑ ${v.uptime || '—'}`;
  vMode.textContent = (v.asr ? '🎤 ASR' : '⌨ TXT') + '  ' + (v.tts ? '🔊 TTS' : '🔇 TTS');

  if (AUTO_MIC && wsReady() && !v.asr && !autoListenRequested) {
    autoListenRequested = true;
    ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  }
}

// ── voice status ──────────────────────────────────────────────────────────
const VOICE_LABELS = {
  waiting: '⏸  waiting for Aiko…',
  listening: '🎤  listening…',
  transcribing: '⚙  transcribing…',
  idle: '',
};
function applyVoice(status) {
  voiceSt.textContent = VOICE_LABELS[status] ?? '';
  voiceSt.className = status === 'idle' ? '' : status;
  window.aikoSetListening?.(status === 'listening');
  if (status === 'waiting' && chatPhaseActive) showTypingIndicator();
  if (status === 'listening') setPresence('listening');
  else if (status === 'transcribing') setPresence('thinking');
  else if (status === 'idle' && (currentPresence === 'listening' || currentPresence === 'thinking')) {
    if (!typingIndicator) setPresence('idle');
  }
}

// ── TTS playback (binary WAV frames from server) ──────────────────────────
let ttsContext = null;
let ttsQueue = [];
let ttsPlaying = false;
let ttsAnalyser = null;
let ttsAnalyserData = null;
let ttsAnalyserConnected = false;
let ttsMouthLoop = false;
let ttsMouthLevel = 0;

function getTtsContext() {
  if (!ttsContext) ttsContext = new AudioContext();
  return ttsContext;
}

function getTtsAnalyser() {
  const ctx = getTtsContext();
  if (!ttsAnalyser) {
    ttsAnalyser = ctx.createAnalyser();
    ttsAnalyser.fftSize = 1024;
    ttsAnalyser.smoothingTimeConstant = 0.35;
    ttsAnalyserData = new Float32Array(ttsAnalyser.fftSize);
  }
  if (!ttsAnalyserConnected) {
    ttsAnalyser.connect(ctx.destination);
    ttsAnalyserConnected = true;
  }
  return ttsAnalyser;
}

// ── voice waveform: her live TTS audio, drawn next to the stage clock ──
// Flat idle line when she is quiet; dancing bars while she speaks, fed by
// the same TTS analyser node that drives lip-sync.
const voiceWaveCvs = [document.getElementById('voice-wave'), document.getElementById('speech-wave')].filter(Boolean);
const VOICE_BARS = 26;
let voiceWaveFreq = null;
let voiceWaveMax = 0;
const voiceBarLevels = new Array(VOICE_BARS).fill(0);
function drawVoiceWave() {
  let freq = null;
  if (ttsPlaying && ttsAnalyser) {
    if (!voiceWaveFreq || voiceWaveFreq.length !== ttsAnalyser.frequencyBinCount)
      voiceWaveFreq = new Uint8Array(ttsAnalyser.frequencyBinCount);
    ttsAnalyser.getByteFrequencyData(voiceWaveFreq);
    freq = voiceWaveFreq;
  }
  let peak = 0;
  for (let i = 0; i < VOICE_BARS; i++) {
    let target = 0.07; // idle floor: a calm flat line
    if (freq) {
      const b0 = 2 + Math.floor(i * 44 / VOICE_BARS);
      const v = ((freq[b0] || 0) + (freq[b0 + 1] || 0)) / 2 / 255;
      target = 0.07 + Math.min(1, v * 1.7);
    }
    const lv = voiceBarLevels[i];
    const nv = lv + (target - lv) * (target > lv ? 0.55 : 0.3);
    voiceBarLevels[i] = nv;
    if (nv > peak) peak = nv;
  }
  voiceWaveMax = peak;
  for (const cv of voiceWaveCvs) {
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const mid = H / 2, bw = W / VOICE_BARS;
    for (let i = 0; i < VOICE_BARS; i++) {
      const nv = voiceBarLevels[i];
      const h = Math.max(1.5, nv * (H - 2));
      const grad = ctx.createLinearGradient(0, mid - h / 2, 0, mid + h / 2);
      const lt = document.documentElement.dataset.theme === 'light';
      grad.addColorStop(0, lt ? '#F8A9C6' : '#7de9ff');
      grad.addColorStop(1, lt ? '#F06292' : '#1a9ec4');
      ctx.fillStyle = grad;
      ctx.shadowColor = lt ? 'rgba(240,98,146,.55)' : 'rgba(53,224,255,.7)';
      ctx.shadowBlur = 4;
      ctx.fillRect(i * bw + bw * 0.22, mid - h / 2, Math.max(1, bw * 0.56), h);
      ctx.shadowBlur = 0;
    }
  }
}
drawVoiceWave(); // one idle frame before any TTS has played

function startMouthAnalyserLoop() {
  if (ttsMouthLoop) return;
  ttsMouthLoop = true;
  const tick = () => {
    if (!ttsMouthLoop) return;
    let target = 0;
    if (ttsPlaying && ttsAnalyser && ttsAnalyserData) {
      ttsAnalyser.getFloatTimeDomainData(ttsAnalyserData);
      let sum = 0;
      for (let i = 0; i < ttsAnalyserData.length; i++) {
        const v = ttsAnalyserData[i];
        sum += v * v;
      }
      const rms = Math.sqrt(sum / ttsAnalyserData.length);
      target = Math.max(0, Math.min(1, (rms - 0.012) * 9.5));
    }

    const coeff = target > ttsMouthLevel ? 0.65 : 0.28;
    ttsMouthLevel += (target - ttsMouthLevel) * coeff;
    if (window.aikoSetMouthOpen) window.aikoSetMouthOpen(ttsMouthLevel);
    drawVoiceWave();

    if (!ttsPlaying && ttsMouthLevel < 0.01 && voiceWaveMax < 0.09) {
      ttsMouthLoop = false;
      ttsMouthLevel = 0;
      if (window.aikoSetMouthOpen) window.aikoSetMouthOpen(0);
      return;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function enqueueTtsAudio(arrayBuffer) {
  ttsQueue.push(arrayBuffer);
  if (!ttsPlaying) playNextTts();
}

let ttsCurrentSource = null;

async function playNextTts() {
  const buf = ttsQueue.shift();
  if (!buf) { ttsPlaying = false; window.aikoIsSpeaking = false; captionSpeaking(false); if (!typingIndicator && currentPresence === 'speaking') setPresence('idle'); return; }
  ttsPlaying = true;
  window.aikoIsSpeaking = true;
  captionSpeaking(true);
  setPresence('speaking');
  window.AIKO_TTS_STARTED_AT = performance.now();  // S3 echo guard
  try {
    const ctx = getTtsContext();
    if (ctx.state === 'suspended') {
      try { await ctx.resume(); } catch (e) {}
    }
    const audioBuffer = await ctx.decodeAudioData(buf.slice(0));
    const analyser = getTtsAnalyser();
    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(analyser);
    src.onended = playNextTts;
    ttsCurrentSource = src;
    src.start();
    startMouthAnalyserLoop();
  } catch (err) {
    console.error('[tts] decode/play failed:', err);
    playNextTts();
  }
}

function stopTtsPlayback() {
  ttsQueue = [];
  if (ttsCurrentSource) {
    try { ttsCurrentSource.onended = null; ttsCurrentSource.stop(); } catch (e) {}
    ttsCurrentSource = null;
  }
  ttsPlaying = false;
  window.aikoIsSpeaking = false;
  captionSpeaking(false);
  if (!typingIndicator && currentPresence === 'speaking') setPresence('idle');
}
window.stopTtsPlayback = stopTtsPlayback;

// ── mic capture ───────────────────────────────────────────────────────────
let micStream = null;
let micContext = null;
let micSource = null;
let micWorklet = null;
let micFirstFrameSeen = false;
let micStreamingEnabled = false;
let browserVadGate = true;
let micCommandSeq = 0;
let micSecureContextWarned = false;

let micStartPromise = null;
// User intent: the mic button is a kill switch. When micMuted is true the
// hardware stays off no matter what the server asks (barge-in / listen
// loops used to resurrect it right after the user turned it off).
let micMuted = true;
let pendingMicStart = null;
let micGen = 0;
let asrOn = false;

async function startMic() {
  if (micContext) return true;
  if (micStartPromise) return micStartPromise;   // <- dedupe concurrent callers
  const gen = micGen;
  micStartPromise = _startMicInner().finally(() => { micStartPromise = null; });
  const ok = await micStartPromise;
  // A stopMic() (or mute) that landed while getUserMedia/AudioContext were
  // still pending must win: tear down instead of leaving the mic hot.
  if (!ok || gen !== micGen || micMuted) {
    if (micContext) stopMicHardware();
    return false;
  }
  return true;
}

// Hardware teardown without touching intent state (used internally when a
// start that was already in flight loses a generation race).
function stopMicHardware() {
  micStreamingEnabled = false;
  if (window.resetVADState) window.resetVADState();
  if (micWorklet) {
    if (micWorklet.port) micWorklet.port.onmessage = null;
    try { micWorklet.disconnect(); } catch (_) {}
    micWorklet = null;
  }
  if (micSource) { try { micSource.disconnect(); } catch (_) {} micSource = null; }
  if (micContext) { try { micContext.close(); } catch (_) {} micContext = null; }
  if (micStream) { micStream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} }); micStream = null; }
}

async function _startMicInner() {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    console.error('[mic] microphone requires localhost or HTTPS');
    if (!micSecureContextWarned) {
      micSecureContextWarned = true;
      const uiPort = location.port || '8787';
      const localUrl = 'http://localhost:' + uiPort + '/';
      const secureUrl = 'https://' + location.hostname + ':' + uiPort + '/';
      addMessage('sys', 'Microphone blocked — browsers only allow mic access on localhost or HTTPS. Open ' + localUrl + ' on this machine, or restart with WEBUI_HTTPS=1 and use ' + secureUrl + '.');
    }
    syncTalkButton();
    return false;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
    });
    micContext = new AudioContext({ sampleRate: 16000 });
    if (micContext.state === 'suspended') {
      await micContext.resume();
      console.log('[mic] AudioContext was suspended — resumed');
    }
    micSource = micContext.createMediaStreamSource(micStream);

    let _vadQueue = Promise.resolve();
    function pushVADFrame(frame) {
      _vadQueue = _vadQueue.then(() => processVADFrame(frame, ws, browserVadGate)).catch(e => console.error('[mic] VAD error:', e));
    }

    let awok = false;
    try {
      await micContext.audioWorklet.addModule('./pcm-worklet.js');
      micWorklet = new AudioWorkletNode(micContext, 'pcm-capture-processor');
      micFirstFrameSeen = false;
      micWorklet.port.onmessage = (e) => {
        if (!micFirstFrameSeen) {
          micFirstFrameSeen = true;
          console.log('[mic] AudioWorklet is sending PCM frames');
        }
        if (wsReady() && micStreamingEnabled) {
          pushVADFrame(new Float32Array(e.data));
        }
      };
      micSource.connect(micWorklet);
      micWorklet.connect(micContext.destination);
      awok = true;
      console.log('[mic] using AudioWorklet capture');
    } catch (awErr) {
      console.warn('[mic] AudioWorklet failed, falling back to ScriptProcessorNode:', awErr);
    }

    if (!awok) {
      const bufSize = 2048;
      const frameSamples = 512;
      let _spBuf = new Float32Array(0);
      const spNode = micContext.createScriptProcessor(bufSize, 1, 1);
      spNode.onaudioprocess = (e) => {
        if (!wsReady() || !micStreamingEnabled) return;
        const input = e.inputBuffer.getChannelData(0);
        let combined = new Float32Array(_spBuf.length + input.length);
        combined.set(_spBuf);
        combined.set(input, _spBuf.length);
        _spBuf = combined;
        while (_spBuf.length >= frameSamples) {
          const frame = _spBuf.slice(0, frameSamples);
          _spBuf = _spBuf.slice(frameSamples);
          pushVADFrame(frame);
        }
      };
      micSource.connect(spNode);
      spNode.connect(micContext.destination);
      micWorklet = spNode;
      console.log('[mic] using ScriptProcessorNode capture');
    }

    vadDot.className = 'dot on';
    vadStatus.textContent = 'mic ready';
    vadStatus.className = 'ready';
    syncTalkButton();
    return true;
  } catch (err) {
    console.error('[mic] getUserMedia/AudioWorklet failed:', err);
    addMessage('sys', 'Microphone access failed — check browser permissions.');
    syncTalkButton();
    return false;
  }
}

function stopMic() {
  micCommandSeq++;   // cancel any in-flight server-driven start
  micGen++;          // cancel any in-flight _startMicInner
  stopMicHardware();
  vadDot.className = 'dot on';
  vadStatus.textContent = 'vad ready';
  vadStatus.className = 'ready';
  syncTalkButton();
}

// ── text input ────────────────────────────────────────────────────────────
function submitInput() {
  const text = input.value.trim();
  if (!text || !wsReady()) return;
  window.dispatchEvent(new CustomEvent('aiko:msg-sent'));
  clearAuxRows();
  flushStream();
  showTypingIndicator();
  ws.send(JSON.stringify({ type: 'user_input', text }));
  input.value = '';
}

async function captureImage() {
  if (!wsReady()) {
    addMessage('sys', 'WebSocket bridge is offline. Cannot use the camera.');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    addMessage('sys', 'This browser does not support camera capture.');
    return;
  }

  cameraBtn.disabled = true;
  toolStatus.textContent = '  📷  opening camera…';
  let stream;
  let submitted = false;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const video = document.createElement('video');
    video.srcObject = stream;
    video.playsInline = true;
    await video.play();
    await new Promise(resolve => setTimeout(resolve, 250));
    const canvas = document.createElement('canvas');
    const maxWidth = 1024;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL('image/jpeg', 0.85);
    const text = input.value.trim();
    if (!wsReady()) {
      addMessage('sys', 'WebSocket bridge is offline. Cannot submit the camera image.');
      return;
    }
    ws.send(JSON.stringify({ type: 'image_input', image, text }));
    submitted = true;
    input.value = '';
    addMessage('user', text ? `📷 Camera image — ${text}` : '📷 Camera image');
  } catch (err) {
    console.error('[camera] capture failed:', err);
    addMessage('sys', 'Camera access was unavailable. Allow camera permission and try again.');
  } finally {
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (!submitted) cameraBtn.disabled = false;
    input.focus();
  }
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitInput(); }
});
sendBtn.addEventListener('click', submitInput);
cameraBtn.addEventListener('click', captureImage);
function syncTalkButton() {
  const live = !micMuted && !!micContext;
  micBtn.classList.toggle('on', live);
  const talk = document.getElementById('talk-btn');
  if (talk) talk.classList.toggle('on', live);
}
function handleMicStart(msg) {
  if (micMuted) { pendingMicStart = msg; return; }
  const seq = ++micCommandSeq;
  browserVadGate = msg.browser_vad_gate !== false;
  window.AIKO_BARGE_IN_ENABLED = !!msg.barge_in_enabled;
  window.AIKO_BARGE_ECHO_GUARD_MS = msg.echo_guard_ms ?? 450;
  startMic().then((ok) => {
    if (!ok || seq !== micCommandSeq) return;
    if (window.resetVADState) window.resetVADState();
    micStreamingEnabled = true;
    vadDot.className = 'dot vad';
    vadStatus.textContent = browserVadGate ? 'vad active' : 'raw mic';
    vadStatus.className = 'active';
  });
}
async function toggleMic() {
  if (!wsReady()) {
    addMessage('sys', 'WebSocket bridge is offline. Cannot toggle voice mode.');
    return;
  }

  if (!micMuted) {
    // User wants it OFF — kill switch: stays off until they turn it back on.
    micMuted = true;
    pendingMicStart = null;
    stopMic();
    if (asrOn) ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  } else {
    const toggleGen = micGen;
    micMuted = false;
    const ok = await startMic();
    if (toggleGen !== micGen) return;
    if (!ok) { micMuted = true; syncTalkButton(); return; }
    if (pendingMicStart) {
      const start = pendingMicStart;
      pendingMicStart = null;
      handleMicStart(start);
    } else if (!asrOn) ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  }
  syncTalkButton();
  input.focus();
}
micBtn.addEventListener('click', toggleMic);
const talkBtn = document.getElementById('talk-btn');
if (talkBtn) talkBtn.addEventListener('click', toggleMic);

// ── WebSocket ─────────────────────────────────────────────────────────────
let ws = null;
let wsReconnectTimer = null;

function wsReady() { return ws && ws.readyState === WebSocket.OPEN; }
// Small public surface for companion.js. Keeping the socket itself private
// prevents auxiliary UI modules from mutating its lifecycle.
window.aikoSendWS = (payload) => {
  if (!wsReady()) return false;
  ws.send(JSON.stringify(payload));
  return true;
};

function websocketURL() {
  const params = new URLSearchParams(location.search);
  const wsHost = params.get("ws_host") || location.hostname;
  const wsPortParam = params.get("ws");
  const protoOverride = (params.get("ws_proto") || "").toLowerCase();
  const wsProto = protoOverride === "ws" || protoOverride === "wss"
    ? protoOverride + ":"
    : location.protocol === "https:" ? "wss:" : "ws:";

  if (wsHost.endsWith(".ts.net")) {
    return wsProto + "//" + wsHost + "/ws";
  }

  if (wsPortParam) {
    return wsProto + "//" + wsHost + ":" + wsPortParam + "/";
  }

  const portPart = location.port ? ":" + location.port : "";
  return wsProto + "//" + wsHost + portPart + "/ws";
}

function connectWS() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
  if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
  const wsUrl = websocketURL();
  const socket = new WebSocket(wsUrl);
  ws = socket;
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => {
    wsDot.className = 'dot on';
    wsLabel.textContent = 'ws connected';
    if (AUTO_MIC) startMic();
  };

  socket.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      enqueueTtsAudio(e.data);
      return;
    }
    let msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }

    if (!chatPhaseActive && ['chat', 'token'].includes(msg.type)) {
      switchToChat();
    }

    switch (msg.type) {
      case 'step': handleStep(msg); break;
      case 'phase': if (msg.value === 'chat') switchToChat(); break;
      case 'chat': addMessage(msg.sender, msg.text); break;
      case 'vision':
        window.aikoCompanionVision?.(msg);
        if (msg.status === 'working') toolStatus.textContent = `  👁  analyzing ${msg.source === 'screen' ? 'screen' : 'camera'} image…`;
        else if (msg.status === 'done') { toolStatus.textContent = ''; cameraBtn.disabled = false; }
        else if (msg.status === 'error') { toolStatus.textContent = ''; cameraBtn.disabled = false; addMessage('sys', msg.message); }
        else if (msg.status === 'busy') addMessage('sys', msg.message);
        break;
      case 'token': appendToken(msg.text); break;
      case 'sources': renderSources(msg.items || []); break;
      case 'files': renderFiles(msg.items || []); break;
      case 'meta':
        if (msg.emotion && window.aikoSetExpression) {
          const expr = msg.emotion === 'neutral' ? 'neutral' : msg.emotion;
          window.aikoSetExpression(expr, 0.9);
          spawnEmotionParticles(expr);
          updateEmotionBadge(expr);
        }
        break;
      case 'commit': flushStream(); hideTypingIndicator(); break;
      case 'tool': toolStatus.textContent = msg.status ? `  ⚙  ${msg.status}` : ''; break;
      case 'vitals': applyVitals(msg); break;
      case 'voice':
        if (msg.status === 'idle') pendingMicStart = null;
        applyVoice(msg.status);
        break;
      case 'mic':
        if (msg.action === 'start') {
          handleMicStart(msg);
        } else if (msg.action === 'stop') {
          pendingMicStart = null;
          micCommandSeq++;
          micStreamingEnabled = false;
          if (window.resetVADState) window.resetVADState();
          vadDot.className = 'dot on';
          vadStatus.textContent = 'mic ready';
          vadStatus.className = 'ready';
        }
        break;
      case 'expression':
        if (window.aikoSetExpression) {
          window.aikoSetExpression(msg.name, msg.intensity ?? 1.0);
          spawnEmotionParticles(msg.name);
          updateEmotionBadge(msg.name);
        }
        break;
      case 'viseme': if (window.aikoSetViseme) window.aikoSetViseme(msg.viseme, msg.weight ?? 1.0); break;
      case 'pose': if (window.aikoSetPose) window.aikoSetPose(msg.name, msg.active); break;
      case 'gesture': if (window.aikoPlayGesture) window.aikoPlayGesture(msg.name); break;
    }
  };

  socket.onclose = () => {
    if (ws !== socket) return;
    ws = null;
    wsDot.className = 'dot';
    wsLabel.textContent = 'ws offline';
    pendingMicStart = null;
    stopMic();
    if (wsUrl.startsWith("wss:")) {
      toolStatus.textContent = "  ws offline: open " + wsUrl.replace("wss:", "https:") + " once to accept the WSS certificate";
    } else {
      toolStatus.textContent = "  ws offline: " + wsUrl;
    }
    if (!wsReconnectTimer) wsReconnectTimer = setTimeout(() => { wsReconnectTimer = null; connectWS(); }, 3000);
  };
  socket.onerror = () => {
    console.error('[ws] connection failed:', wsUrl);
    socket.close();
  };
}

// ── OAuth Login ──────────────────────────────────────────────────────────
const authOverlay = document.getElementById('auth-overlay');
const authStatus = document.getElementById('auth-status');

async function checkAuth() {
  try {
    const res = await fetch('/api/auth/me', { credentials: 'include' });
    if (res.ok) {
      let data = {};
      try { data = await res.json(); } catch (_) { /* no body / non-JSON */ }
      window.currentUsername = data.username || 'You';
      const aiNameEl = document.getElementById('vrm-ai-name');
      if (aiNameEl) aiNameEl.textContent = data.ai_name || 'Aiko';
      const userNameEl = document.getElementById('vrm-user-name');
      if (userNameEl) userNameEl.textContent = window.currentUsername;
      hideAuthOverlay();
      if (data.accepted_terms === false) {
        showTermsOverlay();
      } else {
        connectWS();
      }
      return true;
    }
  } catch (_) { }
  return false;
}

function hideAuthOverlay() {
  authOverlay.classList.add('hidden');
  setTimeout(() => authOverlay.style.display = 'none', 600);
}

function setAuthStatus(msg) {
  authStatus.textContent = msg;
}

function loginGitHub() {
  window.location.href = '/auth/github/login';
}

function loginPatreon() {
  window.location.href = '/auth/patreon/login';
}

const termsOverlay = document.getElementById('terms-overlay');
const termsCheckbox = document.getElementById('terms-checkbox');
const termsContinueBtn = document.getElementById('terms-continue');

function showTermsOverlay() {
  termsOverlay.style.display = 'flex';
  termsOverlay.classList.remove('hidden');
}

function hideTermsOverlay() {
  termsOverlay.classList.add('hidden');
  setTimeout(() => termsOverlay.style.display = 'none', 600);
}

termsCheckbox.addEventListener('change', () => {
  termsContinueBtn.disabled = !termsCheckbox.checked;
});

termsContinueBtn.addEventListener('click', async () => {
  if (!termsCheckbox.checked) return;
  termsContinueBtn.disabled = true;
  try {
    await fetch('/api/auth/accept-terms', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accepted: true }),
    });  }
    catch (err) {
    console.error('[terms] failed to record acceptance:', err);
  }
  hideTermsOverlay();
  connectWS();
});

fetch('/api/auth/config')
  .then(r => {
    if (!r.ok) throw new Error('Failed to load auth config');
    return r.json();
  })
  .then(cfg => {
    window.OAUTH_CONFIG = cfg;
    return checkAuth();
  })
  .then(authenticated => {
    if (!authenticated) {
      authOverlay.classList.remove('hidden');
      setAuthStatus('Authentication required. Please log in.');
    }
  })
  .catch(err => {
    console.error('[auth] initialization error:', err);
    authOverlay.classList.remove('hidden');
    setAuthStatus('Failed to load authentication system.');
  });
