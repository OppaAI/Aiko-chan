/* ── Aiko growth — Fenestra-style raising sim ──────────────────────────
   Real focus time becomes her growth. Four stats:
     intelligence · sensitivity · morality · bond
   XP sources: focus sessions (the TRAIN selector picks which stat the
   session develops, with a spillover to the routine's natural stat),
   completed tasks (morality), and chatting with her (bond).
   Level curve: xpForLevel(l) = 50·(l-1)²  →  L1:0 L2:50 L3:200 L4:450 …
   Stored locally; the server never sees it. */

(() => {
  'use strict';

  const STORE_KEY = 'aiko-growth-v1';
  const STATS = [
    { id: 'intelligence', label: 'Intelligence' },
    { id: 'sensitivity',  label: 'Sensitivity'  },
    { id: 'morality',     label: 'Morality'     },
    { id: 'bond',         label: 'Bond'         },
  ];
  // routine → the stat it naturally nurtures (spillover alongside TRAIN)
  const ROUTINE_STAT = { coding: 'intelligence', study: 'intelligence', writing: 'sensitivity', gaming: 'sensitivity', quiet: 'morality' };

  const xpForLevel = l => 50 * (l - 1) * (l - 1);
  const levelOf = xp => 1 + Math.floor(Math.sqrt(Math.max(0, xp) / 50));
  const levelProgress = xp => {
    const l = levelOf(xp);
    const lo = xpForLevel(l), hi = xpForLevel(l + 1);
    return { level: l, fill: Math.min(1, Math.max(0, (xp - lo) / (hi - lo))) };
  };

  function loadState() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const st = JSON.parse(raw);
        if (st && st.xp) return st;
      }
    } catch (_) {}
    const base = { xp: { intelligence: 0, sensitivity: 0, morality: 0, bond: 0 }, bondDay: '', bondCount: 0, focusMinutesTotal: 0, focusHistory: [] };
    // backfill for older saves
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) { const st = JSON.parse(raw); if (st && st.xp) { st.focusHistory = Array.isArray(st.focusHistory) ? st.focusHistory : []; return st; } }
    } catch (_) {}
    return base;
  }
  let state = loadState();
  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (_) {} }

  const grid = document.getElementById('growth-grid');
  const flash = document.getElementById('growth-flash');
  const trainSel = document.getElementById('train-stat');

  const SG_SHORT = { intelligence: 'INT', sensitivity: 'SEN', morality: 'MOR' };
  const BOND_TITLES = [
    [1, 'First Hello',   'Aiko is getting to know you.'],
    [2, 'Warming Up',    'Aiko looks forward to seeing you.'],
    [4, 'Deep Connection', 'Aiko feels safe, understood, and truly seen.'],
    [6, 'True Partners', 'Aiko lights up whenever you are near.'],
    [99, 'Soulbound',    'Aiko would cross worlds to stay by your side.'],
  ];
  function bondTitle(level) {
    for (const [cap, title, line] of BOND_TITLES) if (level <= cap) return { title, line };
    return { title: 'Soulbound', line: 'Aiko would cross worlds to stay by your side.' };
  }

  // build the 10 bond segments once
  const bondSegs = document.getElementById('bond-segs');
  if (bondSegs && !bondSegs.children.length) {
    for (let i = 0; i < 10; i++) { const d = document.createElement('span'); d.className = 'seg'; bondSegs.appendChild(d); }
  }

  function drawSparkline() {
    const cv = document.getElementById('phase-spark');
    if (!cv) return;
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const hist = (state.focusHistory || []).slice(-40);
    ctx.strokeStyle = 'rgba(53,224,255,0.14)';
    ctx.lineWidth = 1;
    for (let g = 1; g < 4; g++) { ctx.beginPath(); ctx.moveTo(0, H * g / 4); ctx.lineTo(W, H * g / 4); ctx.stroke(); }
    if (hist.length < 2) {
      ctx.strokeStyle = 'rgba(53,224,255,0.35)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, H - 6); ctx.lineTo(W, H - 6); ctx.stroke();
      ctx.setLineDash([]);
      return;
    }
    const max = Math.max(...hist, 1);
    ctx.beginPath();
    hist.forEach((v, i) => {
      const x = 4 + (i / (hist.length - 1)) * (W - 8);
      const y = H - 6 - (v / max) * (H - 14);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = '#35e0ff';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = 'rgba(53,224,255,0.8)';
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  function render() {
    // GROW panel: stat bars (level progress) + values
    for (const st of STATS) {
      if (st.id === 'bond') continue;
      const row = document.querySelector(`.grow-row[data-stat="${st.id}"]`);
      if (!row) continue;
      const xp = state.xp[st.id] || 0;
      const { level, fill } = levelProgress(xp);
      const fillEl = row.querySelector('.grow-fill');
      const valEl = row.querySelector('.grow-val');
      if (fillEl) fillEl.style.width = `${Math.round(fill * 100)}%`;
      if (valEl) valEl.textContent = `Lv ${level}`;
      row.title = `${st.label}: ${xp} XP`;
    }
    // bond row
    const bondLevel = levelOf(state.xp.bond || 0);
    const bondFill = Math.min(10, bondLevel);
    if (bondSegs) [...bondSegs.children].forEach((d, i) => d.classList.toggle('on', i < bondFill));
    const bondVal = document.getElementById('bond-val');
    if (bondVal) bondVal.textContent = `${bondFill} /10`;
    const { title, line } = bondTitle(bondLevel);
    const bst = document.getElementById('bond-status-text');
    if (bst) bst.textContent = title;
    const blt = document.getElementById('bond-line-text');
    if (blt) blt.textContent = line;

    // stage mini HUD: plain level numbers like the reference
    const stage = document.getElementById('stage-growth');
    if (stage) {
      for (const id of ['intelligence', 'sensitivity', 'morality']) {
        const row = stage.querySelector(`.sg-row[data-stat="${id}"]`);
        if (!row) continue;
        const xp = state.xp[id] || 0;
        const { level, fill } = levelProgress(xp);
        const fillEl = row.querySelector('.sg-fill');
        const valEl = row.querySelector('.sg-val');
        if (fillEl) fillEl.style.width = `${Math.round(fill * 100)}%`;
        if (valEl) valEl.textContent = `${level}`;
        row.title = `${SG_SHORT[id]}: ${xp} XP`;
      }
    }
    // phase block
    const totalMin = Math.floor(state.focusMinutesTotal || 0);
    const phase = Math.floor(totalMin / 100);
    const phaseNum = document.getElementById('phase-num');
    if (phaseNum) phaseNum.textContent = String(phase).padStart(2, '0');
    const phaseFill = document.getElementById('sg-phase-fill');
    if (phaseFill) phaseFill.style.width = `${totalMin % 100}%`;
    const phaseVal = document.getElementById('sg-phase-val');
    if (phaseVal) phaseVal.textContent = `${totalMin % 100}m/100m`;
    drawSparkline();
  }

  function levelUpCelebration(statId, level) {
    const label = STATS.find(s => s.id === statId)?.label || statId;
    if (flash) {
      flash.hidden = false;
      flash.textContent = `✨ ${label} grew to Lv ${level}!`;
      clearTimeout(levelUpCelebration._t);
      levelUpCelebration._t = setTimeout(() => { flash.hidden = true; }, 6000);
    }
    try {
      window.aikoPlayGesture?.('happyBounce');
      window.aikoSetExpression?.('happy', 0.9);
    } catch (_) {}
  }

  function addXp(statId, amount, reason) {
    if (!STATS.some(s => s.id === statId) || !(amount > 0)) return;
    const before = levelOf(state.xp[statId] || 0);
    state.xp[statId] = Math.round((state.xp[statId] || 0) + amount);
    const after = levelOf(state.xp[statId]);
    save();
    render();
    if (after > before) levelUpCelebration(statId, after);
    else if (flash && reason) {
      flash.hidden = false;
      flash.textContent = `+${Math.round(amount)} ${STATS.find(s => s.id === statId).label} — ${reason}`;
      clearTimeout(addXp._t);
      addXp._t = setTimeout(() => { flash.hidden = true; }, 4000);
    }
  }

  function todayKey() {
    const d = new Date();
    return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
  }

  // ── event wiring ──
  window.addEventListener('aiko:focus-start', () => {
    try { window.setPresence?.('focus'); } catch (_) {}
  });

  window.addEventListener('aiko:focus-stop', e => {
    const { seconds = 0, routine = 'coding' } = e.detail || {};
    const minutes = seconds / 60;
    state.focusMinutesTotal = (state.focusMinutesTotal || 0) + minutes;
    state.focusHistory = [...(state.focusHistory || []), Math.round(minutes * 10) / 10].slice(-40);
    const train = (trainSel && trainSel.value) || 'intelligence';
    const main = Math.max(1, Math.round(minutes));
    addXp(train, main, `${Math.round(minutes)} min focus`);
    const natural = ROUTINE_STAT[routine];
    if (natural && natural !== train) addXp(natural, Math.max(1, Math.round(minutes / 2)), `${routine} flow`);
    if (seconds >= 1500) addXp(train, 10, 'pomodoro bonus 🎯');
    try { if (window.setPresence) window.setPresence('idle'); } catch (_) {}
  });

  window.addEventListener('aiko:task-done', () => {
    addXp('morality', 3, 'task completed');
  });

  window.addEventListener('aiko:msg-sent', () => {
    const day = todayKey();
    if (state.bondDay !== day) { state.bondDay = day; state.bondCount = 0; }
    if (state.bondCount >= 40) return;   // soft daily cap keeps it about time, not spam
    state.bondCount++;
    addXp('bond', 1);
  });

  window.AikoGrowth = {
    addXp,
    stats() { return { ...state.xp }; },
    levels() {
      const out = {};
      for (const st of STATS) out[st.id] = levelOf(state.xp[st.id] || 0);
      return out;
    },
  };

  render();
})();
