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
    return { xp: { intelligence: 0, sensitivity: 0, morality: 0, bond: 0 }, bondDay: '', bondCount: 0, focusMinutesTotal: 0 };
  }
  let state = loadState();
  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (_) {} }

  const grid = document.getElementById('growth-grid');
  const flash = document.getElementById('growth-flash');
  const trainSel = document.getElementById('train-stat');

  const SG_SHORT = { intelligence: 'INT', sensitivity: 'SEN', morality: 'MOR' };
  function render() {
    if (grid) {
      grid.innerHTML = '';
      for (const st of STATS) {
        const xp = state.xp[st.id] || 0;
        const { level, fill } = levelProgress(xp);
        const el = document.createElement('div');
        el.className = 'growth-stat';
        el.dataset.stat = st.id;
        el.innerHTML =
          `<div class="growth-top"><span class="growth-name">${st.label}</span>` +
          `<span class="growth-lv">Lv ${level}</span></div>` +
          `<div class="growth-bar"><div class="growth-fill" style="width:${Math.round(fill * 100)}%"></div></div>`;
        el.title = `${st.label}: ${xp} XP`;
        grid.appendChild(el);
      }
    }
    // Fenestra-style mini HUD on the stage: thin bars + phase
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
        if (valEl) valEl.textContent = `Lv${level}`;
        row.title = `${SG_SHORT[id]}: ${xp} XP`;
      }
      const totalMin = Math.floor(state.focusMinutesTotal || 0);
      const phase = Math.floor(totalMin / 100);
      const phaseText = document.getElementById('sg-phase-text');
      const phaseFill = document.getElementById('sg-phase-fill');
      const phaseVal = document.getElementById('sg-phase-val');
      if (phaseText) phaseText.textContent = `PHASE ${phase}`;
      if (phaseFill) phaseFill.style.width = `${totalMin % 100}%`;
      if (phaseVal) phaseVal.textContent = `${totalMin % 100}m/100m`;
    }
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
