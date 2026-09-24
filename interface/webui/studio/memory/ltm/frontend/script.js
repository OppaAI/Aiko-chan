/* Neural LTM — organic neural-field view (reference: neural memory demo).
 *
 * Deterministic one-shot force layout (seeded, bounded ticks, no live
 * simulation loop), glossy 3D sphere nodes via one shared radial gradient
 * per hue, dotted synapse connectors whose brightness follows both endpoint
 * sizes. Size + brightness track score / tendency to retain.
 */
const API_BASE = GraphBoot.apiBase();

let graph = { nodes: [], edges: [], meta: {} };
let zoomBehavior = null;
let userEntityId = null;
let currentUsername = 'OppaAI';

/* Live selections — rebuilt only when new data arrives. */
let nodeSel = null;   // d3 selection of node <g> elements
let linkSel = null;   // d3 selection of edge <line> elements
let visibleNodes = [];
let nodeById = new Map();

async function getCurrentUser() {
  try {
    const resp = await fetch('/api/auth/me', { credentials: 'include' });
    if (resp.ok) {
      const data = await resp.json();
      currentUsername = data.username || currentUsername;
    }
  } catch (e) {}
  return currentUsername;
}

function qs() {
  const p = new URLSearchParams();
  p.set('limit', document.getElementById('limit').value || '200');
  p.set('include_history', document.getElementById('include-history').checked);
  p.set('include_entities', document.getElementById('include-entities').checked);
  p.set('include_knowledge', document.getElementById('include-knowledge').checked);
  p.set('include_experience', document.getElementById('include-experience').checked);
  p.set('include_episodes', document.getElementById('include-episodes').checked);
  const df = document.getElementById('date-from').value;
  const dt = document.getElementById('date-to').value;
  if (df) p.set('date_from', df);
  if (dt) p.set('date_to', dt);
  return p.toString();
}

async function loadGraph() {
  document.getElementById('status').textContent = 'Loading…';
  try {
    const resp = await fetch(`${API_BASE}/graph?` + qs());
    graph = await resp.json();
    stretchRetain(graph.nodes);
    const m = graph.meta || {};
    document.getElementById('stats').innerHTML =
      `neurons: ${m.memory_count ?? 0}<br/>entities: ${m.entity_count ?? 0}<br/>knowledge: ${(graph.nodes||[]).filter(n=>n.type==='knowledge').length}<br/>experience: ${(graph.nodes||[]).filter(n=>n.type==='experience').length}<br/>episodes: ${m.episode_count ?? 0}<br/>synapses: ${m.edge_count ?? 0}`;
    document.getElementById('status').textContent =
      `${(graph.nodes||[]).length} nodes · ${(graph.edges||[]).length} edges`;
    document.getElementById('graph-stats').textContent =
      `${(graph.nodes||[]).length} neurons · ${(graph.edges||[]).length} synapses · drag nodes · scroll zoom`;
    await getCurrentUser();
    userEntityId = `ent:${currentUsername.toLowerCase()}`;
    buildGraph();
  } catch (e) {
    document.getElementById('status').textContent = 'Load failed';
    console.error(e);
  }
}

/** Retain in [0,1] — drives BOTH size and brightness */
function retainOf(d) {
  if (d._dispRetain != null) return d._dispRetain;
  const sc = d.scores || {};
  if (sc.retain != null) return Math.max(0, Math.min(1, Number(sc.retain)));
  // Entities carry importance/access scores instead of a retain estimate.
  if (d.type === 'entity') {
    if (sc.importance != null) return Math.max(0, Math.min(1, Number(sc.importance)));
    if (sc.access != null) return Math.max(0, Math.min(1, Number(sc.access)));
  }
  if (d.size != null) {
    // backend size ≈ 0.18 + 1.27 * retain^1.18 → approximate invert
    const s = Math.max(0.18, Number(d.size));
    const t = Math.max(0, Math.min(1, (s - 0.18) / 1.27));
    return Math.pow(t, 1 / 1.18);
  }
  return d.type === 'entity' ? 0.4 : 0.35;
}

/**
 * Contrast-stretch score values per node type across the visible node set,
 * so size/brightness differences are actually visible when scores cluster
 * (raw retain is often ~flat, e.g. 0.48 for half the nodes).
 * Stronger curve + slight valence lift so cyan/gold nodes don't stay tiny/dim.
 */
function stretchRetain(nodes) {
  for (const n of nodes) if (n._dispRetain != null) delete n._dispRetain;
  // Contrast-stretch per node type so size/brightness differences stay
  // visible when scores cluster (raw retain is often ~flat).
  for (const t of ['memory', 'entity']) {
    const grp = (nodes || []).filter(n => n.type === t);
    const r = grp.map(retainOf).filter(v => isFinite(v));
    if (r.length < 3) continue;
    let lo = Math.min.apply(null, r);
    let hi = Math.max.apply(null, r);
    // Force a usable dynamic range even when scores are tightly clustered
    if (hi - lo < 0.18) { lo = Math.max(0, lo - 0.22); hi = Math.min(1, hi + 0.22); }
    const span = (hi - lo) || 1;
    for (const n of grp) {
      const v = retainOf(n);
      // Power curve expands mid/high retain; floor keeps low nodes visible
      let stretched = 0.08 + 0.92 * Math.pow((v - lo) / span, 0.72);
      // Mild valence lift so pos/neg still read larger/brighter than pure neutrals
      const hue = valenceHue(n);
      if (hue === 'pos' || hue === 'neg') stretched = Math.min(1, stretched + 0.07);
      if (n.pinned) stretched = Math.max(stretched, 0.78);
      n._dispRetain = Math.max(0.08, Math.min(1, stretched));
    }
  }
}

function valenceHue(d) {
  if (d.type === 'entity') return 'entity';
  if (d.type === 'knowledge') return 'knowledge';
  if (d.type === 'experience') return 'experience';
  if (d.type === 'episode') return 'episode';
  if (d.imprint) return 'imprint';
  let v = (d.valence_tag || 'neutral').toLowerCase();
  const vs = d.valence_score;
  if ((v === 'neutral' || !v) && vs != null && vs !== '') {
    const s = Number(vs);
    if (s <= -1) v = 'neg';
    else if (s >= 1) v = 'pos';
  }
  if (v === 'neg') return 'neg';
  if (v === 'pos') return 'pos';
  return 'neutral';
}

function hueColor(hue) {
  if (hue === 'entity') return '#554f7a';
  if (hue === 'knowledge') return '#5f7d6b';
  if (hue === 'experience') return '#7d6b57';
  if (hue === 'episode') return '#4a6b70';
  if (hue === 'imprint') return '#6e5a70';
  if (hue === 'neg') return '#385c70';
  if (hue === 'pos') return '#796a3c';
  return '#646c7a';
}

/** Translucent glass fill — dim nodes ghost into the background. */
function nodeOpacity(d) {
  let r = retainOf(d);
  let o = 0.18 + Math.pow(r, 0.9) * 0.55;
  const hue = valenceHue(d);
  if (hue === 'pos' || hue === 'neg') o = Math.min(1, o + 0.08);
  if (d.pinned) o = Math.max(o, 0.92);
  if (d.type === 'memory' && (d.status === 'superseded' || d.is_tip === false)) {
    o = Math.min(o, 0.35);
  }
  return o;
}

function nodeRadius(d) {
  let r = retainOf(d);
  if (d.pinned) r = Math.max(r, 0.72);
  const hue = valenceHue(d);
  if ((hue === 'pos' || hue === 'neg') && d.type === 'memory') {
    r = Math.min(1, r + 0.06);
  }
  // Continuous, score-proportional radius for every node type.
  if (d.type === 'entity') return 3.5 + 10 * Math.pow(r, 1.2);
  return 2.5 + 10.5 * Math.pow(r, 1.25);
}

function edgeOpacity(e) {
  const s = nodeById.get(typeof e.source === 'object' ? e.source.id : e.source);
  const t = nodeById.get(typeof e.target === 'object' ? e.target.id : e.target);
  const rs = s ? retainOf(s) : 0.25;
  const rt = t ? retainOf(t) : 0.25;
  const mid = Math.pow(Math.max(0.05, rs * rt), 0.45);
  const w = Math.max(0, Math.min(1, Number(e.weight) || 0.4));
  const wBoost = 0.35 + 0.65 * w;
  if (e.type === 'supersedes') return Math.min(0.85, (0.25 + mid * 0.7) * wBoost);
  if (e.type === 'mentions' || e.type === 'grounded_in' || e.type === 'practiced_in' || e.type === 'distilled_into') {
    return Math.min(0.8, (0.06 + mid * 0.75) * wBoost);
  }
  return Math.min(0.9, (0.10 + mid * 0.75) * wBoost);
}

function edgeColor(e) {
  if (e.type === 'supersedes') return '#a68b4f';
  if (e.type === 'distilled_into') return '#4a6b70';
  return '#54606f';
}

function lineageText(d, graph) {
  if (!d || d.type !== 'memory') return '';
  const edges = (graph.edges || []).filter(e => e.type === 'supersedes');
  const nodes = Object.fromEntries((graph.nodes || []).map(n => [n.id, n]));
  const idOf = (x) => (x && typeof x === 'object' ? x.id : x);
  const textOf = (n) => String((n && (n.text || n.label)) || '').slice(0, 100);
  // export: source=newer → target=older
  const olderId = idOf(edges.find(e => idOf(e.source) === d.id)?.target);
  const newerId = idOf(edges.find(e => idOf(e.target) === d.id)?.source);
  const bits = [];
  const selfText = textOf(d);
  if (d.status === 'superseded' || d.is_tip === false) {
    bits.push('(superseded)');
    bits.push('Was: ' + selfText);
    if (newerId && nodes[newerId]) bits.push('Now: ' + textOf(nodes[newerId]));
  } else {
    if (olderId && nodes[olderId]) bits.push('Was: ' + textOf(nodes[olderId]));
    bits.push('Now: ' + selfText);
  }
  return bits.join('\n');
}

function showDetails(d) {
  const el = document.getElementById('details');
  el.style.display = 'block';
  const sc = d.scores || {};
  const scoreRows = ['salience', 'spacing', 'connectivity', 'valence', 'access'].map(k => {
    const v = Math.max(0, Math.min(1, Number(sc[k] || 0)));
    return `<div class="detail-label">${k}</div>
      <div class="score-bar"><i style="width:${(v*100).toFixed(0)}%"></i></div>
      <div class="detail-value">${v.toFixed(3)}</div>`;
  }).join('');
  const rows = [
    ['type', d.type],
    ['id', d.id],
    ['kind', d.kind],
    ['status', d.status],
    ['pinned', d.pinned ? 'yes' : 'no'],
    ['valence', d.valence_tag || '—'],
    ['retain', retainOf(d).toFixed(3)],
    ['created', d.created_at || '—'],
    ['entities', (d.entities || []).join(', ') || '—'],
    ['supersedes', d.supersedes_id || '—'],
    ['text', d.text || d.label || ''],
  ];
  if (d.type === 'episode') {
    rows.splice(6, 0,
      ['when', d.created_at || d.date || '—'],
      ['date', d.date || '—'],
      ['salience', d.salience_score != null ? d.salience_score : '—'],
      ['arousal', d.arousal_score != null ? d.arousal_score : '—'],
      ['recalls', d.access_count ?? 0],
      ['distilled', d.distilled_at || '—'],
      ['source', d.source || '—'],
      ['session', d.session_id || '—'],
    );
  }
  let html = '<h3>Neuron</h3>' + rows.map(([k,v]) =>
    `<div class="detail-label">${k}</div><div class="detail-value">${escapeHtml(String(v))}</div>`
  ).join('') + '<h3 style="margin-top:14px">Factor scores</h3>' + scoreRows;

  if (d.type === 'episode') {
    const extra = [
      ['recall_count', d.recall_count != null ? d.recall_count : '—'],
      ['distilled', d.distilled ? 'yes' : 'no'],
      ['distilled_at', d.distilled_at || '—'],
      ['distilled_into', (d.distilled_into || []).join(', ') || '—'],
    ];
    html += '<h3 style="margin-top:14px">Episodic</h3>' + extra.map(([k,v]) =>
      `<div class="detail-label">${k}</div><div class="detail-value">${escapeHtml(String(v))}</div>`
    ).join('');
  }

  const lin = lineageText(d, graph);
  if (lin) {
    html += '<h3 style="margin-top:14px">Lineage</h3>'
      + `<div class="detail-value" style="white-space:pre-wrap">${escapeHtml(lin)}</div>`;
  }
  el.innerHTML = html;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function runSearch() {
  const q = document.getElementById('search-q').value.trim();
  const box = document.getElementById('search-stats');
  if (!q) { box.textContent = 'type a query'; return; }
  box.textContent = 'Searching…';
  try {
    const p = new URLSearchParams();
    p.set('q', q);
    const uidEl = document.getElementById('user-id');
    if (uidEl && uidEl.value.trim()) p.set('user_id', uidEl.value.trim());
    p.set('limit', '10');
    const resp = await fetch(`${API_BASE}/search?` + p.toString());
    const data = await resp.json();
    const hits = data.hits || [];
    if (!hits.length) { box.textContent = 'No hits'; return; }
    box.innerHTML = hits.map(h =>
      `<div style="margin-bottom:10px;padding:8px;border:1px solid var(--dim);border-radius:4px;background:#120a1c">
         <div style="color:var(--pink);text-transform:uppercase;font-size:9px;letter-spacing:.1em">
           ${escapeHtml(h.store)} · ${escapeHtml(h.kind)} · ${Number(h.score||0).toFixed(3)}
         </div>
         <div style="margin-top:4px;color:var(--text)">${escapeHtml(h.text)}</div>
       </div>`
    ).join('');
  } catch (e) {
    box.textContent = 'Search failed';
    console.error(e);
  }
}

/* ── filtering (pure, no DOM) ─────────────────────────────────────────── */
function filteredNodes() {
  const st = document.getElementById('filter-status')?.value || 'all';
  const val = document.getElementById('filter-valence')?.value || 'all';
  const minR = parseFloat(document.getElementById('filter-min-retain')?.value || '0') || 0;
  const entQ = (document.getElementById('filter-entity')?.value || '').trim().toLowerCase();
  const showMem = document.getElementById('layer-memory')?.checked !== false;
  const showEnt = document.getElementById('layer-entity')?.checked !== false;
  const showKb = document.getElementById('layer-knowledge')?.checked !== false;
  const showExp = document.getElementById('layer-experience')?.checked !== false;
  const showEp = document.getElementById('layer-episodes')?.checked !== false;

  const layerOn = { memory: showMem, entity: showEnt, knowledge: showKb, experience: showExp, episode: showEp };

  let nodes = (graph.nodes || []).filter(d => {
    if (!layerOn[d.type]) return false;
    if (d.type === 'memory') {
      if (st !== 'all' && (d.status || 'active') !== st) return false;
      if (val !== 'all' && (d.valence_tag || 'neutral') !== val) return false;
      if (retainOf(d) < minR) return false;
      if (entQ) {
        const ents = (d.entities || []).map(e => String(e).toLowerCase());
        if (!ents.some(e => e.includes(entQ))) return false;
      }
    }
    if (d.type === 'entity' && entQ &&
        !String(d.label || d.text || '').toLowerCase().includes(entQ)) return false;
    return true;
  });

  // Pull in entity hubs connected to any kept node (bounded to entity type
  // so filtered-out memories can't leak back in).
  const keep = new Set(nodes.map(n => n.id));
  const nodeTypeOf = new Map((graph.nodes || []).map(n => [n.id, n.type]));
  let grew = true;
  while (grew) {
    grew = false;
    for (const e of (graph.edges || [])) {
      const sid = e.source, tid = e.target;
      if (keep.has(sid) && !keep.has(tid) && nodeTypeOf.get(tid) === 'entity') { keep.add(tid); grew = true; }
      else if (keep.has(tid) && !keep.has(sid) && nodeTypeOf.get(sid) === 'entity') { keep.add(sid); grew = true; }
    }
  }
  nodes = (graph.nodes || []).filter(n => {
    if (!keep.has(n.id)) return false;
    if (!layerOn[n.type]) return false;
    if (n.type === 'entity' && entQ &&
        !String(n.label || n.text || '').toLowerCase().includes(entQ)) return false;
    return true;
  });

  const keep2 = new Set(nodes.map(n => n.id));
  const links = (graph.edges || []).filter(e => keep2.has(e.source) && keep2.has(e.target));
  return { nodes, links };
}

/* Deterministic hash for stable jitter (no Math.random per render). */
function hashStr(s) {
  let h = 2166136261;
  const str = String(s);
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

/* Deterministic organic layout — reference style: type clusters relax into a
 * neural field. One-shot force simulation (bounded ticks, seeded RNG, no
 * live tick loop): positions are computed synchronously, then the
 * simulation is stopped. Drag/zoom/pan stay fully interactive. */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Lighten (amt>0) or darken (amt<0) a #rrggbb color, amt in [-1, 1]. */
function shade(hex, amt) {
  const n = parseInt(String(hex).slice(1), 16);
  let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  const t = amt < 0 ? 0 : 255, p = Math.abs(Math.max(-1, Math.min(1, amt)));
  r = Math.round(r + (t - r) * p); g = Math.round(g + (t - g) * p); b = Math.round(b + (t - b) * p);
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/* Organic neural-field layout — deterministic, one-shot, no live loop.
 *
 * Connected components are placed on a golden-angle spiral (organic by
 * construction — never rows, columns, or chains) and members are
 * scattered inside their component disc. Overlaps relax with collision
 * forces only: there is no link force, so edge chains can never pull
 * nodes into straight lines. */
function layoutOrganic(nodes, links, w, h) {
  const cx = w / 2, cy = h / 2;
  const R = Math.min(w, h) * 0.34;

  // 1. Connected components (union-find) over visible nodes/links.
  const parent = new Map(nodes.map(n => [n.id, n.id]));
  const find = x => {
    let r = x;
    while (parent.get(r) !== r) r = parent.get(r);
    let c = x;
    while (parent.get(c) !== r) { const nx = parent.get(c); parent.set(c, r); c = nx; }
    return r;
  };
  const idOf = v => (v && typeof v === 'object' ? v.id : v);
  const live = new Set(nodes.map(n => n.id));
  for (const l of links) {
    const s = idOf(l.source), t = idOf(l.target);
    if (s !== t && live.has(s) && live.has(t)) parent.set(find(s), find(t));
  }
  const byRoot = new Map();
  for (const n of nodes) {
    const r = find(n.id);
    if (!byRoot.has(r)) byRoot.set(r, []);
    byRoot.get(r).push(n);
  }
  const comps = [...byRoot.values()].sort((a, b) => b.length - a.length);
  if (!comps.length) return;

  const compRadius = comp => {
    let area = 0;
    for (const n of comp) { const r = nodeRadius(n) + 6; area += r * r; }
    return Math.sqrt(area / Math.PI) * 1.08 + 4;
  };

  // 2. Component centers on a golden-angle spiral.
  const GOLDEN = Math.PI * (3 - Math.sqrt(5));
  const rnd = mulberry32(90210);
  const c0 = { x: cx, y: cy, r: compRadius(comps[0]) };
  const centers = [c0, ...comps.slice(1).map((comp, j) => {
    const i = j + 1;
    const cr = compRadius(comp);
    const a = i * GOLDEN + (rnd() - 0.5) * 0.6;
    const dist = c0.r + cr + R * 0.10 * Math.sqrt(i) + 12;
    const m = 14;
    return {
      x: Math.max(cr + m, Math.min(w - cr - m, cx + Math.cos(a) * dist)),
      y: Math.max(cr + m, Math.min(h - cr - m, cy + Math.sin(a) * dist)),
      r: cr,
    };
  })];

  // 3. Hub-cloud scatter: entity hubs (boosted) and top-degree nodes become
  //    sub-cluster centers; their 1–2 hop neighbors form a soft gaussian
  //    cloud around them, so similar nodes clump together. Unattached nodes
  //    fill the disc. All jitter is isotropic — clumps form, but straight
  //    lines cannot.
  const adj = new Map(nodes.map(n => [n.id, []]));
  const deg = new Map(nodes.map(n => [n.id, 0]));
  for (const l of links) {
    const s = idOf(l.source), t = idOf(l.target);
    if (s === t || !adj.has(s) || !adj.has(t)) continue;
    adj.get(s).push(t); adj.get(t).push(s);
    deg.set(s, deg.get(s) + 1); deg.set(t, deg.get(t) + 1);
  }
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  for (let i = 0; i < comps.length; i++) {
    const c = centers[i], comp = comps[i];
    if (comp.length === 1) {
      const n = comp[0];
      n.x = n._ax = c.x; n.y = n._ay = c.y; n.vx = n.vy = 0;
      continue;
    }
    const score = n => deg.get(n.id) + (n.type === 'entity' ? 100000 : 0);
    const hubs = [...comp].sort((a, b) => score(b) - score(a))
      .slice(0, Math.max(1, Math.min(5, Math.round(comp.length / 6))));
    const hubSet = new Set(hubs.map(h => h.id));
    const hubOf = new Map(hubs.map(h => [h.id, h]));
    hubs.forEach((h, hi) => {
      const a = hi * GOLDEN + (rnd() - 0.5) * 0.9;
      const rr = c.r * 0.30 * Math.sqrt((hi + 1) / hubs.length);
      h._ax = c.x + Math.cos(a) * rr;
      h._ay = c.y + Math.sin(a) * rr;
    });
    for (const n of comp) {
      if (hubSet.has(n.id)) continue;
      let hid = null, hd = -1;
      const consider = id => {
        if (hubSet.has(id) && deg.get(id) > hd) { hd = deg.get(id); hid = id; }
      };
      for (const nb of adj.get(n.id)) consider(nb);
      if (!hid) for (const nb of adj.get(n.id)) for (const nn of adj.get(nb)) consider(nn);
      const hub = hid ? hubOf.get(hid) : null;
      if (hub) {
        const spread = Math.min(c.r * 0.35, 10 + 5 * Math.sqrt(adj.get(hub.id).length));
        n._ax = hub._ax + gauss() * spread * 0.30;
        n._ay = hub._ay + gauss() * spread * 0.30;
      } else {
        const a = rnd() * Math.PI * 2;
        const rr = Math.sqrt(rnd()) * Math.max(6, c.r - nodeRadius(n));
        n._ax = c.x + Math.cos(a) * rr;
        n._ay = c.y + Math.sin(a) * rr;
      }
    }
    for (const n of comp) {
      const maxR = Math.max(6, c.r - nodeRadius(n));
      const dx = n._ax - c.x, dy = n._ay - c.y;
      const d = Math.hypot(dx, dy);
      if (d > maxR) { n._ax = c.x + dx / d * maxR; n._ay = c.y + dy / d * maxR; }
      n.x = n._ax; n.y = n._ay; n.vx = 0; n.vy = 0;
    }
  }

  // 4. Relax overlaps (collision only) around each node's anchor.
  const sim = d3.forceSimulation(nodes)
    .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 1.5).iterations(4))
    .force('x', d3.forceX(d => d._ax).strength(0.45))
    .force('y', d3.forceY(d => d._ay).strength(0.45))
    .randomSource(mulberry32(1777))
    .stop();
  for (let i = 0; i < 150; i++) sim.tick();
  sim.stop();
  for (const n of nodes) { delete n._ax; delete n._ay; }

  // 5. Scale the field to fill the canvas disc.
  const ds = nodes.map(n => Math.hypot(n.x - cx, n.y - cy)).sort((a, b) => a - b);
  const p90 = ds[Math.floor(ds.length * 0.9)] || 1;
  const k = (R * 0.95) / Math.max(1, p90);
  for (const n of nodes) {
    n.x = cx + (n.x - cx) * k;
    n.y = cy + (n.y - cy) * k;
  }
}

function nodeLabel(d) {
  const t = d.label || d.id;
  return t.length > 22 ? t.slice(0, 20) + '…' : t;
}

/* ── build (data changed) ───────────────────────────────────────────── */
function buildGraph() {
  const svg = d3.select('#canvas');
  svg.selectAll('*').remove();
  const area = document.getElementById('canvas-area');
  const w = area.clientWidth || 1000;
  const h = area.clientHeight || 700;
  svg.attr('viewBox', `0 0 ${w} ${h}`);

  const { nodes, links } = filteredNodes();
  visibleNodes = nodes;
  nodeById = new Map(nodes.map(n => [n.id, n]));

  const g = svg.append('g');
  zoomBehavior = GraphBoot.makeZoom({ scaleExtent: [0.15, 4], target: g });
  svg.call(zoomBehavior);

  defs_(svg);

  if (!nodes.length) {
    svg.append('text').attr('x', w/2).attr('y', h/2).attr('text-anchor','middle')
      .attr('fill','var(--dim)').text('No memories yet');
    nodeSel = linkSel = null;
    return;
  }

  layoutOrganic(nodes, links, w, h);

  const nebR = Math.min(w, h) * 0.58;
  g.append('circle')
    .attr('cx', w / 2).attr('cy', h / 2).attr('r', nebR)
    .attr('fill', 'url(#nebula)').attr('pointer-events', 'none');

  // Which nodes carry visible labels: hottest ~75 by retain, small and dim
  // like the reference (labels ride on the bright nodes, not the whole field).
  const hotIds = new Set(
    [...nodes].sort((a, b) => retainOf(b) - retainOf(a)).slice(0, 75).map(n => n.id)
  );

  // dotted synapse connectors — brightness follows both endpoint sizes
  // (dim when both ends are small); supersede edges stay solid + directed.
  linkSel = g.append('g').selectAll('line').data(links).join('line')
    .attr('class', 'edge')
    .attr('stroke', d => d.type === 'supersedes' ? '#a68b4f' : shade(edgeColor(d), 0.12))
    .attr('stroke-width', d => {
      if (d.type === 'supersedes') return 1.2;
      const wgt = Math.max(0, Math.min(1, Number(d.weight) || 0.4));
      return 0.5 + wgt * 0.6;
    })
    .attr('stroke-opacity', d => edgeOpacity(d) * 0.9)
    .attr('stroke-linecap', 'round')
    .attr('stroke-dasharray', d => d.type === 'supersedes' ? null : '1.5,2.5')
    .attr('marker-end', d => d.type === 'supersedes' ? 'url(#arrow-sup)' : null)
    .attr('x1', d => nodeById.get(d.source).x)
    .attr('y1', d => nodeById.get(d.source).y)
    .attr('x2', d => nodeById.get(d.target).x)
    .attr('y2', d => nodeById.get(d.target).y);

  // glossy 3D spheres — one shared gradient per hue, brightness from retain
  nodeSel = g.append('g').selectAll('g').data(nodes, d => d.id).join('g')
    .attr('class', 'node-group')
    .style('cursor', 'pointer')
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .on('click', (event, d) => { event.stopPropagation(); showDetails(d); });

  nodeSel.append('circle')
    .attr('class', 'sphere')
    .attr('r', nodeRadius)
    .attr('fill', d => `url(#gloss-${valenceHue(d)})`)
    .attr('fill-opacity', nodeOpacity)
    .attr('stroke', d => d.pinned ? '#ffffff' : shade(hueColor(valenceHue(d)), -0.25))
    .attr('stroke-width', d => d.pinned ? 2 : 1)
    .attr('stroke-opacity', d => d.pinned ? 0.9 : 0.35)
    .attr('stroke-dasharray', d =>
      (d.type === 'memory' && (d.status === 'superseded' || d.is_tip === false)) ? '3,2' : null);

  // soft glass sheen on bright nodes — frosted, not plastic
  nodeSel.append('ellipse')
    .attr('class', 'glare')
    .attr('cx', d => -nodeRadius(d) * 0.28)
    .attr('cy', d => -nodeRadius(d) * 0.32)
    .attr('rx', d => nodeRadius(d) * 0.48)
    .attr('ry', d => nodeRadius(d) * 0.30)
    .attr('fill', 'url(#glare)')
    .attr('opacity', d => 0.10 + 0.28 * retainOf(d))
    .attr('pointer-events', 'none');

  nodeSel.append('text').attr('class', 'node-label')
    .attr('dy', d => -(nodeRadius(d) + 6))
    .attr('text-anchor', 'middle')
    .style('opacity', d => (hotIds.has(d.id) || d.pinned) ? 0.6 : 0)
    .text(nodeLabel);

  // free drag without a force simulation — move node + its edges
  nodeSel.call(d3.drag()
    .on('start', function (event, d) { d3.select(this).raise(); })
    .on('drag', function (event, d) {
      d.x = event.x; d.y = event.y;
      d3.select(this).attr('transform', `translate(${d.x},${d.y})`);
      linkSel
        .filter(l => l.source === d.id || l.target === d.id)
        .attr('x1', l => nodeById.get(l.source).x)
        .attr('y1', l => nodeById.get(l.source).y)
        .attr('x2', l => nodeById.get(l.target).x)
        .attr('y2', l => nodeById.get(l.target).y);
    }));

  svg.on('click', () => { document.getElementById('details').style.display = 'none'; });
}

/* Shared defs: one arrowhead marker + one radial gloss gradient per hue
 * (reused by every node — no per-node defs) + one shared white glare. */
const HUES = ['neg', 'pos', 'neutral', 'entity', 'knowledge', 'experience', 'episode', 'imprint'];
function defs_(svg) {
  const defs = svg.append('defs');
  defs.append('marker').attr('id','arrow-sup').attr('viewBox','0 0 10 10')
    .attr('refX', 9).attr('refY', 5).attr('markerWidth', 5).attr('markerHeight', 5).attr('orient','auto')
    .append('path').attr('d','M 0 1 L 10 5 L 0 9 Z').attr('fill', '#a68b4f').attr('opacity', 0.8);
  // Glassy orbs: soft sheen, feathered edge that melts into
  // the dark background (outer stop fades to transparent).
  for (const h of HUES) {
    const base = hueColor(h);
    const g = defs.append('radialGradient')
      .attr('id', 'gloss-' + h)
      .attr('cx', '34%').attr('cy', '28%').attr('r', '78%');
    g.append('stop').attr('offset', '0%').attr('stop-color', shade(base, 0.38));
    g.append('stop').attr('offset', '40%').attr('stop-color', shade(base, 0.10));
    g.append('stop').attr('offset', '72%').attr('stop-color', base);
    g.append('stop').attr('offset', '100%').attr('stop-color', base).attr('stop-opacity', 0.30);
  }
  // Shared specular glare dabbed on hot nodes.
  const glare = defs.append('radialGradient').attr('id', 'glare').attr('cx', '50%').attr('cy', '50%').attr('r', '50%');
  glare.append('stop').attr('offset', '0%').attr('stop-color', '#ffffff').attr('stop-opacity', 0.85);
  glare.append('stop').attr('offset', '100%').attr('stop-color', '#ffffff').attr('stop-opacity', 0);
  // Soft nebula glow behind the node cluster (reference style).
  const neb = defs.append('radialGradient').attr('id', 'nebula').attr('cx', '50%').attr('cy', '50%').attr('r', '50%');
  neb.append('stop').attr('offset', '0%').attr('stop-color', '#2c3c58').attr('stop-opacity', 0.34);
  neb.append('stop').attr('offset', '55%').attr('stop-color', '#1b2540').attr('stop-opacity', 0.14);
  neb.append('stop').attr('offset', '100%').attr('stop-color', '#1b2540').attr('stop-opacity', 0);
}

/* ── re-filter: organic re-layout of the visible set (full rebuild) ─────── */
function refilter() {
  buildGraph();
}

document.getElementById('refresh').onclick = loadGraph;
document.getElementById('apply').onclick = loadGraph;
document.getElementById('apply-filters').onclick = refilter;
document.getElementById('search-btn').onclick = runSearch;
document.getElementById('search-q').addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(); });
document.getElementById('export').onclick = () => {
  const blob = new Blob([JSON.stringify(graph, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'neural-memory.json';
  a.click();
};
document.getElementById('zoom-in').onclick = () => d3.select('#canvas').transition().call(zoomBehavior.scaleBy, 1.3);
document.getElementById('zoom-out').onclick = () => d3.select('#canvas').transition().call(zoomBehavior.scaleBy, 0.7);
document.getElementById('zoom-fit').onclick = () => {
  d3.select('#canvas').transition().duration(400).call(zoomBehavior.transform, d3.zoomIdentity);
};

loadGraph();
