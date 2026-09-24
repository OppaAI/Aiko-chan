/* Neural LTM — layered neural-network view.
 *
 * Deterministic column layout (input → hidden → output), one flat-filled
 * circle per node, straight connector lines. No force simulation, no
 * per-node gradients/filters, no decorative DOM — the whole graph is
 * N circles + M lines + K layer labels.
 */
const API_BASE = GraphBoot.apiBase();

/* Layers left → right: raw traces in, distilled knowledge middle, entity hubs out. */
const LAYERS = [
  { key: 'memory',     label: 'Episodic Memory' },
  { key: 'episode',    label: 'Episodes' },
  { key: 'experience', label: 'Experience' },
  { key: 'knowledge',  label: 'Knowledge' },
  { key: 'entity',     label: 'Entity Hubs' },
];
const LAYER_INDEX = Object.fromEntries(LAYERS.map((l, i) => [l.key, i]));

let graph = { nodes: [], edges: [], meta: {} };
let zoomBehavior = null;
let userEntityId = null;
let currentUsername = 'OppaAI';

/* Live selections — rebuilt only when new data arrives. */
let nodeSel = null;   // d3 selection of node <g> elements
let linkSel = null;   // d3 selection of edge <line> elements
let labelSel = null;  // d3 selection of layer label <text> elements
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
  if (d.size != null) {
    // backend size ≈ 0.18 + 1.27 * retain^1.18 → approximate invert
    const s = Math.max(0.18, Number(d.size));
    const t = Math.max(0, Math.min(1, (s - 0.18) / 1.27));
    return Math.pow(t, 1 / 1.18);
  }
  return d.type === 'entity' ? 0.4 : 0.35;
}

/**
 * Contrast-stretch memory retain values across the visible node set, so
 * size/brightness differences are actually visible when scores cluster
 * (raw retain is often ~flat, e.g. 0.48 for half the nodes).
 * Stronger curve + slight valence lift so cyan/gold nodes don't stay tiny/dim.
 */
function stretchRetain(nodes) {
  for (const n of nodes) if (n._dispRetain != null) delete n._dispRetain;
  const mem = (nodes || []).filter(n => n.type === 'memory');
  const r = mem.map(retainOf).filter(v => isFinite(v));
  if (r.length < 3) return;
  let lo = Math.min.apply(null, r);
  let hi = Math.max.apply(null, r);
  // Force a usable dynamic range even when scores are tightly clustered
  if (hi - lo < 0.18) { lo = Math.max(0, lo - 0.22); hi = Math.min(1, hi + 0.22); }
  const span = (hi - lo) || 1;
  for (const n of mem) {
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
  if (hue === 'entity') return '#b794f6';
  if (hue === 'knowledge') return '#4ade80';
  if (hue === 'experience') return '#fb923c';
  if (hue === 'episode') return '#51d4c8';
  if (hue === 'imprint') return '#c651a8';
  if (hue === 'neg') return '#3de0ff';
  if (hue === 'pos') return '#f0c14a';
  return '#8a9bb8';
}

/** Flat-fill opacity — score-proportional brightness, no gradients needed. */
function nodeOpacity(d) {
  let r = retainOf(d);
  let o = 0.25 + Math.pow(r, 0.9) * 0.75;
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
  if (d.type === 'entity') return 3.5 + 7 * Math.pow(r, 1.15);
  return 2.5 + 7.5 * Math.pow(r, 1.2);
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
  return Math.min(0.6, (0.04 + mid * 0.55) * wBoost);
}

function edgeColor(e) {
  if (e.type === 'supersedes') return '#f59e0b';
  if (e.type === 'distilled_into') return '#51d4c8';
  return '#3de0ff';
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

/* Deterministic layered layout: fixed x per column, even vertical spread
 * with stable jitter. Highest-retain nodes sit nearest the column center. */
function layoutLayered(nodes, w, h) {
  const topPad = 64, botPad = 30, sidePad = 70;
  const usableH = Math.max(120, h - topPad - botPad);
  const cols = LAYERS.map(l => ({ ...l, x: 0, nodes: [] }));
  for (const n of nodes) {
    const idx = LAYER_INDEX[n.type] ?? 0;
    cols[idx].nodes.push(n);
    n._col = idx;
  }
  cols.forEach((c, i) => {
    c.x = cols.length === 1 ? w / 2
      : sidePad + i * ((w - 2 * sidePad) / (cols.length - 1));
  });
  for (const c of cols) {
    c.nodes.sort((a, b) => retainOf(b) - retainOf(a));
    const n = c.nodes.length;
    c.nodes.forEach((d, j) => {
      const slot = usableH / Math.max(1, n);
      const jitter = (hashStr(d.id) - 0.5) * slot * 0.55;
      d.x = c.x + (hashStr(d.id + ':x') - 0.5) * 26;
      d.y = topPad + slot * (j + 0.5) + jitter;
    });
  }
  return cols;
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
    nodeSel = linkSel = labelSel = null;
    return;
  }

  layoutLayered(nodes, w, h);

  // layer labels
  const cols = LAYERS.map((l, i) => ({ ...l, i }));
  labelSel = g.append('g').selectAll('text').data(cols).join('text')
    .attr('class', 'layer-label')
    .attr('x', (c) => columnX(c.i, w))
    .attr('y', 30)
    .attr('text-anchor', 'middle')
    .text(c => `${c.label} (${countByLayer(nodes, c.key)})`);

  // straight connector lines — one <line> per edge
  linkSel = g.append('g').selectAll('line').data(links).join('line')
    .attr('class', 'edge')
    .attr('stroke', edgeColor)
    .attr('stroke-width', d => {
      if (d.type === 'supersedes') return 1.4;
      const wgt = Math.max(0, Math.min(1, Number(d.weight) || 0.4));
      return 0.6 + wgt * 1.4;
    })
    .attr('stroke-opacity', edgeOpacity)
    .attr('stroke-linecap', 'round')
    .attr('stroke-dasharray', d => d.type === 'supersedes' ? '5,3' : null)
    .attr('x1', d => nodeById.get(d.source).x)
    .attr('y1', d => nodeById.get(d.source).y)
    .attr('x2', d => nodeById.get(d.target).x)
    .attr('y2', d => nodeById.get(d.target).y);

  // one flat-filled circle per node
  nodeSel = g.append('g').selectAll('g').data(nodes, d => d.id).join('g')
    .attr('class', 'node-group')
    .style('cursor', 'pointer')
    .attr('transform', d => `translate(${d.x},${d.y})`)
    .on('click', (event, d) => { event.stopPropagation(); showDetails(d); });

  nodeSel.append('circle')
    .attr('r', nodeRadius)
    .attr('fill', d => hueColor(valenceHue(d)))
    .attr('fill-opacity', nodeOpacity)
    .attr('stroke', d => d.pinned ? '#ffffff' : hueColor(valenceHue(d)))
    .attr('stroke-width', d => d.pinned ? 2 : 1)
    .attr('stroke-opacity', d => d.pinned ? 0.9 : 0.55)
    .attr('stroke-dasharray', d =>
      (d.type === 'memory' && (d.status === 'superseded' || d.is_tip === false)) ? '3,2' : null);

  nodeSel.append('text').attr('class', 'node-label')
    .attr('dy', d => nodeRadius(d) + 11)
    .attr('text-anchor', 'middle')
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

/* Shared defs: one arrowhead marker only. No per-node gradients or filters. */
function defs_(svg) {
  const defs = svg.append('defs');
  defs.append('marker').attr('id','arrow-sup').attr('viewBox','0 0 10 10')
    .attr('refX', 9).attr('refY', 5).attr('markerWidth', 5).attr('markerHeight', 5).attr('orient','auto')
    .append('path').attr('d','M 0 1 L 10 5 L 0 9 Z').attr('fill', '#f59e0b').attr('opacity', 0.8);
}

function columnX(i, w) {
  const sidePad = 70;
  return LAYERS.length === 1 ? w / 2 : sidePad + i * ((w - 2 * sidePad) / (LAYERS.length - 1));
}

function countByLayer(nodes, key) {
  let c = 0;
  for (const n of nodes) if (n.type === key) c++;
  return c;
}

/* ── re-filter in place (no data reload, no DOM rebuild) ────────────── */
function refilter() {
  if (!nodeSel) { buildGraph(); return; }
  const area = document.getElementById('canvas-area');
  const w = area.clientWidth || 1000;
  const h = area.clientHeight || 700;

  const { nodes, links } = filteredNodes();
  visibleNodes = nodes;
  nodeById = new Map(nodes.map(n => [n.id, n]));
  const keepIds = new Set(nodes.map(n => n.id));

  layoutLayered(nodes, w, h);

  // nodes: toggle visibility, move to new slots
  nodeSel.each(function (d) { d._keepTransform = d3.select(this).attr('transform'); });
  nodeSel
    .style('display', d => keepIds.has(d.id) ? null : 'none')
    .transition().duration(280)
    .attr('transform', d => keepIds.has(d.id) ? `translate(${d.x},${d.y})` : d._keepTransform);

  // edges: keep only edges whose endpoints are visible, update endpoints
  const keepLinks = new Set(links.map(l => l.source + '→' + l.target));
  linkSel
    .style('display', d => (keepLinks.has(d.source + '→' + d.target)) ? null : 'none')
    .transition().duration(280)
    .attr('x1', d => nodeById.get(d.source)?.x ?? 0)
    .attr('y1', d => nodeById.get(d.source)?.y ?? 0)
    .attr('x2', d => nodeById.get(d.target)?.x ?? 0)
    .attr('y2', d => nodeById.get(d.target)?.y ?? 0)
    .attr('stroke-opacity', edgeOpacity);

  // layer counts
  labelSel.text(c => `${c.label} (${countByLayer(nodes, c.key)})`);

  document.getElementById('status').textContent =
    `${nodes.length} nodes · ${links.length} edges`;
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
