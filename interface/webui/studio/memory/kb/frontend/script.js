/* Knowledge Graph Studio — knowledge + entity nodes only.
 *
 * Organic neural-field view (reference: graph-explorer hairball):
 * deterministic one-shot force layout (seeded, bounded ticks, no live
 * simulation loop), glossy 3D spheres via one shared radial gradient per
 * type, thin faint connectors whose brightness follows both endpoint sizes.
 * Size + brightness track importance.
 */
const API_BASE = (window.KNOWLEDGE_API_BASE || GraphBoot.apiBase()).replace(/\/+$/, '');
const API_ROOT = API_BASE.endsWith('/api') ? API_BASE : API_BASE + '/api';

let graph = { nodes: [], edges: [], meta: {} };
let zoomBeh = null;
let viewport = null;   // zoom target group
let edgesG = null;     // lines layer
let nodesG = null;     // node layer
let linkSel = null;    // current line selection (for drag updates)
let currentNodes = [];
let currentLinks = [];

const COL_CHUNK = '#4ade80';
const COL_ENTITY = '#a78bfa';
const EDGE_ABOUT = '#6ee7a8';
const EDGE_SAMEDOC = '#5b4a6e';

function importanceOf(d) {
  const sc = d.scores || {};
  if (sc.importance != null) return Math.max(0, Math.min(1, Number(sc.importance)));
  if (sc.retain != null) return Math.max(0, Math.min(1, Number(sc.retain)));
  if (d.size != null) {
    const s = Math.max(0.20, Number(d.size));
    const t = Math.max(0, Math.min(1, (s - 0.20) / 1.10));
    return Math.pow(t, 1 / 1.25);
  }
  return d.type === 'entity' ? 0.4 : 0.3;
}

function nodeRadius(d) {
  const r = importanceOf(d);
  if (d.type === 'entity') return 5 + 16 * Math.pow(r, 1.25);
  return 5 + 24 * Math.pow(r, 1.35);
}

function nodeOpacity(d) {
  return 0.22 + importanceOf(d) * 0.78;
}

/* Edge brightness follows both endpoint sizes — dim when both ends are small. */
function edgeOpacity(l) {
  const rs = importanceOf(l.source);
  const rt = importanceOf(l.target);
  const mid = Math.pow(Math.max(0.05, rs * rt), 0.45);
  const base = l.type === 'about' ? 0.05 : 0.03;
  const gain = l.type === 'about' ? 0.60 : 0.45;
  return Math.min(0.65, base + mid * gain);
}

function nodeColor(d) {
  return d.type === 'entity' ? COL_ENTITY : COL_CHUNK;
}

function nodeLabel(d) {
  return String(d.label || d.doc_title || d.text || d.id || '').slice(0, 80);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function loadGraph() {
  const limit = document.getElementById('limit').value || 200;
  const includeHistory = document.getElementById('include-history').checked;
  const includeEntities = document.getElementById('include-entities').checked;
  const dateFrom = document.getElementById('date-from').value || '';
  const dateTo = document.getElementById('date-to').value || '';

  const params = new URLSearchParams({
    limit,
    include_history: includeHistory ? 'true' : 'false',
    include_entities: includeEntities ? 'true' : 'false',
  });
  if (dateFrom) params.append('date_from', dateFrom);
  if (dateTo) params.append('date_to', dateTo);

  const res = await fetch(`${API_ROOT}/graph?${params.toString()}`, { headers: { 'Accept': 'application/json' } });
  if (!res.ok) throw new Error(`graph request failed: ${res.status}`);
  graph = await res.json();
  if (graph && graph.meta && graph.meta.error) {
    throw new Error(`graph export failed: ${graph.meta.error}`);
  }
  update();
  const count = (graph.nodes || []).length;
  const details = document.getElementById('details');
  if (!count) {
    details.style.display = 'block';
    details.textContent = graph.meta?.error
      ? `No knowledge nodes loaded: ${graph.meta.error}`
      : 'No knowledge nodes found for the current user and filters.';
  }
}

/* Single-pass filter over the raw payload — no object cloning. Returns the
 * node objects themselves plus links resolved to node references. */
function filteredNodesEdges() {
  const minI = parseFloat(document.getElementById('min-imp').value || '0') || 0;
  const entQ = (document.getElementById('filter-entity').value || '').trim().toLowerCase();
  const all = graph.nodes || [];
  const rawEdges = graph.edges || [];

  // Pass 1: knowledge chunks that survive the importance / entity filters.
  const keepChunk = new Set();
  for (const n of all) {
    if (n.type === 'entity') continue;
    if (importanceOf(n) < minI) continue;
    if (entQ) {
      let hit = false;
      const ents = n.entities || [];
      for (const e of ents) {
        if (String(e).toLowerCase().includes(entQ)) { hit = true; break; }
      }
      if (!hit) continue;
    }
    keepChunk.add(n.id);
  }

  // Pass 2: entities referenced by surviving chunks via 'about' edges.
  const keepEnt = new Set();
  for (const e of rawEdges) {
    if (e.type === 'about' && keepChunk.has(e.source)) keepEnt.add(e.target);
  }

  // Pass 3: final node list + id lookup.
  const nodes = [];
  const byId = new Map();
  for (const n of all) {
    if (n.type === 'entity') {
      if (!keepEnt.has(n.id)) continue;
      if (entQ && !String(n.label || n.text || '').toLowerCase().includes(entQ)) continue;
    } else if (!keepChunk.has(n.id)) {
      continue;
    }
    nodes.push(n);
    byId.set(n.id, n);
  }

  // Pass 4: links resolved to node objects.
  const links = [];
  for (const e of rawEdges) {
    const s = byId.get(e.source);
    const t = byId.get(e.target);
    if (s && t) links.push({ source: s, target: t, type: e.type });
  }
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

/* Deterministic organic layout: type clusters seeded on a ring, relaxed by
 * a one-shot bounded force simulation (no live tick loop). */
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
    return Math.sqrt(area / Math.PI) * 1.5 + 8;
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

  // 3. Seeded scatter inside each component disc.
  for (let i = 0; i < comps.length; i++) {
    const c = centers[i];
    for (const n of comps[i]) {
      const a = rnd() * Math.PI * 2;
      const rr = Math.sqrt(rnd()) * Math.max(6, c.r - nodeRadius(n));
      n.x = c.x + Math.cos(a) * rr;
      n.y = c.y + Math.sin(a) * rr;
      n.vx = 0; n.vy = 0;
      n._ax = n.x; n._ay = n.y;
    }
  }

  // 4. Relax overlaps (collision only) around each node's anchor.
  const sim = d3.forceSimulation(nodes)
    .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 5).iterations(4))
    .force('x', d3.forceX(d => d._ax).strength(0.12))
    .force('y', d3.forceY(d => d._ay).strength(0.12))
    .randomSource(mulberry32(1777))
    .stop();
  for (let i = 0; i < 80; i++) sim.tick();
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

function showDetails(d) {
  const el = document.getElementById('details');
  el.style.display = 'block';
  const sc = d.scores || {};
  const keys = ['importance', 'access', 'recency', 'connectivity'];
  const bars = keys.map(k => {
    const v = Math.max(0, Math.min(1, Number(sc[k] || 0)));
    return `<div class="detail-label">${k}</div>
      <div class="score-bar"><i style="width:${(v * 100).toFixed(0)}%"></i></div>
      <div class="detail-value">${v.toFixed(3)}</div>`;
  }).join('');
  const rows = [
    ['type', d.type],
    ['id', d.id],
    ['title', d.doc_title || d.label || ''],
    ['source', d.doc_source || '—'],
    ['status', d.status || '—'],
    ['access', d.access_count ?? '—'],
    ['entities', (d.entities || []).join(', ') || '—'],
    ['text', d.text || ''],
  ];
  el.innerHTML = '<h3 style="margin:0 0 8px;font-size:13px">Node</h3>' +
    rows.map(([k, v]) =>
      `<div class="detail-label">${k}</div><div class="detail-value">${escapeHtml(String(v))}</div>`
    ).join('') +
    '<h3 style="margin:12px 0 6px;font-size:13px">Scores</h3>' + bars;
}

function ensureScene() {
  if (viewport) return;
  const svg = d3.select('#svg');
  const defs = svg.append('defs');
  // One shared glassmorphic-sphere gradient per node type + one shared glare.
  for (const [key, base] of [['chunk', COL_CHUNK], ['entity', COL_ENTITY]]) {
    const g = defs.append('radialGradient')
      .attr('id', 'gloss-' + key)
      .attr('cx', '34%').attr('cy', '28%').attr('r', '78%');
    g.append('stop').attr('offset', '0%').attr('stop-color', shade(base, 0.45));
    g.append('stop').attr('offset', '35%').attr('stop-color', shade(base, 0.12));
    g.append('stop').attr('offset', '70%').attr('stop-color', base);
    g.append('stop').attr('offset', '100%').attr('stop-color', shade(base, -0.38));
  }
  const glare = defs.append('radialGradient').attr('id', 'kb-glare').attr('cx', '50%').attr('cy', '50%').attr('r', '50%');
  glare.append('stop').attr('offset', '0%').attr('stop-color', '#ffffff').attr('stop-opacity', 0.85);
  glare.append('stop').attr('offset', '100%').attr('stop-color', '#ffffff').attr('stop-opacity', 0);

  viewport = svg.append('g').attr('id', 'viewport');
  zoomBeh = GraphBoot.makeZoom({ scaleExtent: [0.15, 4], target: viewport });
  svg.call(zoomBeh);

  edgesG = viewport.append('g').attr('id', 'edges');
  nodesG = viewport.append('g').attr('id', 'nodes');

  nodesG.append('title').text('Drag nodes to rearrange · scroll to zoom');
}

function edgeKey(l) {
  return l.source.id + '|' + l.target.id + '|' + (l.type || '');
}

function positionEdges(sel) {
  sel
    .attr('x1', l => l.source.x).attr('y1', l => l.source.y)
    .attr('x2', l => l.target.x).attr('y2', l => l.target.y);
}

function makeNodeDrag() {
  return d3.drag()
    .on('drag', (ev, d) => {
      d.x = ev.x;
      d.y = ev.y;
      d3.select(ev.sourceEvent.target.closest('g.node'))
        .attr('transform', `translate(${d.x},${d.y})`);
      if (linkSel) positionEdges(linkSel);
    });
}

/* Update in place: keyed joins, deterministic layout, no teardown, no
 * simulation. Cheap enough to run on every filter change. */
function update() {
  ensureScene();
  const svg = d3.select('#svg');
  const canvas = document.getElementById('canvas');
  const w = canvas.clientWidth || 800;
  const h = canvas.clientHeight || 600;
  svg.attr('viewBox', [0, 0, w, h]);

  const { nodes, links } = filteredNodesEdges();
  currentNodes = nodes;
  currentLinks = links;
  layoutOrganic(nodes, links, w, h);
  const nChunks = nodes.filter(n => n.type !== 'entity').length;
  const nEnts = nodes.length - nChunks;
  const statsEl = document.getElementById('graph-stats');
  if (statsEl) statsEl.textContent = `${nodes.length} nodes · ${links.length} edges · drag nodes · scroll zoom`;

  // Edges: thin straight connectors — brightness follows both endpoint
  // sizes (dim when both ends are small).
  linkSel = edgesG.selectAll('line').data(links, edgeKey);
  linkSel = linkSel.join(
    enter => enter.append('line')
      .attr('stroke', l => (l.type === 'about' ? EDGE_ABOUT : EDGE_SAMEDOC))
      .attr('stroke-width', l => (l.type === 'about' ? 1.1 : 0.7)),
    updateE => updateE,
    exit => exit.remove()
  );
  linkSel.attr('stroke-opacity', edgeOpacity);
  positionEdges(linkSel);

  // Hot nodes carry visible labels (reference style); the rest reveal on hover.
  const hotIds = new Set(
    [...nodes].sort((a, b) => importanceOf(b) - importanceOf(a)).slice(0, 30).map(n => n.id)
  );

  // Nodes: glossy 3D spheres — one shared gradient per type, brightness
  // from importance.
  const nodeSel = nodesG.selectAll('g.node').data(nodes, d => d.id);
  const entered = nodeSel.join(
    enter => {
      const g = enter.append('g')
        .attr('class', 'node')
        .style('cursor', 'grab')
        .call(makeNodeDrag())
        .on('click', (ev, d) => { ev.stopPropagation(); showDetails(d); });
      g.append('circle').attr('class', 'sphere');
      g.append('ellipse').attr('class', 'glare').attr('fill', 'url(#kb-glare)').attr('pointer-events', 'none');
      g.append('text').attr('class', 'node-label').attr('text-anchor', 'middle');
      g.append('title');
      return g;
    },
    updateN => updateN,
    exit => exit.remove()
  );
  entered
    .attr('transform', d => `translate(${d.x},${d.y})`);
  entered.select('circle.sphere')
    .attr('r', nodeRadius)
    .attr('fill', d => `url(#gloss-${d.type === 'entity' ? 'entity' : 'chunk'})`)
    .attr('fill-opacity', nodeOpacity)
    .attr('stroke', d => shade(nodeColor(d), -0.45))
    .attr('stroke-width', 1)
    .attr('stroke-opacity', 0.6);
  entered.select('ellipse.glare')
    .attr('cx', d => -nodeRadius(d) * 0.28)
    .attr('cy', d => -nodeRadius(d) * 0.32)
    .attr('rx', d => nodeRadius(d) * 0.48)
    .attr('ry', d => nodeRadius(d) * 0.30)
    .attr('opacity', d => 0.10 + 0.28 * importanceOf(d));
  entered.select('text.node-label')
    .attr('dy', d => -(nodeRadius(d) + 6))
    .style('opacity', d => hotIds.has(d.id) ? 0.85 : 0)
    .text(d => nodeLabel(d).slice(0, 24));
  entered.select('title')
    .text(d => `${nodeLabel(d)} — importance ${importanceOf(d).toFixed(2)}`);
}

function svgZoom(k) {
  if (!zoomBeh) return;
  d3.select('#svg').transition().call(zoomBeh.scaleBy, k);
}

let resizeTimer = null;
function onResize() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { resizeTimer = null; update(); }, 150);
}

function init() {
  document.getElementById('reload').onclick = () => {
    loadGraph().catch(err => {
      const el = document.getElementById('details');
      el.style.display = 'block';
      el.textContent = 'Load failed: ' + err;
    });
  };
  document.getElementById('refilter').onclick = () => update();
  document.getElementById('z-in').onclick = () => svgZoom(1.25);
  document.getElementById('z-out').onclick = () => svgZoom(0.8);
  document.getElementById('z-fit').onclick = () => {
    if (!zoomBeh) return;
    d3.select('#svg').transition().call(zoomBeh.transform, d3.zoomIdentity);
  };
  window.addEventListener('resize', onResize);

  loadGraph().catch(err => {
    const el = document.getElementById('details');
    el.style.display = 'block';
    el.textContent = 'Load failed: ' + err;
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
