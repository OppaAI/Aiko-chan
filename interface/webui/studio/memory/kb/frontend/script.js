/* Knowledge Graph Studio — knowledge + entity nodes only.
 *
 * Neural-net layered view: knowledge chunks form the left column, entity
 * hubs the right column, joined by thin straight connectors. Layout is
 * fully deterministic (no force simulation): fixed x per column, y spread
 * evenly by importance, so re-filtering is instant and costs no per-tick
 * DOM churn. One flat circle per node, one shared subtle glow, no
 * per-node gradient defs.
 */
const API_BASE = (window.KNOWLEDGE_API_BASE || GraphBoot.apiBase()).replace(/\/+$/, '');
const API_ROOT = API_BASE.endsWith('/api') ? API_BASE : API_BASE + '/api';

let graph = { nodes: [], edges: [], meta: {} };
let zoomBeh = null;
let viewport = null;   // zoom target group
let edgesG = null;     // lines layer
let nodesG = null;     // node layer
let labelsG = null;    // layer labels
let linkSel = null;    // current line selection (for drag updates)
let currentNodes = [];
let currentLinks = [];

const COL_CHUNK = '#4ade80';
const COL_ENTITY = '#a78bfa';
const EDGE_ABOUT = '#6ee7a8';
const EDGE_SAMEDOC = '#5b4a6e';
const LAYER_TOP = 70;
const LAYER_BOTTOM_PAD = 48;

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

/* Deterministic layered layout: fixed x per column, y spread evenly by
 * importance (most important near the top). O(n), runs once per update. */
function layoutLayered(nodes, w, h) {
  const chunks = [];
  const ents = [];
  for (const n of nodes) (n.type === 'entity' ? ents : chunks).push(n);
  chunks.sort((a, b) => importanceOf(b) - importanceOf(a));
  ents.sort((a, b) => importanceOf(b) - importanceOf(a));
  const top = LAYER_TOP;
  const bottom = Math.max(top + 60, h - LAYER_BOTTOM_PAD);
  placeColumn(chunks, w * 0.32, top, bottom);
  placeColumn(ents, w * 0.68, top, bottom);
  return { chunks: chunks.length, ents: ents.length };
}

function placeColumn(list, x, top, bottom) {
  const n = list.length;
  for (let i = 0; i < n; i++) {
    const d = list[i];
    d.x = x;
    d.y = n === 1 ? (top + bottom) / 2 : top + ((bottom - top) * i) / (n - 1);
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
  // One shared subtle glow for the whole node layer — a single offscreen
  // pass instead of per-node filters.
  GraphBoot.addGlowFilter(defs, 'netglow', 2.2);

  viewport = svg.append('g').attr('id', 'viewport');
  zoomBeh = GraphBoot.makeZoom({ scaleExtent: [0.15, 4], target: viewport });
  svg.call(zoomBeh);

  labelsG = viewport.append('g').attr('id', 'layer-labels');
  edgesG = viewport.append('g').attr('id', 'edges');
  nodesG = viewport.append('g').attr('id', 'nodes').attr('filter', 'url(#netglow)');

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
  const counts = layoutLayered(nodes, w, h);

  // Layer labels + faint column guides.
  const labelData = [
    { x: w * 0.32, text: `KNOWLEDGE · ${counts.chunks}` },
    { x: w * 0.68, text: `ENTITIES · ${counts.ents}` },
  ];
  const labels = labelsG.selectAll('text.layer-label').data(labelData, d => d.text.split(' ·')[0]);
  labels.join(
    enter => enter.append('text')
      .attr('class', 'layer-label')
      .attr('text-anchor', 'middle')
      .attr('y', 34)
      .attr('fill', '#8b7a9e')
      .attr('font-size', '11px')
      .attr('letter-spacing', '0.18em')
      .attr('font-weight', '600'),
    updateLbl => updateLbl,
    exit => exit.remove()
  )
    .attr('x', d => d.x)
    .text(d => d.text);

  const guides = labelsG.selectAll('line.col-guide').data(labelData, d => d.text.split(' ·')[0]);
  guides.join(
    enter => enter.append('line')
      .attr('class', 'col-guide')
      .attr('stroke', '#3d2f4f')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3 6')
      .attr('stroke-opacity', 0.5),
    updateG => updateG,
    exit => exit.remove()
  )
    .attr('x1', d => d.x).attr('x2', d => d.x)
    .attr('y1', LAYER_TOP - 18)
    .attr('y2', Math.max(LAYER_TOP + 42, h - LAYER_BOTTOM_PAD));

  // Edges: one thin straight line each.
  linkSel = edgesG.selectAll('line').data(links, edgeKey);
  linkSel = linkSel.join(
    enter => enter.append('line')
      .attr('stroke', l => (l.type === 'about' ? EDGE_ABOUT : EDGE_SAMEDOC))
      .attr('stroke-width', l => (l.type === 'about' ? 1.1 : 0.7))
      .attr('stroke-opacity', 0.30),
    updateE => updateE,
    exit => exit.remove()
  );
  positionEdges(linkSel);

  // Nodes: one flat circle each.
  const nodeSel = nodesG.selectAll('g.node').data(nodes, d => d.id);
  const entered = nodeSel.join(
    enter => {
      const g = enter.append('g')
        .attr('class', 'node')
        .style('cursor', 'grab')
        .call(makeNodeDrag())
        .on('click', (ev, d) => { ev.stopPropagation(); showDetails(d); });
      g.append('circle');
      g.append('title');
      return g;
    },
    updateN => updateN,
    exit => exit.remove()
  );
  entered
    .attr('transform', d => `translate(${d.x},${d.y})`);
  entered.select('circle')
    .attr('r', nodeRadius)
    .attr('fill', nodeColor)
    .attr('fill-opacity', nodeOpacity)
    .attr('stroke', nodeColor)
    .attr('stroke-width', 1)
    .attr('stroke-opacity', 0.85);
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
