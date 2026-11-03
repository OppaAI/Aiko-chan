/* Aiko Graph Studio — n8n-style canvas editor.
 *
 * Talks to interface/webui/studio/dag/backend/api.py:
 *   /api/playbooks   CRUD + validate + run
 *   /api/nodes       palette categories + per-node parameter forms
 *   /api/templates   starter blueprints
 *
 * Design notes
 *   - Node positions live on the node (`node.position`) and are saved with the
 *     workflow, so the canvas you arrange is the canvas you get back.
 *   - Parameter forms are generated from the backend catalog, so adding a new
 *     tool in Python makes it configurable here with zero JS changes.
 *   - Everything edits the in-memory workflow immediately; Save persists.
 */

(function () {
  "use strict";

  // ── constants ────────────────────────────────────────────────────────────
  const NODE_W = 200;
  const NODE_H = 62;
  const PORT_R = 4.5;
  const PORT_HIT_R = 11;
  const COL_GAP = 260;
  const ROW_GAP = 130;
  const STICKY_W = 210;
  const STICKY_H = 96;
  const SUB_W = 148;
  const SUB_H = 46;
  const SUB_GAP = 10;

  // GraphBoot ships with the studio shell; fall back gracefully if absent.
  const Boot = window.GraphBoot || {
    apiBase: () => {
      const path = window.location.pathname.replace(/\/$/, "");
      return path.endsWith("/studio/dag") ? `${path}/api` : "/api";
    },
    makeZoom: (opts) =>
      d3.zoom().scaleExtent(opts.scaleExtent || [0.3, 3]).on("zoom", (event) => {
        opts.target.attr("transform", event.transform);
        if (opts.onZoom) opts.onZoom(event);
      }),
  };
  const API = Boot.apiBase();

  // ── state ────────────────────────────────────────────────────────────────
  let playbooks = [];
  let catalog = { categories: [], groups: {}, nodes: {}, featured: [] };
  let templates = [];
  let current = null;          // selected workflow (mutable working copy)
  let selectedNodeId = null;
  let selectedEdgeKey = null;
  let selectedTool = null;
  let lastRun = null;          // { nodes: {id: result}, ... }
  let activeTab = "params";
  let showAdvanced = false;
  let dirty = false;
  let zoom = null;
  let svgRoot = null;
  let viewport = null;
  let dragging = null;
  let edgeFlowOn = true;
  let graphNodeById = {};
  let resizeTimer = null;

  // Every node gets a real icon: catalog glyph, or the label's initial when the
  // catalog only has the generic ◆ fallback.
  function nodeIcon(meta) {
    const icon = meta && meta.icon;
    if (icon && icon !== "◆") return icon;
    const label = (meta && (meta.label || meta.name)) || "?";
    return label.trim().charAt(0).toUpperCase() || "◆";
  }

  // ── attached sub-nodes (n8n: Chat Model / Memory / Tool under an AI Agent) ─
  // A node with `attached_to` renders docked under its parent instead of as a
  // free card; the depends_on edge is drawn as a dashed connector.
  function isAttached(node) { return !!(node && node.attached_to); }

  function parentOf(node) {
    if (!isAttached(node) || !current) return null;
    return current.nodes.find((n) => n.id === node.attached_to) || null;
  }

  function subKindOf(node) {
    const meta = nodeMeta(node.tool);
    return node.sub_kind || meta.sub_kind || "tool";
  }

  function attachedChildren(parentId) {
    if (!current) return [];
    const order = { chat_model: 0, memory: 1, tool: 2 };
    return current.nodes
      .filter((n) => n.attached_to === parentId)
      .sort((a, b) => (order[subKindOf(a)] ?? 3) - (order[subKindOf(b)] ?? 3));
  }

  function subLabel(node) {
    const meta = nodeMeta(node.tool);
    const kind = subKindOf(node);
    if (kind === "chat_model") return "Chat Model*";
    if (kind === "memory") return "Memory";
    const sub = meta.subtitle && meta.subtitle !== "Not connected" ? meta.subtitle : "";
    return sub || meta.label || node.tool;
  }

  function attachedPos(parent, index) {
    const kids = attachedChildren(parent.id);
    const totalW = kids.length * SUB_W + Math.max(0, kids.length - 1) * SUB_GAP;
    const startX = parent.position.x + (NODE_W - totalW) / 2;
    return {
      x: Math.round(startX + index * (SUB_W + SUB_GAP)),
      y: Math.round(parent.position.y + NODE_H + 30),
    };
  }

  function layoutAttached(parent, nodeById) {
    attachedChildren(parent.id).forEach((child, index) => {
      child.position = attachedPos(parent, index);
      const sel = d3.select(`[data-node="${child.id}"]`);
      if (!sel.empty()) sel.attr("transform", `translate(${child.position.x}, ${child.position.y})`);
      refreshEdgesFor(child.id, nodeById);
    });
  }

  // ── tiny helpers ─────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  function esc(value) {
    const div = document.createElement("div");
    div.textContent = value === undefined || value === null ? "" : String(value);
    return div.innerHTML;
  }

  function status(text, tone) {
    const el = $("header-status");
    if (!el) return;
    el.textContent = text || "";
    el.style.color = tone === "bad" ? "var(--pink)" : tone === "good" ? "var(--green)" : "var(--dim)";
  }

  function markDirty() {
    dirty = true;
    status("Unsaved ●");
  }

  async function api(path, options) {
    const response = await fetch(`${API}${path}`, options);
    let data = null;
    try { data = await response.json(); } catch (_) { data = null; }
    if (!response.ok) {
      const message = (data && (data.detail || data.error)) || response.statusText;
      throw new Error(message);
    }
    return data;
  }

  function categoryOf(toolName) {
    const entry = catalog.nodes[toolName];
    return (entry && entry.category) || "other";
  }

  function categoryColor(categoryId) {
    const found = (catalog.categories || []).find((c) => c.id === categoryId);
    return (found && found.color) || "#6b5f85";
  }

  function nodeMeta(toolName) {
    return catalog.nodes[toolName] || {
      name: toolName, label: toolName, category: "other", icon: "◆",
      summary: "", params: [], defaults: {},
    };
  }

  function isDecorative(node) {
    return node && node.tool === "sticky_note";
  }

  // ── workflow list ────────────────────────────────────────────────────────
  async function fetchPlaybooks(keepSelection) {
    try {
      playbooks = await api("/playbooks");
    } catch (err) {
      $("playbooks-list").innerHTML =
        `<div style="color:var(--pink);font-size:11px">Failed to load workflows: ${esc(err.message)}</div>`;
      return;
    }
    if (keepSelection && current) {
      const fresh = playbooks.find((p) => p.id === current.id);
      if (fresh && !dirty) selectPlaybook(fresh);
      else if (!fresh) { current = null; clearCanvas(); }
    }
    renderPlaybookList();
  }

  function renderPlaybookList() {
    const container = $("playbooks-list");
    container.innerHTML = "";
    if (!playbooks.length) {
      container.innerHTML = '<div class="hint">No workflows yet — start from a template.</div>';
      return;
    }
    playbooks.forEach((pb) => {
      const count = Array.isArray(pb.nodes) ? pb.nodes.length : 0;
      const button = document.createElement("button");
      button.className = "playbook-item" + (current && current.id === pb.id ? " active" : "");
      const badge = pb.readonly
        ? '<span class="pb-badge">spec</span>'
        : pb.source === "studio" ? '<span class="pb-badge">mine</span>' : "";
      button.innerHTML =
        `<div class="pb-name">${esc(pb.name || pb.id)} ${badge}</div>` +
        `<div class="pb-id">${esc(pb.id)}</div>` +
        `<div class="pb-meta"><span>${count} nodes</span></div>`;
      button.onclick = () => {
        if (dirty && !window.confirm("Discard unsaved changes?")) return;
        selectPlaybook(pb);
      };
      container.appendChild(button);
    });
  }

  function selectPlaybook(pb) {
    current = JSON.parse(JSON.stringify(pb));
    current.nodes = current.nodes || [];
    current.edges = current.edges || [];
    selectedNodeId = null;
    selectedEdgeKey = null;
    lastRun = null;
    dirty = false;
    status("");
    ensurePositions();
    renderPlaybookList();
    renderGraph();
    fitToView();
    hideDetails();
  }

  function clearCanvas() {
    d3.select("#canvas").selectAll("*").remove();
    $("graph-info").textContent = "Select a workflow";
  }

  // ── layout ───────────────────────────────────────────────────────────────
  function autoLayout(nodes, edges) {
    const positions = {};
    if (!nodes.length) return positions;
    const incoming = {};
    const outgoing = {};
    nodes.forEach((n) => { incoming[n.id] = []; outgoing[n.id] = []; });
    edges.forEach((e) => {
      if (e.type && e.type !== "depends_on") return;
      if (outgoing[e.source] && incoming[e.target]) {
        outgoing[e.source].push(e.target);
        incoming[e.target].push(e.source);
      }
    });

    const level = {};
    const seen = new Set();
    function assign(id, depth) {
      if (seen.has(id)) { level[id] = Math.max(level[id] || 0, depth); return; }
      seen.add(id);
      level[id] = depth;
      (outgoing[id] || []).forEach((child) => assign(child, depth + 1));
    }
    nodes.filter((n) => !(incoming[n.id] || []).length).forEach((n) => assign(n.id, 0));
    nodes.forEach((n) => { if (level[n.id] === undefined) level[n.id] = 0; });

    const byLevel = {};
    nodes.forEach((n) => {
      const depth = level[n.id] || 0;
      (byLevel[depth] = byLevel[depth] || []).push(n.id);
    });
    Object.entries(byLevel).forEach(([depth, ids]) => {
      const column = parseInt(depth, 10);
      ids.forEach((id, row) => {
        positions[id] = { x: 80 + column * COL_GAP, y: 120 + row * ROW_GAP };
      });
    });
    return positions;
  }

  function ensurePositions() {
    if (!current) return;
    const missing = current.nodes.filter((n) => !n.position || typeof n.position.x !== "number");
    if (!missing.length) return;
    const derived = autoLayout(current.nodes, current.edges || []);
    current.nodes.forEach((n) => {
      if (!n.position || typeof n.position.x !== "number") {
        n.position = derived[n.id] || { x: 120, y: 160 };
      }
    });
  }

  function tidyLayout() {
    if (!current) return;
    const derived = autoLayout(current.nodes.filter((n) => !isDecorative(n)), current.edges || []);
    current.nodes.forEach((n) => { if (derived[n.id]) n.position = derived[n.id]; });
    markDirty();
    renderGraph();
  }

  // ── edges ────────────────────────────────────────────────────────────────
  function rebuildEdges() {
    if (!current) return;
    const ids = new Set(current.nodes.map((n) => n.id));
    const edges = [];
    current.nodes.forEach((node) => {
      (node.depends_on || []).forEach((dep) => {
        if (ids.has(dep)) edges.push({ source: dep, target: node.id, type: "depends_on" });
      });
      if (node.loop_to && ids.has(node.loop_to)) {
        edges.push({ source: node.id, target: node.loop_to, type: "loop_to" });
      }
      if (node.fallback_to && ids.has(node.fallback_to)) {
        edges.push({ source: node.id, target: node.fallback_to, type: "fallback_to" });
      }
    });
    current.edges = edges;
  }

  function edgeKey(edge) { return `${edge.source}->${edge.target}:${edge.type || "depends_on"}`; }

  function edgePath(source, target) {
    const x1 = source.position.x + NODE_W;
    const y1 = source.position.y + NODE_H / 2;
    const x2 = target.position.x;
    const y2 = target.position.y + NODE_H / 2;
    const distance = Math.hypot(x2 - x1, y2 - y1);
    const curve = Math.min(Math.max(distance * 0.35, 30), 110);
    return `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`;
  }

  // Dashed vertical connector from a composite parent to a docked sub-node.
  function attachedConnectorPath(parent, child) {
    const x1 = parent.position.x + NODE_W / 2;
    const y1 = parent.position.y + NODE_H;
    const x2 = child.position.x + SUB_W / 2;
    const y2 = child.position.y - 4;
    const midY = (y1 + y2) / 2;
    return `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;
  }

  // Edges whose source node ran ok in the last run: green + animated flow.
  function executedEdgeKeys() {
    const keys = new Set();
    if (!lastRun || !lastRun.byId || !current) return keys;
    (current.edges || []).forEach((edge) => {
      if ((edge.type || "depends_on") !== "depends_on") return;
      const result = lastRun.byId[edge.source];
      if (result && result.ok) keys.add(edgeKey(edge));
    });
    return keys;
  }

  function itemCountFor(nodeId) {
    const result = lastRun && lastRun.byId && lastRun.byId[nodeId];
    if (!result || !result.content) return null;
    try {
      const parsed = JSON.parse(result.content);
      const items = Array.isArray(parsed) ? parsed : parsed.items;
      return Array.isArray(items) ? items.length : null;
    } catch (_) {
      return null;
    }
  }

  // Hover "+" on an edge: opens the node picker to insert between the nodes.
  let edgeInsertTimer = null;
  function edgeMidpoint(edge) {
    const line = d3.select(`[data-edge-line="${CSS.escape(edgeKey(edge))}"]`).node();
    if (!line || !line.getTotalLength) return null;
    return line.getPointAtLength(line.getTotalLength() / 2);
  }
  function viewportToArea(point) {
    const t = d3.zoomTransform(svgRoot.node());
    return { x: t.x + point.x * t.k, y: t.y + point.y * t.k };
  }
  function showEdgeInsert(edge) {
    hideEdgeInsert();
    if (!viewport) return;
    const mid = edgeMidpoint(edge);
    if (!mid) return;
    const area = viewportToArea(mid);
    const g = viewport.append("g")
      .attr("id", "edge-insert-btn")
      .attr("class", "edge-insert")
      .attr("transform", `translate(${mid.x}, ${mid.y})`)
      .on("click", (event) => {
        event.stopPropagation();
        hideEdgeInsert();
        openNodePicker(area.x, area.y, { insertEdge: edge });
      })
      .on("mouseenter", () => { if (edgeInsertTimer) { clearTimeout(edgeInsertTimer); edgeInsertTimer = null; } })
      .on("mouseleave", () => scheduleHideEdgeInsert());
    g.append("circle").attr("r", 11);
    g.append("text").attr("y", 0.5).text("+");
  }
  function scheduleHideEdgeInsert() {
    if (edgeInsertTimer) clearTimeout(edgeInsertTimer);
    edgeInsertTimer = setTimeout(() => hideEdgeInsert(), 140);
  }
  function hideEdgeInsert() {
    if (edgeInsertTimer) { clearTimeout(edgeInsertTimer); edgeInsertTimer = null; }
    if (viewport) viewport.select("#edge-insert-btn").remove();
  }

  // ── canvas render ────────────────────────────────────────────────────────
  function renderGraph() {
    const svg = d3.select("#canvas");
    const prevTransform = svgRoot ? d3.zoomTransform(svgRoot.node()) : null;
    svg.selectAll("*").remove();
    if (!current) { clearCanvas(); return; }

    rebuildEdges();
    ensurePositions();

    const area = $("canvas-area");
    const width = area.clientWidth || 1200;
    const height = area.clientHeight || 800;
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    svgRoot = svg;

    if (!current.nodes.length) {
      svg.append("text")
        .attr("x", width / 2).attr("y", height / 2)
        .attr("text-anchor", "middle").attr("fill", "var(--dim)")
        .attr("font-size", "14px")
        .text("Empty workflow — pick a node from the palette and press Add Node");
      $("graph-info").textContent = "0 nodes";
      return;
    }

    const defs = svg.append("defs");
    ["depends", "loop", "fallback", "exec"].forEach((kind) => {
      const color = kind === "loop" ? "var(--pink)"
        : kind === "fallback" ? "var(--amber)"
        : kind === "exec" ? "var(--n8n-green)" : "var(--dim)";
      defs.append("marker")
        .attr("id", `arrow-${kind}`).attr("viewBox", "0 0 10 10")
        .attr("refX", 9).attr("refY", 5)
        .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
        .append("path").attr("d", "M 0 1 L 10 5 L 0 9 Z").attr("fill", color);
    });
    const grid = defs.append("pattern")
      .attr("id", "grid").attr("width", 28).attr("height", 28)
      .attr("patternUnits", "userSpaceOnUse");
    grid.append("circle").attr("cx", 1.5).attr("cy", 1.5).attr("r", 1.4)
      .attr("fill", "var(--n8n-dot)");

    const g = svg.append("g");
    viewport = g;
    g.append("rect")
      .attr("x", -4000).attr("y", -4000).attr("width", 12000).attr("height", 12000)
      .attr("fill", "url(#grid)")
      .on("click", () => { closeNodePicker(); hideDetails(); updateSelection(); });

    zoom = Boot.makeZoom({ scaleExtent: [0.25, 2.5], target: g, onZoom: () => {} });
    svg.call(zoom);
    if (prevTransform) svg.call(zoom.transform, prevTransform);

    const edgeLayer = g.append("g").attr("class", "edges");
    const nodeLayer = g.append("g").attr("class", "nodes");
    const nodeById = {};
    current.nodes.forEach((n) => { nodeById[n.id] = n; });
    graphNodeById = nodeById;
    const tooltip = $("tooltip");

    // Docked sub-nodes are positioned from their parents. Recalculate them
    // BEFORE drawing edges so attached connectors use fresh positions on
    // first render (saved/template positions may be stale).
    current.nodes.forEach((n) => {
      if (isAttached(n)) {
        const parent = parentOf(n);
        if (parent) {
          const index = attachedChildren(parent.id).findIndex((c) => c.id === n.id);
          if (index >= 0) n.position = attachedPos(parent, index);
        }
      }
    });

    // edges
    const executed = executedEdgeKeys();
    (current.edges || []).forEach((edge) => {
      const source = nodeById[edge.source];
      const target = nodeById[edge.target];
      if (!source || !target) return;
      const type = edge.type || "depends_on";
      const key = edgeKey(edge);

      // Attached sub-nodes (Chat Model / Memory / Tool) dock under their
      // parent: a dashed connector instead of a regular edge.
      const attachedPair = (target.attached_to && target.attached_to === source.id) ||
        (source.attached_to && source.attached_to === target.id);
      if (attachedPair) {
        const parent = target.attached_to === source.id ? source : target;
        const child = parent === source ? target : source;
        edgeLayer.append("path")
          .attr("class", "edge-line edge-attached")
          .attr("d", attachedConnectorPath(parent, child))
          .attr("data-edge-line", key)
          .style("pointer-events", "none");
        return;
      }

      const cls = type === "loop_to" ? "edge-loop" : type === "fallback_to" ? "edge-fallback" : "edge-depends";
      const isExec = executed.has(key);
      const marker = type === "loop_to" ? "url(#arrow-loop)"
        : type === "fallback_to" ? "url(#arrow-fallback)"
        : isExec ? "url(#arrow-exec)" : "url(#arrow-depends)";
      const path = edgePath(source, target);

      edgeLayer.append("path")
        .attr("class", "edge-hit").attr("d", path)
        .attr("data-edge", key)
        .on("click", (event) => {
          event.stopPropagation();
          selectedEdgeKey = key;
          selectedNodeId = null;
          updateSelection();
          showEdgeDetails(edge);
        })
        .on("mouseenter", (event) => {
          tooltip.textContent = `${edge.source} → ${edge.target} · ${type}`;
          tooltip.style.left = `${event.offsetX + 12}px`;
          tooltip.style.top = `${event.offsetY - 22}px`;
          tooltip.classList.add("visible");
          showEdgeInsert(edge);
        })
        .on("mouseleave", () => { tooltip.classList.remove("visible"); scheduleHideEdgeInsert(); });

      const line = edgeLayer.append("path")
        .attr("class", "edge-line " + cls
          + (selectedEdgeKey === key ? " edge-highlight" : "")
          + (isExec ? " edge-executed" : "")
          + (edgeFlowOn && isExec ? " edge-flow" : ""))
        .attr("d", path).attr("marker-end", marker)
        .attr("data-edge-line", key)
        .style("pointer-events", "none");

      // n8n-style item-count pill on executed edges.
      const count = isExec ? itemCountFor(edge.source) : null;
      if (count !== null && count !== undefined) {
        const lineNode = line.node();
        const mid = lineNode.getPointAtLength(lineNode.getTotalLength() / 2);
        const pill = edgeLayer.append("g").attr("class", "edge-count")
          .attr("transform", `translate(${mid.x}, ${mid.y})`)
          .style("pointer-events", "none");
        const label = String(count);
        const w = Math.max(18, label.length * 7 + 10);
        pill.append("rect").attr("x", -w / 2).attr("y", -8).attr("width", w).attr("height", 16).attr("rx", 8);
        pill.append("text").attr("y", 0.5).text(label);
      }
    });

    // nodes
    current.nodes.forEach((node) => {
      // Docked sub-nodes (Chat Model / Memory / Tool) are positioned from
      // their parent and move with it; they are not draggable on their own.
      let dockedParent = null;
      if (isAttached(node)) {
        const parent = parentOf(node);
        if (parent) {
          const index = attachedChildren(parent.id).findIndex((c) => c.id === node.id);
          if (index >= 0) {
            node.position = attachedPos(parent, index);
            dockedParent = parent;
          }
        }
      }

      const group = nodeLayer.append("g")
        .attr("class", "node-group")
        .attr("data-node", node.id)
        .attr("transform", `translate(${node.position.x}, ${node.position.y})`);

      if (isDecorative(node)) {
        renderSticky(group, node);
      } else if (dockedParent) {
        renderSubNode(group, node, tooltip);
      } else {
        renderNode(group, node, tooltip);
      }

      if (dockedParent) return;

      group.call(d3.drag()
        .filter((event) => !event.target.classList.contains("port-hit"))
        .on("start", () => { dragging = { id: node.id, moved: false }; })
        .on("drag", (event) => {
          node.position.x += event.dx;
          node.position.y += event.dy;
          dragging.moved = true;
          group.attr("transform", `translate(${node.position.x}, ${node.position.y})`);
          refreshEdgesFor(node.id, nodeById);
          layoutAttached(node, nodeById);
        })
        .on("end", () => {
          if (dragging && dragging.moved) {
            node.position.x = Math.round(node.position.x);
            node.position.y = Math.round(node.position.y);
            markDirty();
          }
          dragging = null;
        }));
    });

    const runnable = current.nodes.filter((n) => !isDecorative(n)).length;
    $("graph-info").textContent =
      `${runnable} nodes · ${(current.edges || []).length} connections`;
    paintRunDots();
  }

  // n8n-style node card: dark rounded card, white glyph in a rounded icon
  // square, bold white title, smaller grey subtitle, thin category accent on
  // top, orange lightning badge for triggers.
  function renderNode(group, node, tooltip) {
    const meta = nodeMeta(node.tool);
    const color = categoryColor(meta.category);
    const disabled = !!node.disabled;
    const isTrigger = meta.category === "trigger";

    group.append("rect")
      .attr("x", 2).attr("y", 3).attr("width", NODE_W).attr("height", NODE_H)
      .attr("rx", 9).attr("fill", "rgba(0,0,0,0.3)");

    group.append("rect")
      .attr("class", "node-card" + (selectedNodeId === node.id ? " selected" : "") + (disabled ? " disabled" : ""))
      .attr("width", NODE_W).attr("height", NODE_H).attr("rx", 9)
      .on("click", (event) => {
        event.stopPropagation();
        selectedNodeId = node.id;
        selectedEdgeKey = null;
        updateSelection();
        showNodeDetails(node);
      })
      .on("mouseenter", (event) => {
        tooltip.innerHTML = `<b>${esc(meta.label)}</b><br>${esc(meta.summary || node.tool)}`;
        tooltip.style.left = `${event.offsetX + 14}px`;
        tooltip.style.top = `${event.offsetY - 10}px`;
        tooltip.classList.add("visible");
      })
      .on("mouseleave", () => tooltip.classList.remove("visible"));

    // thin category accent along the top edge
    group.append("rect")
      .attr("class", "node-top-accent")
      .attr("x", 8).attr("y", 0).attr("width", NODE_W - 16).attr("height", 3)
      .attr("rx", 1.5).attr("fill", color).attr("opacity", disabled ? 0.4 : 0.9);

    // icon square with white glyph
    group.append("rect")
      .attr("class", "node-icon-square")
      .attr("x", 12).attr("y", NODE_H / 2 - 15)
      .attr("width", 30).attr("height", 30).attr("rx", 8)
      .attr("fill", color).attr("fill-opacity", disabled ? 0.35 : 0.92);
    group.append("text")
      .attr("class", "node-icon-glyph")
      .attr("x", 27).attr("y", NODE_H / 2 + 0.5)
      .attr("opacity", disabled ? 0.5 : 1)
      .text(nodeIcon(meta));

    // bold title + grey subtitle
    const title = node.label || meta.label || node.tool;
    group.append("text")
      .attr("class", "node-title")
      .attr("x", 50).attr("y", NODE_H / 2 - 7)
      .text(title.length > 19 ? `${title.slice(0, 18)}…` : title);
    const subtitle = meta.subtitle || categoryLabel(meta.category) || node.id;
    group.append("text")
      .attr("class", "node-subtitle")
      .attr("x", 50).attr("y", NODE_H / 2 + 12)
      .text(subtitle.length > 24 ? `${subtitle.slice(0, 23)}…` : subtitle);

    // trigger badge: orange lightning, top-left overlapping the card
    if (isTrigger) {
      const badge = group.append("g").attr("class", "trigger-badge")
        .attr("transform", "translate(2, 2)");
      badge.append("circle").attr("r", 11).attr("fill", "var(--n8n-orange)");
      badge.append("text")
        .attr("y", 0.5).attr("text-anchor", "middle").attr("dominant-baseline", "central")
        .attr("font-size", "11px").attr("fill", "#fff").text("⚡");
    }

    // small status badges, bottom-right
    const badges = [];
    if (node.loop_to) badges.push({ text: `LOOP ×${node.max_visits || 1}`, color: "var(--pink)" });
    if (node.pinned_data) badges.push({ text: "PINNED", color: "var(--amber)" });
    if (node.needs_approval) badges.push({ text: "🔒", color: "var(--amber)" });
    if (disabled) badges.push({ text: "OFF", color: "var(--dim)" });
    let badgeX = NODE_W - 10;
    badges.slice(0, 3).forEach((badge) => {
      group.append("text")
        .attr("class", "node-badge-text")
        .attr("x", badgeX).attr("y", NODE_H - 9)
        .attr("text-anchor", "end")
        .attr("fill", badge.color)
        .text(badge.text);
      badgeX -= badge.text.length * 5 + 10;
    });

    // ports
    group.append("circle").attr("class", "port-circle port-input")
      .attr("cx", 0).attr("cy", NODE_H / 2).attr("r", PORT_R);
    group.append("circle")
      .attr("class", "port-hit")
      .attr("cx", 0).attr("cy", NODE_H / 2).attr("r", PORT_HIT_R)
      .attr("fill", "transparent").attr("data-target-id", node.id)
      .style("cursor", "crosshair")
      .call(inputPortDrag(node));

    group.append("circle").attr("class", "port-circle port-output")
      .attr("cx", NODE_W).attr("cy", NODE_H / 2).attr("r", PORT_R);
    group.append("circle")
      .attr("class", "port-hit")
      .attr("cx", NODE_W).attr("cy", NODE_H / 2).attr("r", PORT_HIT_R)
      .attr("fill", "transparent").attr("data-source-id", node.id)
      .style("cursor", "crosshair")
      .call(outputPortDrag(node));
  }

  // Docked sub-node card (Chat Model* / Memory / Tool): smaller, with a
  // diamond port on top. Click selects it and opens its parameters.
  function renderSubNode(group, node, tooltip) {
    const meta = nodeMeta(node.tool);
    const disabled = !!node.disabled;
    const label = subLabel(node);

    group.append("rect")
      .attr("x", 1).attr("y", 2).attr("width", SUB_W).attr("height", SUB_H)
      .attr("rx", 8).attr("fill", "rgba(0,0,0,0.3)");

    group.append("rect")
      .attr("class", "node-card sub-card" + (selectedNodeId === node.id ? " selected" : "") + (disabled ? " disabled" : ""))
      .attr("width", SUB_W).attr("height", SUB_H).attr("rx", 8)
      .on("click", (event) => {
        event.stopPropagation();
        selectedNodeId = node.id;
        selectedEdgeKey = null;
        updateSelection();
        showNodeDetails(node);
      })
      .on("mouseenter", (event) => {
        tooltip.innerHTML = `<b>${esc(label)}</b><br>${esc(meta.label || "")}<br>${esc(meta.summary || node.tool)}`;
        tooltip.style.left = `${event.offsetX + 14}px`;
        tooltip.style.top = `${event.offsetY - 10}px`;
        tooltip.classList.add("visible");
      })
      .on("mouseleave", () => tooltip.classList.remove("visible"));

    // diamond port, top center
    const cx = SUB_W / 2;
    group.append("path")
      .attr("class", "sub-diamond")
      .attr("d", `M ${cx} -5 L ${cx + 5} 0 L ${cx} 5 L ${cx - 5} 0 Z`);

    group.append("text")
      .attr("class", "sub-label")
      .attr("x", 12).attr("y", SUB_H / 2 + 0.5)
      .attr("dominant-baseline", "central")
      .text(label.length > 20 ? `${label.slice(0, 19)}…` : label);
  }

  function categoryLabel(categoryId) {
    const found = (catalog.categories || []).find((c) => c.id === categoryId);
    return found ? found.label : "";
  }

  // ── port dragging ────────────────────────────────────────────────────────
  // Output port → another node's input connects; dropping on empty canvas
  // opens the node picker to create-and-connect in one gesture (n8n style).
  function areaPointFromEvent(event) {
    const rect = $("canvas-area").getBoundingClientRect();
    const srcEvent = event.sourceEvent || event;
    return {
      x: (srcEvent.clientX || 0) - rect.left,
      y: (srcEvent.clientY || 0) - rect.top,
    };
  }

  function areaToCanvas(point) {
    const t = d3.zoomTransform(svgRoot.node());
    return { x: (point.x - t.x) / t.k, y: (point.y - t.y) / t.k };
  }

  function drawTempEdge(fromX, fromY, toX, toY) {
    let temp = viewport.select("#temp-edge");
    if (temp.empty()) {
      temp = viewport.append("path").attr("id", "temp-edge")
        .attr("class", "edge-depends").style("pointer-events", "none");
    }
    const curve = Math.min(Math.max(Math.hypot(toX - fromX, toY - fromY) * 0.35, 30), 110);
    temp.attr("d", `M ${fromX} ${fromY} C ${fromX + curve} ${fromY}, ${toX - curve} ${toY}, ${toX} ${toY}`);
  }

  function outputPortDrag(node) {
    return d3.drag()
      .on("start", () => { dragging = { connectFrom: node.id }; })
      .on("drag", (event) => {
        if (!dragging || !dragging.connectFrom) return;
        const point = d3.pointer(event, viewport.node());
        drawTempEdge(node.position.x + NODE_W, node.position.y + NODE_H / 2, point[0], point[1]);
      })
      .on("end", (event) => {
        viewport.select("#temp-edge").remove();
        if (!dragging || !dragging.connectFrom) return;
        const point = d3.pointer(event, viewport.node());
        const target = current.nodes.find((candidate) => {
          if (candidate.id === node.id || isDecorative(candidate) || isAttached(candidate)) return false;
          const cx = candidate.position.x;
          const cy = candidate.position.y + NODE_H / 2;
          return Math.hypot(point[0] - cx, point[1] - cy) < 34;
        });
        const areaPt = areaPointFromEvent(event);
        if (target) connectNodes(node.id, target.id);
        else openNodePicker(areaPt.x, areaPt.y, { connectFrom: node.id });
        dragging = null;
      });
  }

  function inputPortDrag(node) {
    return d3.drag()
      .on("start", () => { dragging = { connectTo: node.id }; })
      .on("drag", (event) => {
        if (!dragging || !dragging.connectTo) return;
        const point = d3.pointer(event, viewport.node());
        drawTempEdge(point[0], point[1], node.position.x, node.position.y + NODE_H / 2);
      })
      .on("end", (event) => {
        viewport.select("#temp-edge").remove();
        if (!dragging || !dragging.connectTo) return;
        const point = d3.pointer(event, viewport.node());
        const source = current.nodes.find((candidate) => {
          if (candidate.id === node.id || isDecorative(candidate) || isAttached(candidate)) return false;
          const cx = candidate.position.x + NODE_W;
          const cy = candidate.position.y + NODE_H / 2;
          return Math.hypot(point[0] - cx, point[1] - cy) < 34;
        });
        const areaPt = areaPointFromEvent(event);
        if (source) connectNodes(source.id, node.id);
        else openNodePicker(areaPt.x, areaPt.y, { connectTo: node.id });
        dragging = null;
      });
  }

  // ── searchable canvas node picker ────────────────────────────────────────
  // Opened by the canvas "+" button, by dropping a port on empty canvas, or
  // by the "+" on a hovered edge (insert mode).
  let pickerState = null;

  function openNodePicker(areaX, areaY, opts) {
    closeNodePicker();
    if (!current) return;
    pickerState = Object.assign({ canvasPos: null }, opts || {});
    if (!pickerState.canvasPos) {
      pickerState.canvasPos = areaToCanvas({ x: areaX, y: areaY });
    }
    const area = $("canvas-area");
    const picker = document.createElement("div");
    picker.className = "node-picker";
    picker.id = "node-picker";
    picker.style.left = `${Math.max(8, Math.min(areaX, area.clientWidth - 300))}px`;
    picker.style.top = `${Math.max(8, Math.min(areaY, area.clientHeight - 390))}px`;
    const hint = pickerState.insertEdge ? "Insert node on connection"
      : pickerState.connectFrom ? "New node (connects from " + pickerState.connectFrom + ")"
      : pickerState.connectTo ? "New node (connects into " + pickerState.connectTo + ")"
      : "Add node";
    picker.innerHTML =
      `<div style="padding:10px 10px 0;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--n8n-sub)">${esc(hint)}</div>` +
      `<input id="node-picker-search" placeholder="Search nodes…" autocomplete="off">` +
      `<div class="node-picker-list" id="node-picker-list"></div>`;
    area.appendChild(picker);

    const renderList = (query) => {
      const list = $("node-picker-list");
      list.innerHTML = "";
      const q = (query || "").toLowerCase();
      (catalog.categories || []).forEach((category) => {
        const entries = (catalog.groups[category.id] || []).filter((entry) =>
          !q || entry.name.toLowerCase().includes(q) ||
          (entry.label || "").toLowerCase().includes(q) ||
          (entry.summary || "").toLowerCase().includes(q));
        if (!entries.length) return;
        const head = document.createElement("div");
        head.className = "cat-head";
        head.style.color = category.color;
        head.innerHTML = `<span class="cat-dot" style="background:${category.color}"></span>${esc(category.label)}`;
        list.appendChild(head);
        entries.slice(0, q ? 30 : 20).forEach((entry) => {
          const button = document.createElement("button");
          button.className = "picker-item";
          button.innerHTML =
            `<span class="chip-icon" style="background:${categoryColor(entry.category)}">${esc(nodeIcon(entry))}</span>` +
            `<span class="chip-text"><span class="chip-name">${esc(entry.label)}</span><br>` +
            `<span class="chip-desc">${esc(entry.summary || entry.name)}</span></span>`;
          button.onclick = () => createNodeFromPicker(entry);
          list.appendChild(button);
        });
      });
    };
    renderList("");
    const search = $("node-picker-search");
    search.addEventListener("input", (event) => renderList(event.target.value));
    setTimeout(() => search.focus(), 30);
    picker.addEventListener("mousedown", (event) => event.stopPropagation());
  }

  function closeNodePicker() {
    const picker = $("node-picker");
    if (picker) picker.remove();
    pickerState = null;
  }

  function uniqueNodeId(toolName) {
    const base = toolName.replace(/[^A-Za-z0-9]+/g, "_").toLowerCase().slice(0, 24) || "node";
    const taken = new Set(current.nodes.map((n) => n.id));
    let id = base;
    let suffix = 1;
    while (taken.has(id)) { suffix += 1; id = `${base}_${suffix}`; }
    return id;
  }

  function createNodeFromPicker(entry) {
    const opts = pickerState || {};
    const id = uniqueNodeId(entry.name);
    const meta = nodeMeta(entry.name);
    const node = {
      id,
      tool: entry.name,
      args: JSON.parse(JSON.stringify(meta.defaults || {})),
      depends_on: [],
      position: {
        x: Math.round((opts.canvasPos && opts.canvasPos.x) || 120) - NODE_W / 2,
        y: Math.round((opts.canvasPos && opts.canvasPos.y) || 160) - NODE_H / 2,
      },
    };
    if (opts.connectFrom) {
      node.depends_on = [opts.connectFrom];
    } else if (opts.insertEdge && (opts.insertEdge.type || "depends_on") === "depends_on") {
      const edge = opts.insertEdge;
      node.depends_on = [edge.source];
      const target = current.nodes.find((n) => n.id === edge.target);
      if (target) {
        target.depends_on = (target.depends_on || []).map((dep) => (dep === edge.source ? id : dep));
        if (!target.depends_on.includes(id)) target.depends_on.push(id);
      }
    }
    current.nodes.push(node);
    if (opts.connectTo) {
      const target = current.nodes.find((n) => n.id === opts.connectTo);
      if (target) {
        target.depends_on = target.depends_on || [];
        if (!target.depends_on.includes(id)) target.depends_on.push(id);
      }
    }
    closeNodePicker();
    selectedNodeId = id;
    selectedEdgeKey = null;
    activeTab = "params";
    markDirty();
    renderGraph();
    showNodeDetails(node);
  }

  function renderSticky(group, node) {
    group.append("rect")
      .attr("class", "sticky-card")
      .attr("width", STICKY_W).attr("height", STICKY_H).attr("rx", 6)
      .on("click", (event) => {
        event.stopPropagation();
        selectedNodeId = node.id;
        selectedEdgeKey = null;
        updateSelection();
        showNodeDetails(node);
      });
    const text = String((node.args && node.args.content) || "Note");
    const lines = [];
    text.split("\n").forEach((raw) => {
      let line = raw;
      while (line.length > 30) { lines.push(line.slice(0, 30)); line = line.slice(30); }
      lines.push(line);
    });
    lines.slice(0, 6).forEach((line, index) => {
      group.append("text").attr("class", "sticky-text")
        .attr("x", 12).attr("y", 22 + index * 14)
        .text(line);
    });
  }

  function refreshEdgesFor(nodeId, nodeById) {
    (current.edges || []).forEach((edge) => {
      if (edge.source !== nodeId && edge.target !== nodeId) return;
      const source = nodeById[edge.source];
      const target = nodeById[edge.target];
      if (!source || !target) return;
      const key = edgeKey(edge);
      // Attached sub-node pairs use the dashed vertical connector, not the
      // regular horizontal edge path (otherwise dragging a composite parent
      // redraws its docked links as normal curves).
      const attachedPair = (target.attached_to && target.attached_to === source.id) ||
        (source.attached_to && source.attached_to === target.id);
      let path;
      if (attachedPair) {
        const parent = target.attached_to === source.id ? source : target;
        const child = parent === source ? target : source;
        path = attachedConnectorPath(parent, child);
      } else {
        path = edgePath(source, target);
      }
      d3.selectAll(`[data-edge="${key}"]`).attr("d", path);
      d3.selectAll(`[data-edge-line="${key}"]`).attr("d", path);
    });
  }

  // ── incremental updates (no full rebuild) ──────────────────────────────

  // Selection changes only toggle classes on the existing SVG.
  function updateSelection() {
    d3.selectAll(".node-group").each(function () {
      const g = d3.select(this);
      const selected = g.attr("data-node") === selectedNodeId;
      g.select(".node-card").classed("selected", selected);
      g.select(".sticky-card").classed("selected", selected);
    });
    d3.selectAll(".edge-line").each(function () {
      d3.select(this).classed("edge-highlight",
        d3.select(this).attr("data-edge-line") === selectedEdgeKey);
    });
  }

  // n8n-style run badges: green check / red failure overlay at the top-right.
  function paintRunDots() {
    d3.selectAll(".node-group").each(function () {
      const g = d3.select(this);
      const id = g.attr("data-node");
      const node = graphNodeById[id];
      g.selectAll(".run-badge-ok, .run-badge-fail").remove();
      const result = lastRun && lastRun.byId && lastRun.byId[id];
      if (!result) return;
      const bx = node && isAttached(node) ? SUB_W - 2 : NODE_W - 2;
      const badge = g.append("g")
        .attr("class", result.ok ? "run-badge-ok" : "run-badge-fail")
        .attr("transform", `translate(${bx}, -8)`);
      badge.append("circle").attr("r", 10);
      badge.append("text").attr("y", 0.5).text(result.ok ? "✓" : "✕");
    });
  }

  // While a run is in flight, runnable node cards pulse (see .is-running CSS).
  function setRunning(on) {
    d3.selectAll(".node-group").each(function () {
      const g = d3.select(this);
      const node = graphNodeById[g.attr("data-node")];
      if (!node || isDecorative(node)) return;
      g.classed("is-running", on);
    });
  }

  function updateViewBox() {
    if (!svgRoot) return;
    const area = $("canvas-area");
    const width = area.clientWidth || 1200;
    const height = area.clientHeight || 800;
    svgRoot.attr("viewBox", `0 0 ${width} ${height}`);
  }

  function setEdgeFlow(on) {
    edgeFlowOn = on;
    const btn = $("flow-btn");
    if (btn) btn.classList.toggle("btn-active", on);
    // Flow animates only executed edges; re-render to apply/remove it.
    if (current) renderGraph();
  }

  function connectNodes(sourceId, targetId) {
    const target = current.nodes.find((n) => n.id === targetId);
    if (!target) return;
    target.depends_on = target.depends_on || [];
    if (target.depends_on.includes(sourceId)) return;
    target.depends_on.push(sourceId);
    markDirty();
    renderGraph();
  }

  function fitToView() {
    if (!current || !current.nodes.length || !zoom || !svgRoot) return;
    const xs = current.nodes.map((n) => n.position.x);
    const ys = current.nodes.map((n) => n.position.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs) + NODE_W;
    const maxY = Math.max(...ys) + NODE_H;
    const area = $("canvas-area");
    const width = area.clientWidth || 1200;
    const height = area.clientHeight || 800;
    const scale = Math.min(width / (maxX - minX + 180), height / (maxY - minY + 180), 1.1);
    const tx = (width - (maxX - minX) * scale) / 2 - minX * scale;
    const ty = (height - (maxY - minY) * scale) / 2 - minY * scale;
    svgRoot.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  // ── details panel ────────────────────────────────────────────────────────
  function hideDetails() {
    $("details-panel").hidden = true;
    selectedNodeId = null;
    selectedEdgeKey = null;
  }

  function showNodeDetails(node) {
    const panel = $("details-panel");
    const meta = nodeMeta(node.tool);
    panel.hidden = false;
    const color = categoryColor(meta.category);
    $("details-title").innerHTML =
      `<span class="node-head"><span class="nh-icon" style="background:${color}">${esc(nodeIcon(meta))}</span>` +
      `<span><span class="nh-title">${esc(node.label || meta.label || node.tool)}</span><br>` +
      `<span class="nh-sub">${esc(meta.subtitle || categoryLabel(meta.category) || "")}</span> ` +
      `<span class="nh-tool">${esc(node.id)}</span></span></span>`;
    document.querySelectorAll("#details-tabs .tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === activeTab);
    });
    const body = $("details-content");
    if (activeTab === "params") body.innerHTML = paramsForm(node, meta);
    else if (activeTab === "settings") body.innerHTML = settingsForm(node);
    else body.innerHTML = outputView(node);
    wireDetailInputs(node, meta);
  }

  function fieldId(name) { return `field_${name.replace(/[^A-Za-z0-9_]/g, "_")}`; }

  function renderParam(param, value) {
    const id = fieldId(param.name);
    const label =
      `<div class="detail-label"><span>${esc(param.label || param.name)}` +
      `${param.required ? ' <span class="req">*</span>' : ""}</span>` +
      `<span style="opacity:.55">${esc(param.name)}</span></div>`;
    const help = param.help ? `<div class="detail-help">${esc(param.help)}</div>` : "";
    let control = "";

    if (param.type === "boolean") {
      control =
        `<div class="detail-check"><input type="checkbox" id="${id}" data-param="${esc(param.name)}" ` +
        `data-type="boolean" ${value ? "checked" : ""}><span>${esc(param.label || param.name)}</span></div>`;
      return `<div class="detail-row">${control}${help}</div>`;
    }
    if (param.type === "select") {
      const options = (param.options || [])
        .map((opt) => `<option value="${esc(opt)}"${String(value) === String(opt) ? " selected" : ""}>${esc(opt)}</option>`)
        .join("");
      control = `<select class="detail-input" id="${id}" data-param="${esc(param.name)}" data-type="select">${options}</select>`;
    } else if (param.type === "number") {
      control = `<input class="detail-input" type="number" id="${id}" data-param="${esc(param.name)}" data-type="number" value="${esc(value)}">`;
    } else if (param.type === "json" || param.type === "code" || param.type === "textarea") {
      const cls = param.type === "code" ? "detail-textarea code" : "detail-textarea";
      const text = typeof value === "object" && value !== null ? JSON.stringify(value, null, 2) : value;
      control = `<textarea class="${cls}" id="${id}" data-param="${esc(param.name)}" data-type="${param.type}">${esc(text)}</textarea>`;
    } else {
      control = `<input class="detail-input" id="${id}" data-param="${esc(param.name)}" data-type="string" value="${esc(value)}">`;
    }
    return `<div class="detail-row">${label}${control}${help}</div>`;
  }

  function paramsForm(node, meta) {
    const args = node.args || {};
    const params = meta.params || [];
    const basic = params.filter((p) => !p.advanced);
    const advanced = params.filter((p) => p.advanced);

    let html = meta.summary ? `<div class="detail-help" style="margin-bottom:10px">${esc(meta.summary)}</div>` : "";
    if (meta.outputs) {
      html += `<div class="detail-row"><div class="detail-label"><span>Gate values</span></div>` +
        `<div class="detail-value">${meta.outputs.map((o) => `<code>${esc(o)}</code>`).join(" ")}</div>` +
        `<div class="detail-help">Gate a downstream node with run_if {"node":"${esc(node.id)}","equals":"…"}.</div></div>`;
    }
    if (!params.length) {
      const raw = JSON.stringify(args, null, 2);
      html += `<div class="detail-row"><div class="detail-label"><span>Args (JSON)</span></div>` +
        `<textarea class="detail-textarea" id="raw-args" data-param="__raw__" data-type="rawargs">${esc(raw)}</textarea>` +
        `<div class="detail-help">No form is registered for this tool — edit the arguments directly.</div></div>`;
      return html;
    }

    basic.forEach((param) => {
      const value = args[param.name] !== undefined ? args[param.name] : param.default;
      html += renderParam(param, value === undefined || value === null ? "" : value);
    });
    if (advanced.length) {
      html += `<button class="adv-toggle" id="adv-toggle">${showAdvanced ? "▾" : "▸"} Advanced (${advanced.length})</button>`;
      if (showAdvanced) {
        advanced.forEach((param) => {
          const value = args[param.name] !== undefined ? args[param.name] : param.default;
          html += renderParam(param, value === undefined || value === null ? "" : value);
        });
      }
    }
    if (meta.loop_hint) {
      html += `<div class="panel-actions"><button class="btn" id="apply-loop">Make this a loop node</button></div>`;
    }
    return html;
  }

  function settingsForm(node) {
    const jsonField = (key, label, help) => {
      const value = node[key] ? JSON.stringify(node[key], null, 2) : "";
      return `<div class="detail-row"><div class="detail-label"><span>${esc(label)}</span></div>` +
        `<textarea class="detail-textarea" data-setting="${key}" data-type="json">${esc(value)}</textarea>` +
        (help ? `<div class="detail-help">${esc(help)}</div>` : "") + `</div>`;
    };
    const textField = (key, label, help, type) =>
      `<div class="detail-row"><div class="detail-label"><span>${esc(label)}</span></div>` +
      `<input class="detail-input" data-setting="${key}" data-type="${type || "string"}" ` +
      `value="${esc(node[key] === undefined || node[key] === null ? "" : node[key])}">` +
      (help ? `<div class="detail-help">${esc(help)}</div>` : "") + `</div>`;
    const checkField = (key, label) =>
      `<div class="detail-row"><div class="detail-check"><input type="checkbox" data-setting="${key}" ` +
      `data-type="boolean" ${node[key] ? "checked" : ""}><span>${esc(label)}</span></div></div>`;

    return (
      textField("id", "Node id", "Used by depends_on, run_if and $result references.") +
      textField("tool", "Tool", "Swap the underlying tool — parameters re-render on apply.") +
      textField("label", "Display label", "Optional. Overrides the catalog label on the canvas.") +
      checkField("disabled", "Disabled (skipped at run time, dependants rewired)") +
      jsonField("run_if", "Run if", 'e.g. {"node":"check","equals":"true"}') +
      jsonField("when", "When (state gate)", 'e.g. {"state":"kp_index","gte":5}') +
      textField("loop_to", "Loop to", "Node id to jump back to when the loop condition holds.") +
      jsonField("loop_condition", "Loop condition", 'e.g. {"not":{"contains":"\\"done\\": true"}}') +
      textField("max_visits", "Max visits", "Hard cap on loop passes.", "number") +
      textField("fallback_to", "Fallback to", "Node to run when this one fails outright.") +
      textField("timeout_seconds", "Timeout (s)", "", "number") +
      textField("max_retries", "Max retries", "", "number") +
      textField("retry_backoff_seconds", "Retry backoff (s)", "", "number") +
      checkField("interrupt", "Interrupt (pause the run here for input)") +
      checkField("needs_approval", "Require approval before running") +
      `<div class="detail-row"><div class="detail-label"><span>Pinned data</span></div>` +
      `<textarea class="detail-textarea" data-setting="pinned_data" data-type="raw">${esc(node.pinned_data || "")}</textarea>` +
      `<div class="detail-help">JSON items to use instead of running this node — iterate downstream without re-hitting an API.</div></div>` +
      `<div class="detail-row"><div class="detail-label"><span>Notes</span></div>` +
      `<textarea class="detail-textarea" data-setting="notes" data-type="raw">${esc(node.notes || "")}</textarea></div>` +
      `<div class="panel-actions"><button class="btn btn-primary" id="apply-settings">Apply</button>` +
      `<button class="btn" id="run-from-here">▶ Run from here</button></div>`
    );
  }

  function outputView(node) {
    const result = lastRun && lastRun.byId && lastRun.byId[node.id];
    if (!result) {
      return `<div class="detail-help">No run output yet. Press <b>▶ Run</b>, or use <b>Run from here</b> in Settings.</div>`;
    }
    const verdict = result.ok
      ? '<span class="run-ok">✓ ok</span>'
      : `<span class="run-fail">✗ ${esc(result.error || "failed")}</span>`;
    let pretty = result.content || "";
    try { pretty = JSON.stringify(JSON.parse(pretty), null, 2); } catch (_) { /* plain text */ }
    return `<div class="detail-row"><div class="detail-label"><span>Status</span></div>` +
      `<div class="detail-value">${verdict}</div></div>` +
      `<div class="detail-row"><div class="detail-label"><span>Output</span></div>` +
      `<pre class="out">${esc(pretty)}</pre></div>` +
      `<div class="panel-actions"><button class="btn" id="pin-output">📌 Pin this output</button></div>`;
  }

  function parseValue(element) {
    const type = element.dataset.type;
    if (type === "boolean") return element.checked;
    const raw = element.value;
    if (type === "number") {
      if (raw === "") return undefined;
      const parsed = Number(raw);
      return Number.isNaN(parsed) ? undefined : parsed;
    }
    if (type === "json") {
      if (!raw.trim()) return undefined;
      try {
        const parsed = JSON.parse(raw);
        element.classList.remove("invalid");
        return parsed;
      } catch (_) {
        element.classList.add("invalid");
        return raw; // keep the text so the user can fix it
      }
    }
    return raw;
  }

  function wireDetailInputs(node, meta) {
    const panel = $("details-content");

    panel.querySelectorAll("[data-param]").forEach((element) => {
      const handler = () => {
        const name = element.dataset.param;
        if (name === "__raw__") {
          try {
            node.args = JSON.parse(element.value || "{}");
            element.classList.remove("invalid");
            markDirty();
          } catch (_) { element.classList.add("invalid"); }
          return;
        }
        const type = element.dataset.type;
        node.args = node.args || {};
        if (type === "json") {
          // JSON params are stored as strings: the Python side json.loads them.
          const raw = element.value;
          try { JSON.parse(raw || "{}"); element.classList.remove("invalid"); }
          catch (_) { element.classList.add("invalid"); }
          node.args[name] = raw;
        } else {
          const value = parseValue(element);
          if (value === undefined || value === "") delete node.args[name];
          else node.args[name] = value;
        }
        markDirty();
      };
      element.addEventListener("input", handler);
      element.addEventListener("change", handler);
    });

    const advToggle = $("adv-toggle");
    if (advToggle) advToggle.onclick = () => { showAdvanced = !showAdvanced; showNodeDetails(node); };

    const applyLoop = $("apply-loop");
    if (applyLoop) applyLoop.onclick = () => {
      const hint = meta.loop_hint || {};
      node.loop_to = hint.loop_to === "self" ? node.id : hint.loop_to;
      node.loop_condition = hint.loop_condition;
      node.max_visits = hint.max_visits || 25;
      markDirty();
      renderGraph();
      showNodeDetails(node);
    };

    const applySettings = $("apply-settings");
    if (applySettings) applySettings.onclick = () => applyNodeSettings(node);

    const runFromHere = $("run-from-here");
    if (runFromHere) runFromHere.onclick = () => runWorkflow(node.id);

    const pinOutput = $("pin-output");
    if (pinOutput) pinOutput.onclick = () => {
      const result = lastRun && lastRun.byId && lastRun.byId[node.id];
      if (!result) return;
      node.pinned_data = result.content || "";
      markDirty();
      renderGraph();
      activeTab = "settings";
      showNodeDetails(node);
    };
  }

  function applyNodeSettings(node) {
    const panel = $("details-content");
    const oldId = node.id;
    let newId = oldId;
    let failed = null;

    panel.querySelectorAll("[data-setting]").forEach((element) => {
      const key = element.dataset.setting;
      const type = element.dataset.type;
      let value;
      if (type === "boolean") value = element.checked;
      else if (type === "number") value = element.value === "" ? undefined : Number(element.value);
      else if (type === "json") {
        const raw = element.value.trim();
        if (!raw) value = undefined;
        else {
          try { value = JSON.parse(raw); element.classList.remove("invalid"); }
          catch (err) { element.classList.add("invalid"); failed = `${key}: ${err.message}`; return; }
        }
      } else value = element.value;

      if (key === "id") { newId = String(value || "").trim() || oldId; return; }
      if (value === undefined || value === "" || value === false) delete node[key];
      else node[key] = value;
    });

    if (failed) { window.alert(`Invalid JSON — ${failed}`); return; }

    if (newId !== oldId) {
      if (current.nodes.some((n) => n.id === newId)) { window.alert("That node id already exists"); return; }
      current.nodes.forEach((other) => {
        other.depends_on = (other.depends_on || []).map((dep) => (dep === oldId ? newId : dep));
        if (other.loop_to === oldId) other.loop_to = newId;
        if (other.fallback_to === oldId) other.fallback_to = newId;
        if (other.attached_to === oldId) other.attached_to = newId;
        Object.entries(other.args || {}).forEach(([key, value]) => {
          if (typeof value === "string" && value === `$result:${oldId}`) other.args[key] = `$result:${newId}`;
        });
        if (other.run_if && other.run_if.node === oldId) other.run_if.node = newId;
      });
      node.id = newId;
      selectedNodeId = newId;
    }
    markDirty();
    renderGraph();
    showNodeDetails(node);
  }

  function showEdgeDetails(edge) {
    const panel = $("details-panel");
    panel.hidden = false;
    $("details-title").textContent = "Connection";
    document.querySelectorAll("#details-tabs .tab").forEach((tab) => tab.classList.remove("active"));
    const type = edge.type || "depends_on";
    $("details-content").innerHTML =
      `<div class="detail-row"><div class="detail-label"><span>Type</span></div>` +
      `<select class="detail-input" id="edge-type">` +
      ["depends_on", "loop_to", "fallback_to"].map((option) =>
        `<option value="${option}"${option === type ? " selected" : ""}>${option}</option>`).join("") +
      `</select></div>` +
      `<div class="detail-row"><div class="detail-label"><span>From</span></div>` +
      `<div class="detail-value"><code>${esc(edge.source)}</code></div></div>` +
      `<div class="detail-row"><div class="detail-label"><span>To</span></div>` +
      `<div class="detail-value"><code>${esc(edge.target)}</code></div></div>` +
      `<div class="panel-actions"><button class="btn btn-primary" id="edge-apply">Apply</button>` +
      `<button class="btn btn-danger" id="edge-delete">Delete</button></div>`;

    $("edge-apply").onclick = () => {
      const newType = $("edge-type").value;
      removeEdge(edge);
      const source = current.nodes.find((n) => n.id === edge.source);
      const target = current.nodes.find((n) => n.id === edge.target);
      if (newType === "depends_on" && target) {
        target.depends_on = target.depends_on || [];
        if (!target.depends_on.includes(edge.source)) target.depends_on.push(edge.source);
      } else if (newType === "loop_to" && source) {
        source.loop_to = edge.target;
        source.loop_condition = source.loop_condition || { not: { contains: '"done": true' } };
        source.max_visits = source.max_visits || 25;
      } else if (newType === "fallback_to" && source) {
        source.fallback_to = edge.target;
      }
      markDirty();
      renderGraph();
      hideDetails();
    };
    $("edge-delete").onclick = () => { removeEdge(edge); markDirty(); renderGraph(); hideDetails(); };
  }

  function removeEdge(edge) {
    const type = edge.type || "depends_on";
    const source = current.nodes.find((n) => n.id === edge.source);
    const target = current.nodes.find((n) => n.id === edge.target);
    if (type === "depends_on" && target) {
      target.depends_on = (target.depends_on || []).filter((dep) => dep !== edge.source);
    } else if (type === "loop_to" && source) {
      delete source.loop_to; delete source.loop_condition;
    } else if (type === "fallback_to" && source) {
      delete source.fallback_to;
    }
  }

  // ── palette ──────────────────────────────────────────────────────────────
  async function fetchCatalog() {
    try {
      catalog = await api("/nodes");
    } catch (err) {
      console.error("node catalog failed", err);
      catalog = { categories: [], groups: {}, nodes: {}, featured: [] };
    }
    renderPalette("");
  }

  function renderPalette(filter) {
    const container = $("tool-palette");
    if (!container) return;
    container.innerHTML = "";
    const query = (filter || "").toLowerCase();

    const addChip = (entry) => {
      const button = document.createElement("button");
      button.className = "playbook-item" + (selectedTool === entry.name ? " active" : "");
      button.title = entry.summary || entry.name;
      button.innerHTML =
        `<div class="node-chip">` +
        `<span class="chip-icon" style="color:${categoryColor(entry.category)}">${esc(nodeIcon(entry))}</span>` +
        `<span class="chip-text"><span class="chip-name">${esc(entry.label)}` +
        `${entry.needs_approval ? " 🔒" : ""}</span>` +
        `<span class="chip-desc">${esc(entry.summary || entry.name)}</span></span></div>`;
      button.onclick = () => { selectedTool = entry.name; renderPalette($("palette-search").value); };
      button.ondblclick = () => { selectedTool = entry.name; addNode(); };
      container.appendChild(button);
    };

    if (!query && (catalog.featured || []).length) {
      const head = document.createElement("div");
      head.className = "cat-head";
      head.style.color = "var(--mauve)";
      head.innerHTML = `<span class="cat-dot" style="background:var(--mauve)"></span>Common`;
      container.appendChild(head);
      catalog.featured.forEach((name) => {
        const entry = catalog.nodes[name];
        if (entry) addChip(entry);
      });
    }

    (catalog.categories || []).forEach((category) => {
      const entries = (catalog.groups[category.id] || []).filter((entry) =>
        !query ||
        entry.name.toLowerCase().includes(query) ||
        (entry.label || "").toLowerCase().includes(query) ||
        (entry.summary || "").toLowerCase().includes(query));
      if (!entries.length) return;
      const head = document.createElement("div");
      head.className = "cat-head";
      head.style.color = category.color;
      head.innerHTML = `<span class="cat-dot" style="background:${category.color}"></span>${esc(category.label)}`;
      container.appendChild(head);
      entries.slice(0, query ? 40 : 24).forEach(addChip);
    });
  }

  function addNode() {
    if (!current) { window.alert("Select or create a workflow first"); return; }
    const toolName = selectedTool || "set_fields";
    const meta = nodeMeta(toolName);
    const base = toolName.replace(/[^A-Za-z0-9]+/g, "_").toLowerCase().slice(0, 24) || "node";
    let id = base;
    let suffix = 1;
    const taken = new Set(current.nodes.map((n) => n.id));
    while (taken.has(id)) { suffix += 1; id = `${base}_${suffix}`; }

    const last = current.nodes[current.nodes.length - 1];
    const position = last && last.position
      ? { x: last.position.x + COL_GAP, y: last.position.y }
      : { x: 120, y: 160 };

    const node = {
      id,
      tool: toolName,
      args: JSON.parse(JSON.stringify(meta.defaults || {})),
      depends_on: selectedNodeId && selectedNodeId !== id ? [selectedNodeId] : [],
      position,
    };
    current.nodes.push(node);
    selectedNodeId = id;
    activeTab = "params";
    markDirty();
    renderGraph();
    showNodeDetails(node);
  }

  function deleteSelected() {
    if (!current) return;
    if (selectedNodeId) {
      if (!window.confirm(`Delete node ${selectedNodeId}?`)) return;
      // Deleting a composite parent takes its docked sub-nodes with it.
      const doomed = new Set([selectedNodeId]);
      current.nodes.forEach((n) => { if (n.attached_to === selectedNodeId) doomed.add(n.id); });
      current.nodes = current.nodes.filter((n) => !doomed.has(n.id));
      current.nodes.forEach((node) => {
        node.depends_on = (node.depends_on || []).filter((dep) => !doomed.has(dep));
        if (doomed.has(node.loop_to)) delete node.loop_to;
        if (doomed.has(node.fallback_to)) delete node.fallback_to;
      });
      selectedNodeId = null;
      markDirty();
      hideDetails();
      renderGraph();
      return;
    }
    if (selectedEdgeKey) {
      const edge = (current.edges || []).find((candidate) => edgeKey(candidate) === selectedEdgeKey);
      if (edge) { removeEdge(edge); markDirty(); renderGraph(); }
      hideDetails();
      return;
    }
    window.alert("Select a node or a connection first");
  }

  // ── persistence ──────────────────────────────────────────────────────────
  async function savePlaybook() {
    if (!current) return;
    if (current.readonly) {
      window.alert("This is a Spec-generated graph. Duplicate it first, then edit the copy.");
      return;
    }
    rebuildEdges();
    try {
      const payload = JSON.parse(JSON.stringify(current));
      delete payload.edges;
      const data = await api(`/playbooks/${encodeURIComponent(current.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      current = Object.assign(current, data.playbook);
      dirty = false;
      status((data.warnings || []).length ? `Saved · ${data.warnings.length} warning(s)` : "Saved", "good");
      if ((data.warnings || []).length) showRunPanel("Save warnings",
        data.warnings.map((w) => `<div class="run-dim">⚠ ${esc(w)}</div>`).join(""));
      await fetchPlaybooks(false);
      renderPlaybookList();
    } catch (err) {
      window.alert(`Could not save: ${err.message}`);
      status("Save failed", "bad");
    }
  }

  // ── validate / run ───────────────────────────────────────────────────────
  function showRunPanel(title, html) {
    $("run-title").textContent = title;
    $("run-content").innerHTML = html;
    $("run-panel").hidden = false;
  }

  async function validateWorkflow() {
    if (!current) return;
    if (dirty) await savePlaybook();
    try {
      const data = await api(`/playbooks/${encodeURIComponent(current.id)}/validate`, { method: "POST" });
      const head = data.ok
        ? '<div class="run-ok">✓ Valid</div>'
        : `<div class="run-fail">✗ ${esc((data.errors || []).join("; "))}</div>`;
      const warnings = (data.warnings || [])
        .map((warning) => `<div class="run-dim">⚠ ${esc(warning)}</div>`).join("");
      showRunPanel("Validation",
        `${head}${warnings}<div class="run-dim" style="margin-top:6px">` +
        `${data.runnable_nodes} runnable node(s) · entries: ${esc((data.entry_points || []).join(", ") || "—")}</div>`);
      status(data.ok ? "Valid" : "Invalid", data.ok ? "good" : "bad");
    } catch (err) {
      window.alert(`Validate failed: ${err.message}`);
    }
  }

  async function runWorkflow(startNode) {
    if (!current) return;
    if (dirty) await savePlaybook();
    const promptText = window.prompt("Dry-run prompt (substituted for $prompt):",
      current.goal || `Studio dry-run of ${current.id}`);
    if (promptText === null) return;

    showRunPanel("Run", '<div class="run-dim">Running… (bounded, 1 worker)</div>');
    status("Running…");
    setRunning(true);
    try {
      const data = await api(`/playbooks/${encodeURIComponent(current.id)}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptText, timeout_s: 90, start_node: startNode || "" }),
      });
      lastRun = { byId: {}, raw: data };
      (data.nodes || []).forEach((result) => { lastRun.byId[result.id] = result; });

      const rows = (data.nodes || []).map((result) => {
        const mark = result.ok ? '<span class="rn-mark run-ok">✓</span>' : '<span class="rn-mark run-fail">✗</span>';
        const body = esc((result.content || result.error || "").slice(0, 120)).replace(/\n/g, " ");
        return `<div class="run-node-row" data-run-node="${esc(result.id)}">${mark}` +
          `<span class="rn-id">${esc(result.id)}</span>` +
          `<span class="rn-tool">${esc(result.tool)}</span>` +
          `<span class="rn-body">${body}</span></div>`;
      }).join("");

      const failed = (data.nodes || []).filter((n) => !n.ok).length;
      showRunPanel("Run result",
        `<div class="${failed ? "run-fail" : "run-ok"}">${failed ? `✗ ${failed} node(s) failed` : "✓ All nodes ok"}` +
        ` · goal score ${data.goal_score === null || data.goal_score === undefined ? "—" : data.goal_score}</div>` +
        `<div class="run-dim" style="margin:5px 0">state keys: ${esc((data.state_keys || []).join(", ") || "—")}</div>` +
        rows +
        `<pre class="out">${esc((data.final_answer || "").slice(0, 1500))}</pre>`);

      document.querySelectorAll("[data-run-node]").forEach((row) => {
        row.onclick = () => {
          const node = current.nodes.find((n) => n.id === row.dataset.runNode);
          if (!node) return;
          selectedNodeId = node.id;
          activeTab = "output";
          updateSelection();
          showNodeDetails(node);
        };
      });
      status(failed ? "Run: failures" : "Run ok", failed ? "bad" : "good");
      renderGraph();
    } catch (err) {
      showRunPanel("Run failed", `<div class="run-fail">${esc(err.message)}</div>`);
      status("Run failed", "bad");
    } finally {
      setRunning(false);
    }
  }

  // ── templates ────────────────────────────────────────────────────────────
  async function openTemplates() {
    $("template-modal").hidden = false;
    const list = $("template-list");
    list.innerHTML = '<div class="hint">Loading…</div>';
    try {
      const data = await api("/templates");
      templates = data.templates || [];
    } catch (err) {
      list.innerHTML = `<div class="run-fail">${esc(err.message)}</div>`;
      return;
    }
    list.innerHTML = "";
    templates.forEach((template) => {
      const card = document.createElement("button");
      card.className = "template-card";
      card.innerHTML =
        `<div class="tpl-cat">${esc(template.category)}</div>` +
        `<div class="tpl-name">${esc(template.name)}</div>` +
        `<div class="tpl-desc">${esc(template.description)}</div>` +
        `<div class="tpl-meta">${template.node_count} nodes · ${esc(template.tools.slice(0, 4).join(", "))}</div>`;
      card.onclick = () => createFromTemplate(template);
      list.appendChild(card);
    });
  }

  async function createFromTemplate(template) {
    const suggested = template.id.replace(/^tpl_/, "") + "_1";
    const id = window.prompt("New workflow id:", suggested);
    if (!id) return;
    try {
      const data = await api(`/templates/${encodeURIComponent(template.id)}/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: id.trim(), name: template.name }),
      });
      $("template-modal").hidden = true;
      await fetchPlaybooks(false);
      const created = playbooks.find((p) => p.id === data.playbook.id);
      if (created) selectPlaybook(created);
    } catch (err) {
      window.alert(`Could not create workflow: ${err.message}`);
    }
  }

  // ── header wiring ────────────────────────────────────────────────────────
  function wireHeader() {
    $("back-btn").onclick = (event) => {
      event.preventDefault();
      if (window.history.length > 1) window.history.back();
      else window.location.href = "/";
    };
    $("refresh-btn").onclick = () => fetchPlaybooks(true);
    $("save-btn").onclick = savePlaybook;
    $("validate-btn").onclick = validateWorkflow;
    $("run-btn").onclick = () => runWorkflow("");
    $("add-node-btn").onclick = addNode;
    $("delete-selected-btn").onclick = deleteSelected;
    $("tidy-btn").onclick = tidyLayout;
    $("flow-btn").onclick = () => setEdgeFlow(!edgeFlowOn);
    $("template-btn").onclick = openTemplates;
    $("template-close").onclick = () => { $("template-modal").hidden = true; };
    $("details-close").onclick = hideDetails;
    $("run-close").onclick = () => { $("run-panel").hidden = true; };

    document.querySelectorAll("#details-tabs .tab").forEach((tab) => {
      tab.onclick = () => {
        activeTab = tab.dataset.tab;
        const node = current && current.nodes.find((n) => n.id === selectedNodeId);
        if (node) showNodeDetails(node);
      };
    });

    $("palette-search").addEventListener("input", (event) => renderPalette(event.target.value));

    $("new-graph-btn").onclick = async () => {
      const id = window.prompt("New workflow id (letters, numbers, _ -):", "my_workflow");
      if (!id) return;
      const name = window.prompt("Display name:", id) || id;
      try {
        const data = await api("/playbooks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: id.trim(), name, goal: name }),
        });
        await fetchPlaybooks(false);
        const created = playbooks.find((p) => p.id === data.playbook.id);
        if (created) selectPlaybook(created);
      } catch (err) {
        window.alert(`Could not create workflow: ${err.message}`);
      }
    };

    $("duplicate-graph-btn").onclick = async () => {
      if (!current) { window.alert("Select a workflow first"); return; }
      const newId = window.prompt("New id for the copy:", `${current.id}_copy`);
      if (!newId) return;
      try {
        const data = await api(`/playbooks/${encodeURIComponent(current.id)}/duplicate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_id: newId.trim() }),
        });
        await fetchPlaybooks(false);
        const created = playbooks.find((p) => p.id === data.playbook.id);
        if (created) selectPlaybook(created);
      } catch (err) {
        window.alert(`Could not duplicate: ${err.message}`);
      }
    };

    $("delete-graph-btn").onclick = async () => {
      if (!current) return;
      if (!window.confirm(`Delete workflow ${current.id}? Built-ins are protected.`)) return;
      try {
        await api(`/playbooks/${encodeURIComponent(current.id)}`, { method: "DELETE" });
        current = null;
        dirty = false;
        hideDetails();
        clearCanvas();
        await fetchPlaybooks(false);
      } catch (err) {
        window.alert(`Could not delete: ${err.message}`);
      }
    };

    $("export-btn").onclick = () => {
      if (!current) return;
      rebuildEdges();
      const payload = JSON.parse(JSON.stringify(current));
      delete payload.edges;
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${current.id}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
    };

    $("import-btn").onclick = () => $("import-file").click();
    $("import-file").addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      try {
        const parsed = JSON.parse(await file.text());
        const payload = Array.isArray(parsed) ? parsed[0] : (parsed.playbook || parsed);
        if (!payload || !payload.id) throw new Error("JSON must contain a workflow object with an id");
        const data = await api("/playbooks", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        await fetchPlaybooks(false);
        const created = playbooks.find((p) => p.id === data.playbook.id);
        if (created) selectPlaybook(created);
      } catch (err) {
        window.alert(`Import failed: ${err.message}`);
      }
      event.target.value = "";
    });

    $("zoom-in").onclick = () => svgRoot && svgRoot.transition().duration(250).call(zoom.scaleBy, 1.3);
    $("zoom-out").onclick = () => svgRoot && svgRoot.transition().duration(250).call(zoom.scaleBy, 0.75);
    $("zoom-fit").onclick = fitToView;

    $("canvas-add").onclick = (event) => {
      if (!current) { window.alert("Select or create a workflow first"); return; }
      const rect = $("canvas-area").getBoundingClientRect();
      const areaX = event.clientX - rect.left;
      const areaY = event.clientY - rect.top;
      const center = areaToCanvas({ x: rect.width / 2, y: rect.height / 2 });
      openNodePicker(areaX + 14, areaY + 14, { canvasPos: center });
    };

    document.addEventListener("click", (event) => {
      const picker = $("node-picker");
      if (picker && !picker.contains(event.target) && event.target.id !== "canvas-add") {
        closeNodePicker();
      }
    });

    document.addEventListener("keydown", (event) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if (event.key === "Escape") {
        $("template-modal").hidden = true;
        closeNodePicker();
        hideDetails();
        return;
      }
      if (typing) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault(); savePlaybook(); return;
      }
      if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); deleteSelected(); return; }
      if (event.key === "n" || event.key === "N") addNode();
    });

    window.addEventListener("beforeunload", (event) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });

    window.addEventListener("resize", () => {
      if (!current) return;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(updateViewBox, 150);
    });
  }

  // ── boot ─────────────────────────────────────────────────────────────────
  wireHeader();
  fetchCatalog().then(() => fetchPlaybooks(false));
})();
