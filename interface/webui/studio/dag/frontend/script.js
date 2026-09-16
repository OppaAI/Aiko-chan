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

  // ── canvas render ────────────────────────────────────────────────────────
  function renderGraph() {
    const svg = d3.select("#canvas");
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
    ["depends", "loop", "fallback"].forEach((kind) => {
      const color = kind === "loop" ? "var(--pink)" : kind === "fallback" ? "var(--amber)" : "var(--dim)";
      defs.append("marker")
        .attr("id", `arrow-${kind}`).attr("viewBox", "0 0 10 10")
        .attr("refX", 9).attr("refY", 5)
        .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
        .append("path").attr("d", "M 0 1 L 10 5 L 0 9 Z").attr("fill", color);
    });
    const grid = defs.append("pattern")
      .attr("id", "grid").attr("width", 40).attr("height", 40)
      .attr("patternUnits", "userSpaceOnUse");
    grid.append("path").attr("d", "M 40 0 L 0 0 0 40")
      .attr("fill", "none").attr("stroke", "var(--dimmer)")
      .attr("stroke-width", 0.5).attr("opacity", 0.5);

    const g = svg.append("g");
    viewport = g;
    g.append("rect")
      .attr("x", -4000).attr("y", -4000).attr("width", 12000).attr("height", 12000)
      .attr("fill", "url(#grid)")
      .on("click", () => { hideDetails(); renderGraph(); });

    zoom = Boot.makeZoom({ scaleExtent: [0.25, 2.5], target: g, onZoom: () => {} });
    svg.call(zoom);

    const edgeLayer = g.append("g").attr("class", "edges");
    const nodeLayer = g.append("g").attr("class", "nodes");
    const nodeById = {};
    current.nodes.forEach((n) => { nodeById[n.id] = n; });
    const tooltip = $("tooltip");

    // edges
    (current.edges || []).forEach((edge) => {
      const source = nodeById[edge.source];
      const target = nodeById[edge.target];
      if (!source || !target) return;
      const type = edge.type || "depends_on";
      const cls = type === "loop_to" ? "edge-loop" : type === "fallback_to" ? "edge-fallback" : "edge-depends";
      const marker = type === "loop_to" ? "url(#arrow-loop)"
        : type === "fallback_to" ? "url(#arrow-fallback)" : "url(#arrow-depends)";
      const path = edgePath(source, target);
      const key = edgeKey(edge);

      edgeLayer.append("path")
        .attr("class", "edge-hit").attr("d", path)
        .attr("data-edge", key)
        .on("click", (event) => {
          event.stopPropagation();
          selectedEdgeKey = key;
          selectedNodeId = null;
          showEdgeDetails(edge);
        })
        .on("mouseenter", (event) => {
          tooltip.textContent = `${edge.source} → ${edge.target} · ${type}`;
          tooltip.style.left = `${event.offsetX + 12}px`;
          tooltip.style.top = `${event.offsetY - 22}px`;
          tooltip.classList.add("visible");
        })
        .on("mouseleave", () => tooltip.classList.remove("visible"));

      edgeLayer.append("path")
        .attr("class", cls + (selectedEdgeKey === key ? " edge-highlight" : ""))
        .attr("d", path).attr("marker-end", marker)
        .attr("data-edge-line", key)
        .style("pointer-events", "none");
    });

    // nodes
    current.nodes.forEach((node) => {
      const group = nodeLayer.append("g")
        .attr("class", "node-group")
        .attr("data-node", node.id)
        .attr("transform", `translate(${node.position.x}, ${node.position.y})`);

      if (isDecorative(node)) {
        renderSticky(group, node);
      } else {
        renderNode(group, node, tooltip);
      }

      group.call(d3.drag()
        .filter((event) => !event.target.classList.contains("port-hit"))
        .on("start", () => { dragging = { id: node.id, moved: false }; })
        .on("drag", (event) => {
          node.position.x += event.dx;
          node.position.y += event.dy;
          dragging.moved = true;
          group.attr("transform", `translate(${node.position.x}, ${node.position.y})`);
          refreshEdgesFor(node.id, nodeById);
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
    fitToView();
  }

  function renderNode(group, node, tooltip) {
    const meta = nodeMeta(node.tool);
    const color = categoryColor(meta.category);
    const disabled = !!node.disabled;

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
        renderGraph();
        showNodeDetails(node);
      })
      .on("mouseenter", (event) => {
        tooltip.innerHTML = `<b>${esc(meta.label)}</b><br>${esc(meta.summary || node.tool)}`;
        tooltip.style.left = `${event.offsetX + 14}px`;
        tooltip.style.top = `${event.offsetY - 10}px`;
        tooltip.classList.add("visible");
      })
      .on("mouseleave", () => tooltip.classList.remove("visible"));

    group.append("rect")
      .attr("class", "node-accent")
      .attr("x", 0).attr("y", 10).attr("width", 4).attr("height", NODE_H - 20)
      .attr("rx", 2).attr("fill", color).attr("opacity", disabled ? 0.4 : 1);

    group.append("text")
      .attr("x", 18).attr("y", NODE_H / 2 - 7)
      .attr("font-size", "13px").attr("fill", color)
      .style("pointer-events", "none")
      .text(meta.icon || "◆");

    const label = node.label || meta.label || node.tool;
    group.append("text")
      .attr("class", "node-tool-label")
      .attr("x", 40).attr("y", NODE_H / 2 - 6)
      .text(label.length > 20 ? `${label.slice(0, 19)}…` : label);

    group.append("text")
      .attr("class", "node-id-label")
      .attr("x", 40).attr("y", NODE_H / 2 + 13)
      .text(node.id.length > 24 ? `${node.id.slice(0, 23)}…` : node.id);

    // badges (right edge, top row)
    const badges = [];
    if (!(node.depends_on || []).length) badges.push({ text: "ENTRY", color: "var(--cyan)" });
    if (node.loop_to) badges.push({ text: `LOOP ×${node.max_visits || 1}`, color: "var(--pink)" });
    if (node.pinned_data) badges.push({ text: "PINNED", color: "var(--amber)" });
    if (node.needs_approval) badges.push({ text: "🔒", color: "var(--amber)" });
    if (disabled) badges.push({ text: "OFF", color: "var(--dim)" });
    let badgeX = NODE_W - 8;
    badges.slice(0, 3).forEach((badge) => {
      const node_text = group.append("text")
        .attr("class", "node-badge-text")
        .attr("x", badgeX).attr("y", 15)
        .attr("text-anchor", "end")
        .attr("fill", badge.color)
        .text(badge.text);
      badgeX -= badge.text.length * 5 + 10;
      return node_text;
    });

    // run status dot
    const result = lastRun && lastRun.byId && lastRun.byId[node.id];
    if (result) {
      group.append("circle")
        .attr("class", result.ok ? "status-ok" : "status-fail")
        .attr("cx", NODE_W - 11).attr("cy", NODE_H - 12).attr("r", 4);
    }

    // ports
    group.append("circle").attr("class", "port-circle port-input")
      .attr("cx", 0).attr("cy", NODE_H / 2).attr("r", PORT_R);
    group.append("circle")
      .attr("class", "port-hit")
      .attr("cx", 0).attr("cy", NODE_H / 2).attr("r", PORT_HIT_R)
      .attr("fill", "transparent").attr("data-target-id", node.id)
      .style("cursor", "crosshair");

    group.append("circle").attr("class", "port-circle port-output")
      .attr("cx", NODE_W).attr("cy", NODE_H / 2).attr("r", PORT_R);
    group.append("circle")
      .attr("class", "port-hit")
      .attr("cx", NODE_W).attr("cy", NODE_H / 2).attr("r", PORT_HIT_R)
      .attr("fill", "transparent").attr("data-source-id", node.id)
      .style("cursor", "crosshair")
      .call(d3.drag()
        .on("start", () => {
          dragging = { connectFrom: node.id };
        })
        .on("drag", (event) => {
          if (!dragging || !dragging.connectFrom) return;
          let temp = svgRoot.select("#temp-edge");
          if (temp.empty()) {
            temp = viewport.append("path").attr("id", "temp-edge")
              .attr("class", "edge-depends").style("pointer-events", "none");
          }
          const point = d3.pointer(event, viewport.node());
          const x1 = node.position.x + NODE_W;
          const y1 = node.position.y + NODE_H / 2;
          const curve = Math.min(Math.max(Math.hypot(point[0] - x1, point[1] - y1) * 0.35, 30), 110);
          temp.attr("d", `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${point[0] - curve} ${point[1]}, ${point[0]} ${point[1]}`);
        })
        .on("end", (event) => {
          svgRoot.select("#temp-edge").remove();
          if (!dragging || !dragging.connectFrom) return;
          const point = d3.pointer(event, viewport.node());
          const target = current.nodes.find((candidate) => {
            if (candidate.id === node.id || isDecorative(candidate)) return false;
            const cx = candidate.position.x;
            const cy = candidate.position.y + NODE_H / 2;
            return Math.hypot(point[0] - cx, point[1] - cy) < 34;
          });
          if (target) connectNodes(node.id, target.id);
          dragging = null;
        }));
  }

  function renderSticky(group, node) {
    group.append("rect")
      .attr("class", "sticky-card")
      .attr("width", STICKY_W).attr("height", STICKY_H).attr("rx", 6)
      .on("click", (event) => {
        event.stopPropagation();
        selectedNodeId = node.id;
        selectedEdgeKey = null;
        renderGraph();
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
      const path = edgePath(source, target);
      const key = edgeKey(edge);
      d3.selectAll(`[data-edge="${key}"]`).attr("d", path);
      d3.selectAll(`[data-edge-line="${key}"]`).attr("d", path);
    });
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
    $("details-title").textContent = `${meta.icon || "◆"} ${meta.label || node.tool}`;
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
        `<span class="chip-icon" style="color:${categoryColor(entry.category)}">${esc(entry.icon || "◆")}</span>` +
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
      current.nodes = current.nodes.filter((n) => n.id !== selectedNodeId);
      current.nodes.forEach((node) => {
        node.depends_on = (node.depends_on || []).filter((dep) => dep !== selectedNodeId);
        if (node.loop_to === selectedNodeId) delete node.loop_to;
        if (node.fallback_to === selectedNodeId) delete node.fallback_to;
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
          renderGraph();
          showNodeDetails(node);
        };
      });
      status(failed ? "Run: failures" : "Run ok", failed ? "bad" : "good");
      renderGraph();
    } catch (err) {
      showRunPanel("Run failed", `<div class="run-fail">${esc(err.message)}</div>`);
      status("Run failed", "bad");
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

    document.addEventListener("keydown", (event) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if (event.key === "Escape") {
        $("template-modal").hidden = true;
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

    window.addEventListener("resize", () => { if (current) renderGraph(); });
  }

  // ── boot ─────────────────────────────────────────────────────────────────
  wireHeader();
  fetchCatalog().then(() => fetchPlaybooks(false));
})();
