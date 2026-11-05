/* Settings Studio: renders the /api/settings catalog as tabbed groups.
   Secrets are write-only: the browser only ever learns key names and
   set/unset state; an empty secret field means "leave unchanged". */
(function () {
  "use strict";

  const tabsEl = document.getElementById("set-tabs");
  const groupsEl = document.getElementById("set-groups");
  const saveBtn = document.getElementById("set-save");
  const statusEl = document.getElementById("set-status");
  const dirty = new Set();
  let baseline = new Map();
  let currentTab = "";
  let revision = null;
  const searchEl = document.getElementById("set-search");
  const TAB_KEY = "aiko-settings-tab";

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function badgesFor(s) {
    let out = "";
    if (s.secret && s.is_set) out += ' <span class="badge">set</span>';
    if (s.env_overridden) out += ' <span class="badge warn">env override</span>';
    if (s.readonly) out += ' <span class="badge">read-only</span>';
    if (s.uses_default) out += ' <span class="badge">code default</span>';
    if (s.apply) out += ' <span class="badge ' + (s.restart_required ? '' : 'live') + '">' + esc(s.apply) + '</span>';
    if (s.pending_restart) out += ' <span class="badge warn">restart pending</span>';
    return out;
  }

  function fieldFor(s) {
    const id = "set-" + s.key;
    const locked = s.env_overridden || s.readonly;
    let input;
    if (s.type === "bool") {
      input = '<input type="checkbox" id="' + id + '" data-kind="setting"' +
        ' data-key="' + esc(s.key) + '"' +
        (String(s.value) === "1" ? " checked" : "") +
        (locked ? " disabled" : "") + ">";
    } else if (s.type === "enum") {
      input = '<select id="' + id + '" data-kind="setting" data-key="' +
        esc(s.key) + '"' + (locked ? " disabled" : "") + ">" +
        s.options.map((o) =>
          '<option value="' + esc(o) + '"' +
          (String(s.value) === o ? " selected" : "") + ">" + esc(o) + "</option>"
        ).join("") + "</select>";
    } else if (s.type === "password") {
      input = '<input id="' + id + '" data-kind="setting" data-secret="1"' +
        ' data-key="' + esc(s.key) + '" type="password" value=""' +
        ' placeholder="' + (s.is_set ? "(set — leave blank to keep)" : "not set") + '"' +
        ' autocomplete="new-password"' + (locked ? " disabled" : "") + ">";
    } else if (s.type === "text" || s.type === "json") {
      input = '<textarea rows="4" id="' + id + '" data-kind="setting" data-key="' + esc(s.key) + '"' + (locked ? " disabled" : "") + '>' + esc(s.value) + '</textarea>';
    } else {
      const num = s.type === "int" || s.type === "float";
      input = '<input id="' + id + '" data-kind="setting" data-key="' +
        esc(s.key) + '" type="' + (num ? "number" : "text") + '"' +
        (s.type === "float" ? ' step="any"' : "") +
        (s.min !== undefined ? ' min="' + esc(s.min) + '"' : '') +
        (s.max !== undefined ? ' max="' + esc(s.max) + '"' : '') +
        ' value="' + esc(s.value) + '"' + (locked ? " disabled" : "") + ">";
    }
    if (!locked && (s.type === "float" || s.type === "int") && s.min !== undefined && s.max !== undefined) {
      input = '<div class="set-numeric"><input type="range" aria-label="' + esc(s.label) + ' slider" data-pair="' + id + '" min="' + esc(s.min) + '" max="' + esc(s.max) + '" step="' + (s.type === 'int' ? '1' : '0.01') + '" value="' + esc(s.value) + '">' + input + '</div>';
    }
    let desc = s.desc ? esc(s.desc) : "";
    if (s.env_overridden && s.env_value !== undefined) {
      desc += (desc ? " " : "") +
        "Currently overridden by the process environment (" +
        esc(s.env_value) + "); editing the file has no effect.";
    }
    return '<div class="set-row">' +
      '<div class="set-label"><label for="' + id + '">' + esc(s.label) +
      "</label>" + badgesFor(s) +
      '<div class="set-key">' + esc(s.key) + (s.file ? " · " + esc(s.file) : "") + "</div>" +
      (desc ? '<div class="set-desc">' + desc + "</div>" : "") +
      '</div><div class="set-input">' + input + "</div></div>";
  }

  function secretFieldFor(k) {
    const id = "sec-" + k.key;
    const locked = k.env_overridden || k.yaml_managed;
    return '<div class="set-row">' +
      '<div class="set-label"><label for="' + id + '">' + esc(k.key) +
      "</label>" + (k.is_set ? ' <span class="badge">set</span>' : "") +
      (locked ? ' <span class="badge warn">' + (k.yaml_managed ? "edit in YAML group" : "env override") + "</span>" : "") +
      ' <span class="badge">Restart required</span>' +
      (k.pending_restart ? ' <span class="badge warn">restart pending</span>' : '') +
      '<div class="set-key">.env.age</div></div>' +
      '<div class="set-input"><input id="' + id + '" data-kind="secret"' +
      ' data-key="' + esc(k.key) + '" type="password" value=""' +
      ' placeholder="' + (k.is_set ? "(set — leave blank to keep)" : "not set") + '"' +
      ' autocomplete="new-password"' + (locked ? " disabled" : "") +
      "></div></div>";
  }

  function secretsSection(secrets) {
    let html = '<section class="set-group" data-group="Secrets" hidden>' +
      "<h2>Secrets</h2>";
    if (!secrets.available) {
      html += '<p class="set-note">The encrypted secrets file is not ' +
        "available on this machine (age tool or identity file missing), " +
        "so secrets cannot be edited here.</p>";
    } else {
      html += '<p class="set-note">Stored in the encrypted .env.age file. ' +
        "Values are never shown — leave a field blank to keep the current " +
        "value.</p>";
      html += secrets.keys.map(secretFieldFor).join("");
      html += '<div class="set-row"><div class="set-label">' +
        "<label>Add a new secret</label>" +
        '<div class="set-desc">Key name and value; saved with the rest.</div></div>' +
        '<div class="set-input"><div class="set-newkey">' +
        '<input id="sec-new-key" type="text" placeholder="KEY_NAME" autocomplete="off">' +
        '<input id="sec-new-value" type="password" placeholder="value" autocomplete="new-password">' +
        "</div></div></div>";
    }
    return html + "</section>";
  }

  function activateTab(name) {
    currentTab = name;
    searchEl.value = "";
    try { localStorage.setItem(TAB_KEY, name); } catch (e) {}
    tabsEl.querySelectorAll(".set-tab").forEach((t) =>
      { t.classList.toggle("active", t.dataset.group === name);
        t.setAttribute("aria-selected", String(t.dataset.group === name)); });
    groupsEl.querySelectorAll(".set-group").forEach((s) => {
      s.hidden = s.dataset.group !== name;
      s.querySelectorAll(".set-row").forEach(row => { row.hidden = false; });
    });
  }

  async function load() {
    const r = await fetch("api/settings");
    if (!r.ok) {
      statusEl.textContent = r.status === 403 ? "Settings require a creator or administrator account." : "Could not load settings. Sign in and try again.";
      statusEl.className = "err";
      saveBtn.disabled = true;
      return;
    }
    const data = await r.json();
    revision = data.revision;
    const names = data.groups.map((g) => g.group).concat(["Secrets"]);
    tabsEl.innerHTML = "";
    names.forEach((name) => {
      const t = document.createElement("button");
      t.type = "button";
      t.setAttribute("role", "tab");
      t.className = "set-tab";
      t.dataset.group = name;
      t.textContent = name;
      t.addEventListener("click", () => activateTab(name));
      tabsEl.appendChild(t);
    });
    groupsEl.innerHTML = data.groups.map((g) =>
      '<section class="set-group" data-group="' + esc(g.group) +
      '" hidden><h2>' + esc(g.group) + "</h2>" +
      g.settings.map(fieldFor).join("") + "</section>"
    ).join("") + secretsSection(data.secrets);
    let active = "Conscience & safety";
    try { active = localStorage.getItem(TAB_KEY) || active; } catch (e) {}
    if (!names.includes(active)) active = names[0];
    activateTab(active);
    document.getElementById("set-coverage").textContent = data.groups.reduce((n, g) => n + g.settings.length, 0) + " runtime settings · encrypted secrets";
    const sources = document.getElementById("set-sources");
    sources.innerHTML = (data.sources || []).map(source => '<span class="badge">' + esc(source.file) + ': ' + esc(source.editable ? source.settings + ' settings' : source.reason) + '</span>').join(' ');
    baseline = new Map();
    groupsEl.querySelectorAll('[data-kind]').forEach(el => baseline.set(el.id, valueOf(el)));
    dirty.clear();
    updateCount();
  }

  async function save() {
    const settings = {};
    groupsEl.querySelectorAll('[data-kind="setting"]').forEach((el) => {
      if (el.disabled || !dirty.has(el.id)) return;
      if (el.dataset.secret && el.value === "") return; // blank = unchanged
      settings[el.dataset.key] = el.type === "checkbox"
        ? (el.checked ? "1" : "0") : el.value;
    });
    const secrets = {};
    groupsEl.querySelectorAll('[data-kind="secret"]').forEach((el) => {
      if (el.disabled || el.value === "") return;
      secrets[el.dataset.key] = el.value;
    });
    const newKey = document.getElementById("sec-new-key");
    const newVal = document.getElementById("sec-new-value");
    if (newKey && newVal && newKey.value.trim() !== "" && newVal.value !== "") {
      secrets[newKey.value.trim()] = newVal.value;
    }
    if (!Object.keys(settings).length && !Object.keys(secrets).length) return;
    const invalid = Array.from(groupsEl.querySelectorAll('[data-kind]')).find(el => dirty.has(el.id) && !el.checkValidity());
    if (invalid) { invalid.reportValidity(); return; }
    saveBtn.disabled = true;
    groupsEl.inert = true;
    statusEl.textContent = "saving…";
    statusEl.className = "";
    try {
      const r = await fetch("api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: settings, secrets: secrets, revision: revision }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
      await load();
      statusEl.textContent = "Saved. " + data.applied_live.length + " applied live; " + data.restart_keys.length + " require restart.";
      statusEl.className = "ok";
      groupsEl.querySelectorAll('[data-kind="secret"], [data-secret="1"]')
        .forEach((el) => { el.value = ""; });
      if (newKey) newKey.value = "";
      if (newVal) newVal.value = "";
    } catch (e) {
      statusEl.textContent = "save failed: " + e.message;
      statusEl.className = "err";
    }
    groupsEl.inert = false;
    updateCount();
  }

  function valueOf(el) { return el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value; }
  function updateCount() {
    const newKey = document.getElementById("sec-new-key");
    const newVal = document.getElementById("sec-new-value");
    const n = dirty.size + (newKey && newVal && newKey.value && newVal.value ? 1 : 0);
    document.getElementById("set-count").textContent = n ? n + " unsaved change(s)." : "No unsaved changes.";
    saveBtn.disabled = n === 0;
  }
  groupsEl.addEventListener("input", event => {
    let el = event.target;
    if (el.dataset.pair) {
      const pair = document.getElementById(el.dataset.pair);
      pair.value = el.value;
      el = pair;
    } else if (el.dataset.kind === "setting") {
      const range = groupsEl.querySelector('[data-pair="' + el.id + '"]');
      if (range) range.value = el.value;
    }
    if (baseline.has(el.id)) {
      if (valueOf(el) === baseline.get(el.id)) dirty.delete(el.id);
      else dirty.add(el.id);
      el.closest(".set-row").classList.toggle("is-dirty", dirty.has(el.id));
    }
    updateCount();
  });
  searchEl.addEventListener("input", () => {
    const query = searchEl.value.trim().toLowerCase();
    groupsEl.querySelectorAll(".set-group").forEach(section => {
      let matches = 0;
      section.querySelectorAll(".set-row").forEach(row => {
        row.hidden = query ? !row.textContent.toLowerCase().includes(query) : false;
        if (!row.hidden) matches++;
      });
      section.hidden = query ? matches === 0 : section.dataset.group !== currentTab;
    });
  });
  tabsEl.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const tabs = Array.from(tabsEl.querySelectorAll("button"));
    let index = tabs.indexOf(document.activeElement);
    index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    event.preventDefault(); tabs[index].focus(); tabs[index].click();
  });
  document.getElementById("set-theme").addEventListener("click", () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("aiko-theme", theme); } catch (e) {}
  });
  window.addEventListener("beforeunload", event => {
    if (!saveBtn.disabled) { event.preventDefault(); event.returnValue = ""; }
  });
  saveBtn.addEventListener("click", save);
  load().catch(() => { statusEl.textContent = "Could not load settings. Check your connection and reload."; });
})();
