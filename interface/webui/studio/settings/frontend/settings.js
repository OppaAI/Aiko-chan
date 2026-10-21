/* Settings Studio: renders the /api/settings catalog as tabbed groups.
   Secrets are write-only: the browser only ever learns key names and
   set/unset state; an empty secret field means "leave unchanged". */
(function () {
  "use strict";

  const tabsEl = document.getElementById("set-tabs");
  const groupsEl = document.getElementById("set-groups");
  const saveBtn = document.getElementById("set-save");
  const statusEl = document.getElementById("set-status");
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
    } else {
      const num = s.type === "int" || s.type === "float";
      input = '<input id="' + id + '" data-kind="setting" data-key="' +
        esc(s.key) + '" type="' + (num ? "number" : "text") + '"' +
        (s.type === "float" ? ' step="any"' : "") +
        ' value="' + esc(s.value) + '"' + (locked ? " disabled" : "") + ">";
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
      '<div class="set-key">' + esc(s.key) + "</div>" +
      (desc ? '<div class="set-desc">' + desc + "</div>" : "") +
      '</div><div class="set-input">' + input + "</div></div>";
  }

  function secretFieldFor(k) {
    const id = "sec-" + k.key;
    const locked = k.env_overridden;
    return '<div class="set-row">' +
      '<div class="set-label"><label for="' + id + '">' + esc(k.key) +
      "</label>" + (k.is_set ? ' <span class="badge">set</span>' : "") +
      (locked ? ' <span class="badge warn">env override</span>' : "") +
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
    try { localStorage.setItem(TAB_KEY, name); } catch (e) {}
    tabsEl.querySelectorAll(".set-tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.group === name));
    groupsEl.querySelectorAll(".set-group").forEach((s) => {
      s.hidden = s.dataset.group !== name;
    });
  }

  async function load() {
    const r = await fetch("api/settings");
    if (!r.ok) {
      statusEl.textContent = "Could not load settings (sign-in required?).";
      statusEl.className = "err";
      return;
    }
    const data = await r.json();
    const names = data.groups.map((g) => g.group).concat(["Secrets"]);
    tabsEl.innerHTML = "";
    names.forEach((name) => {
      const t = document.createElement("button");
      t.type = "button";
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
  }

  async function save() {
    const settings = {};
    groupsEl.querySelectorAll('[data-kind="setting"]').forEach((el) => {
      if (el.disabled) return;
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
    saveBtn.disabled = true;
    statusEl.textContent = "saving…";
    statusEl.className = "";
    try {
      const r = await fetch("api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: settings, secrets: secrets }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
      const n = data.updated.length + data.secrets_updated.length;
      statusEl.textContent = "saved " + n + " change(s) — restart Aiko to apply.";
      statusEl.className = "ok";
      groupsEl.querySelectorAll('[data-kind="secret"], [data-secret="1"]')
        .forEach((el) => { el.value = ""; });
      if (newKey) newKey.value = "";
      if (newVal) newVal.value = "";
    } catch (e) {
      statusEl.textContent = "save failed: " + e.message;
      statusEl.className = "err";
    }
    saveBtn.disabled = false;
  }

  saveBtn.addEventListener("click", save);
  load();
})();
