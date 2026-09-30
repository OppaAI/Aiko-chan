/* Aiko Calendar Studio — agenda, work board, and full scheduler control. */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const dialog = $('#editor');
const form = $('#item-form');
const scheduleDialog = $('#schedule-editor');
const scheduleForm = $('#schedule-form');

let data = { items: [], schedules: [] };
let editing = null;          // calendar item being edited (or null for new)
let editingSchedule = null;  // scheduler record being edited (or null for new)

const text = (value) => value || 'No details yet';
const fmt = (value) => value
  ? new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(value))
  : '';

function escapeHtml(value) {
  const d = document.createElement('div');
  d.textContent = value || '';
  return d.innerHTML;
}

/* ── loading & rendering ─────────────────────────────────────────────── */

async function load() {
  const response = await fetch('api/overview');
  if (!response.ok) throw new Error('Could not load calendar');
  data = await response.json();
  render();
}

const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function scheduleSummary(s) {
  const parts = [];
  const freq = (s.frequency || 'once').toLowerCase();
  if (freq === 'interval' && s.interval_seconds) {
    const mins = Math.round(s.interval_seconds / 60);
    parts.push(mins >= 60 ? `every ${mins / 60}h` : `every ${mins} min`);
  } else {
    parts.push(freq.replace('_', ' '));
  }
  if (s.time_of_day && freq !== 'interval') parts.push(s.time_of_day);
  if ((freq === 'weekly' || freq === 'custom_weekdays') && s.days_of_week && s.days_of_week.length) {
    parts.push(s.days_of_week.map((d) => DAY_NAMES[d] || d).join(', '));
  }
  if (s.timezone) parts.push(s.timezone);
  parts.push(s.action || 'announce');
  if (s.requires_idle) parts.push(`idle ≥ ${Math.round((s.idle_seconds || 600) / 60)} min`);
  return parts.join(' · ');
}

function render() {
  const { brief } = data;
  $('#today').textContent = new Intl.DateTimeFormat(undefined, {
    weekday: 'long', month: 'long', day: 'numeric',
  }).format(new Date(`${data.today}T12:00`));
  $('#open-count').textContent = brief.open_count;
  $('#due-count').textContent = brief.due_today.length;
  $('#overdue-count').textContent = brief.overdue.length;
  $('#next-up').textContent = brief.next_up[0]
    ? `${brief.next_up[0].title}${brief.next_up[0].due_at ? ` · due ${fmt(brief.next_up[0].due_at)}` : ''}`
    : 'Nothing queued — make space for deep work.';

  // Agenda
  const dated = data.items
    .filter((i) => i.kind === 'appointment' || i.start_at)
    .sort((a, b) => new Date(a.start_at || a.due_at || 0) - new Date(b.start_at || b.due_at || 0));
  $('#agenda-list').innerHTML = dated.length
    ? dated.map((i) => `<article class="agenda-row" data-id="${i.id}"><b>${escapeHtml(i.title)}</b><small>${fmt(i.start_at || i.due_at)} · ${escapeHtml(i.project || i.kind)}</small></article>`).join('')
    : '<p class="empty">No appointments yet. Add time for what matters.</p>';

  // Scheduler — every record, disabled ones dimmed with a paused badge.
  const schedules = (data.schedules || []).slice().sort((a, b) => {
    const ea = a.enabled !== false, eb = b.enabled !== false;
    if (ea !== eb) return ea ? -1 : 1;
    return new Date(a.next_due || 0) - new Date(b.next_due || 0);
  });
  $('#schedule-list').innerHTML = schedules.length
    ? schedules.map((s) => {
        const paused = s.enabled === false;
        return `<article class="schedule-row${paused ? ' paused' : ''}" data-schedule-id="${escapeHtml(s.id)}">` +
          `<b>${escapeHtml(s.title)}${paused ? ' <span class="badge">paused</span>' : ''}</b>` +
          `<small>${escapeHtml(scheduleSummary(s))}</small>` +
          (s.next_due && !paused ? `<small class="next-due">next ${fmt(s.next_due)}</small>` : '') +
          `</article>`;
      }).join('')
    : '<p class="empty">No scheduled tasks yet.</p>';

  // Work board
  const groups = [['backlog', 'Backlog'], ['todo', 'To do'], ['doing', 'In progress'], ['waiting', 'Waiting']];
  $('#board-columns').innerHTML = groups.map(([status, label]) =>
    `<section class="column"><h3>${label}</h3>${data.items.filter((i) => i.status === status).map(card).join('')}</section>`
  ).join('');

  document.querySelectorAll('[data-id]').forEach((el) => {
    el.onclick = () => openEditor(data.items.find((i) => i.id === el.dataset.id));
  });
  document.querySelectorAll('[data-schedule-id]').forEach((el) => {
    el.onclick = () => openScheduleEditor(data.schedules.find((s) => String(s.id) === el.dataset.scheduleId));
  });
}

function card(i) {
  return `<article class="card" data-id="${i.id}"><span class="tag">${escapeHtml(i.kind)} · ${escapeHtml(i.priority)}</span>` +
    `<p>${escapeHtml(i.title)}</p><small>${escapeHtml(i.project || text(i.due_at && `Due ${fmt(i.due_at)}`))}</small></article>`;
}

/* ── calendar item editor (unchanged behavior) ───────────────────────── */

function openEditor(item) {
  editing = item || null;
  form.reset();
  $('#modal-title').textContent = item ? 'Edit plan' : 'Plan an item';
  $('#delete-item').hidden = !item;
  if (item) {
    for (const [key, value] of Object.entries(item)) {
      const field = form.elements[key];
      if (field && value) field.value = key.endsWith('_at') ? value.slice(0, 16) : value;
    }
  }
  dialog.showModal();
}

$('#new-item').onclick = () => openEditor();
$('#refresh').onclick = () => load().catch(showError);
$('#delete-item').onclick = async () => {
  if (!editing || !confirm('Delete this plan and its linked reminder?')) return;
  await fetch(`api/items/${editing.id}`, { method: 'DELETE' });
  dialog.close();
  load();
};

form.onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const payload = Object.fromEntries(fd.entries());
  payload.schedule_enabled = undefined;
  if (fd.get('schedule_enabled')) {
    payload.schedule = {
      time_of_day: fd.get('schedule_time'),
      frequency: fd.get('schedule_frequency'),
      action: 'announce',
    };
  }
  delete payload.schedule_time;
  delete payload.schedule_frequency;
  delete payload.schedule_enabled;
  for (const key of ['start_at', 'due_at']) if (!payload[key]) payload[key] = null;
  const response = await fetch(editing ? `api/items/${editing.id}` : 'api/items', {
    method: editing ? 'PUT' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) return showError(await response.text());
  dialog.close();
  load();
};

/* ── scheduler record editor ─────────────────────────────────────────── */

function syncScheduleFieldVisibility() {
  const freq = scheduleForm.elements['s_frequency'].value;
  $('#interval-row').hidden = freq !== 'interval';
  $('#weekday-row').hidden = !(freq === 'weekly' || freq === 'custom_weekdays');
  const needsIdle = scheduleForm.elements['s_requires_idle'].checked;
  $('#idle-row').hidden = !needsIdle;
}

function openScheduleEditor(record) {
  editingSchedule = record || null;
  scheduleForm.reset();
  $('#schedule-modal-title').textContent = record ? 'Edit scheduled task' : 'New scheduled task';
  $('#delete-schedule').hidden = !record;
  if (record) {
    scheduleForm.elements['s_title'].value = record.title || '';
    scheduleForm.elements['s_task'].value = record.task || '';
    scheduleForm.elements['s_time'].value = (record.time_of_day || '09:00').slice(0, 5);
    scheduleForm.elements['s_frequency'].value = record.frequency || 'daily';
    scheduleForm.elements['s_interval_minutes'].value = record.interval_seconds ? Math.round(record.interval_seconds / 60) : 15;
    scheduleForm.elements['s_timezone'].value = record.timezone || '';
    scheduleForm.elements['s_action'].value = record.action || 'agentic';
    scheduleForm.elements['s_enabled'].checked = record.enabled !== false;
    scheduleForm.elements['s_requires_idle'].checked = !!record.requires_idle;
    scheduleForm.elements['s_idle_minutes'].value = record.idle_seconds ? Math.round(record.idle_seconds / 60) : 10;
    const days = new Set(record.days_of_week || []);
    $$('input[name="s_weekday"]').forEach((box) => { box.checked = days.has(Number(box.value)); });
    $('#schedule-next-due').textContent = record.next_due
      ? `Next run: ${fmt(record.next_due)}${record.last_ran_at ? ` · last ran ${fmt(record.last_ran_at)}` : ''}`
      : '';
  } else {
    $('#schedule-next-due').textContent = '';
  }
  syncScheduleFieldVisibility();
  scheduleDialog.showModal();
}

scheduleForm.elements['s_frequency'].onchange = syncScheduleFieldVisibility;
scheduleForm.elements['s_requires_idle'].onchange = syncScheduleFieldVisibility;

$('#new-schedule').onclick = () => openScheduleEditor();

$('#delete-schedule').onclick = async () => {
  if (!editingSchedule || !confirm(`Delete "${editingSchedule.title}" permanently?`)) return;
  const response = await fetch(`api/schedules/${editingSchedule.id}`, { method: 'DELETE' });
  if (!response.ok) return showError(await response.text());
  scheduleDialog.close();
  load();
};

scheduleForm.onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(scheduleForm);
  const payload = {
    title: fd.get('s_title'),
    task: fd.get('s_task') || '',
    time_of_day: fd.get('s_time') || '09:00',
    frequency: fd.get('s_frequency'),
    timezone: fd.get('s_timezone') || null,
    action: fd.get('s_action'),
    enabled: !!fd.get('s_enabled'),
    requires_idle: !!fd.get('s_requires_idle'),
  };
  if (payload.frequency === 'interval') {
    payload.interval_seconds = Math.max(60, Math.round(Number(fd.get('s_interval_minutes') || 15) * 60));
  }
  if (payload.frequency === 'weekly' || payload.frequency === 'custom_weekdays') {
    payload.days_of_week = fd.getAll('s_weekday').map(Number);
  }
  if (payload.requires_idle) {
    payload.idle_seconds = Math.max(60, Math.round(Number(fd.get('s_idle_minutes') || 10) * 60));
  }
  const url = editingSchedule ? `api/schedules/${editingSchedule.id}` : 'api/schedules';
  const response = await fetch(url, {
    method: editingSchedule ? 'PUT' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) return showError(await response.text());
  scheduleDialog.close();
  load();
};

function showError(err) {
  alert(typeof err === 'string' ? err : err.message);
}

load().catch(showError);
