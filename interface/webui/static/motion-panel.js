// Local browser library: JSON is parsed as data, never imported as JavaScript.
import { validateMotionClip } from './motion-clip.js';

export function setupMotionPanel({ play, stop }) {
  const file = document.getElementById('motion-file');
  const select = document.getElementById('motion-select');
  const status = document.getElementById('motion-status');
  const clips = new Map();
  let db = null;
  const show = text => { status.textContent = text; };
  const add = clip => {
    clips.set(clip.name, clip);
    if (![...select.options].some(o => o.value === clip.name)) select.add(new Option(clip.name, clip.name));
    select.value = clip.name;
  };
  let request;
  try { request = indexedDB.open('aiko-motion-clips', 1); } catch (_) {
    show('Browser storage unavailable; imports work for this session.');
  }
  if (request) request.onupgradeneeded = () => request.result.createObjectStore('clips', { keyPath: 'name' });
  if (request) request.onerror = () => show('Browser storage unavailable; imports work for this session.');
  if (request) request.onsuccess = () => {
    db = request.result;
    const read = db.transaction('clips').objectStore('clips').getAll();
    read.onsuccess = () => {
      for (const clip of read.result) { try { add(validateMotionClip(clip)); } catch (_) { /* skip old/corrupt entries */ } }
      show(clips.size ? `${clips.size} saved motion clips` : 'Import a motion JSON from the video tool.');
    };
  };
  file.addEventListener('change', async () => {
    const chosen = file.files?.[0];
    if (!chosen) return;
    try {
      if (chosen.size > 20 * 1024 * 1024) throw new Error('Use a clip under 20 MB.');
      const clip = validateMotionClip(JSON.parse(await chosen.text()));
      clip.name = String(clip.name || chosen.name).slice(0, 100);
      add(clip);
      if (db) {
        const tx = db.transaction('clips', 'readwrite');
        tx.objectStore('clips').put(clip);
        tx.oncomplete = () => show(`Saved ${clip.name} in this browser.`);
        tx.onerror = () => show('Loaded for this session; browser storage is full.');
      } else show('Loaded for this session.');
    } catch (err) { show(err.message); }
    file.value = '';
  });
  const playNamed = (name, options = {}) => {
    const clip = clips.get(name);
    if (!clip) throw new Error('Import and select a motion clip first.');
    play(clip, { loop: document.getElementById('motion-loop').checked,
      speed: Number(document.getElementById('motion-speed').value), ...options });
    show(`Playing ${name}`);
  };
  window.aikoPlayMotionClip = playNamed;
  document.getElementById('motion-play').addEventListener('click', () => {
    try { playNamed(select.value); } catch (err) { show(err.message); }
  });
  document.getElementById('motion-stop').addEventListener('click', () => { stop(); show('Stopped.'); });
  document.getElementById('motion-delete').addEventListener('click', () => {
    stop();
    const name = select.value;
    clips.delete(name);
    select.selectedOptions[0]?.remove();
    if (db) {
      const tx = db.transaction('clips', 'readwrite');
      tx.objectStore('clips').delete(name);
      tx.oncomplete = () => show('Clip removed.');
      tx.onerror = () => show('Removed for this session; browser storage could not be updated.');
    } else show('Clip removed.');
  });
}
