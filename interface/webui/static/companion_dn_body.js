/* Stage 6.1: poll Fly Studio body drive and drive VRM expression. */
(function () {
  const url = "/studio/fly/api/body";
  let lastJson = "";
  let inFlight = false;
  async function tick() {
    if (inFlight) return;
    inFlight = true;
    try {
      const r = await fetch(url, { credentials: "same-origin" });
      if (!r.ok) return;
      const d = await r.json();
      const key = JSON.stringify(d.body || {});
      if (key === lastJson) return;
      lastJson = key;
      if (window.aikoApplyDnBody) {
        window.aikoApplyDnBody(d.body || {}, d.avatar_intents || []);
      } else if (d.body && !d.body.cancelled && window.aikoSetExpression) {
        const e = Number(d.body.expression_intensity || 0.5);
        const name = e >= 0.62 ? "happy" : e <= 0.32 ? "sorrow" : "neutral";
        window.aikoSetExpression(name, Math.max(0.1, Math.min(1, e)));
      }
    } catch (_) {
      /* studio may be offline */
    } finally {
      inFlight = false;
    }
  }
  if (!window.aikoApplyDnBody) {
    const s = document.createElement("script");
    s.src = "/static/dn_body_client.js";
    s.async = true;
    s.addEventListener("load", () => {
      lastJson = "";
      void tick();
    });
    document.head.appendChild(s);
  }
  setInterval(tick, 2500);
  setTimeout(tick, 1200);
})();
