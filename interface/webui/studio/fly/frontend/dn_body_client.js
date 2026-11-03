/* Stage 6.1: DN body drive → VRM expression / gesture amplitude */
(function () {
  window.aikoApplyDnBody = function (body, intents) {
    try {
      const b = body || {};
      if (b.cancelled) {
        if (typeof window.aikoSetExpression === "function") window.aikoSetExpression("neutral", 0.15);
        if (typeof window.aikoSetPose === "function") window.aikoSetPose("thinking", false);
        return;
      }
      const expr = Number(b.expression_intensity);
      let name = "neutral";
      if (expr >= 0.62) name = "happy";
      else if (expr <= 0.32) name = "sorrow";
      const intensity = Math.max(0.05, Math.min(1.0, Number.isFinite(expr) ? expr : 0.5));
      if (typeof window.aikoSetExpression === "function") window.aikoSetExpression(name, intensity);
      if (Array.isArray(intents)) {
        for (const it of intents) {
          if (it && it.kind === "expression" && typeof window.aikoSetExpression === "function") {
            window.aikoSetExpression(it.name || "neutral", Number(it.intensity) || 0.5);
          }
        }
      }
    } catch (e) {
      console.debug("aikoApplyDnBody skipped", e);
    }
  };
})();
