/* WinMatic status.js (safe, prod-friendly) */
(function() {
  function byId(id) { return document.getElementById(id); }

  const DEFAULT_BASE = "https://football-predictor-winmatic.onrender.com";
  function apiBase() {
    try {
      const qs = new URLSearchParams(window.location.search);
      const override = (qs.get("api") || "").trim();
      if (override) return override.replace(/\/$/, "");
    } catch (e) {}
    try {
      const origin = window.location.origin || "";
      if (origin && (origin.includes("onrender.com") || origin.includes("localhost"))) return origin.replace(/\/$/, "");
    } catch (e) {}
    return DEFAULT_BASE;
  }

  async function ping() {
    const pill = byId("wm-api-pill") || byId("api-status") || byId("apiStatus");
    if (pill) pill.textContent = "API: checking…";
    try {
      const res = await fetch(apiBase() + "/health");
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error("HTTP " + res.status);
      if (pill) pill.textContent = (data && data.ok) ? "API: OK" : "API: issue";
    } catch (e) {
      if (pill) pill.textContent = "API: offline";
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ping);
  } else {
    ping();
  }
})();
