/* WinMatic site helpers (schema-tolerant + never-stuck loading) */
(function () {
  const WM = (window.WM = window.WM || {});

  WM.pick = function (obj, keys, fallback = null) {
    try {
      for (const k of keys) {
        if (!obj) continue;
        const v = obj[k];
        if (v !== undefined && v !== null && v !== "") return v;
      }
    } catch (e) {}
    return fallback;
  };

  WM.teamName = function (fx, side) {
    // side: 'home'|'away'
    return WM.pick(fx, [`${side}_name`, `${side}_team`, side], "") ||
      WM.pick(fx?.teams?.[side] || null, ["name"], "") ||
      WM.pick(fx?.fixture?.[side] || null, ["name"], "");
  };

  WM.teamId = function (fx, side) {
    return WM.pick(fx, [`${side}_id`, `${side}Id`], null) ||
      WM.pick(fx?.teams?.[side] || null, ["id"], null);
  };

  WM.teamLogo = function (fx, side) {
    const direct = WM.pick(fx, [`${side}_logo`, `${side}Logo`], null) ||
      WM.pick(fx?.teams?.[side] || null, ["logo"], null);
    if (direct) return direct;
    const id = WM.teamId(fx, side);
    return id ? `/team-logo/${id}.png` : "/team-logo/default.png";
  };

  WM.kickoff = function (fx) {
    return WM.pick(fx, ["kickoff_utc", "kickoff", "date"], "") ||
      WM.pick(fx?.fixture || null, ["date"], "");
  };

  WM.fixtureId = function (fx) {
    return WM.pick(fx, ["fixture_id", "id"], null) ||
      WM.pick(fx?.fixture || null, ["id"], null);
  };

  WM.num = function (v, fallback = null) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  };

  WM.pct = function (v, digits = 1) {
    const n = WM.num(v, null);
    if (n === null) return "—";
    return (n * 100).toFixed(digits) + "%";
  };

  WM.sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  WM.apiGet = async function (path, params = {}, opts = {}) {
    const timeoutMs = opts.timeoutMs ?? 20000;
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params || {})) {
      if (v === undefined || v === null || v === "") continue;
      qs.set(k, String(v));
    }
    const url = qs.toString() ? `${path}?${qs}` : path;

    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(url, { signal: controller.signal, headers: opts.headers || {} });
      const text = await res.text();
      let data = null;
      try { data = JSON.parse(text); } catch (e) {}
      if (!res.ok) {
        const msg = (data && (data.detail || data.error)) ? (data.detail || data.error) : (text || res.statusText);
        throw new Error(`${res.status} ${msg}`.trim());
      }
      return data ?? {};
    } finally {
      clearTimeout(t);
    }
  };

  WM.hideLoading = function () {
    const ids = ["loading", "spinner", "page-loading", "overlay-loading"];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) el.style.display = "none";
    }
    document.querySelectorAll("[data-loading]").forEach((el) => (el.style.display = "none"));
  };

  WM.showError = function (msg) {
    const m = (msg && msg.message) ? msg.message : String(msg || "Unknown error");
    // Try page-specific containers first
    const candidates = ["error", "status", "message", "toast", "alert"];
    for (const id of candidates) {
      const el = document.getElementById(id);
      if (el) {
        el.textContent = m;
        el.style.display = "block";
        el.style.color = "crimson";
        return;
      }
    }
    // Create a top banner if nothing exists
    let banner = document.getElementById("__wm_error_banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "__wm_error_banner";
      banner.style.position = "sticky";
      banner.style.top = "0";
      banner.style.zIndex = "9999";
      banner.style.padding = "10px 12px";
      banner.style.borderBottom = "1px solid rgba(0,0,0,0.1)";
      banner.style.background = "rgba(255, 235, 238, 0.98)";
      banner.style.color = "#b00020";
      banner.style.fontFamily = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
      banner.style.fontSize = "14px";
      document.body.prepend(banner);
    }
    banner.textContent = m;
  };

  // Global handlers so pages NEVER stay stuck on "loading"
  window.addEventListener("error", function (e) {
    WM.hideLoading();
    WM.showError(e?.error || e?.message || "Script error");
  });
  window.addEventListener("unhandledrejection", function (e) {
    WM.hideLoading();
    WM.showError(e?.reason || "Unhandled promise rejection");
  });
})();
