(function(){
  const KEY = "wm_client_key";
  const getKey = () => localStorage.getItem(KEY) || "";
  const setKey = (v) => localStorage.setItem(KEY, v);
  const clearKey = () => localStorage.removeItem(KEY);

  async function verify(code){
    const res = await fetch("/auth/verify-tier", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({code})
    });
    if (!res.ok) return {ok:false};
    return await res.json(); // {ok:true, valid:bool, tier:"free|pro|premium"}
  }

  function applyTierUI(tier){
    document.documentElement.classList.toggle("wm-tier-pro", tier === "pro");
    document.documentElement.classList.toggle("wm-tier-premium", tier === "premium");
    const btn = document.querySelector(".wm-upgrade-btn");
    if (btn){
      btn.textContent = tier === "premium" ? "Premium ✓"
        : tier === "pro" ? "Pro ✓"
        : "Sign in / Upgrade";
    }
  }

  function openModal(){
    const existing = document.querySelector(".wm-premium-modal");
    if (existing) existing.remove();

    const wrap = document.createElement("div");
    wrap.className = "wm-premium-modal";
    wrap.innerHTML = `
      <div class="wm-premium-card">
        <div class="wm-premium-head">
          <div class="wm-premium-title">Unlock Pro / Premium</div>
          <button class="wm-premium-x" aria-label="Close">✕</button>
        </div>

        <div class="wm-premium-sub">
          Enter your access code to unlock higher tiers.
          <div class="wm-tier-hint">
            <div><strong>Free</strong>: top 3 value bets</div>
            <div><strong>Pro</strong>: top 10 value bets</div>
            <div><strong>Premium</strong>: full list</div>
          </div>
        </div>

        <input class="wm-premium-input" placeholder="e.g. PRO-XXXX or PREM-XXXX" autocomplete="off" />
        <button class="wm-premium-cta">Verify</button>
        <div class="wm-premium-msg" aria-live="polite"></div>
        <button class="wm-premium-logout">Remove code</button>
      </div>
    `;
    document.body.appendChild(wrap);

    const input = wrap.querySelector(".wm-premium-input");
    const msg = wrap.querySelector(".wm-premium-msg");
    const cta = wrap.querySelector(".wm-premium-cta");
    const x = wrap.querySelector(".wm-premium-x");
    const logout = wrap.querySelector(".wm-premium-logout");

    x.onclick = () => wrap.remove();
    wrap.addEventListener("click", (e)=>{ if(e.target === wrap) wrap.remove(); });

    logout.onclick = () => {
      clearKey();
      msg.textContent = "Code removed. You are on Free.";
      applyTierUI("free");
    };

    cta.onclick = async () => {
      const code = (input.value || "").trim();
      msg.textContent = "Checking…";
      cta.disabled = true;
      const out = await verify(code).catch(()=>({ok:false}));
      cta.disabled = false;

      if (out && out.ok && out.valid) {
        setKey(code);
        msg.textContent = out.tier === "premium" ? "Unlocked Premium ✅" : "Unlocked Pro ✅";
        applyTierUI(out.tier);
        setTimeout(()=>wrap.remove(), 500);
      } else {
        msg.textContent = "Invalid code (Free).";
        applyTierUI("free");
      }
    };

    input.value = getKey();
    input.focus();
  }

  function mountButton(){
    const btn = document.createElement("button");
    btn.className = "wm-upgrade-btn";
    btn.type = "button";
    btn.textContent = "Sign in / Upgrade";
    btn.addEventListener("click", openModal);
    document.body.appendChild(btn);
  }

  // Preview banner logic for value endpoints
  function ensurePreviewBanner(data){
    if (!data || !data.locked) return;

    const existing = document.querySelector(".wm-preview-banner");
    if (existing) return;

    const banner = document.createElement("div");
    banner.className = "wm-preview-banner";
    const shown = (data.count ?? (data.fixtures && data.fixtures.length) ?? 0);
    const plim = data.preview_limit ?? 3;
    const tier = data.tier || "free";

    banner.innerHTML = `
      <div class="wm-preview-text">
        <strong>${tier.toUpperCase()} preview</strong> — showing ${shown} (limit ${plim}).
        Unlock Pro or Premium to see more.
      </div>
      <button class="wm-preview-cta" type="button">Unlock</button>
    `;
    banner.querySelector(".wm-preview-cta").onclick = openModal;

    const target = document.querySelector("main, .main, .wm-landing-main, body");
    if (target) target.prepend(banner);
  }

  // Attach X-Client-Key to same-origin requests + sniff value responses
  const _fetch = window.fetch.bind(window);

// WM_API_BASE_V1 (Pages -> Render API)
const WM_API_BASE = (window.WM_API_BASE || "").replace(/\/$/, "");

function wmAbsUrl(input) {
  try {
    // fetch("/path")
    if (typeof input === "string") {
      if (WM_API_BASE && input.startsWith("/")) return WM_API_BASE + input;
      return input;
    }
    // fetch(Request)
    if (input && typeof input === "object" && input.url) {
      const u = String(input.url);
      // Only rewrite same-origin relative-style URLs (Pages)
      if (WM_API_BASE && u.startsWith(location.origin + "/")) {
        return WM_API_BASE + u.substring(location.origin.length);
      }
    }
  } catch (e) {}
  return input;
}

  window.fetch = async function(input, init){
    let url = "";
    try { url = (typeof input === "string") ? input : (input && input.url) || ""; } catch {}
    init = init || {};
    init.headers = init.headers || {};

    const k = getKey();
    if (k && (url.startsWith("/") || url.startsWith(window.location.origin))) {
      init.headers["X-Client-Key"] = k;
    }

    const resp = await _fetch(wmAbsUrl(input), init);

    try {
      if (url.includes("/value-bets") || url.includes("/value/upcoming")) {
        resp.clone().json().then(ensurePreviewBanner).catch(()=>{});
      }
    } catch {}

    return resp;
  };

  document.addEventListener("DOMContentLoaded", () => {
    mountButton();

    // apply tier class based on stored code (best-effort)
    const k = getKey();
    if (!k) applyTierUI("free");
    else verify(k).then(out => applyTierUI(out && out.valid ? out.tier : "free")).catch(()=>applyTierUI("free"));
  });

  window.WM_Premium = { getKey, openModal };
})();


/* WM_MATCH_MODAL_V1 - Match Details Modal (shared across Predictor/Value) */
(function(){
  function ensureModal(){
    if (document.getElementById("wm-modal-overlay")) return;

    const overlay = document.createElement("div");
    overlay.id = "wm-modal-overlay";
    overlay.className = "wm-modal-overlay";
    overlay.innerHTML = `
      <div class="wm-modal-sheet" role="dialog" aria-modal="true" aria-label="Match details">
        <button class="wm-modal-close" aria-label="Close">×</button>
        <div class="wm-modal-body"></div>
      </div>
    `;
    document.body.appendChild(overlay);

    const close = () => overlay.classList.remove("is-open");
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    overlay.querySelector(".wm-modal-close").addEventListener("click", close);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  }

  function pct(x){
    if (typeof x !== "number") return "—";
    return Math.round(x * 100) + "%";
  }

  function get(obj, ...paths){
    for (const path of paths){
      try{
        let v = obj;
        for (const k of path) v = v?.[k];
        if (v !== undefined && v !== null) return v;
      }catch(e){}
    }
    return undefined;
  }

  window.wmOpenMatchModal = function(fx, opts){
    ensureModal();
    opts = opts || {};

    const overlay = document.getElementById("wm-modal-overlay");
    const body = overlay.querySelector(".wm-modal-body");

    const home = get(fx, ["home_name"], ["home"]) || "Home";
    const away = get(fx, ["away_name"], ["away"]) || "Away";
    const kickoff = get(fx, ["kickoff_utc"], ["kickoff"]) || "";

    const probs = get(fx, ["model_probs"], ["predictions","model_probs"], ["predictions","probs"]) || {};
    const odds  = get(fx, ["bookmaker_odds"], ["odds"]) || {};
    const best  = get(fx, ["value_pick"], ["best_side"], ["best_pick"]) || "—";
    const ev    = (get(fx, ["value_pick_ev"], ["best_edge"]) ?? (get(fx, ["evs", best]) ?? null));

    const reasoning = get(fx, ["reasoning"], ["explanation"]) || "";
    const locked = !!opts.locked;

    body.innerHTML = `
      <div class="wm-modal-head">
        <div class="wm-modal-title">${home} <span>vs</span> ${away}</div>
        <div class="wm-modal-sub">${kickoff}</div>
      </div>

      <div class="wm-modal-kpi">
        <div class="wm-kpi">
          <div class="wm-kpi-label">Best</div>
          <div class="wm-kpi-val">${String(best).toUpperCase()}</div>
        </div>
        <div class="wm-kpi">
          <div class="wm-kpi-label">EV</div>
          <div class="wm-kpi-val">${(typeof ev === "number") ? ev.toFixed(3) : "—"}</div>
        </div>
      </div>

      ${locked ? `
        <div class="wm-modal-card">
          <div class="wm-modal-card-title">Locked</div>
          <div class="wm-modal-text">Upgrade to PRO/PREMIUM to view full value breakdown and all picks for this match.</div>
        </div>
      ` : `
        <div class="wm-modal-card">
          <div class="wm-modal-card-title">Model probabilities</div>
          <div class="wm-modal-grid">
            <div><div class="k">Home</div><div class="v">${pct(probs.home)}</div></div>
            <div><div class="k">Draw</div><div class="v">${pct(probs.draw)}</div></div>
            <div><div class="k">Away</div><div class="v">${pct(probs.away)}</div></div>
          </div>
        </div>

        <div class="wm-modal-card">
          <div class="wm-modal-card-title">Book odds</div>
          <div class="wm-modal-grid">
            <div><div class="k">Home</div><div class="v">${odds.home ?? "—"}</div></div>
            <div><div class="k">Draw</div><div class="v">${odds.draw ?? "—"}</div></div>
            <div><div class="k">Away</div><div class="v">${odds.away ?? "—"}</div></div>
          </div>
        </div>

        ${reasoning ? `<div class="wm-modal-card"><div class="wm-modal-card-title">Reasoning</div><div class="wm-modal-text">${reasoning}</div></div>` : ``}
      `}
    `;

    overlay.classList.add("is-open");
  };
})();
