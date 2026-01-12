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
  window.fetch = async function(input, init){
    let url = "";
    try { url = (typeof input === "string") ? input : (input && input.url) || ""; } catch {}
    init = init || {};
    init.headers = init.headers || {};

    const k = getKey();
    if (k && (url.startsWith("/") || url.startsWith(window.location.origin))) {
      init.headers["X-Client-Key"] = k;
    }

    const resp = await _fetch(input, init);

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
