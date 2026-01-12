(function(){
  const KEY = "wm_client_key";

  const getKey = () => localStorage.getItem(KEY) || "";
  const setKey = (v) => localStorage.setItem(KEY, v);
  const clearKey = () => localStorage.removeItem(KEY);

  async function verify(code){
    const res = await fetch("/auth/verify-code", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({code})
    });
    if (!res.ok) return {ok:false};
    return await res.json();
  }

  function openModal(){
    const existing = document.querySelector(".wm-premium-modal");
    if (existing) existing.remove();

    const wrap = document.createElement("div");
    wrap.className = "wm-premium-modal";
    wrap.innerHTML = `
      <div class="wm-premium-card">
        <div class="wm-premium-head">
          <div class="wm-premium-title">Unlock Premium</div>
          <button class="wm-premium-x" aria-label="Close">✕</button>
        </div>
        <div class="wm-premium-sub">Enter your access code.</div>
        <input class="wm-premium-input" placeholder="e.g. WM-XXXX-XXXX" autocomplete="off" />
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
      msg.textContent = "Code removed.";
      const topBtn = document.querySelector(".wm-upgrade-btn");
      if (topBtn) topBtn.textContent = "Sign in / Upgrade";
      document.documentElement.classList.remove("wm-premium");
    };

    cta.onclick = async () => {
      const code = (input.value || "").trim();
      msg.textContent = "Checking…";
      cta.disabled = true;
      const out = await verify(code).catch(()=>({ok:false}));
      cta.disabled = false;

      if (out && out.ok) {
        setKey(code);
        msg.textContent = "Unlocked ✅";
        document.documentElement.classList.add("wm-premium");
        const topBtn = document.querySelector(".wm-upgrade-btn");
        if (topBtn) topBtn.textContent = "Premium ✓";
        setTimeout(()=>wrap.remove(), 450);
      } else {
        msg.textContent = "Invalid code.";
      }
    };

    input.value = getKey();
    input.focus();
  }

  function mountButton(){
    const btn = document.createElement("button");
    btn.className = "wm-upgrade-btn";
    btn.type = "button";
    btn.textContent = getKey() ? "Premium ✓" : "Sign in / Upgrade";
    btn.addEventListener("click", openModal);
    document.body.appendChild(btn);
  }

  function ensurePreviewBanner(data){
    // show banner on Value + anywhere value endpoints are used
    if (!data || !data.locked) return;

    const existing = document.querySelector(".wm-preview-banner");
    if (existing) return;

    const banner = document.createElement("div");
    banner.className = "wm-preview-banner";
    const shown = (data.count ?? (data.fixtures && data.fixtures.length) ?? 0);
    const limit = data.preview_limit ?? 3;

    banner.innerHTML = `
      <div class="wm-preview-text">
        <strong>Preview mode</strong> — showing top ${shown} (limit ${limit}). Unlock Premium to see all value bets.
      </div>
      <button class="wm-preview-cta" type="button">Unlock</button>
    `;
    banner.querySelector(".wm-preview-cta").onclick = openModal;

    // Try to insert near top of content
    const target = document.querySelector("main, .main, .wm-landing-main, body");
    if (target) target.prepend(banner);
  }

  // Attach X-Client-Key to same-origin requests + sniff value responses for banner
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
    if (getKey()) document.documentElement.classList.add("wm-premium");
    mountButton();
  });

  // expose for other scripts
  window.WM_Premium = { getKey, openModal };
})();
