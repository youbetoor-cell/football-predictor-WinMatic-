(function(){
  const LINKS = [
    { href: "/static/index.html",     label: "Home",      icon: "home" },
    { href: "/static/predictor.html", label: "Predict",   icon: "spark" },
    { href: "/static/value.html",     label: "Value",     icon: "bolt" },
    { href: "/static/results.html",   label: "Results",   icon: "chart" },
  ];

  function iconSvg(name){
    const common = 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
    if(name === "home"){
      return `<svg viewBox="0 0 24 24" ${common}><path d="M3 10.5 12 3l9 7.5"/><path d="M5 10v10h14V10"/></svg>`;
    }
    if(name === "spark"){
      return `<svg viewBox="0 0 24 24" ${common}><path d="M12 2l1.2 5.2L18 9l-4.8 1.8L12 16l-1.2-5.2L6 9l4.8-1.8L12 2z"/><path d="M4 14l.8 3.2L8 18l-3.2.8L4 22l-.8-3.2L0 18l3.2-.8L4 14z"/></svg>`;
    }
    if(name === "bolt"){
      return `<svg viewBox="0 0 24 24" ${common}><path d="M13 2 3 14h8l-1 8 10-12h-8l1-8z"/></svg>`;
    }
    return `<svg viewBox="0 0 24 24" ${common}><path d="M4 19V5"/><path d="M20 19V5"/><path d="M4 12h16"/><path d="M8 8h8"/><path d="M8 16h8"/></svg>`;
  }

  function currentPath(){
    // normalize: keep only /static/xxx.html
    const p = location.pathname || "";
    if (p.includes("/static/")) return p;
    return "/static/predictor.html";
  }

  function setActiveLink(el, href){
    const p = currentPath();
    const active = p.endsWith(href.replace("/static/","/static/"));
    if(active) el.setAttribute("aria-current", "page");
  }

  function ensureMetaTheme(){
    if (!document.querySelector('meta[name="theme-color"]')) {
      const m = document.createElement("meta");
      m.name = "theme-color";
      m.content = "#020617";
      document.head.appendChild(m);
    }
  }

  function wrapTables(){
    const tables = Array.from(document.querySelectorAll("table"));
    for(const t of tables){
      if (t.closest(".wm-table-scroll")) continue;
      const wrap = document.createElement("div");
      wrap.className = "wm-table-scroll";
      t.parentNode.insertBefore(wrap, t);
      wrap.appendChild(t);
    }
  }

  function addBottomNav(){
    if (document.querySelector(".wm-bottom-nav")) return;
    const nav = document.createElement("nav");
    nav.className = "wm-bottom-nav";
    nav.setAttribute("aria-label", "Bottom navigation");

    for(const l of LINKS){
      const a = document.createElement("a");
      a.href = l.href;
      a.innerHTML = `${iconSvg(l.icon)}<span>${l.label}</span>`;
      setActiveLink(a, l.href);
      nav.appendChild(a);
    }
    document.body.appendChild(nav);
  }

  function addDrawer(){
    if (document.querySelector(".wm-drawer")) return;

    const scrim = document.createElement("div");
    scrim.className = "wm-scrim";
    scrim.addEventListener("click", () => document.body.classList.remove("wm-nav-open"));

    const drawer = document.createElement("aside");
    drawer.className = "wm-drawer";
    drawer.setAttribute("aria-label", "Menu");

    const title = document.createElement("div");
    title.className = "wm-drawer-title";
    title.textContent = "WinMatic Menu";
    drawer.appendChild(title);

    for(const l of LINKS){
      const a = document.createElement("a");
      a.href = l.href;
      a.innerHTML = `${iconSvg(l.icon)}<span>${l.label}</span>`;
      setActiveLink(a, l.href);
      drawer.appendChild(a);
    }

    const close = document.createElement("button");
    close.className = "wm-close";
    close.type = "button";
    close.textContent = "Close";
    close.addEventListener("click", () => document.body.classList.remove("wm-nav-open"));
    drawer.appendChild(close);

    document.body.appendChild(scrim);
    document.body.appendChild(drawer);

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") document.body.classList.remove("wm-nav-open");
    });
  }

  function addHamburger(){
    // Try to attach into existing topbar; otherwise we skip (bottom nav still works)
    const topbar = document.querySelector(".topbar");
    if (!topbar) return;
    if (topbar.querySelector(".wm-nav-toggle")) return;

    const btn = document.createElement("button");
    btn.className = "wm-nav-toggle";
    btn.type = "button";
    btn.setAttribute("aria-label", "Open menu");
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h16"/></svg>`;
    btn.addEventListener("click", () => document.body.classList.toggle("wm-nav-open"));

    // Put it on the right side of topbar, or end of brand row
    const right = topbar.querySelector(".topbar-right");
    if (right) {
      topbar.insertBefore(btn, right);
    } else {
      topbar.appendChild(btn);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    ensureMetaTheme();
    wrapTables();
    addDrawer();
    addHamburger();
    addBottomNav();
  });
})();
