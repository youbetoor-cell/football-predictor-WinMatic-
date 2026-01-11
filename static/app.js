(() => {
  const nav = document.querySelector('[data-wm-bottom-nav]');
  if (!nav) return;

  const path = (location.pathname || '').toLowerCase();
  const links = Array.from(nav.querySelectorAll('a[data-route]'));
  let active = null;

  for (const a of links) {
    const route = (a.getAttribute('data-route') || '').toLowerCase();
    if (!route) continue;
    if (path.endsWith('/' + route) || path.endsWith(route)) { active = a; break; }
  }

  if (!active && (path === '/' || path === '')) {
    active = links.find(a => (a.getAttribute('data-route') || '').toLowerCase() === 'index.html') || null;
  }

  if (active) active.dataset.active = "1";
})();
