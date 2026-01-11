const CACHE_VERSION = 'wm-static-v1';
const PRECACHE = [
  '/static/app.css?v=1',
  '/static/app.js?v=1',
  '/static/manifest.json',
  '/static/icons/icon.svg',
  '/static/icons/maskable.svg',
  '/static/index.html',
  '/static/predictor.html',
  '/static/value.html',
  '/static/bets.html',
  '/static/results.html',
  '/static/progress.html',
  '/static/metrics.html',
  '/static/all-matches.html'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then(cache => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(k => (k === CACHE_VERSION ? null : caches.delete(k))))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Only handle /static/ scope
  if (!url.pathname.startsWith('/static/')) return;

  // Network-first for HTML (so updates propagate)
  if (req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html')) {
    event.respondWith(
      fetch(req).then(resp => {
        const copy = resp.clone();
        caches.open(CACHE_VERSION).then(cache => cache.put(req, copy));
        return resp;
      }).catch(() => caches.match(req))
    );
    return;
  }

  // Cache-first for static assets
  event.respondWith(
    caches.match(req).then(hit => hit || fetch(req).then(resp => {
      const copy = resp.clone();
      caches.open(CACHE_VERSION).then(cache => cache.put(req, copy));
      return resp;
    }))
  );
});
