// landing.js
// Home page metrics pulled from /model-info.
// Make sure you've run /train at least once so /model-info has data.

const DEFAULT_LEAGUE = 39;

function $(id) {
  return document.getElementById(id);
}

function formatPercent(x, decimals = 0) {
  if (x == null || isNaN(x)) return "–";
  const pct = x * 100;
  return pct.toFixed(decimals) + "%";
}

function formatNumber(x) {
  if (x == null || isNaN(x)) return "–";
  return x.toLocaleString();
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

async function loadMetrics() {
  try {
    // Use /model-info as the single source of truth
    const resp = await fetchJson(`/model-info?league=${DEFAULT_LEAGUE}`);

    // Your earlier JSON looked like: { ok: true, info: { league, seasons, metrics: {...} } }
    const info = resp.info || resp;
    const mRoot = info.metrics || info;

    const hitModel = mRoot.hit_rate_actual ?? mRoot.hit_rate_expected;
    const hitMarket = mRoot.market_hit_rate ?? mRoot.baseline_hit_rate;
    const edge = mRoot.edge_vs_market ?? mRoot.edge_vs_baseline;
    const samplesTest = mRoot.samples_test;
    const samplesTrain = mRoot.samples_train ?? mRoot.samples_total;
    const logloss = mRoot.logloss_1x2;
    const brier = mRoot.brier_1x2;

    if ($("metric-hit-model"))
      $("metric-hit-model").textContent = formatPercent(hitModel);
    if ($("metric-hit-ring"))
      $("metric-hit-ring").textContent = formatPercent(hitModel);
    if ($("metric-hit-market"))
      $("metric-hit-market").textContent = formatPercent(hitMarket);
    if ($("metric-edge"))
      $("metric-edge").textContent =
        edge == null || isNaN(edge) ? "–" : `+${(edge * 100).toFixed(0)}%`;
    if ($("metric-samples"))
      $("metric-samples").textContent = formatNumber(samplesTest);
    if ($("metric-samples-train"))
      $("metric-samples-train").textContent = formatNumber(samplesTrain);
    if ($("metric-logloss"))
      $("metric-logloss").textContent =
        logloss == null || isNaN(logloss) ? "–" : logloss.toFixed(3);
    if ($("metric-brier"))
      $("metric-brier").textContent =
        brier == null || isNaN(brier) ? "–" : brier.toFixed(3);
  } catch (err) {
    console.warn("[landing] Failed to load /model-info metrics:", err);
    // If this fails, we just leave the placeholders as "–"
  }
}

// Scroll reveal – show sections by adding the CSS class your stylesheet expects
function initReveal() {
  const elements = document.querySelectorAll(".wm-reveal");

  if (!("IntersectionObserver" in window)) {
    elements.forEach((el) => el.classList.add("wm-reveal--visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("wm-reveal--visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  elements.forEach((el) => observer.observe(el));
}

document.addEventListener("DOMContentLoaded", () => {
  initReveal();
  loadMetrics();
});
