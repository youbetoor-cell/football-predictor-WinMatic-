// static/js/status.js
// Lightweight status widget that only hits /health (same-origin).
// Render free tier can cold-start: first request may briefly fail. We retry once.

async function fetchHealthOnce() {
  return fetch("/health", { cache: "no-store" });
}

async function updateApiStatus() {
  const dot = document.getElementById("api-dot");
  const text = document.getElementById("api-text");

  if (!dot || !text) return;

  try {
    let res = await fetchHealthOnce();
    if (!res.ok) {
      // Retry once (cold start / transient proxy)
      await new Promise(r => setTimeout(r, 800));
      res = await fetchHealthOnce();
    }

    if (!res.ok) {
      // Backend up but returned error
      dot.style.backgroundColor = "#ffa000"; // amber
      text.textContent = "API: error";
      return;
    }

    const data = await res.json().catch(() => ({}));

    if (data && data.ok) {
      dot.style.backgroundColor = "#00c853"; // green
      text.textContent = "API: online";
    } else {
      dot.style.backgroundColor = "#ffa000"; // amber
      text.textContent = "API: degraded";
    }
  } catch (err) {
    // Server not reachable
    dot.style.backgroundColor = "#d32f2f"; // red
    text.textContent = "API: offline";
  }
}

function initApiStatus() {
  const existing = document.getElementById("api-status");
  if (existing) return;

  const container = document.createElement("div");
  container.id = "api-status";
  container.style = `
    position: fixed;
    top: 10px;
    right: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: #333;
    z-index: 9999;
  `;

  container.innerHTML = `
    <span id="api-dot" style="
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background-color: gray;
      display: inline-block;
    "></span>
    <span id="api-text">API: checking...</span>
  `;

  document.body.appendChild(container);

  // Call ONCE on load (no setInterval)
  updateApiStatus();
}

document.addEventListener("DOMContentLoaded", initApiStatus);
