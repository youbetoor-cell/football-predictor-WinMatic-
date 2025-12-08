// static/js/status.js
async function updateApiStatus() {
  const dot = document.getElementById("api-dot");
  const text = document.getElementById("api-text");

  if (!dot || !text) return;

  try {
    const res = await fetch("http://127.0.0.1:8000/value-bets?league=39&days_ahead=1&min_edge=0.05");
    if (res.ok) {
      const data = await res.json();
      if (data && data.meta && data.meta.cache_mode === "live") {
        dot.style.backgroundColor = "#00c853"; // green
        text.textContent = "API: live";
      } else {
        dot.style.backgroundColor = "#ffa000"; // amber
        text.textContent = "API: cached";
      }
    } else {
      dot.style.backgroundColor = "#ffa000";
      text.textContent = "API: cached";
    }
  } catch (err) {
    dot.style.backgroundColor = "#d32f2f"; // red
    text.textContent = "API: offline";
  }
}

function initApiStatus() {
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
  updateApiStatus();
  setInterval(updateApiStatus, 60000);
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", initApiStatus);
