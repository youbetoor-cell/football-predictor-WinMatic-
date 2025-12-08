// static/assets/metrics.js
document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = window.location.origin.replace(/\/$/, "");

  const leagueSelect = document.getElementById("leagueMetricsSelector");
  const viewSelect = document.getElementById("viewSelector");
  const refreshBtn = document.getElementById("refreshMetricsBtn");

  const strengthView = document.getElementById("strengthView");
  const historyView = document.getElementById("historyView");

  const strengthMatchesEl = document.getElementById("strengthMatches");
  const strengthTableBody = document.getElementById("strengthTableBody");

  const historyCountEl = document.getElementById("historyCount");
  const historyList = document.getElementById("historyList");

  // shared theme toggle (same as predictor)
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    if (localStorage.getItem("theme") === "neon") {
      document.body.classList.add("theme-neon");
    }

    themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("theme-neon");
      if (document.body.classList.contains("theme-neon")) {
        localStorage.setItem("theme", "neon");
      } else {
        localStorage.setItem("theme", "default");
      }
    });
  }

  function setView(view) {
    if (view === "strength") {
      strengthView.classList.remove("hidden");
      historyView.classList.add("hidden");
    } else {
      strengthView.classList.add("hidden");
      historyView.classList.remove("hidden");
    }
  }

  async function loadTeamStrength() {
    const league = leagueSelect ? leagueSelect.value : "39";
    strengthTableBody.innerHTML = `<tr><td colspan="8">Loading…</td></tr>`;
    strengthMatchesEl.textContent = "--";

    try {
      const res = await fetch(
        `${API_BASE}/team-strength?league=${league}&limit=1000`
      );
      const data = await res.json();

      if (!data.ok || !data.teams || !data.teams.length) {
        strengthTableBody.innerHTML = `<tr><td colspan="8">No strength data yet. Generate predictions first.</td></tr>`;
        strengthMatchesEl.textContent = "0";
        return;
      }

      const teams = data.teams;
      const totalMatches = teams.reduce((acc, t) => acc + t.matches, 0);
      strengthMatchesEl.textContent = String(totalMatches);

      strengthTableBody.innerHTML = teams
        .map((t, idx) => {
          return `
            <tr>
              <td>${idx + 1}</td>
              <td>${t.team_name}</td>
              <td>${t.matches}</td>
              <td>${t.gf}</td>
              <td>${t.ga}</td>
              <td>${t.attack_strength.toFixed(2)}</td>
              <td>${t.defense_strength.toFixed(2)}</td>
              <td>${t.rating.toFixed(2)}</td>
            </tr>
          `;
        })
        .join("");
    } catch (err) {
      console.error(err);
      strengthTableBody.innerHTML = `<tr><td colspan="8">Error loading team strength.</td></tr>`;
      strengthMatchesEl.textContent = "0";
    }
  }

  function fmtDateTime(isoStr) {
    try {
      const d = new Date(isoStr);
      return (
        d.toLocaleDateString("en-GB", {
          day: "2-digit",
          month: "short",
          year: "numeric",
        }) +
        " " +
        d.toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    } catch {
      return isoStr;
    }
  }

  function shortEdgeLabel(preds) {
    const homeG = preds.home_goals ?? 0;
    const awayG = preds.away_goals ?? 0;
    if (homeG > awayG + 0.7) return "Home Edge";
    if (awayG > homeG + 0.7) return "Away Edge";
    return "Even";
  }

  async function loadHistory() {
    const league = leagueSelect ? leagueSelect.value : "39";
    historyList.innerHTML = "Loading…";
    historyCountEl.textContent = "--";

    try {
      const res = await fetch(`${API_BASE}/history?league=${league}&limit=100`);
      const data = await res.json();

      if (!data.ok || !data.fixtures || !data.fixtures.length) {
        historyList.innerHTML =
          "No history yet. Generate predictions on the Predictor page first.";
        historyCountEl.textContent = "0";
        return;
      }

      const fixtures = data.fixtures;
      historyCountEl.textContent = String(fixtures.length);

      historyList.innerHTML = fixtures
        .map((fx) => {
          const preds = fx.predictions || {};
          const edge = shortEdgeLabel(preds);
          const dt = fmtDateTime(fx.kickoff_utc);

          return `
            <div class="history-card">
              <div class="history-main-row">
                <div class="history-teams">
                  <span class="history-team history-team--home">${fx.home}</span>
                  <span class="history-vs">vs</span>
                  <span class="history-team history-team--away">${fx.away}</span>
                </div>
                <div class="history-meta">
                  <span>${dt}</span>
                  <span class="history-edge">${edge}</span>
                </div>
              </div>
              <div class="history-stats">
                <span>Predicted xG: ${preds.home_goals?.toFixed(2) ?? "–"} : ${
            preds.away_goals?.toFixed(2) ?? "–"
          }</span>
                <span>Shots on target: ${preds.home_sot ?? "–"} : ${
            preds.away_sot ?? "–"
          }</span>
                <span>Corners: ${preds.home_corners ?? "–"} : ${
            preds.away_corners ?? "–"
          }</span>
              </div>
            </div>
          `;
        })
        .join("");
    } catch (err) {
      console.error(err);
      historyList.innerHTML = "Error loading history.";
      historyCountEl.textContent = "0";
    }
  }

  async function refreshCurrentView() {
    const view = viewSelect ? viewSelect.value : "strength";
    setView(view);

    if (view === "strength") {
      await loadTeamStrength();
    } else {
      await loadHistory();
    }
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", refreshCurrentView);
  }

  if (viewSelect) {
    viewSelect.addEventListener("change", refreshCurrentView);
  }

    // ==========================
  // 🔢 PnL / Kelly performance
  // ==========================
  const pnlTotalBetsEl = document.getElementById("pnlTotalBets");
  const pnlTotalProfitEl = document.getElementById("pnlTotalProfit");
  const pnlRoiPerBetEl = document.getElementById("pnlRoiPerBet");
  const pnlList = document.getElementById("pnlList");
  const refreshPnlBtn = document.getElementById("refreshPnlBtn");
  const pnlChartCanvas = document.getElementById("pnlChart");

  let pnlChart = null;

  function fmtDateTime(dtStr) {
    if (!dtStr) return "";
    const d = new Date(dtStr);
    if (Number.isNaN(d.getTime())) return dtStr;
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function updatePnlChart(points) {
    if (!pnlChartCanvas || !window.Chart) return;
    const labels = points.map((pt) => pt.index);
    const cumProfit = points.map((pt) => pt.cum_profit);

    const config = {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Cumulative profit (units)",
            data: cumProfit,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                `Bankroll: ${ctx.parsed.y.toFixed(2)}u (bet #${ctx.parsed.x})`,
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Bet index",
            },
          },
          y: {
            title: {
              display: true,
              text: "Cumulative profit (units)",
            },
          },
        },
      },
    };

    if (pnlChart) {
      pnlChart.destroy();
    }
    pnlChart = new Chart(pnlChartCanvas.getContext("2d"), config);
  }

  async function loadPnlHistory() {
    if (!pnlList) return; // metrics.html might not have this section

    const league = leagueSelect ? leagueSelect.value : "39";
    pnlList.innerHTML = '<div class="empty">Loading PnL history…</div>';

    try {
      const res = await fetch(
        `${API_BASE}/metrics/pnl-history?league=${league}&min_edge=0.05`
      );
      const data = await res.json();

      if (!data.ok || !data.points || !data.points.length) {
        pnlList.innerHTML =
          '<div class="empty">No settled value bets yet. Once actual_result is filled in predictions_history, we will plot PnL here.</div>';
        if (pnlTotalBetsEl) pnlTotalBetsEl.textContent = "0";
        if (pnlTotalProfitEl) pnlTotalProfitEl.textContent = "0.00";
        if (pnlRoiPerBetEl) pnlRoiPerBetEl.textContent = "0%";
        if (pnlChart) {
          pnlChart.destroy();
          pnlChart = null;
        }
        return;
      }

      if (pnlTotalBetsEl) pnlTotalBetsEl.textContent = String(data.n_bets);
      if (pnlTotalProfitEl)
        pnlTotalProfitEl.textContent = data.total_profit.toFixed(2);
      if (pnlRoiPerBetEl)
        pnlRoiPerBetEl.textContent = (data.roi_flat * 100).toFixed(1) + "%";

      // 🧠 Chart
      updatePnlChart(data.points);

      // 📃 List of bets
      pnlList.innerHTML = data.points
        .map((pt) => {
          const date = fmtDateTime(pt.kickoff_utc);
          const outcome = pt.win ? "WIN" : "LOSS";
          const outcomeColor = pt.win ? "#bbf7d0" : "#fecaca";
          const edgePct = (pt.edge_value * 100).toFixed(1);

          return `
            <article class="match-card expanded">
              <div class="match-header">
                <div class="match-team match-team-home">
                  <div class="team-name">${pt.home_team}</div>
                </div>
                <div class="match-center">
                  <div class="headline">
                    <span>${pt.bet_side.toUpperCase()} @ ${pt.odds_est.toFixed(
                      2
                    )}</span>
                  </div>
                  <div class="kickoff">${date}</div>
                  <div class="match-value-pill">
                    <span>${edgePct}% edge</span>
                    <span>Kelly: ${(pt.kelly_frac * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div class="match-team match-team-away">
                  <div class="team-name">${pt.away_team}</div>
                </div>
              </div>
              <div class="match-body expanded">
                <div class="hero-stats-row">
                  <div class="hero-stat">
                    <strong style="color:${outcomeColor}">${outcome}</strong>
                    <span>· Profit: ${pt.profit.toFixed(2)}u</span>
                  </div>
                  <div class="hero-stat">
                    <span>Bankroll: ${pt.cum_profit.toFixed(2)}u</span>
                  </div>
                  <div class="hero-stat">
                    <span>ROI so far: ${(pt.roi * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </article>
          `;
        })
        .join("");
    } catch (err) {
      console.error("Error loading PnL history:", err);
      pnlList.innerHTML =
        '<div class="empty">Error loading PnL history. Check the backend logs.</div>';
    }
  }

  if (refreshPnlBtn) {
    refreshPnlBtn.addEventListener("click", loadPnlHistory);
  }


  // initial load
if (viewSelect && strengthView && historyView) {
  refreshCurrentView();
}
loadPnlHistory();
});

