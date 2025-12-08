// ---------------------------------------------------------
// WINMATIC – Premium Predictor Frontend
// Big glowing cards, expandable hero view, real stats
// ---------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  const leagueSelect = document.getElementById("leagueSelect");
  const dateInput = document.getElementById("dateSelect");
  const loadBtn = document.getElementById("loadMatchesBtn");
  const matchesContainer = document.getElementById("matchesContainer");
  const matchesSubtitle = document.getElementById("matchesSubtitle");
  const emptyEl = document.getElementById("matchesEmpty");
  const errorEl = document.getElementById("matchesError");
  const confidenceText = document.getElementById("modelConfidenceValue");
  const confidenceDonut = document.querySelector(".wm-accuracy-donut");

  if (!leagueSelect || !dateInput || !loadBtn || !matchesContainer) {
    console.error("Predictor DOM not found");
    return;
  }

  // Auto-set today
  if (!dateInput.value) {
    dateInput.value = new Date().toISOString().split("T")[0];
  }

  const API_BASE = window.location.origin;

  // ----------------- Helpers -----------------
  async function fetchJSON(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("HTTP " + res.status);
      return await res.json();
    } catch (err) {
      console.error("Fetch error:", err);
      return null;
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      loadBtn.classList.add("is-loading");
      loadBtn.disabled = true;
      loadBtn.textContent = "Loading...";
    } else {
      loadBtn.classList.remove("is-loading");
      loadBtn.disabled = false;
      loadBtn.textContent = "Load Matches";
    }
  }

  function clearState() {
    matchesContainer.innerHTML = "";
    emptyEl.style.display = "none";
    errorEl.style.display = "none";
  }

  function leagueNameFromId(id) {
    const map = {
      39: "Premier League",
      140: "La Liga",
      78: "Bundesliga",
      135: "Serie A",
      61: "Ligue 1",
      2: "Championship",
      94: "Primeira Liga",
      19: "Süper Lig",
      848: "Champions League",
      849: "Europa League",
      850: "Conference League",
    };
    const num = Number(id);
    return map[num] || "League " + id;
  }

  function formatKickoff(utc) {
    if (!utc) {
      return { date: "--", time: "--" };
    }
    const d = new Date(utc);
    const date = d.toLocaleDateString(undefined, {
      weekday: "short",
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
    const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return { date, time };
  }

  function shortDateLabel(utcOrIso) {
    if (!utcOrIso) return "--";
    const d = new Date(utcOrIso);
    return d.toLocaleDateString(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  }

  function toPercent(x) {
    if (typeof x !== "number" || !isFinite(x)) return 0;
    if (x <= 1) return Math.round(x * 100);
    return Math.round(x);
  }

  // ----------------- Confidence Ring -----------------
  function updateConfidence(fixtures) {
    if (!confidenceText || !confidenceDonut) return;

    if (!fixtures || fixtures.length === 0) {
      confidenceText.textContent = "--%";
      confidenceDonut.style.setProperty("--confidence-deg", "0deg");
      return;
    }

    let sum = 0;
    fixtures.forEach((fx) => {
      const p = fx.predictions || fx;
      const h = typeof p.home_win_p === "number" ? p.home_win_p : p.prob_home_win || 0;
      const d = typeof p.draw_p === "number" ? p.draw_p : p.prob_draw || 0;
      const a = typeof p.away_win_p === "number" ? p.away_win_p : p.prob_away_win || 0;
      const best = Math.max(h, d, a);
      sum += best;
    });

    const avg = sum / fixtures.length;
    const pct = Math.max(0, Math.min(100, Math.round(avg * 100)));

    confidenceText.textContent = `${pct}%`;
    confidenceDonut.style.setProperty("--confidence-deg", `${pct * 3.6}deg`);
  }

  // ----------------- Rendering -----------------
  function renderMatches(fixtures, leagueId, dateIso, rangeInfo) {
    clearState();

    if (!fixtures || fixtures.length === 0) {
      emptyEl.style.display = "block";
      matchesSubtitle.textContent = `0 matches • ${leagueNameFromId(
        leagueId
      )} • ${shortDateLabel(dateIso || (rangeInfo && rangeInfo.from))}`;
      updateConfidence([]);
      return;
    }

    const labelDate = dateIso || (rangeInfo && rangeInfo.from) || fixtures[0].kickoff_utc;

    matchesSubtitle.textContent = `${fixtures.length} matches • ${leagueNameFromId(
      leagueId
    )} • ${shortDateLabel(labelDate)}`;

    fixtures.forEach((fx, index) => {
      const card = createMatchCard(fx, index === 0);
      matchesContainer.appendChild(card);
    });

    updateConfidence(fixtures);
  }

  function createMatchCard(fx, expandedInitially = false) {
    const card = document.createElement("article");
    card.className = "wm-match-card";
    if (expandedInitially) card.classList.add("is-expanded");

    const p = fx.predictions || fx;

    const homeName = fx.home_name || fx.home || "Home";
    const awayName = fx.away_name || fx.away || "Away";

    const homeLogo = fx.home_logo || (fx.home_id ? `/team-logo/${fx.home_id}.png` : "/team-logo/default.png");
    const awayLogo = fx.away_logo || (fx.away_id ? `/team-logo/${fx.away_id}.png` : "/team-logo/default.png");

    const homeGoals = typeof p.home_goals === "number" ? p.home_goals : fx.pred_home_goals || 0;
    const awayGoals = typeof p.away_goals === "number" ? p.away_goals : fx.pred_away_goals || 0;

    const homeP = toPercent(p.home_win_p ?? p.prob_home_win);
    const awayP = toPercent(p.away_win_p ?? p.prob_away_win);

    const kickoff = formatKickoff(fx.kickoff_utc);

    card.innerHTML = `
      <div class="wm-match-shell">
        <div class="wm-match-row">
          <!-- HOME -->
          <div class="wm-team-col wm-team-col--home">
            <div class="wm-orb">
              <div class="wm-orb-inner">
                <img src="${homeLogo}" alt="${homeName} logo" loading="lazy">
              </div>
            </div>
            <div class="wm-team-meta">
              <span class="wm-team-label">Home</span>
              <span class="wm-team-name">${homeName}</span>
              <span class="wm-team-prob">${homeP}%</span>
            </div>
          </div>

          <!-- CENTER -->
          <div class="wm-match-center">
            <div class="wm-kickoff-date">${kickoff.date}</div>
            <div class="wm-scoreline">${homeGoals.toFixed(1)} – ${awayGoals.toFixed(1)}</div>
            <div class="wm-kickoff-time">${kickoff.time}</div>
          </div>

          <!-- AWAY -->
          <div class="wm-team-col wm-team-col--away">
            <div class="wm-orb wm-orb--away">
              <div class="wm-orb-inner">
                <img src="${awayLogo}" alt="${awayName} logo" loading="lazy">
              </div>
            </div>
            <div class="wm-team-meta">
              <span class="wm-team-label">Away</span>
              <span class="wm-team-name">${awayName}</span>
              <span class="wm-team-prob">${awayP}%</span>
            </div>
          </div>
        </div>

        <!-- EXPANDED -->
        <div class="wm-match-expanded">
          ${buildStatsHtml(p)}
          ${buildPlayersHtml(fx.players_to_score || fx.likely_scorers || [])}
        </div>
      </div>
    `;

    // Expand/collapse on click
    card.addEventListener("click", () => {
      const isExpanded = card.classList.contains("is-expanded");
      document.querySelectorAll(".wm-match-card.is-expanded").forEach((c) => {
        if (c !== card) c.classList.remove("is-expanded");
      });
      if (!isExpanded) {
        card.classList.add("is-expanded");
      } else {
        card.classList.remove("is-expanded");
      }
    });

    return card;
  }

  function buildStatsHtml(p) {
    function safe(n, decimals = 2) {
      if (typeof n !== "number" || !isFinite(n)) return "--";
      return n.toFixed(decimals);
    }

    const homeXg = safe(p.home_goals);
    const awayXg = safe(p.away_goals);

    const homeSot = safe(p.home_sot);
    const awaySot = safe(p.away_sot);

    const homeCorners = safe(p.home_corners);
    const awayCorners = safe(p.away_corners);

    const homeY = safe(p.home_yellows, 2);
    const awayY = safe(p.away_yellows, 2);

    const homeR = safe(p.home_reds, 2);
    const awayR = safe(p.away_reds, 2);

    return `
      <div class="wm-stats-grid">
        <div class="wm-stat-box">
          <div class="wm-stat-label">xG</div>
          <div class="wm-stat-value">${homeXg} – ${awayXg}</div>
        </div>
        <div class="wm-stat-box">
          <div class="wm-stat-label">Shots on Target</div>
          <div class="wm-stat-value">${homeSot} – ${awaySot}</div>
        </div>
        <div class="wm-stat-box">
          <div class="wm-stat-label">Corners</div>
          <div class="wm-stat-value">${homeCorners} – ${awayCorners}</div>
        </div>
        <div class="wm-stat-box">
          <div class="wm-stat-label">Yellows</div>
          <div class="wm-stat-value">${homeY} – ${awayY}</div>
        </div>
        <div class="wm-stat-box">
          <div class="wm-stat-label">Reds</div>
          <div class="wm-stat-value">${homeR} – ${awayR}</div>
        </div>
      </div>
    `;
  }

  function buildPlayersHtml(players) {
    if (!players || !players.length) {
      return `
        <div class="wm-players-row wm-players-row--empty">
          <span>No standout scorers projected.</span>
        </div>
      `;
    }

    const chips = players
      .slice(0, 4)
      .map((pl) => {
        const name = pl.name || "Player";
        const team = pl.team || "";
        const photo = pl.photo || "/static/team-logo/default.png";
        const p = toPercent(pl.xg_anytime || pl.prob || 0);

        return `
          <div class="wm-player-chip">
            <div class="wm-player-avatar">
              <img src="${photo}" alt="${name}" loading="lazy">
            </div>
            <div class="wm-player-meta">
              <span class="wm-player-name">${name}</span>
              <span class="wm-player-team">${team}</span>
            </div>
            <div class="wm-player-prob">${p}%</div>
          </div>
        `;
      })
      .join("");

    return `
      <div class="wm-players-wrapper">
        <h3 class="wm-players-title">Players to score</h3>
        <div class="wm-players-row">
          ${chips}
        </div>
      </div>
    `;
  }

  // ----------------- Load handler -----------------
  async function loadMatches() {
    const league = leagueSelect.value;
    const date = dateInput.value;

    clearState();
    setLoading(true);

    let url;
    if (date) {
      url = `${API_BASE}/predict/by-date?league=${encodeURIComponent(
        league
      )}&from_date=${date}&to_date=${date}`;
    } else {
      url = `${API_BASE}/predict/upcoming?league=${encodeURIComponent(
        league
      )}&days_ahead=7`;
    }

    const data = await fetchJSON(url);

    setLoading(false);

    if (!data || data.ok === false) {
      errorEl.style.display = "block";
      updateConfidence([]);
      return;
    }

    renderMatches(data.fixtures || [], league, date, data.range);
  }

  // Bind
  loadBtn.addEventListener("click", (e) => {
    e.preventDefault();
    loadMatches();
  });

  // Auto-load on open
  loadMatches();
});
