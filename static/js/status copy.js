// static/js/value.js
// Drives the Value tab UI using the /value-bets endpoint

const VALUE_CONFIG = {
    defaultLeague: 39,      // Premier League
    defaultDaysAhead: 7,    // look ahead 7 days
    defaultMinEdge: 0.05    // 5% edge
};

function initValueScanner() {
    const root =
        document.getElementById("value-root") ||
        createDefaultValueLayout();

    const tableBody = root.querySelector("tbody");
    const statusEl = root.querySelector(".value-status");
    const refreshBtn = root.querySelector("#value-refresh-btn");
    const leagueInput = root.querySelector("#value-league");
    const daysInput = root.querySelector("#value-days");
    const edgeInput = root.querySelector("#value-min-edge");

    function setStatus(text) {
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    function buildUrl() {
        const league = parseInt(leagueInput.value || VALUE_CONFIG.defaultLeague, 10);
        const days = parseInt(daysInput.value || VALUE_CONFIG.defaultDaysAhead, 10);
        const edge = parseFloat(edgeInput.value || VALUE_CONFIG.defaultMinEdge);

        const params = new URLSearchParams({
            league: String(league),
            days_ahead: String(days),
            min_edge: String(edge)
        });

        return "/value-bets?" + params.toString();
    }

    async function loadValueBets() {
        if (!tableBody) return;

        tableBody.innerHTML = "";
        setStatus("Loading value bets...");

        try {
            const url = buildUrl();
            const resp = await fetch(url);

            if (!resp.ok) {
                throw new Error("HTTP " + resp.status);
            }

            const data = await resp.json();
            const bets = normalizeBets(data);

            if (!bets || bets.length === 0) {
                setStatus("No value bets found for the current filters.");
                return;
            }

            bets.forEach((bet) => {
                const tr = document.createElement("tr");

                const kickoff = formatKickoff(bet.kickoff || bet.datetime || bet.time);
                const teams = (bet.home_team || bet.home || bet.homeName || "?")
                    + " vs "
                    + (bet.away_team || bet.away || bet.awayName || "?");
                const market = bet.market || bet.bet_type || "1X2";
                const selection = bet.selection || bet.outcome || bet.pick || "-";
                const odds = bet.odds || bet.price || bet.bookmaker_odds || null;
                const modelProb = bet.model_prob || bet.model_probability || null;
                const edge = bet.edge || bet.value || null;
                const bookmaker = bet.bookmaker || bet.book || "";

                tr.innerHTML = `
                    <td>${kickoff}</td>
                    <td>${escapeHtml(teams)}</td>
                    <td>${escapeHtml(market)}</td>
                    <td>${escapeHtml(selection)}</td>
                    <td>${odds !== null ? Number(odds).toFixed(2) : "-"}</td>
                    <td>${modelProb !== null ? (Number(modelProb) * 100).toFixed(1) + "%" : "-"}</td>
                    <td class="edge-cell">${edge !== null ? (Number(edge) * 100).toFixed(1) + "%" : "-"}</td>
                    <td>${escapeHtml(bookmaker)}</td>
                `;

                tableBody.appendChild(tr);
            });

            setStatus("Loaded " + bets.length + " value bets.");
        } catch (err) {
            console.error("[VALUE] Load error", err);
            setStatus("Could not load value bets (check API / quota).");
        }
    }

    if (refreshBtn) {
        refreshBtn.addEventListener("click", (e) => {
            e.preventDefault();
            loadValueBets();
        });
    }

    loadValueBets();
}

function normalizeBets(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    if (Array.isArray(data.bets)) return data.bets;
    if (Array.isArray(data.value_bets)) return data.value_bets;
    if (Array.isArray(data.results)) return data.results;
    return [];
}

function formatKickoff(value) {
    if (!value) return "-";
    try {
        const d = new Date(value);
        if (isNaN(d.getTime())) return String(value);
        return d.toLocaleString(undefined, {
            weekday: "short",
            hour: "2-digit",
            minute: "2-digit",
            month: "short",
            day: "numeric"
        });
    } catch (_) {
        return String(value);
    }
}

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function createDefaultValueLayout() {
    const container = document.createElement("div");
    container.id = "value-root";
    container.className = "value-root";

    container.innerHTML = `
        <div class="value-header">
            <h2>Value Scanner</h2>
            <div class="value-filters">
                <label>
                    League ID
                    <input id="value-league" type="number" value="${VALUE_CONFIG.defaultLeague}" />
                </label>
                <label>
                    Days ahead
                    <input id="value-days" type="number" value="${VALUE_CONFIG.defaultDaysAhead}" />
                </label>
                <label>
                    Min edge
                    <input id="value-min-edge" type="number" step="0.01" value="${VALUE_CONFIG.defaultMinEdge}" />
                </label>
                <button id="value-refresh-btn">Refresh</button>
            </div>
        </div>
        <p class="value-status"></p>
        <div class="value-table-wrapper">
            <table class="value-table">
                <thead>
                    <tr>
                        <th>Kickoff</th>
                        <th>Match</th>
                        <th>Market</th>
                        <th>Selection</th>
                        <th>Odds</th>
                        <th>Model prob</th>
                        <th>Edge</th>
                        <th>Bookmaker</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    `;

    document.body.appendChild(container);
    return container;
}

document.addEventListener("DOMContentLoaded", initValueScanner);
