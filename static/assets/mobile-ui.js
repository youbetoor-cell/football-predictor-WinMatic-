/* WM Mobile UI helpers (cards + affordance) */
(function () {
  const isMobile = () => {
    try { return window.matchMedia("(max-width: 600px)").matches; }
    catch { return (window.innerWidth || 9999) <= 600; }
  };

  function collapseAll() {
    if (!isMobile()) return;
    document.querySelectorAll(".wm-match-card.is-expanded").forEach(c => c.classList.remove("is-expanded"));
  }

  function ensureToggle(card) {
    if (card.querySelector(".wm-card-toggle")) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "wm-card-toggle";
    btn.setAttribute("aria-label", "Toggle match details");
    btn.innerHTML = `
      <span class="wm-card-toggle-text">Details</span>
      <span class="wm-card-toggle-ico" aria-hidden="true">⌄</span>
    `;

    // Prefer putting it inside the shell if present
    const shell = card.querySelector(".wm-match-shell") || card;
    shell.style.position = shell.style.position || "relative";
    shell.appendChild(btn);
  }

  function wireCards() {
    const cards = document.querySelectorAll(".wm-match-card");
    cards.forEach(ensureToggle);

    document.addEventListener("click", (e) => {
      const toggle = e.target.closest(".wm-card-toggle");
      const card = e.target.closest(".wm-match-card");
      if (!card) return;

      // Only toggle on: (a) chevron OR (b) card tap not on links/buttons
      if (!toggle) {
        const blocked = e.target.closest("a, button, input, select, textarea, label");
        if (blocked) return;
      }

      const wasExpanded = card.classList.contains("is-expanded");

      // one-at-a-time
      document.querySelectorAll(".wm-match-card.is-expanded").forEach(c => {
        if (c !== card) c.classList.remove("is-expanded");
      });

      card.classList.toggle("is-expanded", !wasExpanded);

      const btn = card.querySelector(".wm-card-toggle");
      if (btn) {
        const expanded = card.classList.contains("is-expanded");
        btn.querySelector(".wm-card-toggle-text").textContent = expanded ? "Hide" : "Details";
        btn.querySelector(".wm-card-toggle-ico").textContent = expanded ? "⌃" : "⌄";
      }
    }, { passive: true });
  }

  document.addEventListener("DOMContentLoaded", () => {
    collapseAll();
    wireCards();
    // after dynamic renders
    setTimeout(() => { collapseAll(); wireCards(); }, 250);
    setTimeout(() => { collapseAll(); wireCards(); }, 900);
  });
})();
