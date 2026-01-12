/* WinMatic Match Modal (Option 2)
   - Clicking a .wm-match-card opens a modal overlay
   - Prevents inline expansion behavior
*/
(function () {
  function isBlockedTarget(el) {
    return !!el.closest("a, button, input, select, textarea, label");
  }

  function closeModal() {
    const m = document.querySelector(".wm-match-modal-overlay");
    if (m) m.remove();
    document.documentElement.classList.remove("wm-modal-open");
  }

  function openModalFromCard(card) {
    closeModal();

    // Clone card so we keep all existing markup, but show expanded section
    const clone = card.cloneNode(true);
    clone.classList.add("wm-modal-card");
    clone.classList.add("is-expanded"); // important: show .wm-match-expanded content
    clone.querySelectorAll(".wm-card-toggle").forEach(n => n.remove()); // remove chevron if present

    const overlay = document.createElement("div");
    overlay.className = "wm-match-modal-overlay";
    overlay.innerHTML = `
      <div class="wm-match-modal" role="dialog" aria-modal="true" aria-label="Match details">
        <div class="wm-match-modal-head">
          <div class="wm-match-modal-title">Match details</div>
          <button class="wm-match-modal-x" type="button" aria-label="Close">✕</button>
        </div>
        <div class="wm-match-modal-body"></div>
      </div>
    `;

    overlay.querySelector(".wm-match-modal-body").appendChild(clone);

    // Close actions
    overlay.querySelector(".wm-match-modal-x").addEventListener("click", closeModal);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeModal();
    });

    document.addEventListener("keydown", function onKey(e) {
      if (e.key === "Escape") {
        document.removeEventListener("keydown", onKey);
        closeModal();
      }
    });

    document.body.appendChild(overlay);
    document.documentElement.classList.add("wm-modal-open");
  }

  // CAPTURE click early so predictor.js inline expand never fires
  document.addEventListener(
    "click",
    (e) => {
      const card = e.target.closest(".wm-match-card");
      if (!card) return;
      if (isBlockedTarget(e.target)) return;

      // If you only want modal on mobile, uncomment:
      // if (window.innerWidth > 900) return;

      e.preventDefault();
      e.stopPropagation();
      if (e.stopImmediatePropagation) e.stopImmediatePropagation();

      openModalFromCard(card);
    },
    true // capture
  );

  // Optional: prevent body scroll behind modal
  document.addEventListener(
    "touchmove",
    (e) => {
      const overlay = document.querySelector(".wm-match-modal-overlay");
      if (!overlay) return;
      // allow scroll inside modal only
      if (!e.target.closest(".wm-match-modal")) e.preventDefault();
    },
    { passive: false }
  );
})();
