
(function () {
  const btn = document.getElementById("contact-widget-button");
  const panel = document.getElementById("contact-widget-panel");
  const overlay = document.getElementById("contact-widget-overlay");
  const closeBtn = document.getElementById("contact-widget-close");

  if (!btn || !panel || !overlay || !closeBtn) return;

  function toggle(open) {
    const isOpen = open !== undefined ? open : panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !isOpen);
    overlay.classList.toggle("hidden", !isOpen);
    btn.setAttribute("aria-expanded", String(isOpen));
  }

  btn.addEventListener("click", () => toggle());
  closeBtn.addEventListener("click", () => toggle(false));
  overlay.addEventListener("click", () => toggle(false));

  document.addEventListener("keydown", (e) => { if (e.key === "Escape") toggle(false); });
})();
