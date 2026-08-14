// Root controller: hash router, navigation, boot sequence.
//
// The navigation is not hard-coded. It is derived from the server's feature
// flags at boot, so with the NLP pipeline switched off the app really is
// two screens — there is no hidden route to stumble into.

import { flush } from "./sync.js";
import { api } from "./api.js";
import { loadConfig, nlpEnabled } from "./config.js";
import { on, toast } from "./events.js";
import { icon } from "./icons.js";
import { initReminderTimers } from "./reminders.js";
import { initTextSize } from "./textsize.js";
import * as home from "./screens/home.js";
import * as entries from "./screens/entries.js";
import * as recap from "./screens/recap.js";
import * as memories from "./screens/memories.js";
import * as review from "./screens/review.js";
import * as settings from "./screens/settings.js";
import * as timeline from "./screens/timeline.js";

/* ---------- Screen registry ----------
   `feature: null` means always available. Anything else names the flag it
   depends on. */

const SCREENS = [
  {
    name: "home",
    title: "Hatırlaf",
    tab: { label: "Ana", icon: "home" },
    hash: "#/home",
    pattern: /^#\/?$|^#\/home$/,
    render: home.render,
    module: home,
  },
  {
    name: "entries",
    title: "Günlüğüm",
    tab: { label: "Günlüğüm", icon: "book" },
    hash: "#/entries",
    pattern: /^#\/entries$/,
    render: entries.render,
    module: entries,
  },
  {
    name: "timeline",
    title: "Takvim",
    tab: { label: "Takvim", icon: "calendar" },
    hash: "#/timeline",
    pattern: /^#\/timeline$/,
    render: timeline.render,
    module: timeline,
    feature: "nlp",
  },
  {
    name: "recap",
    title: "Özet",
    tab: { label: "Özet", icon: "sparkle" },
    hash: "#/recap",
    pattern: /^#\/recap$/,
    render: recap.render,
    module: recap,
    feature: "nlp",
  },
  {
    name: "memories",
    title: "Anılar",
    pattern: /^#\/memories\/([^/]+)\/(.+)$/,
    params: ["kind", "label"],
    render: memories.render,
    module: memories,
    feature: "nlp",
    titleOf: (params) => decodeURIComponent(params.label || "") || "Anılar",
  },
  {
    name: "review",
    title: "İnceleme",
    pattern: /^#\/review\/(\d+)$/,
    params: ["id"],
    render: review.render,
    module: review,
    feature: "nlp",
  },
  {
    name: "settings",
    title: "Ayarlar",
    hash: "#/settings",
    pattern: /^#\/settings$/,
    render: settings.render,
    module: settings,
  },
];

const screenRoot = document.getElementById("screen-root");
const titleEl = document.getElementById("screen-title");
const backBtn = document.querySelector(".app-back");
const onlineDot = document.getElementById("online-dot");
const navRoot = document.getElementById("app-nav");
const settingsBtn = document.getElementById("settings-btn");
const startupScreen = document.getElementById("startup-screen");
const startupFill = document.getElementById("startup-progress-fill");
const startupPercent = document.getElementById("startup-percent");

let available = [];
let tabOrder = [];
let currentScreen = "";
let privacyState = { password_enabled: false, unlocked: true };

/* ---------- Boot ---------- */

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", async () => {
  initTextSize();
  updateOnline(navigator.onLine);
  setupSwipeNavigation();

  await waitForStartup();
  await loadConfig();
  buildNavigation();

  // Reminders come from extracted calendar events, so they belong to the
  // NLP layer — a capture-only app must not fire notifications about them.
  if (nlpEnabled()) initReminderTimers();

  if (!location.hash) location.hash = "#/home";
  await refreshPrivacyStatus();
  route();

  if (!isPrivacyLocked()) {
    flush();
    if (nlpEnabled()) startEventificationWatcher();
  }
});

backBtn.addEventListener("click", () => history.back());
settingsBtn.addEventListener("click", () => {
  location.hash = currentScreen === "settings" ? "#/home" : "#/settings";
});

on("online-changed", updateOnline);
on("session-uploaded", () => {
  if (nlpEnabled()) pollEventificationStatuses();
});
on("privacy-locked", () => {
  privacyState = { password_enabled: true, unlocked: false };
  renderPrivacyLock();
});
on("privacy-updated", async () => {
  await refreshPrivacyStatus();
});

function buildNavigation() {
  available = SCREENS.filter((s) => !s.feature || nlpEnabled());
  tabOrder = available.filter((s) => s.tab);

  navRoot.innerHTML = "";
  for (const screen of tabOrder) {
    const button = document.createElement("button");
    button.className = "app-nav-btn";
    button.type = "button";
    button.dataset.screen = screen.name;
    button.appendChild(icon(screen.tab.icon, { size: 26 }));
    button.appendChild(document.createTextNode(screen.tab.label));
    button.addEventListener("click", () => {
      location.hash = screen.hash;
    });
    navRoot.appendChild(button);
  }
}

/* ---------- Routing ---------- */

async function route() {
  const hash = location.hash || "#/home";
  const previous = currentScreen;

  for (const screen of SCREENS) {
    if (typeof screen.module.cleanup === "function") screen.module.cleanup();
  }

  if (isPrivacyLocked()) {
    renderPrivacyLock();
    return;
  }

  let matched = null;
  let params = {};
  for (const screen of available) {
    const m = hash.match(screen.pattern);
    if (!m) continue;
    matched = screen;
    params = {};
    (screen.params || []).forEach((p, i) => (params[p] = m[i + 1]));
    break;
  }
  if (!matched) {
    location.hash = "#/home";
    return;
  }

  titleEl.textContent = matched.titleOf ? matched.titleOf(params) : matched.title;
  currentScreen = matched.name;
  backBtn.hidden = matched.name === "home";
  settingsBtn.classList.toggle("active", matched.name === "settings");
  for (const button of navRoot.querySelectorAll(".app-nav-btn")) {
    button.classList.toggle("active", button.dataset.screen === matched.name);
  }

  try {
    await matched.render(screenRoot, { params });
    animateRoute(previous, matched.name);
  } catch (err) {
    console.error(err);
    screenRoot.innerHTML = "";
    const fallback = document.createElement("div");
    fallback.className = "empty-state";
    const title = document.createElement("div");
    title.className = "empty-title";
    title.textContent = "Bu sayfa açılamadı";
    const detail = document.createElement("p");
    detail.textContent = err.message;
    fallback.append(title, detail);
    screenRoot.appendChild(fallback);
  }
}

function animateRoute(previous, next) {
  screenRoot.classList.remove("route-enter", "route-forward", "route-back");
  if (!previous || previous === next) return;
  const names = tabOrder.map((s) => s.name);
  const from = names.indexOf(previous);
  const to = names.indexOf(next);
  const direction = from >= 0 && to >= 0 && to < from ? "route-back" : "route-forward";
  screenRoot.classList.add("route-enter", direction);
  window.requestAnimationFrame(() => {
    window.setTimeout(() => {
      screenRoot.classList.remove("route-enter", "route-forward", "route-back");
    }, 190);
  });
}

/* ---------- Swipe between tabs ---------- */

let touchStartX = 0;
let touchStartY = 0;
let touchStartTime = 0;
let touchTracking = false;

function setupSwipeNavigation() {
  screenRoot.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length !== 1 || isInteractiveTarget(e.target)) return;
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
      touchStartTime = performance.now();
      touchTracking = true;
    },
    { passive: true }
  );

  screenRoot.addEventListener(
    "touchmove",
    (e) => {
      if (!touchTracking || e.touches.length !== 1) return;
      const dx = e.touches[0].clientX - touchStartX;
      const dy = e.touches[0].clientY - touchStartY;
      if (Math.abs(dy) > 32 && Math.abs(dy) > Math.abs(dx)) touchTracking = false;
    },
    { passive: true }
  );

  screenRoot.addEventListener(
    "touchend",
    (e) => {
      if (!touchTracking) return;
      touchTracking = false;
      const names = tabOrder.map((s) => s.name);
      if (!names.includes(currentScreen)) return;
      const dx = e.changedTouches[0].clientX - touchStartX;
      const dy = e.changedTouches[0].clientY - touchStartY;
      const fastEnough = performance.now() - touchStartTime < 520;
      const horizontal = Math.abs(dx) > 72 && Math.abs(dx) > Math.abs(dy) * 1.35;
      if (!fastEnough || !horizontal) return;
      const next = tabOrder[names.indexOf(currentScreen) + (dx < 0 ? 1 : -1)];
      if (next) location.hash = next.hash;
    },
    { passive: true }
  );
}

function isInteractiveTarget(target) {
  return Boolean(
    target?.closest?.(
      "button, a, input, textarea, select, audio, .modal, [contenteditable], [role='button']"
    )
  );
}

/* ---------- Connectivity ---------- */

function updateOnline(isOnline) {
  if (isOnline === undefined) isOnline = navigator.onLine;
  onlineDot.classList.toggle("offline", !isOnline);
  onlineDot.title = isOnline ? "Çevrimiçi" : "Çevrimdışı";
}

/* ---------- Startup screen ---------- */

async function waitForStartup() {
  if (!startupScreen) return;
  let lastProgress = 0;
  // The splash exists to explain a wait. If there was never a wait — no
  // models to warm — it should not manufacture one.
  let sawWaiting = false;

  while (true) {
    if (!navigator.onLine) {
      renderStartup(100);
      break;
    }
    try {
      const health = await api.health();
      const startup = health.startup || { ready: true, progress: 100 };
      lastProgress = Math.max(lastProgress, Number(startup.progress) || 0);
      renderStartup(lastProgress);
      if (startup.ready) break;
      sawWaiting = true;
    } catch (err) {
      console.debug("startup health poll failed", err);
      sawWaiting = true;
      renderStartup(Math.max(lastProgress, 5));
    }
    await sleep(700);
  }

  if (sawWaiting) await sleep(450);
  startupScreen.classList.add("is-done");
  document.body.classList.remove("startup-active");
  setTimeout(() => {
    startupScreen.hidden = true;
  }, 260);
}

function renderStartup(value) {
  const progress = Math.max(0, Math.min(100, Math.round(value || 0)));
  startupFill.style.width = `${progress}%`;
  startupFill.parentElement?.setAttribute("aria-valuenow", String(progress));
  startupPercent.textContent = `${progress}%`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ---------- Privacy lock ---------- */

async function refreshPrivacyStatus() {
  try {
    privacyState = await api.privacyStatus();
  } catch (err) {
    console.debug("privacy status failed", err);
    privacyState = { password_enabled: false, unlocked: true };
  }
}

function isPrivacyLocked() {
  return Boolean(privacyState.password_enabled && !privacyState.unlocked);
}

function renderPrivacyLock() {
  currentScreen = "locked";
  titleEl.textContent = "Kilitli";
  backBtn.hidden = true;
  settingsBtn.classList.remove("active");
  for (const button of navRoot.querySelectorAll(".app-nav-btn")) {
    button.classList.remove("active");
  }
  screenRoot.innerHTML = "";

  const wrap = document.createElement("section");
  wrap.className = "privacy-lock";
  wrap.innerHTML = `
    <div class="privacy-lock-card">
      <div class="privacy-lock-mark" aria-hidden="true">H</div>
      <div>
        <h2 class="privacy-lock-title">Günlüğün kilitli</h2>
        <p class="privacy-lock-copy">Kayıtlarını görmek için parolanı yaz.</p>
      </div>
      <form class="privacy-lock-form">
        <input class="privacy-lock-input" type="password" autocomplete="current-password"
               placeholder="Parola" aria-label="Parola" />
        <button class="cta" type="submit">Kilidi Aç</button>
        <div class="privacy-lock-error" role="alert"></div>
      </form>
    </div>
  `;

  const form = wrap.querySelector("form");
  const input = wrap.querySelector("input");
  const button = wrap.querySelector("button");
  const error = wrap.querySelector(".privacy-lock-error");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    error.textContent = "";
    button.setAttribute("disabled", "");
    button.textContent = "Açılıyor…";
    try {
      privacyState = await api.unlockPrivacy(input.value);
      toast("Kilit açıldı");
      route();
      flush();
      if (nlpEnabled()) startEventificationWatcher();
    } catch (err) {
      error.textContent = readApiError(err);
      button.removeAttribute("disabled");
      button.textContent = "Kilidi Aç";
    }
  });

  screenRoot.appendChild(wrap);
  setTimeout(() => input.focus(), 0);
}

function readApiError(err) {
  const raw = String(err.message || "");
  const match = raw.match(/\{.*\}$/);
  if (!match) return "Hata: " + raw;
  try {
    return JSON.parse(match[0]).detail || raw;
  } catch (_) {
    return "Hata: " + raw;
  }
}

/* ---------- Eventification watcher (NLP only) ---------- */

let eventPollStarted = false;
let eventStatusSnapshot = null;

function startEventificationWatcher() {
  if (eventPollStarted || !nlpEnabled()) return;
  eventPollStarted = true;
  pollEventificationStatuses({ silent: true });
  setInterval(() => pollEventificationStatuses(), 5000);
}

async function pollEventificationStatuses(opts = {}) {
  if (!nlpEnabled() || !navigator.onLine || isPrivacyLocked()) return;

  let sessions = [];
  try {
    const resp = await api.listSessions();
    sessions = Array.isArray(resp) ? resp : resp.results || [];
  } catch (err) {
    console.debug("eventification poll failed", err);
    return;
  }

  const next = new Map(
    sessions.map((s) => [String(s.id), s.eventification_status || "not_started"])
  );

  if (eventStatusSnapshot && !opts.silent) {
    for (const s of sessions) {
      const previous = eventStatusSnapshot.get(String(s.id));
      const current = s.eventification_status || "not_started";
      if (previous && previous !== "completed" && current === "completed") {
        toast("Takvim güncellendi.", { duration: 3200 });
        if ((location.hash || "").startsWith("#/timeline")) route();
      }
    }
  }

  eventStatusSnapshot = next;
}
