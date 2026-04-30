// Root controller: tiny hash-based router + online indicator + sync kickoff.

import { flush } from "./sync.js";
import { api } from "./api.js";
import { on, toast } from "./events.js";
import * as home from "./screens/home.js";
import * as record from "./screens/record.js";
import * as review from "./screens/review.js";
import * as timeline from "./screens/timeline.js";

const TITLES = {
  home: "Hatırlaf",
  record: "Girişler",
  review: "İnceleme",
  timeline: "Takvim",
};

const routes = [
  { pattern: /^#\/?$|^#\/home$/, screen: "home", render: home.render },
  { pattern: /^#\/record$/, screen: "record", render: record.render },
  { pattern: /^#\/review\/(\d+)$/, screen: "review", params: ["id"], render: review.render },
  { pattern: /^#\/timeline$/, screen: "timeline", render: timeline.render },
];
const MAIN_ROUTE_ORDER = ["home", "record", "timeline"];
const MAIN_ROUTE_HASH = {
  home: "#/home",
  record: "#/record",
  timeline: "#/timeline",
};

const screenRoot = document.getElementById("screen-root");
const titleEl = document.getElementById("screen-title");
const backBtn = document.querySelector(".app-back");
const onlineDot = document.getElementById("online-dot");
const navBtns = Array.from(document.querySelectorAll(".app-nav .app-nav-btn"));
const startupScreen = document.getElementById("startup-screen");
const startupFill = document.getElementById("startup-progress-fill");
const startupPercent = document.getElementById("startup-percent");
let currentScreen = "";
let touchStartX = 0;
let touchStartY = 0;
let touchStartTime = 0;
let touchTracking = false;

backBtn.addEventListener("click", () => history.back());

for (const btn of navBtns) {
  btn.addEventListener("click", () => {
    location.hash = btn.dataset.route;
  });
}

on("online-changed", updateOnline);
on("session-uploaded", () => pollEventificationStatuses());

function updateOnline(isOnline) {
  if (isOnline === undefined) isOnline = navigator.onLine;
  onlineDot.classList.toggle("offline", !isOnline);
  onlineDot.title = isOnline ? "Çevrimiçi" : "Çevrimdışı";
}

async function route() {
  const hash = location.hash || "#/home";
  const previousScreen = currentScreen;
  for (const mod of [home, record, review, timeline]) {
    if (typeof mod.cleanup === "function") mod.cleanup();
  }
  let matched = null;
  let params = {};
  for (const r of routes) {
    const m = hash.match(r.pattern);
    if (m) {
      matched = r;
      params = {};
      (r.params || []).forEach((p, i) => (params[p] = m[i + 1]));
      break;
    }
  }
  if (!matched) {
    location.hash = "#/home";
    return;
  }
  titleEl.textContent = TITLES[matched.screen] || "Hatırlaf";
  currentScreen = matched.screen;
  backBtn.hidden = matched.screen === "home";
  for (const btn of navBtns) {
    const btnScreen = (btn.dataset.route || "").replace("#/", "");
    btn.classList.toggle("active", btnScreen === matched.screen);
  }
  try {
    await matched.render(screenRoot, { params });
    animateRoute(previousScreen, matched.screen);
  } catch (err) {
    console.error(err);
    screenRoot.innerHTML = "";
    const fallback = document.createElement("div");
    fallback.className = "empty-state";
    fallback.innerHTML =
      '<div class="empty-title">Ekran yüklenemedi</div>' +
      '<p class="muted">' + escapeHTML(err.message) + "</p>";
    screenRoot.appendChild(fallback);
    toast("Hata: " + err.message);
  }
}

function escapeHTML(s) {
  return String(s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", async () => {
  if (!location.hash) location.hash = "#/home";
  updateOnline(navigator.onLine);
  setupSwipeNavigation();
  await waitForStartup();
  route();
  flush();
  startEventificationWatcher();
});

function setupSwipeNavigation() {
  screenRoot.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1 || isInteractiveTarget(e.target)) return;
    const touch = e.touches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
    touchStartTime = performance.now();
    touchTracking = true;
  }, { passive: true });

  screenRoot.addEventListener("touchmove", (e) => {
    if (!touchTracking || e.touches.length !== 1) return;
    const touch = e.touches[0];
    const dx = touch.clientX - touchStartX;
    const dy = touch.clientY - touchStartY;
    if (Math.abs(dy) > 32 && Math.abs(dy) > Math.abs(dx)) {
      touchTracking = false;
    }
  }, { passive: true });

  screenRoot.addEventListener("touchend", (e) => {
    if (!touchTracking || !MAIN_ROUTE_ORDER.includes(currentScreen)) return;
    touchTracking = false;
    const touch = e.changedTouches[0];
    const dx = touch.clientX - touchStartX;
    const dy = touch.clientY - touchStartY;
    const elapsed = performance.now() - touchStartTime;
    const fastEnough = elapsed < 520;
    const horizontal = Math.abs(dx) > 72 && Math.abs(dx) > Math.abs(dy) * 1.35;
    if (!fastEnough || !horizontal) return;
    goToAdjacentMainRoute(dx < 0 ? 1 : -1);
  }, { passive: true });
}

function isInteractiveTarget(target) {
  return Boolean(target?.closest?.(
    "button, a, input, textarea, select, audio, .modal, [contenteditable], [role='button']"
  ));
}

function goToAdjacentMainRoute(delta) {
  const index = MAIN_ROUTE_ORDER.indexOf(currentScreen);
  if (index < 0) return;
  const next = MAIN_ROUTE_ORDER[index + delta];
  if (!next) return;
  location.hash = MAIN_ROUTE_HASH[next];
}

function animateRoute(previous, next) {
  screenRoot.classList.remove("route-enter", "route-forward", "route-back");
  if (!previous || previous === next) return;
  const prevIndex = MAIN_ROUTE_ORDER.indexOf(previous);
  const nextIndex = MAIN_ROUTE_ORDER.indexOf(next);
  const directionClass = prevIndex >= 0 && nextIndex >= 0 && nextIndex < prevIndex
    ? "route-back"
    : "route-forward";
  screenRoot.classList.add("route-enter", directionClass);
  window.requestAnimationFrame(() => {
    window.setTimeout(() => {
      screenRoot.classList.remove("route-enter", "route-forward", "route-back");
    }, 170);
  });
}

async function waitForStartup() {
  if (!startupScreen) return;
  const minVisibleUntil = performance.now() + 1000;
  let lastProgress = 0;
  while (true) {
    if (!navigator.onLine) {
      renderStartup({
        ready: true,
        progress: 100,
        components: [],
        current: { detail: "Çevrimdışı modda açılıyor." },
      });
      break;
    }
    try {
      const health = await api.health();
      const startup = health.startup || { ready: true, progress: 100, components: [] };
      lastProgress = Math.max(lastProgress, Number(startup.progress) || 0);
      renderStartup({ ...startup, progress: lastProgress });
      if (startup.ready) break;
    } catch (err) {
      console.debug("startup health poll failed", err);
      renderStartup({
        ready: false,
        progress: Math.max(lastProgress, 5),
      });
    }
    await sleep(700);
  }
  const remaining = minVisibleUntil - performance.now();
  if (remaining > 0) await sleep(remaining);
  await sleep(180);
  startupScreen.classList.add("is-done");
  document.body.classList.remove("startup-active");
  setTimeout(() => {
    startupScreen.hidden = true;
  }, 240);
}

function renderStartup(startup) {
  const progress = Math.max(0, Math.min(100, Math.round(startup.progress || 0)));
  startupFill.style.width = `${progress}%`;
  startupFill.parentElement?.setAttribute("aria-valuenow", String(progress));
  startupPercent.textContent = `${progress}%`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let eventPollStarted = false;
let eventStatusSnapshot = null;

function startEventificationWatcher() {
  if (eventPollStarted) return;
  eventPollStarted = true;
  pollEventificationStatuses({ silent: true });
  setInterval(() => pollEventificationStatuses(), 5000);
}

async function pollEventificationStatuses(opts = {}) {
  if (!navigator.onLine) return;
  let sessions = [];
  try {
    const resp = await api.listSessions();
    sessions = Array.isArray(resp) ? resp : resp.results || [];
  } catch (err) {
    console.debug("eventification poll failed", err);
    return;
  }

  const next = new Map();
  for (const s of sessions) {
    next.set(String(s.id), s.eventification_status || "not_started");
  }

  if (eventStatusSnapshot && !opts.silent) {
    for (const s of sessions) {
      const id = String(s.id);
      const prev = eventStatusSnapshot.get(id);
      const cur = s.eventification_status || "not_started";
      if (prev && prev !== "completed" && cur === "completed") {
        toast("Olaylaştırma tamamlandı. Takvim güncellendi.", { duration: 3200 });
        if ((location.hash || "").startsWith("#/timeline")) {
          route();
        }
      }
      if (prev && prev !== "failed" && cur === "failed") {
        toast("Olaylaştırma tamamlanamadı: " + (s.eventification_detail || ""), {
          duration: 4200,
        });
      }
    }
  }

  eventStatusSnapshot = next;
}