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

const screenRoot = document.getElementById("screen-root");
const titleEl = document.getElementById("screen-title");
const backBtn = document.querySelector(".app-back");
const onlineDot = document.getElementById("online-dot");
const navBtns = Array.from(document.querySelectorAll(".app-nav .app-nav-btn"));
const startupScreen = document.getElementById("startup-screen");
const startupDetail = document.getElementById("startup-detail");
const startupFill = document.getElementById("startup-progress-fill");
const startupPercent = document.getElementById("startup-percent");
const startupComponents = document.getElementById("startup-components");

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
  backBtn.hidden = matched.screen === "home";
  for (const btn of navBtns) {
    const btnScreen = (btn.dataset.route || "").replace("#/", "");
    btn.classList.toggle("active", btnScreen === matched.screen);
  }
  try {
    await matched.render(screenRoot, { params });
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
  await waitForStartup();
  route();
  flush();
  startEventificationWatcher();
});

async function waitForStartup() {
  if (!startupScreen) return;
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
        components: [],
        current: { detail: "Backend yanıtı bekleniyor." },
      });
    }
    await sleep(700);
  }
  await sleep(180);
  startupScreen.classList.add("is-done");
  document.body.classList.remove("startup-active");
  setTimeout(() => {
    startupScreen.hidden = true;
  }, 240);
}

function renderStartup(startup) {
  const progress = Math.max(0, Math.min(100, Math.round(startup.progress || 0)));
  const current = startup.current;
  startupDetail.textContent = current?.detail || (startup.ready ? "Hazır." : "Modeller yükleniyor.");
  startupFill.style.width = `${progress}%`;
  startupFill.parentElement?.setAttribute("aria-valuenow", String(progress));
  startupPercent.textContent = `${progress}%`;
  startupComponents.innerHTML = "";
  for (const component of startup.components || []) {
    const row = document.createElement("div");
    row.className = `startup-component ${component.status || "pending"}`;
    const label = document.createElement("span");
    label.className = "startup-component-label";
    label.textContent = component.label || component.key;
    const status = document.createElement("span");
    status.className = "startup-component-status";
    status.textContent = startupStatusText(component.status);
    const meter = document.createElement("span");
    meter.className = "startup-component-meter";
    const meterFill = document.createElement("span");
    meterFill.style.width = `${Math.max(0, Math.min(100, component.progress || 0))}%`;
    meter.appendChild(meterFill);
    row.title = component.detail || "";
    row.append(label, meter, status);
    startupComponents.appendChild(row);
  }
}

function startupStatusText(status) {
  if (status === "ready") return "Hazır";
  if (status === "loading") return "Yükleniyor";
  if (status === "skipped") return "Kapalı";
  if (status === "failed") return "Yedek";
  return "Bekliyor";
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
