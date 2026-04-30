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
  if (typeof review.cleanup === "function") review.cleanup();
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
window.addEventListener("DOMContentLoaded", () => {
  if (!location.hash) location.hash = "#/home";
  updateOnline(navigator.onLine);
  route();
  flush();
  startEventificationWatcher();
});

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
