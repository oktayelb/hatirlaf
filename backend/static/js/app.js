// Root controller: tiny hash-based router + online indicator + sync kickoff.

import { flush } from "./sync.js";
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
});
