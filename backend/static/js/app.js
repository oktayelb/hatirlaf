// Root controller: tiny hash-based router + online indicator + sync kickoff.

import { flush } from "./sync.js";
import { emit, on, toast } from "./events.js";
import { pendingStore } from "./db.js";
import * as home from "./screens/home.js";
import * as record from "./screens/record.js";
import * as review from "./screens/review.js";
import * as timeline from "./screens/timeline.js";
import * as nodes from "./screens/nodes.js";

const TITLES = {
  home: "Hatırlaf",
  record: "Yeni Kayıt",
  review: "İnceleme",
  timeline: "Kişisel Takvim",
  nodes: "Düğümler",
  node: "Düğüm",
};

const routes = [
  { pattern: /^#\/?$|^#\/home$/, screen: "home", render: home.render },
  { pattern: /^#\/record$/, screen: "record", render: record.render },
  { pattern: /^#\/review\/(\d+)$/, screen: "review", params: ["id"], render: review.render },
  { pattern: /^#\/timeline$/, screen: "timeline", render: timeline.render },
  { pattern: /^#\/nodes$/, screen: "nodes", render: nodes.render },
  { pattern: /^#\/nodes\/(\d+)$/, screen: "node", params: ["id"], render: nodes.renderDetail },
];

const screenRoot = document.getElementById("screen-root");
const titleEl = document.getElementById("screen-title");
const backBtn = document.querySelector(".app-back");
const onlineDot = document.getElementById("online-dot");
const tabs = Array.from(document.querySelectorAll(".app-tabbar .tab"));

backBtn.addEventListener("click", () => history.back());

for (const tab of tabs) {
  tab.addEventListener("click", () => {
    location.hash = tab.dataset.route;
  });
}

on("pending-changed", updatePendingChip);
on("online-changed", updateOnline);

async function updatePendingChip() {
  const n = await pendingStore.count();
  const existing = document.getElementById("pending-chip");
  if (existing) existing.remove();
  if (n <= 0) return;
  const tpl = document.getElementById("tpl-pending-chip").content.cloneNode(true);
  const chip = tpl.querySelector(".pending-chip");
  chip.id = "pending-chip";
  chip.querySelector(".pending-text").textContent = `${n} kayıt gönderilmeyi bekliyor`;
  document.querySelector(".app-header").appendChild(chip);
}

function updateOnline(isOnline) {
  if (isOnline === undefined) isOnline = navigator.onLine;
  onlineDot.textContent = isOnline ? "●" : "○";
  onlineDot.classList.toggle("offline", !isOnline);
  onlineDot.title = isOnline ? "Çevrimiçi" : "Çevrimdışı";
}

async function route() {
  const hash = location.hash || "#/home";
  // Allow a previously-rendered screen to clean up (stop polling, release streams).
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
  for (const tab of tabs) {
    const tabScreen = (tab.dataset.route || "").replace("#/", "");
    tab.classList.toggle("active", tabScreen === matched.screen);
  }
  try {
    await matched.render(screenRoot, { params });
  } catch (err) {
    console.error(err);
    screenRoot.innerHTML = "";
    screenRoot.appendChild(
      Object.assign(document.createElement("div"), {
        className: "empty-state",
        innerHTML: `<div class="big-icon">⚠️</div><p>Ekran yüklenemedi: ${err.message}</p>`,
      })
    );
    toast("Hata: " + err.message);
  }
  // Re-render pending chip on each screen switch.
  updatePendingChip();
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", () => {
  if (!location.hash) location.hash = "#/home";
  updateOnline(navigator.onLine);
  updatePendingChip();
  route();
  flush();
});
