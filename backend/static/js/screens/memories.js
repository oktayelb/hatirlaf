import { api } from "../api.js";
import { el, escapeHtml } from "./utils.js";

export async function render(root, ctx) {
  root.innerHTML = "";

  const kind = normalizeKind(ctx.params.kind);
  const label = decodeURIComponent(ctx.params.label || "").trim();
  if (!kind || !label) {
    root.appendChild(emptyState("Anı sayfası yüklenemedi", "Geçersiz kişi veya yer."));
    return;
  }

  const wrap = el("section", { class: "memory-page" }, [
    el("div", { class: "memory-page-hero" }, [
      el("div", { class: "memory-page-kicker" }, [kindLabel(kind)]),
      el("h2", { class: "memory-page-title" }, [label]),
      el("p", { class: "memory-page-sub" }, [
        `${kindLabel(kind)} ile ilişkili tüm anılar burada toplanır.`,
      ]),
    ]),
    el("div", { class: "memory-page-loading" }, [
      el("span", { class: "loading" }),
      el("span", {}, ["Anılar yükleniyor..."]),
    ]),
  ]);
  root.appendChild(wrap);

  let payload;
  try {
    payload = await api.nodeMemories({ kind, label });
  } catch (err) {
    root.innerHTML = "";
    root.appendChild(emptyState("Anılar yüklenemedi", err.message));
    return;
  }

  root.innerHTML = "";
  root.appendChild(
    el("section", { class: "memory-page" }, [
      header(payload),
      stats(payload.stats || {}),
      payload.memories && payload.memories.length
        ? el("div", { class: "memory-list" }, payload.memories.map((item) => memoryCard(item)))
        : emptyState("Henüz anı yok", "Bu kişi veya yer için eşleşen bir kayıt bulunamadı."),
      el("div", { class: "memory-page-actions" }, [
        el("button", {
          class: "cta secondary",
          type: "button",
          onclick: () => (location.hash = "#/timeline"),
        }, ["Takvime dön"]),
        el("button", {
          class: "cta ghost",
          type: "button",
          onclick: () => history.back(),
        }, ["Geri"]),
      ]),
    ])
  );
}

function header(payload) {
  const node = payload.node || {};
  return el("div", { class: "memory-page-hero" }, [
    el("div", { class: "memory-page-kicker" }, [kindLabel(node.kind)]),
    el("h2", { class: "memory-page-title" }, [node.label || "Anılar"]),
    el("p", { class: "memory-page-sub" }, [
      payload.resolved
        ? `Bu sayfa ${node.label} için ayrılmış hafıza izlerini gösterir.`
        : "Bu sayfa, NER tarafından tanınan isim veya yer için çıkarılmış hafızaları gösterir.",
    ]),
  ]);
}

function stats(stats) {
  const items = [
    ["Kayıt", stats.sessions || 0],
    ["Eşleşme", stats.mentions || 0],
    ["Olay", stats.events || 0],
  ];
  return el("div", { class: "memory-stats" },
    items.map(([label, value]) => el("div", { class: "memory-stat" }, [
      el("div", { class: "memory-stat-value" }, [value]),
      el("div", { class: "memory-stat-label" }, [label]),
    ]))
  );
}

function memoryCard(item) {
  const date = item.recorded_at
    ? new Date(item.recorded_at).toLocaleString("tr-TR", {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "";
  const mentions = Array.isArray(item.matched_mentions)
    ? item.matched_mentions.filter((m) => m && m.surface).slice(0, 4)
    : [];
  const events = Array.isArray(item.event_matches) ? item.event_matches : [];

  return el("article", { class: "memory-card" }, [
    el("div", { class: "memory-card-head" }, [
      el("div", { class: "memory-card-date" }, [date]),
      el("button", {
        class: "memory-card-link",
        type: "button",
        onclick: () => (location.hash = `#/review/${item.session_id}`),
      }, ["Kayda git"]),
    ]),
    item.transcript_excerpt
      ? el("p", { class: "memory-card-text", html: highlight(item.transcript_excerpt, item.matched_label) })
      : null,
    mentions.length
      ? el("div", { class: "memory-card-tags" }, mentions.map((m) => tag(m.surface)))
      : null,
    events.length
      ? el("div", { class: "memory-card-events" }, events.map((ev) => eventRow(ev)))
      : null,
  ]);
}

function tag(text) {
  return el("span", { class: "memory-tag" }, [text]);
}

function eventRow(ev) {
  const bits = [ev.date, ev.time].filter(Boolean).join(" · ");
  return el("div", { class: "memory-event" }, [
    el("div", { class: "memory-event-meta" }, [bits]),
    el("div", { class: "memory-event-text" }, [ev.text || ""]),
  ]);
}

function highlight(text, label) {
  const raw = String(text || "");
  const target = String(label || "").trim();
  if (!target) return escapeHtml(raw);
  const parts = raw.split(new RegExp(`(${escapeRegExp(target)})`, "gi"));
  return parts.map((part) => {
    if (part.toLowerCase() === target.toLowerCase()) {
      return `<mark>${escapeHtml(part)}</mark>`;
    }
    return escapeHtml(part);
  }).join("");
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function emptyState(title, text) {
  return el("div", { class: "empty-state memory-empty" }, [
    el("div", { class: "empty-title" }, [title]),
    el("p", { class: "muted" }, [text]),
  ]);
}

function normalizeKind(kind) {
  const upper = String(kind || "").toUpperCase();
  return upper === "PRONOUN" ? "PERSON" : upper;
}

function kindLabel(kind) {
  return {
    PERSON: "Kişi hafızası",
    LOCATION: "Yer hafızası",
    ORG: "Kurum hafızası",
  }[kind] || "Anılar";
}
