import { api } from "../api.js";
import { el, entityMemoryHash } from "./utils.js";

const MONTH_NAMES = [
  "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
  "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
];

export async function render(root) {
  root.innerHTML = "";

  const now = new Date();
  const state = {
    year: now.getFullYear(),
    month: now.getMonth(),
    recap: null,
    loading: true,
  };

  const wrap = el("section", { class: "recap" });
  const header = el("div", { class: "recap-header" });
  const body = el("div", { class: "recap-body" });
  wrap.appendChild(header);
  wrap.appendChild(body);
  root.appendChild(wrap);

  async function load() {
    state.loading = true;
    paintHeader();
    paintBody();
    try {
      const month = `${state.year}-${String(state.month + 1).padStart(2, "0")}`;
      state.recap = await api.recap(month);
    } catch (err) {
      console.error(err);
      state.recap = { error: err.message };
    }
    state.loading = false;
    paintHeader();
    paintBody();
  }

  function changeMonth(delta) {
    const next = new Date(state.year, state.month + delta, 1);
    state.year = next.getFullYear();
    state.month = next.getMonth();
    load();
  }

  function paintHeader() {
    header.innerHTML = "";
    header.appendChild(el("button", {
      class: "recap-nav",
      "aria-label": "Önceki ay",
      onclick: () => changeMonth(-1),
    }, ["‹"]));
    header.appendChild(el("div", { class: "recap-month" }, [
      el("div", { class: "recap-month-title" }, [`${MONTH_NAMES[state.month]} ${state.year}`]),
      el("div", { class: "recap-month-sub" }, [
        state.loading ? "Özet hazırlanıyor..." : "Aylık hafıza kartı",
      ]),
    ]));
    header.appendChild(el("button", {
      class: "recap-nav",
      "aria-label": "Sonraki ay",
      onclick: () => changeMonth(1),
    }, ["›"]));
  }

  function paintBody() {
    body.innerHTML = "";
    if (state.loading) {
      body.appendChild(el("div", { class: "recap-loading" }, [
        el("span", { class: "loading" }),
        el("span", {}, ["Anılar taranıyor..."]),
      ]));
      return;
    }
    if (state.recap?.error) {
      body.appendChild(el("div", { class: "empty-state" }, [
        el("div", { class: "empty-title" }, ["Özet yüklenemedi"]),
        el("p", { class: "muted" }, [state.recap.error]),
      ]));
      return;
    }

    const recap = state.recap;
    body.appendChild(hero(recap));
    body.appendChild(statsGrid(recap.stats || {}));
    body.appendChild(topLists(recap));
    body.appendChild(highlights(recap.highlights || []));
  }

  load();
}

function hero(recap) {
  const stats = recap.stats || {};
  const hasActivity = Number(stats.sessions || 0) > 0;
  return el("section", { class: `recap-hero ${hasActivity ? "" : "empty"}` }, [
    el("div", { class: "recap-kicker" }, ["Bu ayın hikayesi"]),
    el("h2", { class: "recap-title" }, [recap.title || "Henüz sessiz"]),
    el("p", { class: "recap-summary" }, [recap.summary || "Bu ay için kayıt bulunamadı."]),
    hasActivity
      ? el("button", {
          class: "cta secondary",
          onclick: () => (location.hash = "#/timeline"),
        }, ["Takvimde gör"])
      : el("button", {
          class: "cta",
          onclick: () => (location.hash = "#/home"),
        }, ["İlk kaydı ekle"]),
  ]);
}

function statsGrid(stats) {
  const items = [
    ["Kayıt", stats.sessions || 0],
    ["Olay", stats.events || 0],
    ["Aktif gün", stats.active_days || 0],
    ["Kişi", stats.people || 0],
    ["Yer", stats.places || 0],
    ["Bekleyen", stats.unresolved_conflicts || 0],
  ];
  return el("section", { class: "recap-stat-grid" },
    items.map(([label, value]) => el("div", { class: "recap-stat" }, [
      el("div", { class: "recap-stat-value" }, [value]),
      el("div", { class: "recap-stat-label" }, [label]),
    ]))
  );
}

function topLists(recap) {
  return el("section", { class: "recap-columns" }, [
    rankedList("İnsanlar", recap.top_people || [], "Henüz kişi yok", "PERSON"),
    rankedList("Yerler", recap.top_places || [], "Henüz yer yok", "LOCATION"),
    rankedList("Yoğun Günler", formatDays(recap.busiest_days || []), "Henüz gün yok"),
  ]);
}

function rankedList(title, items, emptyText, kind) {
  return el("div", { class: "recap-panel" }, [
    el("h3", {}, [title]),
    items.length
      ? el("div", { class: "recap-rank-list" },
          items.slice(0, 5).map((item, index) => rankedItem(item, index, kind))
        )
      : el("p", { class: "muted" }, [emptyText]),
  ]);
}

function rankedItem(item, index, kind) {
  return el("div", { class: "recap-rank" }, [
    el("span", { class: "recap-rank-num" }, [index + 1]),
    kind && item.label
      ? el("button", {
          class: "recap-rank-label entity-link",
          type: "button",
          onclick: () => (location.hash = entityMemoryHash(kind, item.display_label || item.label)),
        }, [item.display_label || item.label])
      : el("span", { class: "recap-rank-label" }, [item.display_label || item.label]),
    el("span", { class: "recap-rank-count" }, [item.count]),
  ]);
}

function formatDays(days) {
  return days.map((day) => {
    const date = new Date(`${day.date}T00:00:00`);
    return {
      label: date.toLocaleDateString("tr-TR", { day: "numeric", month: "short" }),
      count: day.count,
    };
  });
}

function highlights(items) {
  return el("section", { class: "recap-highlights" }, [
    el("h3", {}, ["Öne çıkan anlar"]),
    items.length
      ? el("div", { class: "recap-memory-list" }, items.map(memoryCard))
      : el("div", { class: "empty-state", style: "padding:18px;" }, [
          el("p", { class: "muted" }, ["Bu ay öne çıkarılacak olay yok."]),
        ]),
  ]);
}

function memoryCard(item) {
  const date = item.date
    ? new Date(`${item.date}T00:00:00`).toLocaleDateString("tr-TR", {
        day: "numeric",
        month: "long",
      })
    : "";
  const people = Array.isArray(item.people)
    ? item.people.filter((p) => p && String(p).toLowerCase() !== "ben").slice(0, 3)
    : [];
  return el("article", { class: "recap-memory" }, [
    el("div", { class: "recap-memory-meta" }, [
      date,
      item.time ? ` · ${item.time}` : "",
      item.place
        ? el("button", {
            class: "entity-link recap-inline-link",
            type: "button",
            onclick: () => (location.hash = entityMemoryHash("LOCATION", item.place)),
          }, [item.place])
        : "",
    ]),
    el("p", {}, [item.text || "Olay"]),
    people.length
      ? el("div", { class: "recap-tags" }, people.map((p) => el("button", {
          class: "entity-link recap-tag",
          type: "button",
          onclick: () => (location.hash = entityMemoryHash("PERSON", p)),
        }, [p])))
      : null,
    item.session_id
      ? el("button", {
          class: "recap-link",
          onclick: () => (location.hash = `#/review/${item.session_id}`),
        }, ["Kayda git"])
      : null,
  ]);
}
