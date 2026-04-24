// Personal History — real monthly calendar. Each day is a box that
// surfaces the events extracted by the NLP + LLM pipeline. Tapping a
// box opens a drawer with the full list for that day (people included).

import { api } from "../api.js";
import { el, modal } from "./utils.js";

const MONTH_NAMES = [
  "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
  "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
];
const WEEKDAY_NAMES = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"];

export async function render(root) {
  root.innerHTML = "";

  const today = new Date();
  const state = {
    year: today.getFullYear(),
    month: today.getMonth(),
    days: {},
    loading: true,
  };

  const wrap = el("div", { class: "calendar" });
  root.appendChild(wrap);

  const header = el("div", { class: "cal-header" });
  const weekdayRow = el("div", { class: "cal-weekdays" },
    WEEKDAY_NAMES.map((w) => el("div", { class: "cal-weekday" }, [w]))
  );
  const grid = el("div", { class: "cal-grid" });
  const legend = el("div", { class: "cal-legend" }, [
    el("span", { class: "cal-legend-item past" }, ["● Geçmiş"]),
    el("span", { class: "cal-legend-item now" }, ["● Bugün"]),
    el("span", { class: "cal-legend-item future" }, ["● Gelecek"]),
  ]);

  wrap.appendChild(header);
  wrap.appendChild(weekdayRow);
  wrap.appendChild(grid);
  wrap.appendChild(legend);

  async function load() {
    state.loading = true;
    paintHeader();
    paintGrid();
    try {
      const month = `${state.year}-${String(state.month + 1).padStart(2, "0")}`;
      const resp = await api.calendar(month);
      state.days = resp.days || {};
    } catch (err) {
      console.error(err);
      state.days = {};
    }
    state.loading = false;
    paintHeader();
    paintGrid();
  }

  function changeMonth(delta) {
    const d = new Date(state.year, state.month + delta, 1);
    state.year = d.getFullYear();
    state.month = d.getMonth();
    load();
  }

  function jumpToday() {
    const t = new Date();
    state.year = t.getFullYear();
    state.month = t.getMonth();
    load();
  }

  function paintHeader() {
    header.innerHTML = "";
    header.appendChild(el("button", { class: "cal-nav", onclick: () => changeMonth(-1), "aria-label": "Önceki ay" }, ["‹"]));
    header.appendChild(
      el("div", { class: "cal-title" }, [
        el("div", { class: "cal-title-month" }, [`${MONTH_NAMES[state.month]} ${state.year}`]),
        el("div", { class: "cal-title-sub" }, [
          state.loading ? "Yükleniyor…" : subtitle(),
        ]),
      ])
    );
    const right = el("div", { class: "cal-header-right" }, [
      el("button", { class: "cal-today", onclick: jumpToday }, ["Bugün"]),
      el("button", { class: "cal-nav", onclick: () => changeMonth(1), "aria-label": "Sonraki ay" }, ["›"]),
    ]);
    header.appendChild(right);
  }

  function subtitle() {
    const total = totalEvents();
    const busyDays = Object.values(state.days).filter((d) => d.length > 0).length;
    if (total === 0) return "Bu ay için kayıt yok";
    return `${total} olay · ${busyDays} gün`;
  }

  function totalEvents() {
    return Object.values(state.days).reduce((sum, evs) => sum + evs.length, 0);
  }

  function paintGrid() {
    grid.innerHTML = "";
    const first = new Date(state.year, state.month, 1);
    const offset = (first.getDay() + 6) % 7; // Monday-first
    const daysInMonth = new Date(state.year, state.month + 1, 0).getDate();

    for (let i = 0; i < offset; i++) {
      grid.appendChild(el("div", { class: "cal-cell cal-cell-empty" }));
    }

    const todayISO = isoLocal(new Date());
    for (let day = 1; day <= daysInMonth; day++) {
      const iso = `${state.year}-${String(state.month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
      const events = state.days[iso] || [];
      grid.appendChild(dayCell(iso, day, events, iso === todayISO));
    }

    // Trailing blanks so the grid always renders complete weeks — gives a
    // cleaner rectangle instead of a ragged last row.
    const filled = offset + daysInMonth;
    const trailing = (7 - (filled % 7)) % 7;
    for (let i = 0; i < trailing; i++) {
      grid.appendChild(el("div", { class: "cal-cell cal-cell-empty" }));
    }
  }

  function dayCell(iso, day, events, isToday) {
    const cls = ["cal-cell"];
    if (events.length > 0) cls.push("has-events");
    if (isToday) cls.push("is-today");

    const bucket = dominantBucket(events);
    if (bucket) cls.push(`bucket-${bucket}`);

    const cell = el("div", {
      class: cls.join(" "),
      onclick: () => openDay(iso, events),
      role: "button",
      tabindex: "0",
    });

    const top = el("div", { class: "cal-day-top" }, [
      el("span", { class: "cal-day-num" }, [String(day)]),
      events.length ? el("span", { class: "cal-day-dot" }, []) : null,
    ].filter(Boolean));
    cell.appendChild(top);

    const preview = el("div", { class: "cal-day-events" });
    for (const ev of events.slice(0, 3)) {
      preview.appendChild(
        el("div", { class: `cal-event-chip ${slugBucket(ev.zaman_dilimi)}`, title: ev.olay || "" }, [
          eventSummary(ev),
        ])
      );
    }
    if (events.length > 3) {
      preview.appendChild(el("div", { class: "cal-event-more" }, [`+${events.length - 3} daha`]));
    }
    cell.appendChild(preview);
    return cell;
  }

  function openDay(iso, events) {
    const d = new Date(iso + "T00:00:00");
    const title = d.toLocaleDateString("tr-TR", { day: "numeric", month: "long", year: "numeric", weekday: "long" });

    const content = el("div", { class: "cal-drawer" });
    content.appendChild(el("h3", {}, [title]));

    if (events.length === 0) {
      content.appendChild(
        el("div", { class: "cal-empty" }, [
          el("div", { class: "cal-empty-icon" }, ["🗒"]),
          el("p", { class: "muted" }, ["Bu güne ait kayıt yok."]),
        ])
      );
    } else {
      content.appendChild(el("div", { class: "muted", style: "font-size:12px; margin-bottom:10px;" }, [`${events.length} olay`]));
      for (const ev of events) content.appendChild(renderEventCard(ev));
    }
    modal(content);
  }

  load();
}

function renderEventCard(ev) {
  const badge = el("span", { class: `cal-bucket-badge ${slugBucket(ev.zaman_dilimi)}` }, [
    ev.zaman_dilimi || "Olay",
  ]);

  const metaBits = [];
  if (ev.saat) metaBits.push(el("span", { class: "cal-meta-bit" }, [`🕒 ${ev.saat}`]));
  if (ev.lokasyon && !/bilinmeyen/i.test(ev.lokasyon)) {
    metaBits.push(el("span", { class: "cal-meta-bit" }, [`📍 ${ev.lokasyon}`]));
  }

  const people = (ev.kisiler || []).filter(Boolean);
  const peopleRow = people.length
    ? el("div", { class: "cal-people" },
        people.map((p) => el("span", { class: "cal-person" }, [p]))
      )
    : null;

  return el("div", { class: "cal-event-card" }, [
    el("div", { class: "cal-event-header" }, [badge]),
    el("div", { class: "cal-event-body" }, [ev.olay || ""]),
    metaBits.length ? el("div", { class: "cal-event-meta" }, metaBits) : null,
    peopleRow,
    ev.session_id
      ? el("button", { class: "cta ghost", style: "margin-top:4px;", onclick: () => (location.hash = `#/review/${ev.session_id}`) }, ["Kayda git →"])
      : null,
  ]);
}

function eventSummary(ev) {
  const who = (ev.kisiler || []).filter((p) => p && p.toLowerCase() !== "ben").slice(0, 2).join(", ");
  const where = ev.lokasyon && !/bilinmeyen/i.test(ev.lokasyon) ? ev.lokasyon : "";
  const head = [who, where].filter(Boolean).join(" · ");
  if (head) return head;
  const text = (ev.olay || "").trim();
  return text.length > 28 ? text.slice(0, 26) + "…" : text;
}

function dominantBucket(events) {
  let bucket = "";
  for (const ev of events) {
    const s = slugBucket(ev.zaman_dilimi);
    if (s === "now") return "now";
    if (s === "future" && bucket !== "now") bucket = "future";
    if (s === "past" && !bucket) bucket = "past";
  }
  return bucket;
}

function slugBucket(turkish) {
  switch ((turkish || "").toLowerCase()) {
    case "geçmiş": return "past";
    case "şu an": return "now";
    case "gelecek": return "future";
    default: return "";
  }
}

function isoLocal(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
