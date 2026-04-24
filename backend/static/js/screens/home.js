import { api } from "../api.js";
import { pendingStore } from "../db.js";
import { el } from "./utils.js";

export async function render(root, ctx) {
  root.innerHTML = "";
  const [sessions, pending] = await Promise.all([
    api.listSessions().catch(() => []),
    pendingStore.all(),
  ]);

  const latest = Array.isArray(sessions) ? sessions.slice(0, 3) : sessions.results?.slice(0, 3) || [];

  const totalMentions = latest.reduce((n, s) => n + (s.mention_count || 0), 0);
  const totalConflicts = latest.reduce((n, s) => n + (s.conflict_count || 0), 0);

  root.appendChild(heroCard(pending.length));
  root.appendChild(statsCard(latest.length, totalMentions, totalConflicts));

  if (latest.length === 0 && pending.length === 0) {
    root.appendChild(
      el("div", { class: "card" }, [
        el("h2", {}, ["Hoş geldin"]),
        el("p", { class: "muted" }, [
          "İlk kaydını yapmak için alttaki Kaydet sekmesine git. " +
          "İnternete erişim yoksa kayıt cihazda güvende tutulur ve sonra otomatik gönderilir.",
        ]),
      ])
    );
    return;
  }

  root.appendChild(el("h2", { style: "margin:14px 4px 8px; font-size:14px; color:var(--fg-muted); letter-spacing:.4px;" }, ["Son Kayıtlar"]));

  if (pending.length > 0) {
    const pendingWrap = el("div", { class: "card" }, [
      el("h2", {}, [`${pending.length} bekleyen kayıt`]),
      el("p", { class: "muted" }, ["Çevrimiçi olunca otomatik senkronize olacak."]),
      el("div", { class: "pending-list" },
        pending.map((p) =>
          el("div", { class: "pending-row" }, [
            el("span", { class: "lbl" }, [new Date(p.recordedAt).toLocaleString("tr-TR")]),
            el("span", {}, [`${Math.round(p.durationSeconds || 0)}s`]),
          ])
        )
      ),
    ]);
    root.appendChild(pendingWrap);
  }

  for (const s of latest) {
    root.appendChild(sessionRow(s, ctx));
  }

  root.appendChild(
    el("button", { class: "cta ghost", onclick: () => (location.hash = "#/timeline") }, [
      "Tüm zaman çizelgesi →",
    ])
  );
}

function heroCard(pendingCount) {
  const card = el("section", { class: "card", style: "background: linear-gradient(135deg, #1f3d8a, #2a59d9); border:0;" }, [
    el("h2", { style: "font-size:18px; margin-bottom:6px;" }, ["Sesli günlük, özel kalır"]),
    el("p", { class: "muted", style: "color:rgba(255,255,255,0.85); font-size:13px;" }, [
      "Kaydet. Yazıya dökülsün. Belirsiz kısımları sen işaretle. Her şey cihazında ve kendi sunucunda.",
    ]),
    el("div", { style: "display:flex; gap:8px; margin-top:12px;" }, [
      el("button", { class: "cta", style: "flex:1; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);", onclick: () => (location.hash = "#/record") }, ["● Yeni Kayıt"]),
      pendingCount > 0
        ? el("button", { class: "cta", style: "flex:0 0 auto; background:rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.25);", onclick: () => (location.hash = "#/home") }, [`${pendingCount} bekliyor`])
        : null,
    ].filter(Boolean)),
  ]);
  return card;
}

function statsCard(sessionCount, mentionCount, conflictCount) {
  return el("div", { class: "card", style: "display:flex; justify-content:space-between; text-align:center;" }, [
    el("div", {}, [
      el("div", { style: "font-size:20px; font-weight:600;" }, [String(sessionCount)]),
      el("div", { class: "muted", style: "font-size:11px;" }, ["Kayıt"]),
    ]),
    el("div", {}, [
      el("div", { style: "font-size:20px; font-weight:600;" }, [String(mentionCount)]),
      el("div", { class: "muted", style: "font-size:11px;" }, ["Bağlantı"]),
    ]),
    el("div", {}, [
      el("div", { style: `font-size:20px; font-weight:600; color:${conflictCount ? "var(--warn)" : "var(--ok)"};` }, [String(conflictCount)]),
      el("div", { class: "muted", style: "font-size:11px;" }, ["Çatışma"]),
    ]),
  ]);
}

function sessionRow(s, ctx) {
  const wrap = el("div", {
    class: "session-row",
    onclick: () => (location.hash = `#/review/${s.id}`),
  });

  const top = el("div", { class: "top" }, [
    el("span", { class: "date" }, [formatDate(s.recorded_at)]),
    statusBadge(s),
  ]);
  const excerpt = el("div", { class: "excerpt" }, [s.transcript || "(Henüz yazıya dökülmedi)"]);
  const chips = el("div", { class: "chips" }, [
    s.conflict_count > 0 ? el("span", { class: "chip unknown" }, [`${s.conflict_count} çatışma`]) : null,
    s.mention_count > 0 ? el("span", { class: "chip" }, [`${s.mention_count} bağlantı`]) : null,
    s.duration_seconds ? el("span", { class: "chip" }, [`${Math.round(s.duration_seconds)}s`]) : null,
  ].filter(Boolean));

  wrap.appendChild(top);
  wrap.appendChild(excerpt);
  wrap.appendChild(chips);
  return wrap;
}

function statusBadge(s) {
  const map = {
    queued: { cls: "", text: "Sırada" },
    transcribing: { cls: "warn", text: "Yazıya çevriliyor" },
    parsing: { cls: "warn", text: "Ayrıştırılıyor" },
    review: { cls: "ok", text: "Hazır" },
    completed: { cls: "ok", text: "Hazır" },
    failed: { cls: "err", text: "Hata" },
  };
  const m = map[s.status] || { cls: "", text: s.status_display || s.status };
  return el("span", { class: `status-badge ${m.cls}` }, [m.text]);
}

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleString("tr-TR", { dateStyle: "medium", timeStyle: "short" });
}
