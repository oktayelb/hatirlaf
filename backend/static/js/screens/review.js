// Review screen: the "human-in-the-loop" UI from the spec.
// Shows the transcript with highlighted conflicts, an audio player, and an
// action sheet per mention ({Assign existing / New / Unknown / Ignore}).

import { api } from "../api.js";
import { emit, toast } from "../events.js";
import { el, escapeHtml, MENTION_LABELS, modal } from "./utils.js";

let pollHandle = null;

export async function render(root, ctx) {
  if (pollHandle) clearInterval(pollHandle);
  root.innerHTML = "";
  const id = ctx.params.id;
  if (!id) {
    root.appendChild(el("div", { class: "empty-state" }, ["Kayıt bulunamadı."]));
    return;
  }

  const loading = el("div", { class: "card" }, [
    el("span", { class: "loading" }),
    el("span", { style: "margin-left:8px;" }, ["Yükleniyor..."]),
  ]);
  root.appendChild(loading);

  let session;
  try {
    session = await api.getSession(id);
  } catch (err) {
    loading.remove();
    root.appendChild(el("div", { class: "empty-state" }, [
      el("div", { class: "empty-title" }, ["Kayıt yüklenemedi"]),
      el("p", { class: "muted" }, [err.message]),
    ]));
    return;
  }
  loading.remove();

  paint(root, session);

  // Poll until processed, so the UI updates as the background thread finishes.
  if (["queued", "transcribing", "parsing"].includes(session.status)) {
    pollHandle = setInterval(async () => {
      try {
        const updated = await api.getSession(id);
        if (!["queued", "transcribing", "parsing"].includes(updated.status)) {
          clearInterval(pollHandle);
          pollHandle = null;
        }
        paint(root, updated);
      } catch (_) {}
    }, 1500);
  }
}

export function cleanup() {
  if (pollHandle) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

function paint(root, session) {
  root.innerHTML = "";

  root.appendChild(headerCard(session));

  if (session.audio_url) {
    const audio = el("audio", {
      class: "audio-player",
      controls: "",
      preload: "metadata",
      src: session.audio_url,
      id: "session-audio",
    });
    root.appendChild(audio);
  }

  root.appendChild(transcriptView(session));

  const actions = el("div", { style: "display:flex; gap:8px; margin-top:12px;" }, [
    el("button", {
      class: "cta secondary",
      onclick: async () => {
        await api.reprocess(session.id);
        toast("Yeniden ayrıştırılıyor…");
      },
    }, ["Yeniden İşle"]),
    el("button", {
      class: "cta ghost",
      onclick: () => (location.hash = "#/timeline"),
    }, ["Zaman çizelgesine dön"]),
  ]);
  root.appendChild(actions);

  root.appendChild(mentionsSection(session));
}

function headerCard(session) {
  const date = new Date(session.recorded_at).toLocaleString("tr-TR", {
    dateStyle: "full",
    timeStyle: "short",
  });
  const statusCls = {
    completed: "ok",
    failed: "err",
    review: "warn",
  }[session.status] || "";

  return el("div", { class: "card" }, [
    el("div", { style: "display:flex; justify-content:space-between; align-items:flex-start; gap:10px;" }, [
      el("div", {}, [
        el("h2", { style: "margin:0 0 4px;" }, [date]),
        el("div", { class: "muted", style: "font-size:12px;" }, [
          `${session.mention_count || 0} bağlantı · ${session.conflict_count || 0} çatışma · ${Math.round(session.duration_seconds || 0)}s`,
        ]),
      ]),
      el("span", { class: `status-badge ${statusCls}` }, [
        session.status_display || session.status,
      ]),
    ]),
    session.status_detail
      ? el("div", { class: "muted", style: "font-size:12px; margin-top:8px;" }, [session.status_detail])
      : null,
  ]);
}

function transcriptView(session) {
  const text = session.transcript || "(Transkript henüz yok)";
  const mentions = (session.mentions || []).slice().sort((a, b) => a.char_start - b.char_start);

  const container = el("div", { class: "review-transcript", id: "transcript" });

  let cursor = 0;
  for (const m of mentions) {
    if (m.char_start < cursor) continue; // overlap guard
    if (m.char_start > cursor) {
      container.appendChild(document.createTextNode(text.slice(cursor, m.char_start)));
    }
    const cls = [
      m.mention_type,
      m.is_conflict ? "is-conflict" : "",
      m.resolved ? "resolved" : "",
    ].join(" ");
    const mark = el(
      "mark",
      {
        class: cls,
        "data-mention": m.id,
        title: `${MENTION_LABELS[m.mention_type] || m.mention_type}${m.is_conflict ? " · çatışma" : ""}`,
      },
      [text.slice(m.char_start, m.char_end)]
    );
    mark.addEventListener("click", () => focusMention(session, m));
    container.appendChild(mark);
    cursor = m.char_end;
  }
  if (cursor < text.length) {
    container.appendChild(document.createTextNode(text.slice(cursor)));
  }
  return container;
}

function mentionsSection(session) {
  const conflicts = (session.mentions || []).filter((m) => m.is_conflict && !m.resolved);
  const resolved = (session.mentions || []).filter((m) => m.resolved);
  const section = el("section", { style: "margin-top:18px;" });

  section.appendChild(
    el("h2", { style: "font-size:14px; color:var(--fg-muted); letter-spacing:.4px; margin:0 0 8px;" }, [
      `ÇATIŞMALAR (${conflicts.length})`,
    ])
  );

  const list = el("div", { class: "mentions-list" });
  if (conflicts.length === 0) {
    list.appendChild(el("div", { class: "empty-state", style: "padding:20px;" }, [
      el("p", { class: "muted" }, ["Bekleyen çatışma yok. Her şey bağlandı."]),
    ]));
  } else {
    conflicts.forEach((m) => list.appendChild(mentionCard(session, m, false)));
  }
  section.appendChild(list);

  if (resolved.length > 0) {
    section.appendChild(
      el("h2", { style: "font-size:14px; color:var(--fg-muted); letter-spacing:.4px; margin:20px 0 8px;" }, [
        `ÇÖZÜLMÜŞ BAĞLANTILAR (${resolved.length})`,
      ])
    );
    const resolvedList = el("div", { class: "mentions-list" });
    resolved.forEach((m) => resolvedList.appendChild(mentionCard(session, m, true)));
    section.appendChild(resolvedList);
  }
  return section;
}

function mentionCard(session, m, isResolved) {
  const card = el("div", {
    class: `mention-card ${m.is_conflict ? "conflict" : ""} ${isResolved ? "resolved" : ""}`,
    "data-mention": m.id,
  });

  const hint = m.conflict_hint || "";
  const typeLabel = MENTION_LABELS[m.mention_type] || m.mention_type;

  card.appendChild(el("div", { style: "display:flex; justify-content:space-between;" }, [
    el("span", { class: "surface" }, [`"${m.surface}"`]),
    el("span", { class: "chip" }, [typeLabel]),
  ]));
  if (hint) card.appendChild(el("div", { class: "hint" }, [hint]));
  if (m.node) {
    const tag = m.node.is_unknown ? " (Bilinmeyen)" : "";
    card.appendChild(el("div", { class: "muted", style: "font-size:12px;" }, [
      `→ ${m.node.label}${tag}`,
    ]));
  }

  if (!isResolved) {
    const actions = el("div", { class: "mention-actions" });
    actions.appendChild(
      el("button", { class: "primary", onclick: () => openResolveModal(session, m) }, ["Bağla"])
    );
    actions.appendChild(
      el("button", { class: "ghost", onclick: () => resolve(session, m, { action: "NEW" }) }, ["Yeni Düğüm"])
    );
    actions.appendChild(
      el("button", { class: "warn", onclick: () => resolve(session, m, { action: "UNKNOWN" }) }, ["Bilinmeyen"])
    );
    actions.appendChild(
      el("button", { class: "ghost", onclick: () => resolve(session, m, { action: "IGNORE" }) }, ["Yoksay"])
    );
    if (m.audio_start != null) {
      actions.appendChild(
        el("button", {
          class: "ghost",
          onclick: () => playRange(m.audio_start, m.audio_end),
        }, ["Dinle"])
      );
    }
    card.appendChild(actions);
  }
  return card;
}

async function openResolveModal(session, mention) {
  const suggestedKind = mentionKindToNodeKind(mention.mention_type);
  const nodes = await api.listNodes({ kind: suggestedKind });
  const input = el("input", {
    type: "text",
    placeholder: `Mevcut bir ${kindLabel(suggestedKind)} ara veya yeni ekle`,
    value: mention.surface,
  });
  const suggestions = el("div", { class: "suggestion-row", id: "suggestion-row" });

  const renderSuggestions = (list) => {
    suggestions.innerHTML = "";
    for (const n of list.slice(0, 10)) {
      suggestions.appendChild(
        el("button", {
          onclick: () => {
            input.value = n.label;
            input.dataset.nodeId = n.id;
          },
        }, [`${n.label}${n.is_unknown ? " (Bilinmeyen)" : ""}`])
      );
    }
  };
  renderSuggestions(nodes);

  input.addEventListener("input", async () => {
    delete input.dataset.nodeId;
    const q = input.value.trim();
    const filtered = await api.listNodes({ kind: suggestedKind, q });
    renderSuggestions(filtered);
  });

  const kindSelect = el("select", {}, [
    ...["PERSON", "LOCATION", "TIME", "EVENT", "ORG", "OTHER"].map((k) =>
      el("option", { value: k, selected: k === suggestedKind ? "" : null }, [kindLabel(k)])
    ),
  ]);

  const content = el("div", {}, [
    el("h3", {}, [`"${mention.surface}" çözümle`]),
    el("div", { class: "muted" }, [mention.conflict_hint || "Bu ifadeyi bir düğüme bağla."]),
    el("label", {}, ["Tür"]),
    kindSelect,
    el("label", {}, ["Etiket"]),
    input,
    el("div", { class: "muted", style: "font-size:12px; margin-top:6px;" }, ["Seçenekler"]),
    suggestions,
    el("div", { style: "display:flex; gap:8px; margin-top:14px;" }, [
      el("button", {
        class: "cta",
        style: "flex:1;",
        onclick: async () => {
          const label = input.value.trim();
          if (!label) return toast("Etiket boş olamaz");
          const existing = input.dataset.nodeId
            ? { action: "ASSIGN", node_id: Number(input.dataset.nodeId) }
            : { action: "NEW", label, kind: kindSelect.value };
          backdrop.close();
          await resolve(null, mention, existing);
        },
      }, ["Kaydet"]),
      el("button", { class: "cta ghost", style: "flex:0 0 auto;", onclick: () => backdrop.close() }, ["Vazgeç"]),
    ]),
  ]);
  const backdrop = modal(content);
}

async function resolve(_session, mention, payload) {
  try {
    await api.resolveMention(mention.id, payload);
    toast("Güncellendi");
    emit("mention-resolved", mention.id);
    const updated = await api.getSession(mention.session);
    const root = document.getElementById("screen-root");
    paint(root, updated);
  } catch (err) {
    toast("Hata: " + err.message);
  }
}

function focusMention(session, m) {
  document.querySelectorAll(".review-transcript mark").forEach((el) => el.classList.remove("active"));
  const mark = document.querySelector(`.review-transcript mark[data-mention="${m.id}"]`);
  if (mark) mark.classList.add("active");
  const card = document.querySelector(`.mention-card[data-mention="${m.id}"]`);
  if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
  if (m.audio_start != null) playRange(m.audio_start, m.audio_end);
}

function playRange(start, end) {
  const audio = document.getElementById("session-audio");
  if (!audio) return;
  audio.currentTime = Math.max(0, start || 0);
  audio.play().catch(() => {});
  if (end != null) {
    const stopAt = end;
    const check = () => {
      if (audio.currentTime >= stopAt) {
        audio.pause();
        audio.removeEventListener("timeupdate", check);
      }
    };
    audio.addEventListener("timeupdate", check);
  }
}

function mentionKindToNodeKind(k) {
  return { PRONOUN: "PERSON" }[k] || k;
}

function kindLabel(k) {
  return { PERSON: "Kişi", LOCATION: "Yer", TIME: "Zaman", EVENT: "Olay", ORG: "Kurum", OTHER: "Diğer" }[k] || k;
}
