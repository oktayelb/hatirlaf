// Entries screen: lists every session the user has captured. Vocal entries
// show an audio player plus the editable STT transcript. Textual entries
// show only the editable transcript. Saving PATCHes the transcript field.

import { api } from "../api.js";
import { toast } from "../events.js";
import { el, fmtRelative } from "./utils.js";

let pollHandle = null;

export async function render(root) {
  cleanup();
  root.innerHTML = "";

  const header = el("div", { class: "entries-header" }, [
    el("h2", { class: "entries-title" }, ["Tüm Girişler"]),
    el("p", { class: "entries-sub muted" }, [
      "Sesli kayıtların hem ses hem de yazı haline buradan ulaşır, dilediğin gibi düzenleyebilirsin.",
    ]),
  ]);
  root.appendChild(header);

  const listWrap = el("div", { class: "entries-list" }, [
    el("div", { class: "card" }, [
      el("span", { class: "loading" }),
      el("span", { style: "margin-left:8px;" }, ["Yükleniyor..."]),
    ]),
  ]);
  root.appendChild(listWrap);

  let sessions;
  try {
    const resp = await api.listSessions();
    sessions = Array.isArray(resp) ? resp : resp.results || [];
  } catch (err) {
    listWrap.innerHTML = "";
    listWrap.appendChild(
      el("div", { class: "empty-state" }, [
        el("div", { class: "empty-title" }, ["Girişler yüklenemedi"]),
        el("p", { class: "muted" }, [err.message]),
      ])
    );
    return;
  }

  paintList(listWrap, sessions);
  if (sessions.some(isProcessing)) startPolling(listWrap);
}

export function cleanup() {
  if (pollHandle) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

function paintList(listWrap, sessions) {
  listWrap.innerHTML = "";
  if (sessions.length === 0) {
    listWrap.appendChild(emptyState());
    return;
  }
  for (const s of sessions) {
    listWrap.appendChild(entryCard(s));
  }
}

function startPolling(listWrap) {
  cleanup();
  pollHandle = setInterval(async () => {
    try {
      const resp = await api.listSessions();
      const sessions = Array.isArray(resp) ? resp : resp.results || [];
      paintList(listWrap, sessions);
      if (!sessions.some(isProcessing)) cleanup();
    } catch (err) {
      console.debug("entry processing poll failed", err);
    }
  }, 1500);
}

function emptyState() {
  return el("div", { class: "empty-state" }, [
    el("div", { class: "empty-title" }, ["Henüz giriş yok"]),
    el("p", { class: "muted" }, [
      "Ana sayfadan ses kaydı yap ya da yazıyla ekle.",
    ]),
  ]);
}

function entryCard(session) {
  const hasAudio = Boolean(session.audio_url);
  const recordedAt = new Date(session.recorded_at);
  const dateStr = recordedAt.toLocaleString("tr-TR", {
    dateStyle: "medium",
    timeStyle: "short",
  });

  const kindBadge = el(
    "span",
    { class: `entry-kind ${hasAudio ? "voice" : "text"}` },
    [hasAudio ? "Sesli" : "Yazılı"]
  );

  const statusBadge = el(
    "span",
    { class: `status-badge ${statusClass(session.status)}` },
    [session.status_display || session.status]
  );
  const eventBadge = el(
    "span",
    { class: `status-badge ${eventStatusClass(session.eventification_status)}` },
    [eventStatusLabel(session)]
  );

  const meta = el("div", { class: "entry-meta" }, [
    el("div", { class: "entry-meta-row" }, [
      kindBadge,
      statusBadge,
      eventBadge,
      el("span", { class: "muted entry-time" }, [fmtRelative(session.recorded_at)]),
    ]),
    el("div", { class: "entry-date muted" }, [dateStr]),
  ]);

  const card = el("article", { class: "entry-card" }, [meta]);
  card.appendChild(processingPanel(session));

  if (hasAudio) {
    card.appendChild(
      el("audio", {
        class: "entry-audio",
        controls: "",
        preload: "metadata",
        src: session.audio_url,
      })
    );
  }

  const transcriptPending = ["queued", "transcribing", "parsing"].includes(session.status);
  const transcript = session.transcript || "";
  const placeholder = hasAudio
    ? (transcriptPending ? "Yazıya çevriliyor…" : "Henüz transkript yok")
    : "Metin boş";

  const textarea = el("textarea", {
    class: "entry-transcript",
    rows: "5",
    placeholder,
  });
  textarea.value = transcript;
  card.appendChild(textarea);

  const status = el("span", { class: "muted entry-status" }, [""]);
  const saveBtn = el(
    "button",
    { class: "cta", disabled: "" },
    ["Kaydet"]
  );
  const reprocessBtn = el(
    "button",
    { class: "cta ghost" },
    ["Yeniden işle"]
  );
  const deleteBtn = el(
    "button",
    { class: "cta danger" },
    ["Sil"]
  );

  let original = transcript;
  textarea.addEventListener("input", () => {
    if (textarea.value !== original) saveBtn.removeAttribute("disabled");
    else saveBtn.setAttribute("disabled", "");
  });

  saveBtn.addEventListener("click", async () => {
    const next = textarea.value;
    saveBtn.setAttribute("disabled", "");
    saveBtn.textContent = "Kaydediliyor…";
    status.textContent = "";
    try {
      await api.updateSession(session.id, { transcript: next });
      original = next;
      saveBtn.textContent = "Kaydet";
      status.textContent = "Kaydedildi";
      toast("Giriş güncellendi");
    } catch (err) {
      console.error(err);
      saveBtn.removeAttribute("disabled");
      saveBtn.textContent = "Kaydet";
      status.textContent = "Hata: " + err.message;
    }
  });

  reprocessBtn.addEventListener("click", async () => {
    reprocessBtn.setAttribute("disabled", "");
    try {
      await api.reprocess(session.id);
      toast("Yeniden işleniyor…");
      if (card.parentElement) startPolling(card.parentElement);
    } catch (err) {
      toast("Hata: " + err.message);
    } finally {
      reprocessBtn.removeAttribute("disabled");
    }
  });

  deleteBtn.addEventListener("click", async () => {
    const ok = window.confirm(
      "Bu girişi silmek istediğine emin misin? Ses dosyası, transkript ve takvim olayları kalıcı olarak silinir."
    );
    if (!ok) return;
    deleteBtn.setAttribute("disabled", "");
    deleteBtn.textContent = "Siliniyor…";
    try {
      await api.deleteSession(session.id);
      const list = card.parentElement;
      card.remove();
      if (list && list.children.length === 0) {
        list.appendChild(emptyState());
      }
      toast("Giriş silindi");
    } catch (err) {
      console.error(err);
      deleteBtn.removeAttribute("disabled");
      deleteBtn.textContent = "Sil";
      toast("Silme başarısız: " + err.message);
    }
  });

  card.appendChild(
    el("div", { class: "entry-actions" }, [saveBtn, reprocessBtn, deleteBtn, status])
  );

  return card;
}

function processingPanel(session) {
  const progress = Math.max(0, Math.min(100, Number(session.processing_progress) || 0));
  const active = isProcessing(session);
  const failed = session.status === "failed" || session.eventification_status === "failed";
  const complete = !failed && progress >= 100;
  const title = failed ? "İşleme tamamlanamadı" : complete ? "İşleme tamamlandı" : processingTitle(session);
  const detail = session.processing_detail || session.status_detail || session.eventification_detail || "";

  return el("div", { class: `entry-processing ${active ? "active" : ""} ${complete ? "complete" : ""} ${failed ? "failed" : ""}` }, [
    el("div", { class: "entry-processing-head" }, [
      el("span", { class: "entry-processing-title" }, [title]),
      el("span", { class: "entry-processing-percent" }, [`${progress}%`]),
    ]),
    el("div", { class: "entry-processing-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(progress) }, [
      el("span", { style: { width: `${progress}%` } }),
    ]),
    detail ? el("div", { class: "entry-processing-detail muted" }, [detail]) : null,
  ]);
}

function isProcessing(session) {
  return (
    ["queued", "transcribing", "parsing"].includes(session.status) ||
    ["queued", "running"].includes(session.eventification_status)
  );
}

function processingTitle(session) {
  if (session.status === "queued") return "Sırada";
  if (session.status === "transcribing") return "Ses yazıya çevriliyor";
  if (session.status === "parsing") return "NLP analizi çalışıyor";
  if (session.eventification_status === "queued") return "LLM sırada";
  if (session.eventification_status === "running") return "LLM olayları çıkarıyor";
  return "İşleniyor";
}

function statusClass(status) {
  return (
    {
      completed: "ok",
      failed: "err",
      review: "warn",
    }[status] || ""
  );
}

function eventStatusLabel(session) {
  switch (session.eventification_status) {
    case "completed":
      return "Takvim hazır";
    case "running":
      return "Olaylaştırılıyor";
    case "queued":
      return "Olay sırada";
    case "failed":
      return "Olay hatası";
    default:
      return "Olay bekliyor";
  }
}

function eventStatusClass(status) {
  return (
    {
      completed: "ok",
      failed: "err",
      queued: "warn",
      running: "warn",
    }[status] || ""
  );
}
