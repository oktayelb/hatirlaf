// Entries screen: lists every session the user has captured. Vocal entries
// show an audio player plus the editable STT transcript. Textual entries
// show only the editable transcript. Saving PATCHes the transcript field.

import { api } from "../api.js";
import { toast } from "../events.js";
import { el, fmtRelative } from "./utils.js";

export async function render(root) {
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

  listWrap.innerHTML = "";
  if (sessions.length === 0) {
    listWrap.appendChild(emptyState());
    return;
  }

  for (const s of sessions) {
    listWrap.appendChild(entryCard(s));
  }
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

  const meta = el("div", { class: "entry-meta" }, [
    el("div", { class: "entry-meta-row" }, [
      kindBadge,
      statusBadge,
      el("span", { class: "muted entry-time" }, [fmtRelative(session.recorded_at)]),
    ]),
    el("div", { class: "entry-date muted" }, [dateStr]),
  ]);

  const card = el("article", { class: "entry-card" }, [meta]);

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

  const isProcessing = ["queued", "transcribing", "parsing"].includes(session.status);
  const transcript = session.transcript || "";
  const placeholder = hasAudio
    ? (isProcessing ? "Yazıya çevriliyor…" : "Henüz transkript yok")
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

function statusClass(status) {
  return (
    {
      completed: "ok",
      failed: "err",
      review: "warn",
    }[status] || ""
  );
}
