// Günlüğüm — every entry you have made, newest first.
//
// A voice entry shows its recording plus the text it was turned into; a
// written entry shows what you wrote. Both are editable. Nothing about how
// the text was produced is shown, and while the NLP switch is off there is
// nothing else to show.

import { api } from "../api.js";
import { nlpEnabled } from "../config.js";
import { toast } from "../events.js";
import { el, fmtRelative, modal } from "./utils.js";

let pollHandle = null;

export async function render(root) {
  cleanup();
  root.innerHTML = "";

  root.appendChild(
    el("header", { class: "entries-header" }, [
      el("h2", { class: "section-title" }, ["Günlüğüm"]),
    ])
  );

  const listWrap = el("div", { class: "entries-list" }, [
    el("div", { class: "card" }, [
      el("span", { class: "loading" }),
      el("span", { style: { marginLeft: "12px" } }, ["Günlüğün yükleniyor…"]),
    ]),
  ]);
  root.appendChild(listWrap);

  if (nlpEnabled()) root.appendChild(reextractAllButton(listWrap));

  let sessions;
  try {
    sessions = await fetchSessions();
  } catch (err) {
    listWrap.innerHTML = "";
    listWrap.appendChild(
      el("div", { class: "empty-state" }, [
        el("div", { class: "empty-title" }, ["Günlüğün yüklenemedi"]),
        el("p", {}, [err.message]),
      ])
    );
    return;
  }

  paintList(listWrap, sessions);
  if (sessions.some(isBusy)) startPolling(listWrap);
}

export function cleanup() {
  if (pollHandle) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

async function fetchSessions() {
  const resp = await api.listSessions();
  return Array.isArray(resp) ? resp : resp.results || [];
}

function paintList(listWrap, sessions) {
  const drafts = captureDrafts(listWrap);
  listWrap.innerHTML = "";
  if (sessions.length === 0) {
    listWrap.appendChild(emptyState());
    return;
  }
  for (const session of sessions) {
    listWrap.appendChild(entryCard(session, drafts.get(String(session.id))));
  }
}

function startPolling(listWrap) {
  cleanup();
  pollHandle = setInterval(async () => {
    try {
      const sessions = await fetchSessions();
      paintList(listWrap, sessions);
      if (!sessions.some(isBusy)) cleanup();
    } catch (err) {
      console.debug("entry poll failed", err);
    }
  }, 2000);
}

function emptyState() {
  return el("div", { class: "empty-state" }, [
    el("div", { class: "empty-title" }, ["Henüz hiç kayıt yok"]),
    el("p", {}, [
      'Ana sayfaya git, yuvarlak düğmeye basıp konuş ya da aşağıdaki kutuya yaz. ' +
        "Kaydettiğin her şey burada birikecek.",
    ]),
  ]);
}

/* ---------- One entry ---------- */

function entryCard(session, draft = null) {
  const hasAudio = Boolean(session.audio_url);
  const recordedAt = new Date(session.recorded_at);

  const card = el("article", { class: "entry-card", "data-session-id": session.id });

  card.appendChild(
    el("div", { class: "entry-head" }, [
      el("div", {}, [
        el("div", { class: "entry-date" }, [
          recordedAt.toLocaleDateString("tr-TR", {
            weekday: "long",
            day: "numeric",
            month: "long",
            year: "numeric",
          }),
        ]),
        el("div", { class: "entry-time" }, [
          recordedAt.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" }) +
            " · " +
            fmtRelative(session.recorded_at),
        ]),
      ]),
    ])
  );

  const progress = progressPanel(session);
  if (progress) card.appendChild(progress);

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

  const transcript = session.transcript || "";
  const textarea = el("textarea", {
    class: "entry-transcript",
    rows: "6",
    "aria-label": hasAudio ? "Yazıya çevrilmiş hâli" : "Yazdıkların",
    placeholder: hasAudio
      ? isTranscribing(session)
        ? "Sesin yazıya çevriliyor, birazdan burada olacak…"
        : "Bu kayıt için henüz yazı yok."
      : "Bu not boş.",
  });
  textarea.value = draft ? draft.value : transcript;
  textarea.defaultValue = transcript;
  if (draft?.focused) {
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(draft.selectionStart || 0, draft.selectionEnd || 0);
    }, 0);
  }

  card.appendChild(textarea);

  card.appendChild(actionRow({ session, card, textarea, transcript }));
  return card;
}

function actionRow({ session, card, textarea, transcript }) {
  const status = el("span", { class: "entry-status" }, [""]);
  const saveBtn = el("button", { class: "cta", type: "button", disabled: "" }, ["Kaydet"]);
  const deleteBtn = el("button", { class: "cta danger", type: "button" }, ["Sil"]);

  let saved = transcript;
  textarea.addEventListener("input", () => {
    if (textarea.value !== saved) saveBtn.removeAttribute("disabled");
    else saveBtn.setAttribute("disabled", "");
  });

  saveBtn.addEventListener("click", async () => {
    saveBtn.setAttribute("disabled", "");
    saveBtn.textContent = "Kaydediliyor…";
    status.textContent = "";
    try {
      await api.updateSession(session.id, { transcript: textarea.value });
      saved = textarea.value;
      saveBtn.textContent = "Kaydet";
      status.textContent = "Değişikliklerin kaydedildi.";
      toast("Kaydedildi");
    } catch (err) {
      console.error(err);
      saveBtn.removeAttribute("disabled");
      saveBtn.textContent = "Kaydet";
      status.textContent = "Kaydedilemedi: " + err.message;
    }
  });

  deleteBtn.addEventListener("click", () => openDeleteModal({ session, card, deleteBtn }));

  const buttons = [saveBtn];
  if (nlpEnabled()) {
    const reprocessBtn = el("button", { class: "cta ghost", type: "button" }, [
      "Yeniden işle",
    ]);
    reprocessBtn.addEventListener("click", async () => {
      reprocessBtn.setAttribute("disabled", "");
      try {
        const updated = await api.reprocess(session.id);
        const list = card.parentElement;
        if (list) {
          card.replaceWith(entryCard(updated));
          startPolling(list);
        }
      } catch (err) {
        toast("Hata: " + err.message);
      } finally {
        reprocessBtn.removeAttribute("disabled");
      }
    });
    buttons.push(reprocessBtn);
  }
  buttons.push(deleteBtn, status);

  return el("div", { class: "entry-actions" }, buttons);
}

function progressPanel(session) {
  // A finished entry needs no status line at all — the text is the status.
  if (session.status === "completed" || session.status === "review") return null;

  const failed = session.status === "failed";
  const percent = Math.max(0, Math.min(100, Number(session.processing_progress) || 0));
  const label = failed
    ? "Bu kayıt yazıya çevrilemedi"
    : session.status === "queued"
    ? "Kaydın sırada bekliyor"
    : "Sesin yazıya çevriliyor";

  return el("div", { class: `entry-progress ${failed ? "failed" : ""}` }, [
    el("div", { class: "entry-progress-label" }, [
      failed ? null : el("span", { class: "loading" }),
      label,
    ]),
    el(
      "div",
      {
        class: "entry-progress-bar",
        role: "progressbar",
        "aria-valuemin": "0",
        "aria-valuemax": "100",
        "aria-valuenow": String(percent),
      },
      [el("span", { style: { width: `${percent}%` } })]
    ),
  ]);
}

function openDeleteModal({ session, card, deleteBtn }) {
  const content = el("div", { class: "confirm-dialog" }, [
    el("h3", {}, ["Bu kaydı silmek istediğine emin misin?"]),
    el("p", {}, [
      "Kayıt kalıcı olarak silinir. Ses dosyası ve yazısı geri getirilemez.",
    ]),
    el("div", { class: "confirm-dialog-actions" }, [
      el("button", { class: "cta ghost", type: "button", onclick: () => dialog.close() }, [
        "Vazgeç",
      ]),
      el(
        "button",
        {
          class: "cta danger",
          type: "button",
          onclick: () => deleteSession({ session, card, deleteBtn, dialog }),
        },
        ["Evet, Sil"]
      ),
    ]),
  ]);
  const dialog = modal(content);
}

async function deleteSession({ session, card, deleteBtn, dialog }) {
  deleteBtn.setAttribute("disabled", "");
  try {
    await api.deleteSession(session.id);
    const list = card.parentElement;
    dialog.close();
    card.remove();
    if (list && list.children.length === 0) list.appendChild(emptyState());
    toast("Kayıt silindi");
  } catch (err) {
    console.error(err);
    deleteBtn.removeAttribute("disabled");
    toast("Silinemedi: " + err.message);
  }
}

function captureDrafts(listWrap) {
  const drafts = new Map();
  const active = document.activeElement;
  for (const card of listWrap.querySelectorAll(".entry-card[data-session-id]")) {
    const textarea = card.querySelector(".entry-transcript");
    if (!textarea) continue;
    const isFocused = textarea === active;
    if (!isFocused && textarea.value === textarea.defaultValue) continue;
    drafts.set(card.dataset.sessionId, {
      value: textarea.value,
      focused: isFocused,
      selectionStart: textarea.selectionStart,
      selectionEnd: textarea.selectionEnd,
    });
  }
  return drafts;
}

function isTranscribing(session) {
  return ["queued", "transcribing", "parsing"].includes(session.status);
}

function isBusy(session) {
  return (
    isTranscribing(session) ||
    ["queued", "running"].includes(session.eventification_status || "")
  );
}

/* ---------- Developer tool, only while the NLP switch is on ---------- */

function reextractAllButton(listWrap) {
  const button = el("button", { class: "cta ghost entries-debug-btn", type: "button" }, [
    "Tüm çıkarımları yenile",
  ]);
  button.addEventListener("click", async () => {
    button.setAttribute("disabled", "");
    button.textContent = "Yenileniyor…";
    try {
      const summary = await api.reextractAll();
      toast(summary.detail || "Çıkarımlar yenilendi");
      const sessions = await fetchSessions();
      paintList(listWrap, sessions);
      if (sessions.some(isBusy)) startPolling(listWrap);
    } catch (err) {
      toast("Hata: " + err.message);
    } finally {
      button.removeAttribute("disabled");
      button.textContent = "Tüm çıkarımları yenile";
    }
  });
  return button;
}
