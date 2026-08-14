// Ana — the whole app in one screen: your photos, a microphone, a box to
// write in. Nothing else. The controls speak for themselves; anything that
// needs saying while recording goes to the toast.

import { Recorder, fileExtFor, fmtDuration } from "../audio.js";
import { uuid } from "../db.js";
import { toast } from "../events.js";
import { icon } from "../icons.js";
import { photoBoard } from "../photos.js";
import { enqueue } from "../sync.js";
import { el } from "./utils.js";

let board = null;
let liveRecorder = null;

export async function render(root) {
  cleanup();
  root.innerHTML = "";

  root.appendChild(greeting());

  board = photoBoard();
  root.appendChild(board.element);

  root.appendChild(recorderPanel());
  root.appendChild(composerPanel());
}

export function cleanup() {
  if (board) {
    board.cleanup();
    board = null;
  }
  if (liveRecorder) {
    // Leaving the screen mid-recording must release the microphone.
    liveRecorder.cancel();
    liveRecorder = null;
  }
}

function greeting() {
  const today = new Date().toLocaleDateString("tr-TR", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
  return el("header", { class: "greeting" }, [
    el("div", { class: "greeting-date" }, [today]),
    el("h2", { class: "greeting-title" }, ["Bugün ne oldu?"]),
  ]);
}

/* ---------- Voice ---------- */

function recorderPanel() {
  const state = { recording: false, elapsed: 0, tickHandle: null, lastTake: null };

  const levelEl = el("div", { class: "recorder-level" });
  const timeEl = el("div", { class: "recorder-time" }, ["00:00"]);
  const micLabel = el("span", { class: "recorder-mic-label" }, ["Başlat"]);
  const micBtn = el(
    "button",
    { class: "recorder-mic", type: "button", "aria-label": "Konuşmaya başla" },
    [icon("mic", { size: 52 }), micLabel]
  );
  const saveBtn = el("button", { class: "cta", type: "button", hidden: "" }, [
    "Günlüğüme Kaydet",
  ]);
  const discardBtn = el("button", { class: "cta ghost", type: "button", hidden: "" }, [
    "Bu Kaydı Sil",
  ]);

  const ui = { levelEl, timeEl, micBtn, micLabel, saveBtn, discardBtn };

  micBtn.addEventListener("click", async () => {
    if (state.recording) await stopRecording(state, ui);
    else await startRecording(state, ui);
  });

  saveBtn.addEventListener("click", async () => {
    if (!state.lastTake) return;
    saveBtn.setAttribute("disabled", "");
    saveBtn.textContent = "Kaydediliyor…";
    await queueAudio(state.lastTake);
    state.lastTake = null;
    resetRecorder(ui);
    setTimeout(() => (location.hash = "#/entries"), 400);
  });

  discardBtn.addEventListener("click", () => {
    state.lastTake = null;
    resetRecorder(ui);
    toast("Kayıt silindi");
  });

  return el("section", { class: "recorder" }, [
    el("h3", { class: "recorder-title" }, ["Konuşarak Anlat"]),
    el("div", { class: "recorder-stage" }, [levelEl, timeEl, micBtn]),
    el("div", { class: "recorder-actions" }, [saveBtn, discardBtn]),
  ]);
}

async function startRecording(state, ui) {
  try {
    liveRecorder = new Recorder({
      onLevel: (level) => {
        ui.levelEl.style.transform = `scale(${1 + Math.min(0.5, level * 1.3)})`;
        ui.levelEl.style.opacity = String(Math.min(0.75, 0.25 + level));
      },
    });
    await liveRecorder.start();
  } catch (err) {
    console.error(err);
    liveRecorder = null;
    // The panel carries no help text any more, so status only goes to the toast.
    toast("Mikrofona erişilemedi");
    return;
  }

  state.recording = true;
  state.elapsed = 0;
  ui.timeEl.textContent = "00:00";
  ui.micBtn.classList.add("is-recording");
  ui.micBtn.setAttribute("aria-label", "Konuşmayı bitir");
  ui.micLabel.textContent = "Bitir";
  ui.saveBtn.hidden = true;
  ui.discardBtn.hidden = true;

  state.tickHandle = setInterval(() => {
    state.elapsed += 1;
    ui.timeEl.textContent = fmtDuration(state.elapsed);
  }, 1000);
}

async function stopRecording(state, ui) {
  if (!liveRecorder) return;
  clearInterval(state.tickHandle);
  state.tickHandle = null;

  const result = await liveRecorder.stop();
  liveRecorder = null;
  state.recording = false;

  ui.micBtn.classList.remove("is-recording");
  ui.micBtn.setAttribute("aria-label", "Konuşmaya başla");
  ui.micLabel.textContent = "Başlat";
  ui.levelEl.style.transform = "scale(1)";
  ui.levelEl.style.opacity = "0";

  if (!result || !result.blob || result.blob.size === 0) {
    toast("Kayıt boş göründü. Bir daha dener misin?");
    return;
  }

  state.lastTake = { blob: result.blob, duration: result.duration, mime: result.mime };
  ui.timeEl.textContent = fmtDuration(result.duration);
  ui.saveBtn.hidden = false;
  ui.discardBtn.hidden = false;
}

function resetRecorder(ui) {
  ui.timeEl.textContent = "00:00";
  ui.saveBtn.hidden = true;
  ui.discardBtn.hidden = true;
  ui.saveBtn.removeAttribute("disabled");
  ui.saveBtn.textContent = "Günlüğüme Kaydet";
}

async function queueAudio({ blob, duration, mime }) {
  const clientUuid = uuid();
  await enqueue({
    clientUuid,
    recordedAt: new Date().toISOString(),
    durationSeconds: duration,
    language: "tr",
    audioBlob: blob,
    audioName: `recording-${clientUuid}.${fileExtFor(mime)}`,
  });
  toast("Kaydın günlüğüne eklendi");
}

/* ---------- Text ---------- */

function composerPanel() {
  const textarea = el("textarea", {
    class: "composer-field",
    "aria-label": "Günlük yazısı",
    placeholder: "Bugün neler yaptın?",
    rows: "8",
  });

  const counter = el("div", { class: "composer-counter" }, ["0 karakter"]);
  const submit = el("button", { class: "cta", type: "button", disabled: "" }, [
    "Günlüğüme Kaydet",
  ]);

  textarea.addEventListener("input", () => {
    const length = textarea.value.trim().length;
    counter.textContent = `${length} karakter`;
    if (length > 0) submit.removeAttribute("disabled");
    else submit.setAttribute("disabled", "");
  });

  submit.addEventListener("click", async () => {
    const text = textarea.value.trim();
    if (!text) return;
    submit.setAttribute("disabled", "");
    submit.textContent = "Kaydediliyor…";
    try {
      await enqueue({
        clientUuid: uuid(),
        recordedAt: new Date().toISOString(),
        durationSeconds: 0,
        language: "tr",
        transcript: text,
      });
      textarea.value = "";
      counter.textContent = "0 karakter";
      toast("Yazın günlüğüne eklendi");
      setTimeout(() => (location.hash = "#/entries"), 350);
    } catch (err) {
      console.error(err);
      toast("Kaydedilemedi: " + err.message);
      submit.removeAttribute("disabled");
      submit.textContent = "Günlüğüme Kaydet";
    }
  });

  return el("section", { class: "composer" }, [
    el("h3", { class: "composer-title" }, ["Yazarak Anlat"]),
    textarea,
    el("div", { class: "composer-footer" }, [counter, submit]),
  ]);
}
