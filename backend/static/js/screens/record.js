import { Recorder, fileExtFor, fmtDuration } from "../audio.js";
import { uuid } from "../db.js";
import { enqueue } from "../sync.js";
import { toast } from "../events.js";
import { el } from "./utils.js";

export async function render(root) {
  root.innerHTML = "";
  root.appendChild(recordUI());
}

function recordUI() {
  const state = {
    recording: false,
    elapsed: 0,
    tickHandle: null,
    recorder: null,
    lastBlob: null,
    manualMode: false,
  };

  const timeEl = el("div", { class: "record-time" }, ["00:00"]);
  const hintEl = el("div", { class: "record-hint" }, ["Kaydetmek için butona dokun"]);
  const levelEl = el("div", { class: "record-meter-level", style: "height:0%;" });
  const inner = el("div", { class: "record-meter-inner" }, [
    timeEl, hintEl,
  ]);
  const meter = el("div", { class: "record-meter" }, [levelEl, inner]);

  const recordBtn = el("button", { class: "record-cta", "aria-label": "Kaydı başlat" }, ["●"]);

  const finishBtn = el("button", { class: "cta", style: "display:none;" }, ["Yüklemeye Ekle"]);
  const discardBtn = el("button", { class: "cta ghost", style: "display:none;" }, ["Vazgeç"]);

  const manualToggle = el("button", { class: "cta ghost", style: "margin-top:24px;" }, [
    "Yazarak Ekle (mikrofonsuz)",
  ]);

  const manualPanel = el("div", { style: "display:none; margin-top:12px;" }, [
    el("textarea", {
      class: "manual-transcript",
      id: "manual-ta",
      placeholder: "Günlük girişini buraya yaz (Türkçe)...",
    }),
    el("button", { class: "cta", style: "margin-top:10px;" }, ["Metni Gönder"]),
  ]);

  const wrap = el("div", { class: "record" }, [meter, recordBtn, finishBtn, discardBtn, manualToggle, manualPanel]);

  recordBtn.addEventListener("click", async () => {
    if (!state.recording) {
      await startRecording(state, { timeEl, hintEl, levelEl, meter, recordBtn, finishBtn, discardBtn });
    } else {
      await stopRecording(state, { timeEl, hintEl, levelEl, meter, recordBtn, finishBtn, discardBtn });
    }
  });

  finishBtn.addEventListener("click", async () => {
    if (!state.lastBlob) return;
    await persist(state.lastBlob);
    state.lastBlob = null;
    resetUI({ timeEl, hintEl, levelEl, meter, recordBtn, finishBtn, discardBtn });
    setTimeout(() => (location.hash = "#/timeline"), 400);
  });

  discardBtn.addEventListener("click", () => {
    state.lastBlob = null;
    resetUI({ timeEl, hintEl, levelEl, meter, recordBtn, finishBtn, discardBtn });
    toast("Kayıt silindi");
  });

  manualToggle.addEventListener("click", () => {
    state.manualMode = !state.manualMode;
    manualPanel.style.display = state.manualMode ? "block" : "none";
    manualToggle.textContent = state.manualMode
      ? "Mikrofon moduna dön"
      : "Yazarak Ekle (mikrofonsuz)";
  });

  manualPanel.querySelector("button").addEventListener("click", async () => {
    const text = manualPanel.querySelector("textarea").value.trim();
    if (!text) {
      toast("Önce metni yaz");
      return;
    }
    await enqueue({
      clientUuid: uuid(),
      recordedAt: new Date().toISOString(),
      durationSeconds: 0,
      language: "tr",
      transcript: text,
    });
    manualPanel.querySelector("textarea").value = "";
    toast("Metin yüklendi");
    setTimeout(() => (location.hash = "#/timeline"), 400);
  });

  return wrap;
}

async function startRecording(state, ui) {
  try {
    state.recorder = new Recorder({
      onLevel: (lvl) => {
        ui.levelEl.style.height = `${Math.min(100, lvl * 100)}%`;
      },
    });
    await state.recorder.start();
  } catch (err) {
    console.error(err);
    toast("Mikrofona erişilemedi — manuel moda geç.");
    return;
  }
  state.recording = true;
  state.elapsed = 0;
  ui.meter.classList.add("active");
  ui.recordBtn.classList.add("is-stop");
  ui.recordBtn.textContent = "■";
  ui.recordBtn.setAttribute("aria-label", "Kaydı bitir");
  ui.hintEl.textContent = "Kayıt sürüyor…";
  ui.finishBtn.style.display = "none";
  ui.discardBtn.style.display = "none";
  state.tickHandle = setInterval(() => {
    state.elapsed += 1;
    ui.timeEl.textContent = fmtDuration(state.elapsed);
  }, 1000);
}

async function stopRecording(state, ui) {
  if (!state.recorder) return;
  clearInterval(state.tickHandle);
  state.tickHandle = null;
  const result = await state.recorder.stop();
  state.recorder = null;
  state.recording = false;
  ui.meter.classList.remove("active");
  ui.levelEl.style.height = "0%";
  ui.recordBtn.classList.remove("is-stop");
  ui.recordBtn.textContent = "●";
  ui.recordBtn.setAttribute("aria-label", "Kaydı başlat");
  if (!result || !result.blob || result.blob.size === 0) {
    ui.hintEl.textContent = "Kayıt boş göründü.";
    return;
  }
  state.lastBlob = { blob: result.blob, duration: result.duration, mime: result.mime };
  ui.hintEl.textContent = `${fmtDuration(result.duration)} kaydedildi — gönderimi onayla.`;
  ui.timeEl.textContent = fmtDuration(result.duration);
  ui.finishBtn.style.display = "block";
  ui.discardBtn.style.display = "block";
}

async function persist({ blob, duration, mime }) {
  const clientUuid = uuid();
  await enqueue({
    clientUuid,
    recordedAt: new Date().toISOString(),
    durationSeconds: duration,
    language: "tr",
    audioBlob: blob,
    audioName: `recording-${clientUuid}.${fileExtFor(mime)}`,
  });
  toast("Sıraya alındı ✓");
}

function resetUI(ui) {
  ui.timeEl.textContent = "00:00";
  ui.hintEl.textContent = "Kaydetmek için butona dokun";
  ui.finishBtn.style.display = "none";
  ui.discardBtn.style.display = "none";
}
