import { Recorder, fileExtFor, fmtDuration } from "../audio.js";
import { uuid } from "../db.js";
import { enqueue } from "../sync.js";
import { toast } from "../events.js";
import { el } from "./utils.js";

export async function render(root) {
  root.innerHTML = "";
  root.appendChild(heroRecorder());
  root.appendChild(textComposer());
}

function heroRecorder() {
  const state = {
    recording: false,
    elapsed: 0,
    tickHandle: null,
    recorder: null,
    lastBlob: null,
  };

  const timeEl = el("div", { class: "hero-time" }, ["00:00"]);
  const hintEl = el("div", { class: "hero-hint" }, ["Kaydetmek için butona dokun"]);
  const levelEl = el("div", { class: "hero-level" });

  const micBtn = el(
    "button",
    {
      class: "hero-mic",
      "aria-label": "Kaydı başlat",
    },
    [el("span", { class: "hero-mic-dot", "aria-hidden": "true" })]
  );

  const finishBtn = el(
    "button",
    { class: "cta", style: "display:none;" },
    ["Günlüğe Ekle"]
  );
  const discardBtn = el(
    "button",
    { class: "cta ghost", style: "display:none;" },
    ["Vazgeç"]
  );

  const hero = el("section", { class: "hero" }, [
    el("div", { class: "hero-inner" }, [
      el("div", { class: "hero-eyebrow" }, ["Hatırlaf"]),
      el("h1", { class: "hero-title" }, ["Sesli günlük, özel kalır"]),
      el("p", { class: "hero-lede" }, [
        "Mikrofonu aç, gününü anlat. Yazıya dökelim, tarih ve kişileri kendimiz çözelim.",
      ]),
      el("div", { class: "hero-stage" }, [
        levelEl,
        timeEl,
        micBtn,
        hintEl,
      ]),
      el("div", { class: "hero-actions" }, [finishBtn, discardBtn]),
    ]),
  ]);

  const ui = { timeEl, hintEl, levelEl, micBtn, finishBtn, discardBtn };

  micBtn.addEventListener("click", async () => {
    if (!state.recording) await startRecording(state, ui);
    else await stopRecording(state, ui);
  });

  finishBtn.addEventListener("click", async () => {
    if (!state.lastBlob) return;
    finishBtn.setAttribute("disabled", "");
    finishBtn.textContent = "Gönderiliyor…";
    await persistAudio(state.lastBlob);
    state.lastBlob = null;
    resetUI(ui);
    setTimeout(() => (location.hash = "#/record"), 400);
  });

  discardBtn.addEventListener("click", () => {
    state.lastBlob = null;
    resetUI(ui);
    toast("Kayıt silindi");
  });

  return hero;
}

async function startRecording(state, ui) {
  try {
    state.recorder = new Recorder({
      onLevel: (lvl) => {
        // Adjust formula for the softer glow effect
        ui.levelEl.style.transform = `scale(${1 + Math.min(0.6, lvl * 1.5)})`;
        ui.levelEl.style.opacity = String(Math.min(0.5, 0.15 + lvl));
      },
    });
    await state.recorder.start();
  } catch (err) {
    console.error(err);
    toast("Mikrofona erişilemedi. Aşağıdan yazarak ekleyebilirsin.");
    return;
  }
  state.recording = true;
  state.elapsed = 0;
  ui.micBtn.classList.add("is-recording");
  ui.micBtn.setAttribute("aria-label", "Kaydı bitir");
  ui.hintEl.textContent = "Kayıt sürüyor — bitirmek için butona dokun";
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
  ui.micBtn.classList.remove("is-recording");
  ui.micBtn.setAttribute("aria-label", "Kaydı başlat");
  
  // Hide glow entirely when stopped
  ui.levelEl.style.transform = "scale(1)";
  ui.levelEl.style.opacity = "0"; 
  
  if (!result || !result.blob || result.blob.size === 0) {
    ui.hintEl.textContent = "Kayıt boş göründü.";
    return;
  }
  state.lastBlob = { blob: result.blob, duration: result.duration, mime: result.mime };
  ui.hintEl.textContent = `${fmtDuration(result.duration)} kaydedildi — göndermeyi onayla.`;
  ui.timeEl.textContent = fmtDuration(result.duration);
  ui.finishBtn.style.display = "block";
  ui.discardBtn.style.display = "block";
}

async function persistAudio({ blob, duration, mime }) {
  const clientUuid = uuid();
  await enqueue({
    clientUuid,
    recordedAt: new Date().toISOString(),
    durationSeconds: duration,
    language: "tr",
    audioBlob: blob,
    audioName: `recording-${clientUuid}.${fileExtFor(mime)}`,
  });
  toast("Sıraya alındı");
}

function resetUI(ui) {
  ui.timeEl.textContent = "00:00";
  ui.hintEl.textContent = "Kaydetmek için butona dokun";
  ui.finishBtn.style.display = "none";
  ui.discardBtn.style.display = "none";
  ui.finishBtn.removeAttribute("disabled");
  ui.finishBtn.textContent = "Günlüğe Ekle";
}

function textComposer() {
  const today = new Date().toLocaleDateString("tr-TR", {
    weekday: "long", day: "numeric", month: "long",
  });

  const textarea = el("textarea", {
    class: "composer-field",
    placeholder:
      "Bugün neler yaptın? Kimlerle görüştün, nereye gittin?\n\nÖrn: Sabah Ayşe ile Bağdat Caddesinde yürüdük. Yarın Ahmet ile üniversitede buluşacağız.",
    rows: "8",
  });

  const submit = el(
    "button",
    { class: "cta", disabled: "" },
    ["Günlüğe Ekle"]
  );
  const counter = el("div", { class: "composer-counter" }, ["0 karakter"]);

  textarea.addEventListener("input", () => {
    const len = textarea.value.trim().length;
    counter.textContent = `${len} karakter`;
    if (len > 0) submit.removeAttribute("disabled");
    else submit.setAttribute("disabled", "");
  });

  submit.addEventListener("click", async () => {
    const text = textarea.value.trim();
    if (!text) {
      toast("Önce metni yaz");
      return;
    }
    submit.setAttribute("disabled", "");
    submit.textContent = "Gönderiliyor…";
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
      toast("Giriş eklendi");
      setTimeout(() => (location.hash = "#/record"), 350);
    } catch (err) {
      console.error(err);
      toast("Hata: " + err.message);
      submit.removeAttribute("disabled");
      submit.textContent = "Günlüğe Ekle";
    }
  });

  return el("section", { class: "composer" }, [
    el("div", { class: "composer-header" }, [
      el("div", { class: "composer-eyebrow" }, ["Yazarak Ekle"]),
      el("div", { class: "composer-date" }, [today]),
    ]),
    el("p", { class: "composer-hint" }, [
      "Mikrofon yerine yazmak istersen buradan ekle. Tarih, zaman, kişi ve yerleri sistem kendisi çıkarır.",
    ]),
    textarea,
    el("div", { class: "composer-footer" }, [counter, submit]),
  ]);
}