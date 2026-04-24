import { api } from "../api.js";
import { Recorder, fileExtFor, fmtDuration } from "../audio.js";
import { uuid } from "../db.js";
import { enqueue } from "../sync.js";
import { toast } from "../events.js";
import { el } from "./utils.js";

export async function render(root) {
  root.innerHTML = "";

  const stats = await loadStats();

  root.appendChild(heroRecorder());
  root.appendChild(statsCard(stats));
  root.appendChild(quickActions());
}

async function loadStats() {
  try {
    const sessions = await api.listSessions();
    const list = Array.isArray(sessions) ? sessions : sessions.results || [];
    const totalConflicts = list.reduce((n, s) => n + (s.conflict_count || 0), 0);
    return {
      sessionCount: list.length,
      conflictCount: totalConflicts,
    };
  } catch (_err) {
    return { sessionCount: 0, conflictCount: 0 };
  }
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
    await persist(state.lastBlob);
    state.lastBlob = null;
    resetUI(ui);
    setTimeout(() => (location.hash = "#/timeline"), 400);
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
        ui.levelEl.style.transform = `scale(${1 + Math.min(0.35, lvl * 0.7)})`;
        ui.levelEl.style.opacity = String(Math.min(0.85, 0.3 + lvl));
      },
    });
    await state.recorder.start();
  } catch (err) {
    console.error(err);
    toast("Mikrofona erişilemedi. Kaydet sekmesinden yazarak ekleyebilirsin.");
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
  ui.levelEl.style.transform = "scale(1)";
  ui.levelEl.style.opacity = "0.3";
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

function statsCard(stats) {
  const conflictColor = stats.conflictCount ? "var(--warn)" : "var(--ok)";
  return el("div", { class: "stats-card" }, [
    stat(stats.sessionCount, "Kayıt"),
    stat(stats.conflictCount, "Çatışma", conflictColor),
  ]);
}

function stat(value, label, color) {
  return el("div", { class: "stat" }, [
    el(
      "div",
      { class: "stat-value", style: color ? `color:${color};` : "" },
      [String(value)]
    ),
    el("div", { class: "stat-label" }, [label]),
  ]);
}

function quickActions() {
  return el("div", { class: "quick-actions" }, [
    el(
      "button",
      {
        class: "quick-action",
        onclick: () => (location.hash = "#/timeline"),
      },
      [
        el("div", { class: "quick-action-title" }, ["Takvimi aç"]),
        el("div", { class: "quick-action-sub" }, [
          "Aylık görünüme geç ve günlere göz at.",
        ]),
      ]
    ),
    el(
      "button",
      {
        class: "quick-action",
        onclick: () => (location.hash = "#/record"),
      },
      [
        el("div", { class: "quick-action-title" }, ["Metin olarak yaz"]),
        el("div", { class: "quick-action-sub" }, [
          "Mikrofonsuz mu? Kaydet sekmesinde yaz, sistem geri kalanı çözsün.",
        ]),
      ]
    ),
  ]);
}
