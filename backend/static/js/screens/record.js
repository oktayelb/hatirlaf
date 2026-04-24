// Text-only entry. The Kaydet screen accepts a Turkish paragraph and
// hands it to the NLP + LLM pipeline through the standard upload path.

import { uuid } from "../db.js";
import { enqueue } from "../sync.js";
import { toast } from "../events.js";
import { el } from "./utils.js";

export async function render(root) {
  root.innerHTML = "";
  root.appendChild(composer());
}

function composer() {
  const today = new Date().toLocaleDateString("tr-TR", {
    weekday: "long", day: "numeric", month: "long",
  });

  const textarea = el("textarea", {
    class: "composer-field",
    placeholder:
      "Bugün neler yaptın? Kimlerle görüştün, nereye gittin?\n\nÖrn: Sabah Ayşe ile Bağdat Caddesinde yürüdük. Yarın Ahmet ile üniversitede buluşacağız.",
    rows: "10",
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
      toast("Giriş eklendi");
      setTimeout(() => (location.hash = "#/timeline"), 350);
    } catch (err) {
      console.error(err);
      toast("Hata: " + err.message);
      submit.removeAttribute("disabled");
      submit.textContent = "Günlüğe Ekle";
    }
  });

  return el("section", { class: "composer" }, [
    el("div", { class: "composer-header" }, [
      el("div", { class: "composer-eyebrow" }, ["Yeni Giriş"]),
      el("div", { class: "composer-date" }, [today]),
    ]),
    el("p", { class: "composer-hint" }, [
      "Gününü kendi cümlelerinle yaz. Tarih, zaman, kişi ve yerleri sistem kendisi çıkaracak.",
    ]),
    textarea,
    el("div", { class: "composer-footer" }, [counter, submit]),
  ]);
}
