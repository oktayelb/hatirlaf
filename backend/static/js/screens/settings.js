import { api } from "../api.js";
import { emit, toast } from "../events.js";
import { el } from "./utils.js";

export async function render(root) {
  root.innerHTML = "";
  const status = await loadStatus();

  root.appendChild(
    el("section", { class: "settings" }, [
      el("div", { class: "settings-panel" }, [
        el("div", { class: "settings-panel-head" }, [
          el("div", {}, [
            el("h2", { class: "settings-title" }, ["Gizlilik"]),
            el("p", { class: "settings-copy" }, [
              "Uygulama parolası bu tarayıcı oturumunda günlük verilerini kilitler. Yerel veritabanındaki hassas alanlar ve ses dosyaları ayrıca disk üzerinde şifreli tutulur.",
            ]),
          ]),
          el("span", { class: `settings-status ${status.password_enabled ? "on" : "off"}` }, [
            status.password_enabled ? "Parola aktif" : "Parola yok",
          ]),
        ]),
        passwordForm(status, root),
      ]),
      status.password_enabled ? lockPanel() : null,
    ].filter(Boolean))
  );
}

async function loadStatus() {
  try {
    return await api.privacyStatus();
  } catch (err) {
    toast("Gizlilik durumu alınamadı: " + err.message);
    return { password_enabled: false, unlocked: true };
  }
}

function passwordForm(status, root) {
  const current = el("input", {
    class: "settings-input",
    type: "password",
    autocomplete: "current-password",
    placeholder: "Mevcut parola",
  });
  const next = el("input", {
    class: "settings-input",
    type: "password",
    autocomplete: "new-password",
    placeholder: status.password_enabled ? "Yeni parola" : "Yeni parola belirle",
  });
  const confirm = el("input", {
    class: "settings-input",
    type: "password",
    autocomplete: "new-password",
    placeholder: "Yeni parolayı tekrar yaz",
  });
  const save = el("button", { class: "cta settings-action", type: "button" }, [
    status.password_enabled ? "Parolayı Değiştir" : "Parola Ekle",
  ]);
  const clear = el("button", { class: "cta ghost settings-action", type: "button" }, [
    "Parolayı Kaldır",
  ]);

  save.addEventListener("click", async () => {
    const newPassword = next.value;
    if (newPassword.length < 6) {
      toast("Parola en az 6 karakter olmalı.");
      return;
    }
    if (newPassword !== confirm.value) {
      toast("Yeni parolalar eşleşmiyor.");
      return;
    }
    save.setAttribute("disabled", "");
    save.textContent = "Kaydediliyor...";
    try {
      await api.setPrivacyPassword({
        current_password: current.value,
        new_password: newPassword,
      });
      toast(status.password_enabled ? "Parola değiştirildi" : "Parola eklendi");
      emit("privacy-updated");
      await render(root);
    } catch (err) {
      toast(readApiError(err));
    } finally {
      save.removeAttribute("disabled");
      save.textContent = status.password_enabled ? "Parolayı Değiştir" : "Parola Ekle";
    }
  });

  clear.addEventListener("click", async () => {
    if (!current.value) {
      toast("Mevcut parolayı yaz.");
      return;
    }
    clear.setAttribute("disabled", "");
    clear.textContent = "Kaldırılıyor...";
    try {
      await api.clearPrivacyPassword(current.value);
      toast("Parola kaldırıldı");
      emit("privacy-updated");
      await render(root);
    } catch (err) {
      toast(readApiError(err));
    } finally {
      clear.removeAttribute("disabled");
      clear.textContent = "Parolayı Kaldır";
    }
  });

  return el("div", { class: "settings-form" }, [
    status.password_enabled
      ? el("label", { class: "settings-field" }, [
          el("span", {}, ["Mevcut parola"]),
          current,
        ])
      : null,
    el("label", { class: "settings-field" }, [
      el("span", {}, [status.password_enabled ? "Yeni parola" : "Parola"]),
      next,
    ]),
    el("label", { class: "settings-field" }, [
      el("span", {}, ["Tekrar"]),
      confirm,
    ]),
    el("div", { class: "settings-actions" }, [
      save,
      status.password_enabled ? clear : null,
    ].filter(Boolean)),
  ].filter(Boolean));
}

function lockPanel() {
  const lock = el("button", { class: "cta danger settings-action", type: "button" }, [
    "Şimdi Kilitle",
  ]);
  lock.addEventListener("click", async () => {
    lock.setAttribute("disabled", "");
    try {
      await api.lockPrivacy();
      emit("privacy-locked");
    } catch (err) {
      toast(readApiError(err));
      lock.removeAttribute("disabled");
    }
  });
  return el("div", { class: "settings-panel compact" }, [
    el("h3", { class: "settings-subtitle" }, ["Oturumu Kilitle"]),
    el("p", { class: "settings-copy" }, [
      "Kilitlendikten sonra kayıtlar, takvim, anılar ve özet ekranları parolayı isteyecek.",
    ]),
    lock,
  ]);
}

function readApiError(err) {
  const raw = String(err.message || "");
  const match = raw.match(/\{.*\}$/);
  if (!match) return "Hata: " + raw;
  try {
    const data = JSON.parse(match[0]);
    return data.detail || raw;
  } catch (_) {
    return "Hata: " + raw;
  }
}
