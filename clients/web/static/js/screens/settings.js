// Ayarlar — reachable from the gear in the header, not from the main tabs.
// Two things live here: how big the writing is, and the app password.

import { api } from "../api.js";
import { emit, toast } from "../events.js";
import { TEXT_SIZES, applyTextSize, currentTextSize } from "../textsize.js";
import { el } from "./utils.js";

export async function render(root) {
  root.innerHTML = "";
  const status = await loadStatus();

  root.appendChild(
    el(
      "section",
      { class: "settings" },
      [textSizePanel(), privacyPanel(status, root), status.password_enabled ? lockPanel() : null].filter(
        Boolean
      )
    )
  );
}

/* ---------- Text size ---------- */

function textSizePanel() {
  const active = currentTextSize();
  const row = el("div", { class: "text-size-row" });

  for (const size of TEXT_SIZES) {
    const button = el(
      "button",
      {
        class: `text-size-btn ${size.id === active.id ? "active" : ""}`,
        type: "button",
        "data-size": size.id,
        "aria-pressed": String(size.id === active.id),
      },
      [
        el("span", { class: "text-size-sample" }, ["Aa"]),
        el("span", { class: "text-size-name" }, [size.label]),
      ]
    );
    button.addEventListener("click", () => {
      applyTextSize(size.id, { persist: true });
      for (const peer of row.querySelectorAll(".text-size-btn")) {
        const isActive = peer.dataset.size === size.id;
        peer.classList.toggle("active", isActive);
        peer.setAttribute("aria-pressed", String(isActive));
      }
      toast(`Yazı boyutu: ${size.label}`);
    });
    row.appendChild(button);
  }

  return el("div", { class: "settings-panel" }, [
    el("h2", { class: "settings-title" }, ["Yazı Boyutu"]),
    el("p", { class: "settings-copy" }, [
      "Uygulamadaki bütün yazıları büyütebilir ya da küçültebilirsin. Seçtiğin boyut hatırlanır.",
    ]),
    row,
  ]);
}

/* ---------- Privacy ---------- */

async function loadStatus() {
  try {
    return await api.privacyStatus();
  } catch (err) {
    toast("Gizlilik durumu alınamadı: " + err.message);
    return { password_enabled: false, unlocked: true };
  }
}

function privacyPanel(status, root) {
  return el("div", { class: "settings-panel" }, [
    el("div", { class: "settings-panel-head" }, [
      el("div", {}, [
        el("h2", { class: "settings-title" }, ["Parola"]),
        el("p", { class: "settings-copy" }, [
          "Bir parola koyarsan, günlüğünü açmak için her seferinde bu parola sorulur. " +
            "Kayıtların bu cihazda şifreli olarak saklanır.",
        ]),
      ]),
      el("span", { class: `settings-status ${status.password_enabled ? "on" : ""}` }, [
        status.password_enabled ? "Parola var" : "Parola yok",
      ]),
    ]),
    passwordForm(status, root),
  ]);
}

function passwordForm(status, root) {
  const current = el("input", {
    class: "settings-input",
    type: "password",
    autocomplete: "current-password",
  });
  const next = el("input", {
    class: "settings-input",
    type: "password",
    autocomplete: "new-password",
  });
  const confirm = el("input", {
    class: "settings-input",
    type: "password",
    autocomplete: "new-password",
  });

  const saveLabel = status.password_enabled ? "Parolayı Değiştir" : "Parola Koy";
  const save = el("button", { class: "cta settings-action", type: "button" }, [saveLabel]);
  const clear = el("button", { class: "cta ghost settings-action", type: "button" }, [
    "Parolayı Kaldır",
  ]);

  save.addEventListener("click", async () => {
    if (next.value.length < 6) {
      toast("Parola en az 6 karakter olmalı.");
      return;
    }
    if (next.value !== confirm.value) {
      toast("İki parola aynı değil.");
      return;
    }
    save.setAttribute("disabled", "");
    save.textContent = "Kaydediliyor…";
    try {
      await api.setPrivacyPassword({
        current_password: current.value,
        new_password: next.value,
      });
      toast(status.password_enabled ? "Parola değiştirildi" : "Parola konuldu");
      emit("privacy-updated");
      await render(root);
      return;
    } catch (err) {
      toast(readApiError(err));
    }
    save.removeAttribute("disabled");
    save.textContent = saveLabel;
  });

  clear.addEventListener("click", async () => {
    if (!current.value) {
      toast("Önce mevcut parolanı yaz.");
      return;
    }
    clear.setAttribute("disabled", "");
    clear.textContent = "Kaldırılıyor…";
    try {
      await api.clearPrivacyPassword(current.value);
      toast("Parola kaldırıldı");
      emit("privacy-updated");
      await render(root);
      return;
    } catch (err) {
      toast(readApiError(err));
    }
    clear.removeAttribute("disabled");
    clear.textContent = "Parolayı Kaldır";
  });

  return el(
    "div",
    { class: "settings-form" },
    [
      status.password_enabled
        ? el("label", { class: "settings-field" }, [
            el("span", {}, ["Şimdiki parolan"]),
            current,
          ])
        : null,
      el("label", { class: "settings-field" }, [
        el("span", {}, [status.password_enabled ? "Yeni parola" : "Parola"]),
        next,
      ]),
      el("label", { class: "settings-field" }, [
        el("span", {}, ["Parolayı bir kez daha yaz"]),
        confirm,
      ]),
      el(
        "div",
        { class: "settings-actions" },
        [save, status.password_enabled ? clear : null].filter(Boolean)
      ),
    ].filter(Boolean)
  );
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
  return el("div", { class: "settings-panel" }, [
    el("h2", { class: "settings-title" }, ["Günlüğü Kilitle"]),
    el("p", { class: "settings-copy" }, [
      "Kilitledikten sonra günlüğünü görmek için parolanı yazman gerekir.",
    ]),
    el("div", { class: "settings-actions" }, [lock]),
  ]);
}

function readApiError(err) {
  const raw = String(err.message || "");
  const match = raw.match(/\{.*\}$/);
  if (!match) return "Hata: " + raw;
  try {
    return JSON.parse(match[0]).detail || raw;
  } catch (_) {
    return "Hata: " + raw;
  }
}
