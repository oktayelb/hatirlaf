// The photo board on the home screen.
//
// One or two pictures the user keeps in front of them while they talk —
// a face to speak to. They live in IndexedDB on this device only: nothing
// here is attached to a diary entry or uploaded anywhere.

import { photoStore } from "./db.js";
import { toast } from "./events.js";
import { icon } from "./icons.js";
import { el } from "./screens/utils.js";

const SLOTS = [0, 1];
const MAX_EDGE = 1280; // plenty for a phone screen, small enough to store
const CAPTIONS = ["1. Fotoğraf", "2. Fotoğraf"];

export function photoBoard() {
  const frames = el("div", { class: "photo-frames" });
  const actions = el("div", { class: "photo-actions" });
  const help = el("p", { class: "help" }, [
    "Konuşurken bakmak istediğin bir fotoğraf ekleyebilirsin. Fotoğraflar yalnızca bu cihazda kalır, hiçbir yere gönderilmez.",
  ]);

  const board = el("section", { class: "photo-board" }, [frames, help, actions]);

  const filePicker = el("input", {
    type: "file",
    accept: "image/*",
    style: { display: "none" },
  });
  board.appendChild(filePicker);

  let objectUrls = [];
  let pickingSlot = 0;

  filePicker.addEventListener("change", async () => {
    const file = filePicker.files && filePicker.files[0];
    filePicker.value = "";
    if (!file) return;
    try {
      const blob = await shrink(file);
      await photoStore.put(pickingSlot, blob);
      toast("Fotoğraf eklendi");
      await paint();
    } catch (err) {
      console.error(err);
      toast("Fotoğraf eklenemedi: " + err.message);
    }
  });

  function choose(slot) {
    pickingSlot = slot;
    filePicker.click();
  }

  async function remove(slot) {
    await photoStore.delete(slot);
    toast("Fotoğraf kaldırıldı");
    await paint();
  }

  async function paint() {
    for (const url of objectUrls) URL.revokeObjectURL(url);
    objectUrls = [];

    const saved = await photoStore.all();
    const bySlot = new Map(saved.map((p) => [p.slot, p]));

    frames.innerHTML = "";
    actions.innerHTML = "";

    // Slot 2 only appears once slot 1 is filled, so a first-time user sees
    // one clear invitation instead of two identical empty boxes.
    const visible = bySlot.has(0) || bySlot.has(1) ? SLOTS : [0];
    frames.classList.toggle("is-pair", visible.length === 2);

    for (const slot of visible) {
      const record = bySlot.get(slot);
      frames.appendChild(record ? filledFrame(slot, record) : emptyFrame(slot));
      if (record) {
        const removeBtn = el("button", { class: "cta ghost", type: "button" }, [
          `${CAPTIONS[slot]}ı kaldır`,
        ]);
        removeBtn.addEventListener("click", () => remove(slot));
        actions.appendChild(removeBtn);
      }
    }

    help.textContent = bySlot.size
      ? "Değiştirmek için fotoğrafın üzerine dokun. Fotoğraflar yalnızca bu cihazda kalır."
      : "Konuşurken bakmak istediğin bir fotoğraf ekleyebilirsin. Fotoğraflar yalnızca bu cihazda kalır, hiçbir yere gönderilmez.";
  }

  function filledFrame(slot, record) {
    const url = URL.createObjectURL(record.blob);
    objectUrls.push(url);
    const img = el("img", { src: url, alt: `${CAPTIONS[slot]}` });
    const frame = el(
      "button",
      {
        class: "photo-frame",
        type: "button",
        "aria-label": `${CAPTIONS[slot]}ı değiştir`,
      },
      [
        el("span", { class: "photo-frame-mat" }, [
          el("span", { class: "photo-frame-inner" }, [img]),
        ]),
        el("span", { class: "photo-frame-caption" }, ["Değiştirmek için dokun"]),
      ]
    );
    frame.addEventListener("click", () => choose(slot));
    return frame;
  }

  function emptyFrame(slot) {
    const frame = el(
      "button",
      {
        class: "photo-frame is-empty",
        type: "button",
        "aria-label": `${CAPTIONS[slot]}ı ekle`,
      },
      [
        el("span", { class: "photo-frame-mat" }, [
          el("span", { class: "photo-frame-inner" }, [
            icon("photo", { size: 44, className: "photo-empty-icon" }),
            el("span", { class: "photo-empty-label" }, ["Fotoğraf Ekle"]),
            el("span", { class: "photo-empty-help" }, [
              slot === 0
                ? "Cihazından bir fotoğraf seç"
                : "İstersen ikinci bir fotoğraf daha ekle",
            ]),
          ]),
        ]),
      ]
    );
    frame.addEventListener("click", () => choose(slot));
    return frame;
  }

  function cleanup() {
    for (const url of objectUrls) URL.revokeObjectURL(url);
    objectUrls = [];
  }

  paint();
  return { element: board, cleanup };
}

/** Downscale to something a phone screen can use, then re-encode as JPEG. */
async function shrink(file) {
  const bitmap = await createImageBitmap(file).catch(() => null);
  if (!bitmap) return file; // exotic format — store it as picked

  const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();

  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.85)
  );
  return blob || file;
}
