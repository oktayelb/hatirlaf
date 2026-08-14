// Reader-controlled text size.
//
// Every font size in the stylesheet is a multiple of --text-scale, so
// changing this one number resizes the whole app without touching layout.

const KEY = "hatirlaf-text-scale";

export const TEXT_SIZES = [
  { id: "small", label: "Küçük", scale: 0.9 },
  { id: "normal", label: "Normal", scale: 1 },
  { id: "large", label: "Büyük", scale: 1.15 },
  { id: "xlarge", label: "En Büyük", scale: 1.3 },
];

export function applyTextSize(id, { persist = false } = {}) {
  const choice = TEXT_SIZES.find((s) => s.id === id) || TEXT_SIZES[1];
  document.documentElement.style.setProperty("--text-scale", String(choice.scale));
  if (persist) {
    try {
      localStorage.setItem(KEY, String(choice.scale));
    } catch (_) {}
  }
  return choice;
}

export function currentTextSize() {
  let stored = "";
  try {
    stored = localStorage.getItem(KEY) || "";
  } catch (_) {}
  const scale = Number(stored);
  return TEXT_SIZES.find((s) => s.scale === scale) || TEXT_SIZES[1];
}

export function initTextSize() {
  return applyTextSize(currentTextSize().id);
}
