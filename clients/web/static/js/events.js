// Ultra-small event bus + toast helper used across screens.

const _listeners = new Map(); // type -> Set<fn>

export function on(type, fn) {
  if (!_listeners.has(type)) _listeners.set(type, new Set());
  _listeners.get(type).add(fn);
  return () => off(type, fn);
}

export function off(type, fn) {
  const set = _listeners.get(type);
  if (set) set.delete(fn);
}

export function emit(type, payload) {
  const set = _listeners.get(type);
  if (!set) return;
  for (const fn of [...set]) {
    try {
      fn(payload);
    } catch (e) {
      console.error("event handler failed", type, e);
    }
  }
}

export function toast(text, opts = {}) {
  const root = document.getElementById("toast-root");
  if (!root) return;
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = text;
  root.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateY(10px)";
    setTimeout(() => el.remove(), 200);
  }, opts.duration || 2400);
}
