// Tiny hyperscript helper — avoids pulling in a virtual-dom framework.

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (k === "html") {
      node.innerHTML = v;
    } else {
      node.setAttribute(k, v);
    }
  }
  for (const c of [].concat(children)) {
    if (c == null || c === false) continue;
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function modal(content, { onClose } = {}) {
  const backdrop = el("div", { class: "modal-backdrop" });
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", onKeyDown);
    backdrop.remove();
    if (onClose) onClose();
  };
  const onKeyDown = (e) => {
    if (e.key === "Escape") close();
  };
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
  const panel = el("div", { class: "modal" });
  const closeBtn = el("button", {
    class: "modal-close",
    type: "button",
    "aria-label": "Kapat",
    onclick: close,
  }, ["×"]);
  panel.appendChild(closeBtn);
  panel.appendChild(content);
  backdrop.appendChild(panel);
  document.body.appendChild(backdrop);
  document.addEventListener("keydown", onKeyDown);
  return { close, panel };
}

export function fmtRelative(iso) {
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "az önce";
  if (diff < 3600) return `${Math.floor(diff / 60)} dk önce`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} saat önce`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} gün önce`;
  return d.toLocaleDateString("tr-TR");
}

export function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export const MENTION_LABELS = {
  PERSON: "Kişi",
  LOCATION: "Yer",
  TIME: "Zaman",
  EVENT: "Olay",
  ORG: "Kurum",
  PRONOUN: "Zamir",
  OTHER: "Diğer",
};

export const NODE_KIND_CLASSES = {
  PERSON: "person",
  LOCATION: "location",
  TIME: "time",
  EVENT: "event",
  ORG: "org",
  OTHER: "",
};
