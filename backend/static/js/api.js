// Thin REST client. All calls go through a single wrapper so failures route
// into the offline-queue system in db.js without scattering try/catch.
const BASE = (window.HATIRLAF_API_BASE || "/api").replace(/\/$/, "");

async function request(path, opts = {}) {
  const url = `${BASE}${path}`;
  const init = { credentials: "same-origin", ...opts };
  if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(init.body);
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const err = new Error(`HTTP ${res.status}: ${text}`);
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null;
  const ct = res.headers.get("Content-Type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

export const api = {
  health: () => request("/health/"),
  listSessions: () => request("/sessions/"),
  getSession: (id) => request(`/sessions/${id}/`),
  uploadSession: (form) => request("/sessions/", { method: "POST", body: form }),
  reprocess: (id) => request(`/sessions/${id}/process/`, { method: "POST" }),
  audioUrl: (id) => `${BASE}/sessions/${id}/audio/`,
  listNodes: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/nodes/${q ? "?" + q : ""}`);
  },
  createNode: (data) => request("/nodes/", { method: "POST", body: data }),
  listMentions: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/mentions/${q ? "?" + q : ""}`);
  },
  resolveMention: (id, payload) =>
    request(`/mentions/${id}/resolve/`, { method: "POST", body: payload }),
  timeline: () => request("/timeline/"),
  calendar: (month) => request(`/calendar/${month ? `?month=${month}` : ""}`),
  graph: () => request("/graph/"),
};
