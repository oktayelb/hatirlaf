import { getApiBase, getSessionCookie, setSessionCookie } from "./config";

export class LockedError extends Error {
  constructor(message = "Uygulama kilitli") {
    super(message);
    this.name = "LockedError";
    this.status = 423;
  }
}

async function request(path, options = {}) {
  const base = await getApiBase();
  const cookie = await getSessionCookie();
  const headers = {
    ...(cookie ? { Cookie: cookie } : {}),
    ...(options.headers || {}),
  };
  const init = {
    ...options,
    headers,
  };

  if (init.body && !(init.body instanceof FormData) && typeof init.body !== "string") {
    init.headers = { "Content-Type": "application/json", ...headers };
    init.body = JSON.stringify(init.body);
  }

  const response = await fetch(`${base}${path}`, init);
  const setCookie = response.headers.get("set-cookie");
  if (setCookie) await setSessionCookie(setCookie);
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    if (response.status === 423) throw new LockedError();
    throw new Error(text || `HTTP ${response.status}`);
  }
  if (response.status === 204) return null;
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) return response.json();
  return response;
}

export const api = {
  health: () => request("/health/"),
  privacyStatus: () => request("/privacy/status/"),
  unlockPrivacy: (password) =>
    request("/privacy/unlock/", { method: "POST", body: { password } }),
  lockPrivacy: () => request("/privacy/lock/", { method: "POST" }),
  setPrivacyPassword: (payload) =>
    request("/privacy/set-password/", { method: "POST", body: payload }),
  clearPrivacyPassword: (currentPassword) =>
    request("/privacy/clear-password/", {
      method: "POST",
      body: { current_password: currentPassword },
    }),
  listSessions: () => request("/sessions/"),
  uploadSession: (form) =>
    request("/sessions/", {
      method: "POST",
      body: form,
    }),
  updateSession: (id, data) =>
    request(`/sessions/${id}/`, { method: "PATCH", body: data }),
  reprocess: (id) => request(`/sessions/${id}/process/`, { method: "POST" }),
  calendar: (month) => request(`/calendar/${month ? `?month=${month}` : ""}`),
  recap: (month) => request(`/recap/${month ? `?month=${month}` : ""}`),
};
