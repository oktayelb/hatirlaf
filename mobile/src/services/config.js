import Constants from "expo-constants";
import * as SecureStore from "expo-secure-store";

const API_BASE_KEY = "hatirlaf.apiBase";
const SESSION_COOKIE_KEY = "hatirlaf.sessionCookie";

export async function getApiBase() {
  const stored = await SecureStore.getItemAsync(API_BASE_KEY);
  return normalizeApiBase(
    stored || Constants.expoConfig?.extra?.defaultApiBase || "http://127.0.0.1:8001/api"
  );
}

export async function setApiBase(value) {
  await SecureStore.setItemAsync(API_BASE_KEY, normalizeApiBase(value));
}

export async function getSessionCookie() {
  return (await SecureStore.getItemAsync(SESSION_COOKIE_KEY)) || "";
}

export async function setSessionCookie(cookie) {
  if (!cookie) return;
  await SecureStore.setItemAsync(SESSION_COOKIE_KEY, cookie);
}

export async function clearSessionCookie() {
  await SecureStore.deleteItemAsync(SESSION_COOKIE_KEY);
}

function normalizeApiBase(value) {
  const trimmed = String(value || "").trim().replace(/\/$/, "");
  if (!trimmed) return "http://127.0.0.1:8001/api";
  return trimmed.endsWith("/api") ? trimmed : `${trimmed}/api`;
}
