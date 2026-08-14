// Server feature flags, fetched once at boot.
//
// The client never assumes a feature is on. If /api/config/ can't be
// reached we fall back to the smallest app — record and review — because
// that is the one thing that always works.

import { api } from "./api.js";

const FALLBACK = { nlp: false };

let features = { ...FALLBACK };

export async function loadConfig() {
  try {
    const data = await api.config();
    features = { ...FALLBACK, ...(data.features || {}) };
  } catch (err) {
    console.debug("feature config unavailable, assuming capture-only", err);
    features = { ...FALLBACK };
  }
  return features;
}

export function isEnabled(name) {
  return Boolean(features[name]);
}

export function nlpEnabled() {
  return isEnabled("nlp");
}
