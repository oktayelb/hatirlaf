// Server feature flags. Same contract as the web client: ask once, and
// assume the smallest app if the answer never arrives.

import { api } from "./api";

const FALLBACK = { nlp: false };

let features = { ...FALLBACK };

export async function loadFeatures() {
  try {
    const data = await api.config();
    features = { ...FALLBACK, ...(data.features || {}) };
  } catch (err) {
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
