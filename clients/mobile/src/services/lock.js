// The app password, kept on the phone.
//
// This used to be a Django session gated by server middleware. With no server
// there is nothing to ask, so the check happens here: a salted SHA-256 of the
// password lives in SecureStore (the iOS keychain / Android keystore) and the
// plaintext is never written anywhere.
//
// Worth being honest about what this does and does not do: it keeps someone
// who picks up the phone out of the diary. It is not encryption — the SQLite
// file and the audio are still readable to anything with filesystem access.

import * as Crypto from "expo-crypto";
import * as SecureStore from "expo-secure-store";

const HASH_KEY = "hatirlaf.lock.hash";
const SALT_KEY = "hatirlaf.lock.salt";

// Unlocking is per-session and deliberately not persisted: closing the app
// relocks it, which is what someone protecting a diary expects.
let unlocked = false;

async function hash(password, salt) {
  return Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    `${salt}:${password}`
  );
}

export async function isPasswordSet() {
  return Boolean(await SecureStore.getItemAsync(HASH_KEY));
}

/** `{ password_enabled, unlocked }` — the shape the screens already expect. */
export async function lockStatus() {
  const enabled = await isPasswordSet();
  return { password_enabled: enabled, unlocked: enabled ? unlocked : true };
}

export async function setPassword(newPassword) {
  const bytes = await Crypto.getRandomBytesAsync(16);
  const salt = Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  await SecureStore.setItemAsync(SALT_KEY, salt);
  await SecureStore.setItemAsync(HASH_KEY, await hash(newPassword, salt));
  unlocked = true;
}

export async function verify(password) {
  const [stored, salt] = await Promise.all([
    SecureStore.getItemAsync(HASH_KEY),
    SecureStore.getItemAsync(SALT_KEY),
  ]);
  if (!stored || !salt) return false;
  return (await hash(password, salt)) === stored;
}

export async function unlock(password) {
  const ok = await verify(password);
  if (ok) unlocked = true;
  return ok;
}

export async function clearPassword(currentPassword) {
  if (!(await verify(currentPassword))) return false;
  await SecureStore.deleteItemAsync(HASH_KEY);
  await SecureStore.deleteItemAsync(SALT_KEY);
  unlocked = true;
  return true;
}

export function lockNow() {
  unlocked = false;
}
