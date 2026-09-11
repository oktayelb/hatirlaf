// Everything the app knows about diary entries. Nothing here talks to a
// network; the phone is the only copy, which is why deleteEntry also unlinks
// the audio file and why saveRecording moves audio out of the cache directory
// the moment it exists.

import * as FileSystem from "expo-file-system/legacy";
import { db } from "./db";

export const AUDIO_DIR = `${FileSystem.documentDirectory}hatirlaf-audio/`;

async function ensureAudioDir() {
  const info = await FileSystem.getInfoAsync(AUDIO_DIR);
  if (!info.exists) await FileSystem.makeDirectoryAsync(AUDIO_DIR, { intermediates: true });
}

export function newId() {
  if (global.crypto?.randomUUID) return global.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Newest first. The whole diary — there is no pagination yet. */
export async function listEntries() {
  const database = await db();
  return database.getAllAsync(
    `SELECT id, recorded_at, transcript, audio_path, duration_ms, source
       FROM entries
      ORDER BY recorded_at DESC`
  );
}

export async function countEntries() {
  const database = await db();
  const row = await database.getFirstAsync(`SELECT COUNT(*) AS n FROM entries`);
  return row?.n ?? 0;
}

export async function saveTextEntry(transcript) {
  const trimmed = String(transcript || "").trim();
  if (!trimmed) throw new Error("Boş yazı kaydedilmez.");
  const now = new Date().toISOString();
  const database = await db();
  const id = newId();
  await database.runAsync(
    `INSERT INTO entries
       (id, recorded_at, transcript, audio_path, duration_ms, source, created_at, updated_at)
     VALUES (?, ?, ?, NULL, 0, 'text', ?, ?)`,
    [id, now, trimmed, now, now]
  );
  return id;
}

/**
 * Store a finished recording.
 *
 * `sourceUri` is where the recorder left the file — usually the OS cache
 * directory, which the system is free to purge. We move it under
 * documentDirectory first and only then write the row, so a row never
 * references audio that has already evaporated.
 */
export async function saveRecording({ sourceUri, transcript = "", durationMs = 0, source = "speech" }) {
  const now = new Date().toISOString();
  const id = newId();
  let storedPath = null;

  if (sourceUri) {
    await ensureAudioDir();
    const ext = (String(sourceUri).split("?")[0].split(".").pop() || "wav").slice(0, 5);
    storedPath = `${AUDIO_DIR}${id}.${ext}`;
    try {
      await FileSystem.moveAsync({ from: sourceUri, to: storedPath });
    } catch (_) {
      // Some platforms hand back a URI that cannot be moved across volumes.
      await FileSystem.copyAsync({ from: sourceUri, to: storedPath });
      await FileSystem.deleteAsync(sourceUri, { idempotent: true }).catch(() => {});
    }
  }

  const database = await db();
  await database.runAsync(
    `INSERT INTO entries
       (id, recorded_at, transcript, audio_path, duration_ms, source, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [id, now, String(transcript || "").trim(), storedPath, Math.round(durationMs), source, now, now]
  );
  return id;
}

// A WAV the recogniser opened but never wrote samples into. Both platforms are
// asked for 16 kHz 16-bit mono — 32 kB of audio per second — so a file this
// small holds under half a second whatever the wall clock claimed. Asking the
// file is the only honest test: a recogniser can hold a session open for a
// minute and put nothing in it.
const MIN_AUDIO_BYTES = 16000;

/** Whether a finished recording actually contains anything. */
export async function hasUsableAudio(uri) {
  if (!uri) return false;
  const info = await FileSystem.getInfoAsync(uri, { size: true }).catch(() => null);
  return Boolean(info?.exists) && (info.size || 0) >= MIN_AUDIO_BYTES;
}

/**
 * Throw away audio we have decided not to keep. The recogniser creates its
 * file before it tries to recognise anything, so a session that dies on its
 * first frame still leaves one behind — an empty WAV that no row will ever
 * point at.
 */
export async function discardRecording(uri) {
  if (!uri) return;
  await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
}

export async function updateTranscript(id, transcript) {
  const database = await db();
  await database.runAsync(
    `UPDATE entries SET transcript = ?, updated_at = ? WHERE id = ?`,
    [String(transcript || ""), new Date().toISOString(), id]
  );
}

export async function deleteEntry(id) {
  const database = await db();
  const row = await database.getFirstAsync(`SELECT audio_path FROM entries WHERE id = ?`, [id]);
  await database.runAsync(`DELETE FROM entries WHERE id = ?`, [id]);
  if (row?.audio_path) {
    await FileSystem.deleteAsync(row.audio_path, { idempotent: true }).catch(() => {});
  }
}

/** Total bytes of stored audio — shown in Ayarlar so the size is never a surprise. */
export async function audioFootprint() {
  const info = await FileSystem.getInfoAsync(AUDIO_DIR);
  if (!info.exists) return 0;
  const names = await FileSystem.readDirectoryAsync(AUDIO_DIR);
  let total = 0;
  for (const name of names) {
    const file = await FileSystem.getInfoAsync(`${AUDIO_DIR}${name}`, { size: true });
    total += file.size || 0;
  }
  return total;
}

export function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  const mb = bytes / (1024 * 1024);
  if (mb < 1024) return `${mb.toFixed(mb < 10 ? 1 : 0)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

/** Orphaned files from a crash between the move and the insert. */
export async function pruneOrphanAudio() {
  const info = await FileSystem.getInfoAsync(AUDIO_DIR);
  if (!info.exists) return 0;
  const database = await db();
  const rows = await database.getAllAsync(
    `SELECT audio_path FROM entries WHERE audio_path IS NOT NULL`
  );
  const known = new Set(rows.map((row) => row.audio_path));
  const names = await FileSystem.readDirectoryAsync(AUDIO_DIR);
  let removed = 0;
  for (const name of names) {
    const path = `${AUDIO_DIR}${name}`;
    if (known.has(path)) continue;
    await FileSystem.deleteAsync(path, { idempotent: true }).catch(() => {});
    removed += 1;
  }
  return removed;
}
