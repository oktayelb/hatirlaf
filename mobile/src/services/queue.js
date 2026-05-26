import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system/legacy";
import { api, LockedError } from "./api";

const QUEUE_KEY = "hatirlaf.pendingSessions";
const AUDIO_DIR = `${FileSystem.documentDirectory}hatirlaf-audio/`;

export async function ensureAudioDir() {
  const info = await FileSystem.getInfoAsync(AUDIO_DIR);
  if (!info.exists) await FileSystem.makeDirectoryAsync(AUDIO_DIR, { intermediates: true });
}

export async function enqueueText(transcript) {
  const item = {
    id: uuid(),
    type: "text",
    transcript,
    recordedAt: new Date().toISOString(),
    durationSeconds: 0,
    language: "tr",
  };
  await appendQueue(item);
  return item;
}

export async function enqueueAudio({ uri, durationSeconds = 0 }) {
  await ensureAudioDir();
  const id = uuid();
  const ext = extensionFromUri(uri);
  const target = `${AUDIO_DIR}${id}.${ext}`;
  await FileSystem.copyAsync({ from: uri, to: target });
  const item = {
    id,
    type: "audio",
    audioUri: target,
    audioName: `recording-${id}.${ext}`,
    recordedAt: new Date().toISOString(),
    durationSeconds,
    language: "tr",
  };
  await appendQueue(item);
  return item;
}

export async function queueCount() {
  return (await readQueue()).length;
}

export async function readQueue() {
  const raw = await AsyncStorage.getItem(QUEUE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

export async function flushQueue() {
  const queue = await readQueue();
  const remaining = [];
  const uploaded = [];

  for (const item of queue) {
    try {
      const form = new FormData();
      form.append("client_uuid", item.id);
      form.append("recorded_at", item.recordedAt);
      form.append("duration_seconds", String(item.durationSeconds || 0));
      form.append("language", item.language || "tr");
      if (item.transcript) form.append("transcript", item.transcript);
      if (item.audioUri) {
        form.append("audio", {
          uri: item.audioUri,
          name: item.audioName || "recording.m4a",
          type: mimeFromName(item.audioName),
        });
      }
      await api.uploadSession(form);
      uploaded.push(item);
      if (item.audioUri) {
        await FileSystem.deleteAsync(item.audioUri, { idempotent: true });
      }
    } catch (err) {
      remaining.push(item);
      if (err instanceof LockedError) break;
    }
  }

  await writeQueue(remaining);
  return { uploaded, remaining };
}

async function appendQueue(item) {
  const queue = await readQueue();
  queue.push(item);
  await writeQueue(queue);
}

async function writeQueue(queue) {
  await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
}

function extensionFromUri(uri) {
  const clean = String(uri || "").split("?")[0];
  const ext = clean.includes(".") ? clean.split(".").pop() : "m4a";
  return ext || "m4a";
}

function mimeFromName(name = "") {
  const lower = name.toLowerCase();
  if (lower.endsWith(".webm")) return "audio/webm";
  if (lower.endsWith(".wav")) return "audio/wav";
  if (lower.endsWith(".mp3")) return "audio/mpeg";
  return "audio/mp4";
}

function uuid() {
  if (global.crypto?.randomUUID) return global.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
