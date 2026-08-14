// Background sync queue. Polls pendingStore and pushes to the API when the
// network is reachable. This mirrors the mobile spec's "queue audio and
// sync when connectivity returns" requirement.

import { api } from "./api.js";
import { pendingStore } from "./db.js";
import { toast, emit } from "./events.js";

let _flushing = false;

export async function enqueue(record) {
  await pendingStore.put(record);
  emit("pending-changed");
  flush();
}

export async function flush() {
  if (_flushing) return;
  if (!navigator.onLine) return;
  _flushing = true;
  try {
    const queued = await pendingStore.all();
    for (const rec of queued) {
      try {
        const form = new FormData();
        form.append("client_uuid", rec.clientUuid);
        form.append("recorded_at", rec.recordedAt);
        form.append("duration_seconds", String(rec.durationSeconds || 0));
        form.append("language", rec.language || "tr");
        if (rec.audioBlob) {
          const name = rec.audioName || "recording.webm";
          form.append("audio", rec.audioBlob, name);
        }
        if (rec.transcript) {
          form.append("transcript", rec.transcript);
        }
        const resp = await api.uploadSession(form);
        await pendingStore.delete(rec.clientUuid);
        emit("pending-changed");
        emit("session-uploaded", resp);
        toast(`Yüklendi: ${formatShort(rec.recordedAt)}`);
      } catch (err) {
        console.warn("sync failed for", rec.clientUuid, err);
        // Leave it in the queue for a later retry. Don't spam toasts.
        break;
      }
    }
  } finally {
    _flushing = false;
  }
}

function formatShort(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    return iso;
  }
}

// Trigger a flush whenever we come back online.
window.addEventListener("online", () => {
  emit("online-changed", true);
  flush();
});
window.addEventListener("offline", () => emit("online-changed", false));

// Periodic retry (every 20s) — cheap insurance when `online` event lies.
setInterval(() => {
  if (navigator.onLine) flush();
}, 20000);
