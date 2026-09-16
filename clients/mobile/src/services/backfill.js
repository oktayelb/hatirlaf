// Giving words to recordings that never got any.
//
// The app keeps the audio whether or not it could transcribe it, so a phone
// that spent weeks without the Turkish pack still has every entry — just
// silent ones. Once the pack arrives, this walks that backlog and fills in
// what the recogniser can now hear, which is the only way those entries ever
// become searchable, readable diary text rather than a list of play buttons.

import * as FileSystem from "expo-file-system/legacy";
import { audioFormat, listUntranscribed, updateTranscript } from "./entries";
import { abortListening, transcribeAudioFile } from "./speech";

// The recogniser is a single shared object: starting a second session while
// one is live fails, and a backfill racing the microphone would lose the
// recording the user is actually making. One run at a time, app-wide.
let running = false;

export function backfillRunning() {
  return running;
}

// Android needs a beat between tearing one recognition session down and
// building the next. Without it the second `start()` lands on a recogniser
// that is still releasing the first and comes back `busy`.
const GAP_MS = 350;

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Work through every entry that has audio but no words.
 *
 * `onProgress` is called before and after each entry so a screen can show the
 * count moving. `shouldStop` is polled between entries, which is how leaving
 * the screen cancels a long run without killing it mid-file.
 *
 * Resolves to a tally rather than throwing: one unreadable file should not
 * abandon the rest of the backlog.
 */
export async function backfillTranscripts({ onProgress, shouldStop } = {}) {
  if (running) return { busy: true, total: 0, written: 0, failed: 0, missing: 0 };
  running = true;

  const tally = { busy: false, total: 0, written: 0, failed: 0, missing: 0, stopped: false };

  try {
    const rows = await listUntranscribed();
    tally.total = rows.length;
    onProgress?.({ done: 0, ...tally });

    for (let index = 0; index < rows.length; index += 1) {
      if (shouldStop?.()) {
        tally.stopped = true;
        break;
      }

      const row = rows[index];
      const done = index + 1;
      onProgress?.({ done, ...tally });

      // The audio directory is ordinary storage the user can clear, and
      // pruneOrphanAudio only deletes the other way round — files with no row.
      const info = await FileSystem.getInfoAsync(row.audio_path).catch(() => null);
      if (!info?.exists) {
        tally.missing += 1;
        onProgress?.({ done, ...tally });
        continue;
      }

      // Insurance: if anything left a session open, this one would come back
      // `busy` and the entry would be counted a failure for no reason.
      abortListening();
      await wait(GAP_MS);

      const { sampleRate, channels } = audioFormat(row);
      const { transcript } = await transcribeAudioFile({
        uri: row.audio_path,
        sampleRate,
        channels,
        durationMs: row.duration_ms,
      });

      if (transcript) {
        await updateTranscript(row.id, transcript);
        tally.written += 1;
      } else {
        // Silence, a format the recogniser could not read, or a model that is
        // still not there. All three leave the entry exactly as it was.
        tally.failed += 1;
      }

      onProgress?.({ done, ...tally });
      await wait(GAP_MS);
    }
  } finally {
    running = false;
    abortListening();
  }

  return tally;
}

/** Turkish copy for how a finished run went. */
export function explainBackfill(tally) {
  if (!tally) return "";
  if (tally.busy) return "Çevirme zaten sürüyor.";
  if (!tally.total) return "Yazıya çevrilecek eski kayıt yok.";

  const parts = [`${tally.written} kayıt yazıya çevrildi`];
  if (tally.failed) parts.push(`${tally.failed} kayıttan söz çıkmadı`);
  if (tally.missing) parts.push(`${tally.missing} kaydın ses dosyası bulunamadı`);
  const body = `${parts.join(", ")}.`;
  return tally.stopped ? `${body} Kalanlar için tekrar başlatabilirsin.` : body;
}
