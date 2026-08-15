// Getting the diary off the phone.
//
// With no server, the phone is the only copy — a lost or reset handset loses
// everything. That makes export a safety feature rather than a convenience,
// so it is deliberately plain: a single readable text file for the writing,
// and the original audio file for any one entry.
//
// What this does not do is bundle every recording into one archive. That
// needs a zip implementation and a folder picker, and doing it badly (a share
// sheet fired once per file) is worse than not offering it. See exportAll's
// return value, which tells the caller how much audio is still phone-only.

import * as FileSystem from "expo-file-system/legacy";
import * as Sharing from "expo-sharing";
import { listEntries } from "./entries";

export async function canShare() {
  return Sharing.isAvailableAsync();
}

function stamp(iso) {
  const date = new Date(iso);
  return `${date.toLocaleDateString("tr-TR", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  })} ${date.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}`;
}

/** Share one entry: its audio if it has any, otherwise its text. */
export async function shareEntry(entry) {
  if (!(await Sharing.isAvailableAsync())) return false;

  if (entry.audio_path) {
    const info = await FileSystem.getInfoAsync(entry.audio_path);
    if (info.exists) {
      await Sharing.shareAsync(entry.audio_path, {
        mimeType: "audio/wav",
        dialogTitle: stamp(entry.recorded_at),
      });
      return true;
    }
  }

  const target = `${FileSystem.cacheDirectory}hatirlaf-kayit.txt`;
  await FileSystem.writeAsStringAsync(
    target,
    `${stamp(entry.recorded_at)}\n\n${entry.transcript || ""}\n`
  );
  await Sharing.shareAsync(target, {
    mimeType: "text/plain",
    dialogTitle: stamp(entry.recorded_at),
  });
  return true;
}

/**
 * Write the whole diary to one text file and hand it to the share sheet.
 *
 * Returns `{ entries, withAudio }` so the caller can say plainly how many
 * recordings were not included.
 */
export async function exportAll() {
  const entries = await listEntries();
  if (!entries.length) return { entries: 0, withAudio: 0, shared: false };

  const lines = [
    "Hatırlaf günlüğü",
    `Dışa aktarma: ${stamp(new Date().toISOString())}`,
    `Toplam kayıt: ${entries.length}`,
    "",
    "".padEnd(60, "="),
    "",
  ];

  let withAudio = 0;
  for (const entry of entries) {
    if (entry.audio_path) withAudio += 1;
    lines.push(stamp(entry.recorded_at));
    if (entry.audio_path) lines.push("(bu kaydın sesi telefonda duruyor)");
    lines.push("");
    lines.push(entry.transcript?.trim() || "(yazı yok)");
    lines.push("");
    lines.push("".padEnd(60, "-"));
    lines.push("");
  }

  const target = `${FileSystem.cacheDirectory}hatirlaf-gunluk.txt`;
  await FileSystem.writeAsStringAsync(target, lines.join("\n"));

  const shared = await Sharing.isAvailableAsync();
  if (shared) {
    await Sharing.shareAsync(target, {
      mimeType: "text/plain",
      dialogTitle: "Günlüğünü dışa aktar",
    });
  }

  return { entries: entries.length, withAudio, shared };
}
