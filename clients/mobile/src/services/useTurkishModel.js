// The Turkish language pack, as a screen sees it.
//
// Ana and Ayarlar both need the same three things — what this phone can do,
// how to ask Android for the pack, and what came back — and two copies of that
// state machine would drift apart the first time either was touched. The
// "asked once" guard is module-level rather than per-hook for the same reason:
// it means once per app run, not once per screen that happens to be mounted.

import { useCallback, useEffect, useRef, useState } from "react";
import { Alert, AppState, Platform } from "react-native";
import {
  DOWNLOAD,
  DOWNLOAD_RUNS_IN_BACKGROUND,
  detectCapabilities,
  downloadTurkishModel,
  isMissingModelError,
  openSpeechSettings,
  rememberTurkishMissing,
} from "./speech";

// Android 14 downloads in the background; 13 opens a system screen instead,
// which is not a thing to spring on someone who has not asked for anything.
const SILENT_MODEL_DOWNLOAD = DOWNLOAD_RUNS_IN_BACKGROUND;

let autoRequested = false;

/**
 * @param canRefresh — false while the microphone is live. Detection talks to
 *   the same recogniser the recording is using, so asking it mid-session is
 *   how you lose the recording.
 */
export function useTurkishModel({ canRefresh = true } = {}) {
  const [capabilities, setCapabilities] = useState(null);
  // `{ status, code? }`, or `{ status: "working" }` while one is in flight.
  const [download, setDownload] = useState(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    let next;
    try {
      next = await detectCapabilities();
    } catch (_) {
      next = { mode: "audio", canDownloadModel: false, reason: "unavailable" };
    }
    if (mountedRef.current) setCapabilities(next);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Installing the pack sends the user out to a system screen — on Android 13
  // the trigger even resolves before the download has finished. Re-asking on
  // the way back is the only way the answer stops being stale without a
  // restart.
  useEffect(() => {
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active" && canRefresh) refresh();
    });
    return () => subscription.remove();
  }, [canRefresh, refresh]);

  const install = useCallback(
    async ({ userAsked = false } = {}) => {
      if (Platform.OS !== "android") return null;
      // Automatic attempts happen once per app run, so a phone that cannot
      // install the pack does not nag. A press always tries.
      if (!userAsked && autoRequested) return null;
      autoRequested = true;

      if (mountedRef.current) setDownload({ status: "working" });
      const result = await downloadTurkishModel({ userAsked });
      if (mountedRef.current) setDownload(result);

      // Only a confirmed install is worth re-detecting for. The rest resolve
      // later, out in the system, and the AppState listener catches those.
      if (result.status === DOWNLOAD.INSTALLED) await refresh();
      return result;
    },
    [refresh]
  );

  // Detection can tell us the pack is missing before anything is recorded.
  useEffect(() => {
    const unproven =
      capabilities?.reason === "model_missing" || capabilities?.reason === "model_unproven";
    if (unproven && SILENT_MODEL_DOWNLOAD) install();
  }, [capabilities?.reason, install]);

  // The phone's own speech settings, for when its recogniser will not take the
  // request. Without this there is no way out of a failed download at all.
  const openSettings = useCallback(async () => {
    if (!(await openSpeechSettings())) {
      Alert.alert(
        "Ayarlar açılamadı",
        "Telefonun ayarlarında “Ses girişi” ya da “Konuşma tanıma” bölümünden Türkçe dil paketini kurabilirsin."
      );
    }
  }, []);

  /**
   * The recogniser gave up mid-session. Its verdict outranks the capability
   * probe, which on many phones cheerfully lists Turkish as installed for a
   * recogniser that then refuses it.
   */
  const reportSpeechFailure = useCallback(
    (code) => {
      const missingModel = Platform.OS === "android" && isMissingModelError(code);
      setCapabilities((current) => ({
        ...(current || {}),
        mode: "audio",
        canDownloadModel: Platform.OS === "android",
        reason: missingModel ? "model_missing" : "speech_failed",
        errorCode: code,
      }));
      if (!missingModel) return;
      // Written down so it survives the refresh on the next return to the app,
      // which is what keeps the install button on screen instead of letting it
      // vanish the moment the recording ends.
      rememberTurkishMissing();
      // By now the user has asked for a transcript by tapping the microphone,
      // so the system screen is no longer an interruption out of nowhere.
      install();
    },
    [install]
  );

  const clearDownload = useCallback(() => setDownload(null), []);

  return {
    capabilities,
    download,
    install,
    openSettings,
    refresh,
    reportSpeechFailure,
    clearDownload,
  };
}
