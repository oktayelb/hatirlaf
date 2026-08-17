// Turkish speech-to-text, on the phone.
//
// The rule this module exists to enforce: audio never leaves the device. Both
// platforms will happily fall back to network recognition — on Android that
// means shipping diary audio to Google — so `requiresOnDeviceRecognition` is
// always true and we would rather record without a transcript than transcribe
// off-device. Everything here is capability detection in service of that.

import { Platform } from "react-native";
import { ExpoSpeechRecognitionModule } from "expo-speech-recognition";

export const LOCALE = "tr-TR";

// Android System Intelligence, which ships Google's offline models. Only used
// as a fallback probe — see `turkishModelState` for why it is not the first
// thing we ask.
const ANDROID_ON_DEVICE_PACKAGE = "com.google.android.as";

// Errors that mean the recogniser will not run at all on this device right
// now. The home screen answers these by dropping to a plain recording rather
// than leaving the user with nothing. `no-speech`, `speech-timeout`, `nomatch`
// and `aborted` are ordinary ends to a session, not failures.
const FATAL_ERRORS = new Set([
  "audio-capture",
  "busy",
  "client",
  "language-not-supported",
  "network",
  "service-not-allowed",
  "unknown",
]);

/**
 * What this device can actually do, in the order the UI cares about.
 *
 * - `mode: "speech"`  — live Turkish transcript plus a saved recording.
 * - `mode: "audio"`   — recording only. Either the device is too old to
 *                       persist audio from the recogniser, or the Turkish
 *                       model is not installed and we refuse to go online.
 * - `canDownloadModel` — Android can be offered the model download.
 */
export async function detectCapabilities() {
  const base = {
    mode: "audio",
    canDownloadModel: false,
    reason: "",
  };

  // Persisting audio needs Android 13+. Without it there is no recording to
  // keep, and a diary that only stores text is not the app we are building.
  if (!ExpoSpeechRecognitionModule.supportsRecording()) {
    return { ...base, reason: "no_recording" };
  }

  if (!ExpoSpeechRecognitionModule.supportsOnDeviceRecognition()) {
    return { ...base, reason: "no_on_device" };
  }

  // `isRecognitionAvailable()` asks whether the device has a *default*
  // recognition service. On Android that is the wrong question: with
  // `requiresOnDeviceRecognition` the library calls
  // `createOnDeviceSpeechRecognizer()`, which bypasses the default service
  // entirely, so a device can answer "no" here and still transcribe Turkish
  // offline perfectly well. iOS has no such split.
  if (Platform.OS !== "android" && !ExpoSpeechRecognitionModule.isRecognitionAvailable()) {
    return { ...base, reason: "unavailable" };
  }

  if (Platform.OS === "android") {
    // iOS reports on-device support truthfully; Android needs the language
    // pack to actually be present, which is a separate question.
    const state = await turkishModelState();
    if (state === "missing") {
      return { ...base, canDownloadModel: true, reason: "model_missing" };
    }
    // "unknown" means the probe could not answer, not that Turkish is absent.
    // Try to transcribe anyway and let `start()` be the judge — a failure
    // there falls back to a plain recording, so nothing is lost by trying,
    // whereas refusing up front silently costs every transcript on devices
    // whose recogniser does not answer capability queries.
    return { ...base, mode: "speech", canDownloadModel: state === "unknown" };
  }

  return { ...base, mode: "speech" };
}

/** "installed" | "missing" | "unknown" */
async function turkishModelState() {
  // Ask the recogniser we actually use first. With no service package named,
  // `getSupportedLocales` goes through `createOnDeviceSpeechRecognizer()` —
  // the same object `start()` builds. Naming `com.google.android.as` instead
  // sends the call down a package-resolution path that throws outright on
  // devices where the on-device recogniser is not published under that
  // package name, which is why it is only the fallback.
  const direct = await installedTurkish(null);
  if (direct !== "unknown") return direct;
  return installedTurkish(ANDROID_ON_DEVICE_PACKAGE);
}

async function installedTurkish(servicePackage) {
  try {
    const { installedLocales } = await ExpoSpeechRecognitionModule.getSupportedLocales(
      servicePackage ? { androidRecognitionServicePackage: servicePackage } : {}
    );
    if (!Array.isArray(installedLocales)) return "unknown";
    if (installedLocales.some((locale) => locale.toLowerCase().startsWith("tr"))) {
      return "installed";
    }
    // An empty list is what Android 12 and below return, and what a recogniser
    // that cannot report its languages returns. That is not the same as
    // knowing Turkish is absent.
    return installedLocales.length ? "missing" : "unknown";
  } catch (_) {
    // Throws as `package_not_found` when the named service does not exist, and
    // as `error_<n>` when the recogniser refuses the query.
    return "unknown";
  }
}

/** Whether a recogniser error means we should fall back to plain recording. */
export function isFatalSpeechError(code) {
  return FATAL_ERRORS.has(code);
}

/** Opens the system flow that installs the Turkish model. Android 13+ only. */
export async function downloadTurkishModel() {
  return ExpoSpeechRecognitionModule.androidTriggerOfflineModelDownload({ locale: LOCALE });
}

export async function requestPermissions() {
  const result = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
  return Boolean(result.granted);
}

/**
 * Begin listening. Audio is written to the recogniser's own directory and
 * moved somewhere permanent by entries.saveRecording once `audioend` fires —
 * see the note there about the cache being purgeable.
 */
export function startListening() {
  ExpoSpeechRecognitionModule.start({
    lang: LOCALE,
    interimResults: true,
    continuous: true,
    requiresOnDeviceRecognition: true,
    addsPunctuation: true,
    recordingOptions: {
      persist: true,
      // iOS defaults to 32-bit float at the device rate, which is four times
      // the size for no benefit to a spoken-word diary. Android already
      // records 16 kHz 16-bit mono and ignores both of these.
      outputSampleRate: 16000,
      outputEncoding: "pcmFormatInt16",
    },
  });
}

export function stopListening() {
  ExpoSpeechRecognitionModule.stop();
}

export function abortListening() {
  try {
    ExpoSpeechRecognitionModule.abort();
  } catch (_) {
    // Aborting a session that already ended is not an error worth surfacing.
  }
}

/** Turkish copy for the states the home screen has to explain. */
export function explainMode(capabilities) {
  switch (capabilities.reason) {
    case "unavailable":
      return "Bu telefonda konuşma tanıma yok. Sesin kaydedilir, yazıya çevrilmez.";
    case "no_recording":
      return "Bu telefonun Android sürümü konuşurken kayıt tutmayı desteklemiyor. Sesin kaydedilir, yazıya çevrilmez.";
    case "no_on_device":
      return "Bu telefon konuşmayı internete göndermeden çeviremiyor. Günlüğün telefonda kalsın diye sesin yalnızca kaydedilir.";
    case "model_missing":
      return "Türkçe dil paketi kurulu değil. Kurarsan konuştuklarını internete hiç bağlanmadan yazıya çevirebilirim.";
    case "speech_failed":
      return "Konuşma tanıma bu telefonda çalışmadı, o yüzden sesin yalnızca kaydediliyor. Türkçe dil paketini kurmayı deneyebilirsin.";
    default:
      return "";
  }
}
