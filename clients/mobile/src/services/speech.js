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

// Google's on-device recogniser. The generic engine on Android routes to the
// network, so we ask for this one by name when checking for installed models.
const ANDROID_ON_DEVICE_PACKAGE = "com.google.android.as";

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

  if (!ExpoSpeechRecognitionModule.isRecognitionAvailable()) {
    return { ...base, reason: "unavailable" };
  }

  // Persisting audio needs Android 13+. Without it there is no recording to
  // keep, and a diary that only stores text is not the app we are building.
  if (!ExpoSpeechRecognitionModule.supportsRecording()) {
    return { ...base, reason: "no_recording" };
  }

  if (!ExpoSpeechRecognitionModule.supportsOnDeviceRecognition()) {
    return { ...base, reason: "no_on_device" };
  }

  if (Platform.OS === "android") {
    // iOS reports on-device support truthfully; Android needs the language
    // pack to actually be present, which is a separate question.
    const installed = await hasTurkishModel();
    if (!installed) {
      return { ...base, canDownloadModel: true, reason: "model_missing" };
    }
  }

  return { ...base, mode: "speech" };
}

async function hasTurkishModel() {
  try {
    const { installedLocales } = await ExpoSpeechRecognitionModule.getSupportedLocales({
      androidRecognitionServicePackage: ANDROID_ON_DEVICE_PACKAGE,
    });
    return installedLocales.some((locale) => locale.toLowerCase().startsWith("tr"));
  } catch (_) {
    // getSupportedLocales throws on Android 12 and below, where there is no
    // on-device story at all. Treat that as "no model".
    return false;
  }
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
      // the size for no benefit to a spoken-word diary.
      outputSampleRate: 16000,
      outputEncoding: "pcmFormatInt16",
    },
  });
}

export function stopListening() {
  ExpoSpeechRecognitionModule.stop();
}

export function abortListening() {
  ExpoSpeechRecognitionModule.abort();
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
    default:
      return "";
  }
}
