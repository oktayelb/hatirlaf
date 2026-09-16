// Turkish speech-to-text, on the phone.
//
// The rule this module exists to enforce: audio never leaves the device. Both
// platforms will happily fall back to network recognition — on Android that
// means shipping diary audio to Google — so `requiresOnDeviceRecognition` is
// always true and we would rather record without a transcript than transcribe
// off-device. Everything here is capability detection in service of that.

import { Linking, Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ExpoSpeechRecognitionModule } from "expo-speech-recognition";

export const LOCALE = "tr-TR";

// Android System Intelligence, which ships Google's offline models. Only used
// as a fallback probe — see `turkishModelState` for why it is not the first
// thing we ask.
const ANDROID_ON_DEVICE_PACKAGE = "com.google.android.as";

// Android 13 is the first version that can be asked for a language pack at
// all, and the first that can persist audio, so the two gates coincide.
const ANDROID_API = Platform.OS === "android" ? Number(Platform.Version) : 0;
const CAN_TRIGGER_DOWNLOAD = ANDROID_API >= 33;

// Android 14 added the progress listener. Below it the trigger opens a system
// screen and tells us nothing, which is not something to spring on someone who
// has not asked for it — see the auto-install gate in HomeScreen.
export const DOWNLOAD_RUNS_IN_BACKGROUND = ANDROID_API >= 34;

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

// What the recogniser itself last said about Turkish, kept across restarts.
//
// `getSupportedLocales` is a probe, and on plenty of phones it lists tr-TR as
// installed for a recogniser that then refuses the language outright. When the
// two disagree, the one that actually tried to transcribe is right. Without
// remembering that, every capability refresh — and there is one on every
// return to the app — believed the probe again and hid the install button, so
// the only moment it was ever visible was the few seconds between the
// recogniser failing and the next refresh. That is the "button only appears
// while recording" bug.
const MISSING_KEY = "stt.turkish.missing";
let missingVerdict = null;
let verdictLoaded = false;

async function turkishReportedMissing() {
  if (verdictLoaded) return missingVerdict;
  try {
    missingVerdict = (await AsyncStorage.getItem(MISSING_KEY)) === "1";
  } catch (_) {
    missingVerdict = false;
  }
  verdictLoaded = true;
  return missingVerdict;
}

/** The recogniser refused Turkish. Remember it past this app run. */
export async function rememberTurkishMissing() {
  verdictLoaded = true;
  if (missingVerdict === true) return;
  missingVerdict = true;
  try {
    await AsyncStorage.setItem(MISSING_KEY, "1");
  } catch (_) {
    // A verdict we cannot write is only a verdict we forget on restart.
  }
}

/** Turkish demonstrably works. Clears any stale verdict above. */
export async function forgetTurkishMissing() {
  verdictLoaded = true;
  if (missingVerdict === false) return;
  missingVerdict = false;
  try {
    await AsyncStorage.removeItem(MISSING_KEY);
  } catch (_) {
    // Same.
  }
}

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
  if (Platform.OS !== "android") {
    if (!ExpoSpeechRecognitionModule.isRecognitionAvailable()) {
      return { ...base, reason: "unavailable" };
    }
    // iOS reports on-device support truthfully and installs its own models.
    return { ...base, mode: "speech" };
  }

  const [state, reportedMissing] = await Promise.all([
    turkishModelState(),
    turkishReportedMissing(),
  ]);

  // The probe is certain: no transcript to be had, and a download to offer.
  if (state === "missing") {
    return { ...base, canDownloadModel: CAN_TRIGGER_DOWNLOAD, reason: "model_missing" };
  }

  // The probe says Turkish is there (or cannot say) but the recogniser refused
  // it last time. Show the install button — that is the part the old code got
  // wrong — yet still attempt a transcript, because that attempt is the only
  // thing that can ever clear a stale verdict: a pack installed from the
  // system settings does not announce itself, and a phone parked in audio-only
  // forever is worse than a second of failing over on each recording.
  if (reportedMissing) {
    return {
      ...base,
      mode: "speech",
      canDownloadModel: CAN_TRIGGER_DOWNLOAD,
      reason: "model_unproven",
    };
  }

  // "unknown" means the probe could not answer, not that Turkish is absent.
  // Try to transcribe and let `start()` be the judge — a failure there falls
  // back to a plain recording and is remembered, so nothing is lost by trying.
  return { ...base, mode: "speech" };
}

/** "installed" | "missing" | "unknown" */
export async function turkishModelState() {
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

/**
 * The one fatal error the app can do something about. Android raises it when
 * the recogniser has no offline model for `LOCALE`, which a download fixes —
 * so it must not be treated like a phone that simply cannot transcribe.
 */
export function isMissingModelError(code) {
  return code === "language-not-supported";
}

/**
 * Outcomes of asking for the language pack, which the UI has to tell apart.
 *
 * The library's own three statuses collapse into something misleading: its
 * `download_canceled` is what the Android 14 listener reports from
 * `onScheduled`, which means the opposite of cancelled — the system accepted
 * the request and will run the download in the background, typically once the
 * phone is on Wi-Fi. Reading that as a refusal is why the old code went
 * silent, changed nothing, and then offered the same button again.
 */
export const DOWNLOAD = {
  INSTALLED: "installed", // Finished. Turkish is on the phone now.
  SCHEDULED: "scheduled", // Queued by Android; it lands in its own time.
  OPENED: "opened", // Android 13 opened its own screen; outcome unknown.
  RUNNING: "running", // One from earlier this run has not reported back.
  UNSUPPORTED: "unsupported", // Too old to be asked at all.
  FAILED: "failed", // The recogniser refused; `code` says how.
};

// A scheduled download never reports back, so the native module's own
// in-flight flag stays set for the rest of the process and rejects every later
// call with `download_in_progress`. Mirroring it here lets automatic attempts
// skip a call we know will be refused, while an explicit press still tries.
let downloadInFlight = false;

/**
 * Ask Android for the Turkish pack. Never throws: every outcome comes back as
 * a `DOWNLOAD` status the home screen can say something honest about.
 */
export async function downloadTurkishModel({ userAsked = false } = {}) {
  if (!CAN_TRIGGER_DOWNLOAD) return { status: DOWNLOAD.UNSUPPORTED };
  if (downloadInFlight && !userAsked) return { status: DOWNLOAD.RUNNING };

  downloadInFlight = true;
  try {
    const result = await ExpoSpeechRecognitionModule.androidTriggerOfflineModelDownload({
      locale: LOCALE,
    });

    if (result?.status === "download_success") {
      downloadInFlight = false;
      await forgetTurkishMissing();
      return { status: DOWNLOAD.INSTALLED };
    }

    // `onScheduled`, despite what the library calls it. Stays in flight.
    if (result?.status === "download_canceled") {
      return { status: DOWNLOAD.SCHEDULED };
    }

    // Android 13: the system screen is open and the real answer arrives when
    // the user comes back, which the home screen re-detects. Nothing is
    // pending natively on this path.
    downloadInFlight = false;
    return { status: DOWNLOAD.OPENED };
  } catch (err) {
    const code = String(err?.code || err?.message || "");
    if (code.includes("download_in_progress")) return { status: DOWNLOAD.RUNNING };
    downloadInFlight = false;
    if (code.includes("not_supported")) return { status: DOWNLOAD.UNSUPPORTED };
    return { status: DOWNLOAD.FAILED, code };
  }
}

// The way in by hand, for phones whose recogniser will not take the request.
// Ordered from the screen that actually lists downloadable voice languages to
// the one every phone has.
const SETTINGS_INTENTS = ["android.settings.VOICE_INPUT_SETTINGS", "android.settings.SETTINGS"];

/** Opens the phone's own speech settings. Returns false if none would open. */
export async function openSpeechSettings() {
  if (Platform.OS !== "android") return false;
  for (const action of SETTINGS_INTENTS) {
    try {
      await Linking.sendIntent(action);
      return true;
    } catch (_) {
      // Not every phone ships every one of these screens.
    }
  }
  return false;
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

/** The shape the speech path records in, and what a backfill should assume. */
export const SPEECH_SAMPLE_RATE = 16000;
export const SPEECH_CHANNELS = 1;

// A file the recogniser never answers about would wedge a whole backfill, so
// every attempt is bounded. The native streamer paces itself at ~15 ms per
// 4 KiB of decoded PCM — comfortably faster than real time — which is why
// twice the clip's own length plus a floor is generous rather than tight.
const FILE_TIMEOUT_FLOOR_MS = 45000;

/**
 * Transcribe audio already on disk, rather than a live microphone.
 *
 * Android streams the file through `MediaCodec`, so any format the phone can
 * decode works — but it decodes to the file's *own* sample rate and channel
 * count, and the recogniser has to be told which those are. Hand it the wrong
 * numbers and it hears a stream at the wrong speed and finds no words in it,
 * which is why entries carry their format rather than the caller guessing.
 *
 * Never throws, and never leaves a listener behind: a backfill runs this once
 * per entry and the subscriptions would otherwise pile up across the run.
 */
export function transcribeAudioFile({ uri, sampleRate, channels, durationMs = 0 }) {
  return new Promise((resolve) => {
    const finals = [];
    const subscriptions = [];
    let errorCode = null;
    let timer = null;
    let settled = false;

    const finish = () => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      for (const subscription of subscriptions) {
        try {
          subscription.remove();
        } catch (_) {
          // An already-removed subscription is not a problem worth raising.
        }
      }
      const transcript = finals.join(" ").replace(/\s+/g, " ").trim();
      resolve({ transcript, error: transcript ? null : errorCode });
    };

    subscriptions.push(
      ExpoSpeechRecognitionModule.addListener("result", (event) => {
        if (!event.isFinal) return;
        const text = event.results?.[0]?.transcript ?? "";
        if (text) finals.push(text);
      })
    );

    subscriptions.push(
      ExpoSpeechRecognitionModule.addListener("error", (event) => {
        // A recording of somebody not saying anything is an ordinary outcome
        // for a diary, not a failure to report back to the user.
        if (event.error === "no-speech" || event.error === "speech-timeout") return;
        errorCode = event.error;
      })
    );

    // `end` arrives after the last final result on every path, including the
    // error ones, so it is the only place this needs to settle.
    subscriptions.push(ExpoSpeechRecognitionModule.addListener("end", finish));

    timer = setTimeout(
      () => {
        abortListening();
        finish();
      },
      Math.max(FILE_TIMEOUT_FLOOR_MS, durationMs * 2)
    );

    try {
      ExpoSpeechRecognitionModule.start({
        lang: LOCALE,
        interimResults: false,
        continuous: true,
        requiresOnDeviceRecognition: true,
        addsPunctuation: true,
        audioSource: {
          uri,
          sampleRate: sampleRate || SPEECH_SAMPLE_RATE,
          audioChannels: channels || SPEECH_CHANNELS,
        },
      });
    } catch (err) {
      errorCode = String(err?.message || err);
      finish();
    }
  });
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
      return "Türkçe dil paketi telefonda yok. Kurulana kadar sesin yalnızca kaydediliyor; aşağıdan kurabilirsin.";
    case "model_unproven":
      // The probe and the recogniser disagree. We try anyway, so the copy has
      // to promise a transcript without depending on one.
      return "Geçen sefer Türkçe dil paketi bulunamadı. Yine de deneyeceğim; olmazsa sesin yalnızca kaydedilir. Aşağıdan kurabilirsin.";
    case "speech_failed": {
      const base =
        "Konuşma tanıma bu telefonda çalışmadı, o yüzden sesin yalnızca kaydediliyor. Türkçe dil paketini kurmayı deneyebilirsin.";
      // There is no console in a release build, so the recogniser's own code
      // is the only clue anyone gets about why this phone refused.
      return capabilities.errorCode ? `${base} (${capabilities.errorCode})` : base;
    }
    default:
      return "";
  }
}

/** Turkish copy for what came back from asking for the language pack. */
export function explainDownload(download) {
  if (!download) return "";
  switch (download.status) {
    case "working":
      return "Türkçe dil paketi isteniyor…";
    case DOWNLOAD.INSTALLED:
      return "Türkçe dil paketi kuruldu. Artık konuştuklarını yazıya çevirebilirim.";
    case DOWNLOAD.SCHEDULED:
      return "Telefon indirmeyi sıraya aldı; kendi zamanında, çoğunlukla Wi-Fi'deyken indirir. Wi-Fi'ye bağlı kal, birkaç dakika sonra uygulamayı kapatıp yeniden aç.";
    case DOWNLOAD.RUNNING:
      return "İndirme sürüyor. Bitmesini bekle, sonra uygulamayı kapatıp yeniden aç. Beklemek istemiyorsan telefonun ayarlarından da kurabilirsin.";
    case DOWNLOAD.OPENED:
      return "Telefonun indirme ekranını açtım. İndirme bitince buraya dön.";
    case DOWNLOAD.UNSUPPORTED:
      return "Bu telefon dil paketini uygulamanın içinden indiremiyor. Telefonun ayarlarından kurman gerekiyor.";
    case DOWNLOAD.FAILED:
      return `Telefon dil paketini indiremedi${download.code ? ` (${download.code})` : ""}. Telefonun ayarlarından elle kurmayı deneyebilirsin.`;
    default:
      return "";
  }
}

/**
 * Whether to offer the manual way in. Anything short of a confirmed install
 * counts — including the Android 13 "opened the system screen" answer, which
 * is a guess: the trigger resolves whether or not a screen ever appeared, and
 * on the phones where none does, the settings route is the only way through.
 */
export function needsManualInstall(download) {
  if (!download) return false;
  return download.status !== "working" && download.status !== DOWNLOAD.INSTALLED;
}
