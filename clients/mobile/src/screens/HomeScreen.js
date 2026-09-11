// Ana — photos, a microphone, a box to write in. Nothing else.
//
// The microphone has two behaviours depending on what the phone can do
// privately. When Turkish on-device recognition is available we record and
// transcribe in one pass; otherwise we record audio only. See speech.js for
// why we never fall back to network recognition.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  AppState,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from "expo-audio";
import { useSpeechRecognitionEvent } from "expo-speech-recognition";
import { Ionicons } from "@expo/vector-icons";
import { Button, Card, Help } from "../ui/Primitives";
import { PhotoBoard } from "../ui/PhotoBoard";
import {
  discardRecording,
  hasUsableAudio,
  saveRecording,
  saveTextEntry,
} from "../services/entries";
import {
  abortListening,
  detectCapabilities,
  downloadTurkishModel,
  explainMode,
  isFatalSpeechError,
  isMissingModelError,
  requestPermissions,
  startListening,
  stopListening,
} from "../services/speech";
import { colors, radius, spacing, type } from "../theme";

// How long to wait for the recogniser's `end` event after asking it to stop.
// Android flushes a final result first, which takes a beat on a cold model.
const STOP_TIMEOUT_MS = 5000;

// Android 14 downloads the language pack in the background; 13 opens a system
// dialog instead, which is not a thing to spring on someone who has not asked
// for anything yet. There the download waits until a recording needs it.
const SILENT_MODEL_DOWNLOAD = Platform.OS === "android" && Number(Platform.Version) >= 34;

export function HomeScreen({ onSaved }) {
  const [capabilities, setCapabilities] = useState(null);
  const [recording, setRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [liveText, setLiveText] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  // Refs, not state: the speech events fire outside React's render cycle and
  // the values have to survive until `end` arrives to be written together.
  const finalsRef = useRef([]);
  const audioUriRef = useRef(null);
  const startedAtRef = useRef(0);
  // Set by the error handler to the recogniser's own error code when it gives
  // up mid-session, read by the `end` handler to decide what to do about it.
  const speechFailureRef = useRef(null);
  const stopWatchdogRef = useRef(null);
  // The model download is asked for at most once per app run, so a phone that
  // cannot install it does not nag on every recording.
  const modelRequestedRef = useRef(false);

  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const audioRecorderState = useAudioRecorderState(audioRecorder);

  const speechMode = capabilities?.mode === "speech";

  const refreshCapabilities = useCallback(async () => {
    try {
      setCapabilities(await detectCapabilities());
    } catch (_) {
      setCapabilities({ mode: "audio", canDownloadModel: false, reason: "unavailable" });
    }
  }, []);

  useEffect(() => {
    refreshCapabilities();
  }, [refreshCapabilities]);

  // Installing the Turkish model sends the user out to a system screen — on
  // Android 13 the download even resolves before it has finished. Re-asking on
  // the way back is the only way the answer stops being stale without a
  // restart. Never while recording: detection talks to the same recogniser.
  useEffect(() => {
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active" && !recording) refreshCapabilities();
    });
    return () => sub.remove();
  }, [recording, refreshCapabilities]);

  // One timer for both capture paths, so the display cannot disagree with
  // itself when the mode changes.
  useEffect(() => {
    if (!recording) return undefined;
    const id = setInterval(() => setElapsedMs(Date.now() - startedAtRef.current), 250);
    return () => clearInterval(id);
  }, [recording]);

  // Recording without transcription — the plain expo-audio path. Used when the
  // phone cannot transcribe privately, and as the landing spot when the
  // recogniser fails part-way through a session.
  const startAudioRecording = useCallback(async () => {
    // iOS refuses to record until the session allows it. Without this the
    // recorder throws RecordingDisabledException and the button does nothing.
    await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
    await audioRecorder.prepareToRecordAsync();
    audioRecorder.record();
  }, [audioRecorder]);

  const clearStopWatchdog = useCallback(() => {
    if (stopWatchdogRef.current) {
      clearTimeout(stopWatchdogRef.current);
      stopWatchdogRef.current = null;
    }
  }, []);

  // Android keeps Turkish speech recognition behind a separate download and
  // the user should not have to know that. Ask for it as soon as we learn it
  // is missing, rather than parking the answer behind a button.
  const installModelQuietly = useCallback(async () => {
    if (modelRequestedRef.current || Platform.OS !== "android") return;
    modelRequestedRef.current = true;
    try {
      const result = await downloadTurkishModel();
      if (result.status === "download_canceled") return;
      if (result.status === "download_success") {
        await refreshCapabilities();
        Alert.alert(
          "Türkçe hazır",
          "Dil paketi kuruldu. Bundan sonra konuştuklarını yazıya çevirebilirim."
        );
        return;
      }
      // Android 13 resolves the moment the system dialog opens; the real
      // answer arrives when the user comes back, which the AppState listener
      // above picks up.
      Alert.alert(
        "Türkçe dil paketi",
        "Konuştuklarını yazıya çevirebilmem için telefonun indirme ekranını açtım. İndirme bitince buraya dön."
      );
    } catch (_) {
      // Nothing to do: the recording is still kept, and Ana explains why
      // there is no transcript.
    }
  }, [refreshCapabilities]);

  // Detection can tell us the model is missing before anything is recorded.
  useEffect(() => {
    if (capabilities?.reason === "model_missing" && SILENT_MODEL_DOWNLOAD) {
      installModelQuietly();
    }
  }, [capabilities?.reason, installModelQuietly]);

  // What a fatal recogniser error means for the rest of the session. This runs
  // whether or not audio came back: `persist: true` writes the file before
  // recognition is attempted, so gating it on "nothing was captured" — as it
  // was — meant the failure was swallowed on every single session, leaving
  // silent transcript-less entries and no explanation on screen.
  const noteSpeechFailure = useCallback(
    (code) => {
      const missingModel = Platform.OS === "android" && isMissingModelError(code);
      setCapabilities((current) => ({
        ...(current || {}),
        mode: "audio",
        canDownloadModel: Platform.OS === "android",
        reason: missingModel ? "model_missing" : "speech_failed",
        errorCode: code,
      }));
      // A missing model is the one failure the app can repair by itself, and
      // by now the user has asked for a transcript by tapping the microphone,
      // so the system dialog is no longer an interruption out of nowhere.
      if (missingModel) installModelQuietly();
    },
    [installModelQuietly]
  );

  // `end` is the only event guaranteed to arrive last — after the final
  // `result` and after `audioend` has released the file — so the entry is
  // written there rather than in the stop handler.
  //
  // The captured values are read and cleared in one step at the top, which is
  // what makes a second call — the stop watchdog racing a late `end` — see an
  // empty session and return instead of writing the entry twice.
  const finishSpeechCapture = useCallback(async () => {
    clearStopWatchdog();
    const transcript = finalsRef.current.join(" ").trim();
    const uri = audioUriRef.current;
    const durationMs = startedAtRef.current ? Date.now() - startedAtRef.current : 0;
    const failure = speechFailureRef.current;
    finalsRef.current = [];
    audioUriRef.current = null;
    speechFailureRef.current = null;

    if (failure) noteSpeechFailure(failure);

    // A session the recogniser refused still leaves a file behind, containing
    // nothing. Saving it fills Günlüğüm with silent, textless rows that look
    // like the app is losing recordings. The file is asked directly rather
    // than trusting the elapsed clock: a recogniser can hold a session open
    // for a minute and never write a sample into it.
    const keepAudio = await hasUsableAudio(uri);
    const worthKeeping = Boolean(transcript) || keepAudio;

    // The recogniser gave up before anything was said. Rather than hand back a
    // button that did nothing, carry the same session on as a plain recording.
    if (failure && !worthKeeping) {
      await discardRecording(uri);
      try {
        await startAudioRecording();
        startedAtRef.current = Date.now();
        setElapsedMs(0);
        setLiveText("");
        return;
      } catch (_) {
        // Nothing was captured either way; fall through to the ordinary reset.
      }
    }

    setRecording(false);
    // Recognition can end on its own without anything having been captured.
    if (!worthKeeping) {
      await discardRecording(uri);
      resetCapture();
      return;
    }

    // Words but no audio: the recogniser heard us and wrote nothing. Keep the
    // text and drop the file, so the entry does not offer a play button that
    // plays silence.
    if (!keepAudio) await discardRecording(uri);

    setBusy(true);
    try {
      await saveRecording({
        sourceUri: keepAudio ? uri : null,
        transcript,
        durationMs: keepAudio ? durationMs : 0,
        source: "speech",
      });
      resetCapture();
      onSaved?.();
      Alert.alert(
        "Kaydedildi",
        transcript ? "Anlattıkların günlüğüne eklendi." : "Sesin günlüğüne eklendi."
      );
    } catch (err) {
      Alert.alert("Kaydedilemedi", String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }, [clearStopWatchdog, noteSpeechFailure, onSaved, startAudioRecording]);

  useEffect(() => clearStopWatchdog, [clearStopWatchdog]);

  useSpeechRecognitionEvent("result", (event) => {
    const transcript = event.results?.[0]?.transcript ?? "";
    if (event.isFinal) {
      if (transcript) finalsRef.current.push(transcript);
      setLiveText(finalsRef.current.join(" "));
    } else {
      setLiveText([...finalsRef.current, transcript].join(" "));
    }
  });

  useSpeechRecognitionEvent("audioend", (event) => {
    audioUriRef.current = event.uri;
  });

  useSpeechRecognitionEvent("error", (event) => {
    // `no-speech` and `speech-timeout` just mean silence; not worth alarming
    // anyone over. Everything falls through to `end`, which keeps the audio.
    if (event.error === "no-speech" || event.error === "speech-timeout") return;

    if (event.error === "not-allowed") {
      Alert.alert(
        "Mikrofon izni gerekli",
        "Konuşarak günlük tutabilmek için mikrofon iznini açman gerekiyor."
      );
      return;
    }

    // The recogniser cannot run on this phone. `end` handles the consequences:
    // it saves whatever audio was captured, or drops to a plain recording so
    // the user can keep talking.
    if (isFatalSpeechError(event.error)) {
      speechFailureRef.current = event.error;
      return;
    }

    Alert.alert("Kayıt sorunu", "Konuşma tanıma durdu. Sesin yine de kaydedildi.");
  });

  useSpeechRecognitionEvent("end", finishSpeechCapture);

  function resetCapture() {
    finalsRef.current = [];
    audioUriRef.current = null;
    speechFailureRef.current = null;
    startedAtRef.current = 0;
    setLiveText("");
    setElapsedMs(0);
    setRecording(false);
  }

  async function start() {
    if (!capabilities) return;
    try {
      if (speechMode) {
        if (!(await requestPermissions())) {
          Alert.alert(
            "Mikrofon izni gerekli",
            "Konuşarak günlük tutabilmek için mikrofon iznini açman gerekiyor."
          );
          return;
        }
        finalsRef.current = [];
        audioUriRef.current = null;
        speechFailureRef.current = null;
        startedAtRef.current = Date.now();
        setLiveText("");
        setElapsedMs(0);
        setRecording(true);
        startListening();
        return;
      }

      const permission = await requestRecordingPermissionsAsync();
      if (!permission.granted) {
        Alert.alert(
          "Mikrofon izni gerekli",
          "Konuşarak günlük tutabilmek için mikrofon iznini açman gerekiyor."
        );
        return;
      }
      await startAudioRecording();
      startedAtRef.current = Date.now();
      setElapsedMs(0);
      setRecording(true);
    } catch (err) {
      setRecording(false);
      Alert.alert("Kayıt başlatılamadı", String(err?.message || err));
    }
  }

  async function stop() {
    if (speechMode) {
      // The rest of the work happens in the `end` handler, once the recogniser
      // has flushed its final result and released the audio file. If `end`
      // never arrives — the session already ended, or the recogniser wedged —
      // force the issue, so the button cannot stay stuck on "Bitir".
      stopListening();
      clearStopWatchdog();
      stopWatchdogRef.current = setTimeout(() => {
        stopWatchdogRef.current = null;
        abortListening();
        finishSpeechCapture();
      }, STOP_TIMEOUT_MS);
      return;
    }

    setBusy(true);
    try {
      await audioRecorder.stop();
      const durationMs = startedAtRef.current ? Date.now() - startedAtRef.current : 0;
      if (!audioRecorder.uri) {
        Alert.alert("Kayıt boş", "Ses kaydedilemedi. Bir daha dener misin?");
        return;
      }
      await saveRecording({ sourceUri: audioRecorder.uri, durationMs, source: "audio" });
      resetCapture();
      onSaved?.();
      Alert.alert("Kaydedildi", "Sesin günlüğüne eklendi.");
    } catch (err) {
      Alert.alert("Kaydedilemedi", String(err?.message || err));
    } finally {
      setRecording(false);
      setBusy(false);
    }
  }

  async function submitText() {
    const trimmed = text.trim();
    if (!trimmed) return;
    setBusy(true);
    try {
      await saveTextEntry(trimmed);
      setText("");
      onSaved?.();
      Alert.alert("Kaydedildi", "Yazın günlüğüne eklendi.");
    } catch (err) {
      Alert.alert("Kaydedilemedi", String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  // The button, for when the automatic attempt was declined or never ran. An
  // explicit press always tries again, whatever happened earlier.
  async function installModel() {
    modelRequestedRef.current = false;
    setBusy(true);
    try {
      await installModelQuietly();
    } finally {
      setBusy(false);
    }
  }

  const today = useMemo(
    () =>
      new Date().toLocaleDateString("tr-TR", {
        weekday: "long",
        day: "numeric",
        month: "long",
        year: "numeric",
      }),
    []
  );

  // The audio-only path has its own duration source; prefer it when it is live.
  const shownMs = !speechMode && recording ? audioRecorderState.durationMillis || elapsedMs : elapsedMs;
  const elapsed = useMemo(() => {
    const seconds = Math.floor(shownMs / 1000);
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }, [shownMs]);

  const note = capabilities ? explainMode(capabilities) : "";

  return (
    <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
      <View>
        <Text style={styles.date}>{today}</Text>
        <Text style={styles.greeting}>Bugün ne oldu?</Text>
      </View>

      <PhotoBoard />

      <Card style={styles.recorder}>
        <Text style={styles.cardTitle}>Konuşarak Anlat</Text>

        <Text style={styles.time}>{elapsed}</Text>

        <Pressable
          accessibilityRole="button"
          accessibilityLabel={recording ? "Konuşmayı bitir" : "Konuşmaya başla"}
          disabled={busy || !capabilities}
          onPress={recording ? stop : start}
          style={({ pressed }) => [
            styles.mic,
            recording && styles.micRecording,
            (busy || !capabilities) && styles.micDisabled,
            pressed && styles.micPressed,
          ]}
        >
          <Ionicons name={recording ? "stop" : "mic"} size={52} color={colors.accentInk} />
          <Text style={styles.micLabel}>{recording ? "Bitir" : "Başlat"}</Text>
        </Pressable>

        {speechMode && (recording || liveText) ? (
          <View style={styles.live}>
            <Text style={styles.liveText}>
              {liveText || "Dinliyorum…"}
            </Text>
          </View>
        ) : null}

        {note ? <Help>{note}</Help> : null}

        {capabilities?.canDownloadModel ? (
          <Button variant="ghost" disabled={busy} onPress={installModel}>
            Türkçe Dil Paketini Kur
          </Button>
        ) : null}
      </Card>

      <Card style={styles.composer}>
        <Text style={styles.cardTitle}>Yazarak Anlat</Text>
        <TextInput
          value={text}
          onChangeText={setText}
          multiline
          accessibilityLabel="Günlük yazısı"
          placeholder={"Bugün neler yaptın?"}
          placeholderTextColor={colors.faint}
          style={styles.input}
        />
        <Button disabled={busy || !text.trim()} onPress={submitText}>
          Günlüğüme Kaydet
        </Button>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
    gap: spacing.lg,
  },
  date: {
    color: colors.muted,
    fontSize: type.base,
    textTransform: "capitalize",
  },
  greeting: {
    color: colors.text,
    fontSize: type.xxl,
    fontWeight: "700",
    marginTop: spacing.xs,
  },
  cardTitle: {
    color: colors.text,
    fontSize: type.xl,
    fontWeight: "700",
    textAlign: "center",
  },
  recorder: {
    alignItems: "center",
    gap: spacing.sm,
  },
  time: {
    color: colors.muted,
    fontSize: type.lg,
    fontWeight: "700",
    fontVariant: ["tabular-nums"],
    marginTop: spacing.sm,
  },
  mic: {
    width: 152,
    height: 152,
    borderRadius: 76,
    backgroundColor: colors.accent,
    borderWidth: 4,
    borderColor: colors.accentDeep,
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.xs,
  },
  micRecording: {
    backgroundColor: colors.clay,
    borderColor: "#7f3a28",
  },
  micDisabled: {
    opacity: 0.5,
  },
  micPressed: {
    transform: [{ scale: 0.98 }],
  },
  micLabel: {
    color: colors.accentInk,
    fontSize: type.base,
    fontWeight: "700",
  },
  live: {
    alignSelf: "stretch",
    backgroundColor: colors.surface2,
    borderColor: colors.lineStrong,
    borderWidth: 2,
    borderRadius: radius.sm,
    padding: spacing.md,
  },
  liveText: {
    color: colors.text,
    fontSize: type.md,
    lineHeight: type.md * 1.5,
  },
  composer: {
    gap: spacing.sm,
  },
  input: {
    minHeight: 190,
    color: colors.text,
    fontSize: type.md,
    lineHeight: type.md * 1.5,
    backgroundColor: colors.surface2,
    borderColor: colors.lineStrong,
    borderWidth: 2,
    borderRadius: radius.sm,
    padding: spacing.md,
    textAlignVertical: "top",
  },
});
