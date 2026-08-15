// Ana — photos, a microphone, a box to write in. Nothing else.
//
// The microphone has two behaviours depending on what the phone can do
// privately. When Turkish on-device recognition is available we record and
// transcribe in one pass; otherwise we record audio only. See speech.js for
// why we never fall back to network recognition.

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
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
import { saveRecording, saveTextEntry } from "../services/entries";
import {
  detectCapabilities,
  downloadTurkishModel,
  explainMode,
  requestPermissions,
  startListening,
  stopListening,
} from "../services/speech";
import { colors, radius, spacing, type } from "../theme";

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

  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const audioRecorderState = useAudioRecorderState(audioRecorder);

  const speechMode = capabilities?.mode === "speech";

  useEffect(() => {
    detectCapabilities().then(setCapabilities).catch(() => setCapabilities({ mode: "audio", reason: "unavailable" }));
  }, []);

  // One timer for both capture paths, so the display cannot disagree with
  // itself when the mode changes.
  useEffect(() => {
    if (!recording) return undefined;
    const id = setInterval(() => setElapsedMs(Date.now() - startedAtRef.current), 250);
    return () => clearInterval(id);
  }, [recording]);

  // `end` is the only event guaranteed to arrive last — after the final
  // `result` and after `audioend` has released the file — so the entry is
  // written there rather than in the stop handler.
  const finishSpeechCapture = useCallback(async () => {
    const transcript = finalsRef.current.join(" ").trim();
    const uri = audioUriRef.current;
    const durationMs = startedAtRef.current ? Date.now() - startedAtRef.current : 0;

    setRecording(false);
    // Recognition can end on its own without anything having been captured.
    if (!uri && !transcript) {
      resetCapture();
      return;
    }

    setBusy(true);
    try {
      await saveRecording({ sourceUri: uri, transcript, durationMs, source: "speech" });
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
  }, [onSaved]);

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
    // `no-speech` just means silence; it is not worth alarming anyone over.
    // Everything else still falls through to `end`, which keeps the audio.
    if (event.error === "no-speech") return;
    Alert.alert("Kayıt sorunu", "Konuşma tanıma durdu. Sesin yine de kaydedildi.");
  });

  useSpeechRecognitionEvent("end", finishSpeechCapture);

  function resetCapture() {
    finalsRef.current = [];
    audioUriRef.current = null;
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
      // iOS refuses to record until the session allows it. Without this the
      // recorder throws RecordingDisabledException and the button does nothing.
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();
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
      // has flushed its final result and released the audio file.
      stopListening();
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

  async function installModel() {
    setBusy(true);
    try {
      const result = await downloadTurkishModel();
      if (result.status === "download_canceled") return;
      setCapabilities(await detectCapabilities());
      if (result.status === "opened_dialog") {
        Alert.alert(
          "Dil paketi",
          "Telefonun indirme ekranını açtı. İndirme bitince buraya dön."
        );
      }
    } catch (_) {
      Alert.alert("İndirilemedi", "Türkçe dil paketi kurulamadı. Sesin yine de kaydedilir.");
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
