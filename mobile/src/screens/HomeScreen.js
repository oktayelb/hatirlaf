// Ana — photos, a microphone, a box to write in. Nothing else.

import React, { useMemo, useState } from "react";
import { Alert, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from "expo-audio";
import { Ionicons } from "@expo/vector-icons";
import { Pressable } from "react-native";
import { Button, Card, Help } from "../ui/Primitives";
import { PhotoBoard } from "../ui/PhotoBoard";
import { enqueueAudio, enqueueText, flushQueue, queueCount } from "../services/queue";
import { colors, radius, spacing, type } from "../theme";

export function HomeScreen({ onQueueChanged }) {
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [queued, setQueued] = useState(0);

  const recording = recorderState.isRecording;

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

  const elapsed = useMemo(() => {
    const seconds = Math.floor((recorderState.durationMillis || 0) / 1000);
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }, [recorderState.durationMillis]);

  async function start() {
    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        "Mikrofon izni gerekli",
        "Konuşarak günlük tutabilmek için mikrofon iznini açman gerekiyor."
      );
      return;
    }
    await recorder.prepareToRecordAsync();
    recorder.record();
  }

  async function stop() {
    setBusy(true);
    try {
      await recorder.stop();
      if (!recorder.uri) return;
      await enqueueAudio({
        uri: recorder.uri,
        durationSeconds: Math.round((recorderState.durationMillis || 0) / 1000),
      });
      await afterSave("Kaydın günlüğüne eklendi.");
    } finally {
      setBusy(false);
    }
  }

  async function submitText() {
    const trimmed = text.trim();
    if (!trimmed) return;
    setBusy(true);
    try {
      await enqueueText(trimmed);
      setText("");
      await afterSave("Yazın günlüğüne eklendi.");
    } finally {
      setBusy(false);
    }
  }

  async function afterSave(message) {
    const result = await flushQueue().catch(() => null);
    const count = await queueCount();
    setQueued(count);
    onQueueChanged?.(count);
    Alert.alert(
      "Kaydedildi",
      result?.uploaded?.length
        ? message
        : `${message} İnternet geldiğinde otomatik olarak gönderilecek.`
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
      <View>
        <Text style={styles.date}>{today}</Text>
        <Text style={styles.greeting}>Bugün ne oldu?</Text>
      </View>

      <PhotoBoard />

      <Card style={styles.recorder}>
        <Text style={styles.cardTitle}>Konuşarak Anlat</Text>
        <Help>
          Sesini kaydet. Söylediklerin daha sonra yazıya çevrilir ve günlüğünde saklanır.
        </Help>

        <Text style={styles.time}>{elapsed}</Text>

        <Pressable
          accessibilityRole="button"
          accessibilityLabel={recording ? "Konuşmayı bitir" : "Konuşmaya başla"}
          disabled={busy}
          onPress={recording ? stop : start}
          style={({ pressed }) => [
            styles.mic,
            recording && styles.micRecording,
            busy && styles.micDisabled,
            pressed && styles.micPressed,
          ]}
        >
          <Ionicons name={recording ? "stop" : "mic"} size={52} color={colors.accentInk} />
          <Text style={styles.micLabel}>{recording ? "Bitir" : "Başlat"}</Text>
        </Pressable>

        <Text style={styles.hint}>
          {recording
            ? "Kayıt sürüyor. Bitirmek için düğmeye tekrar bas."
            : "Yuvarlak düğmeye bas ve konuşmaya başla. Bitince aynı düğmeye tekrar bas."}
        </Text>
      </Card>

      <Card style={styles.composer}>
        <Text style={styles.cardTitle}>Yazarak Anlat</Text>
        <Help>
          Konuşmak istemiyorsan aşağıdaki kutuya yaz. Yazdıkların da günlüğünde saklanır.
        </Help>
        <TextInput
          value={text}
          onChangeText={setText}
          multiline
          accessibilityLabel="Günlük yazısı"
          placeholder={"Bugün neler yaptın? Kimlerle görüştün, nereye gittin?"}
          placeholderTextColor={colors.faint}
          style={styles.input}
        />
        <Button disabled={busy || !text.trim()} onPress={submitText}>
          Günlüğüme Kaydet
        </Button>
      </Card>

      {queued ? (
        <Text style={styles.queue}>
          {queued} kayıt gönderilmeyi bekliyor. İnternet geldiğinde kendiliğinden gidecek.
        </Text>
      ) : null}
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
  hint: {
    color: colors.text,
    fontSize: type.md,
    fontWeight: "600",
    textAlign: "center",
    lineHeight: type.md * 1.4,
    marginTop: spacing.xs,
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
  queue: {
    color: colors.muted,
    fontSize: type.sm,
    textAlign: "center",
  },
});
