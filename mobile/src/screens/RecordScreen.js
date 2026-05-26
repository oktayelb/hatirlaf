import React, { useMemo, useState } from "react";
import { Alert, StyleSheet, Text, TextInput, View } from "react-native";
import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  useAudioRecorder,
  useAudioRecorderState,
} from "expo-audio";
import { Button, Card, Screen } from "../ui/Primitives";
import { enqueueAudio, enqueueText, flushQueue, queueCount } from "../services/queue";
import { colors } from "../theme";

export function RecordScreen({ onQueueChanged }) {
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(recorder);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [queued, setQueued] = useState(0);

  const elapsed = useMemo(() => {
    const ms = recorderState.durationMillis || 0;
    const seconds = Math.floor(ms / 1000);
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }, [recorderState.durationMillis]);

  async function start() {
    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Mikrofon izni gerekli", "Sesli günlük kaydı için mikrofon izni ver.");
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
      await afterEnqueue();
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
      await afterEnqueue();
    } finally {
      setBusy(false);
    }
  }

  async function afterEnqueue() {
    const result = await flushQueue().catch(() => null);
    const count = await queueCount();
    setQueued(count);
    onQueueChanged?.(count);
    if (result?.uploaded?.length) Alert.alert("Yüklendi", "Kayıt sunucuya gönderildi.");
    else Alert.alert("Sıraya alındı", "Bağlantı geldiğinde arka planda gönderilecek.");
  }

  return (
    <Screen title="Hatırlaf" subtitle="Sesli ya da yazılı günlük kaydı ekle.">
      <Card style={styles.recorder}>
        <Text style={styles.time}>{elapsed}</Text>
        <Button
          disabled={busy}
          variant={recorderState.isRecording ? "danger" : "primary"}
          onPress={recorderState.isRecording ? stop : start}
          style={styles.recordButton}
        >
          {recorderState.isRecording ? "Kaydı Bitir" : "Kayda Başla"}
        </Button>
        <Text style={styles.hint}>
          {recorderState.isRecording ? "Kayıt sürüyor." : "Mikrofona dokun ve gününü anlat."}
        </Text>
      </Card>

      <Card style={styles.composer}>
        <Text style={styles.label}>Yazarak ekle</Text>
        <TextInput
          value={text}
          onChangeText={setText}
          multiline
          placeholder="Bugün neler oldu?"
          placeholderTextColor={colors.muted}
          style={styles.input}
        />
        <Button disabled={busy || !text.trim()} onPress={submitText}>
          Günlüğe Ekle
        </Button>
      </Card>

      {queued ? <Text style={styles.queue}>{queued} kayıt çevrimdışı sırada.</Text> : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  recorder: {
    alignItems: "center",
    gap: 14,
  },
  time: {
    color: colors.text,
    fontSize: 36,
    fontWeight: "800",
    fontVariant: ["tabular-nums"],
  },
  recordButton: {
    width: "100%",
  },
  hint: {
    color: colors.muted,
    fontSize: 13,
  },
  composer: {
    gap: 12,
  },
  label: {
    color: colors.accent2,
    fontWeight: "800",
    fontSize: 12,
    textTransform: "uppercase",
  },
  input: {
    minHeight: 140,
    color: colors.text,
    backgroundColor: "#101722",
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    textAlignVertical: "top",
  },
  queue: {
    color: colors.muted,
    textAlign: "center",
  },
});
