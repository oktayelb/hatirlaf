// Günlüğüm — every entry, newest first, with its audio and its text.
// No analysis, no badges, no pipeline vocabulary.

import React, { useCallback, useEffect, useState } from "react";
import {
  Alert,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { setAudioModeAsync, useAudioPlayer, useAudioPlayerStatus } from "expo-audio";
import { Ionicons } from "@expo/vector-icons";
import { deleteEntry, listEntries, updateTranscript } from "../services/entries";
import { shareEntry } from "../services/backup";
import { Button, Card, EmptyState, Loading, Screen } from "../ui/Primitives";
import { colors, radius, spacing, type } from "../theme";

export function EntriesScreen({ locked, reloadKey }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  // Only one recording plays at a time; the cards read this to pause
  // themselves when another one takes over.
  const [playingId, setPlayingId] = useState(null);

  const load = useCallback(async () => {
    if (locked) return;
    setEntries(await listEntries());
  }, [locked]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    load()
      .catch(console.warn)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [load, reloadKey]);

  async function refresh() {
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <Screen title="Günlüğüm">
      {loading ? (
        <Loading label="Günlüğün yükleniyor…" />
      ) : (
        <FlatList
          data={entries}
          keyExtractor={(item) => String(item.id)}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} />}
          contentContainerStyle={entries.length ? styles.list : styles.emptyList}
          renderItem={({ item }) => (
            <EntryCard
              entry={item}
              playingId={playingId}
              onPlay={setPlayingId}
              onChanged={refresh}
            />
          )}
          ListEmptyComponent={
            <EmptyState
              title="Henüz hiç kayıt yok"
              text="Ana sayfaya git, yuvarlak düğmeye basıp konuş ya da oradaki kutuya yaz."
            />
          }
        />
      )}
    </Screen>
  );
}

function EntryCard({ entry, playingId, onPlay, onChanged }) {
  const [text, setText] = useState(entry.transcript || "");
  const [saving, setSaving] = useState(false);
  const hasAudio = Boolean(entry.audio_path);
  const changed = text !== (entry.transcript || "");
  const recordedAt = new Date(entry.recorded_at);

  const player = useAudioPlayer(hasAudio ? { uri: entry.audio_path } : null);
  const status = useAudioPlayerStatus(player);
  const isActive = playingId === entry.id;

  // Another card started playing, so this one steps aside.
  useEffect(() => {
    if (!isActive && status.playing) player.pause();
  }, [isActive, player, status.playing]);

  useEffect(() => {
    if (status.didJustFinish) {
      onPlay(null);
      player.seekTo(0);
    }
  }, [status.didJustFinish, player, onPlay]);

  async function toggle() {
    if (!hasAudio) return;
    if (status.playing) {
      player.pause();
      onPlay(null);
      return;
    }
    // Recording leaves the iOS session in playAndRecord, which routes playback
    // to the earpiece. Handing it back to playback puts it on the speaker.
    await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true }).catch(() => {});
    onPlay(entry.id);
    player.play();
  }

  async function save() {
    setSaving(true);
    try {
      await updateTranscript(entry.id, text);
      await onChanged();
    } finally {
      setSaving(false);
    }
  }

  function confirmDelete() {
    Alert.alert("Bu kaydı sil", "Kayıt ve sesi telefondan tamamen silinecek. Geri alınamaz.", [
      { text: "Vazgeç", style: "cancel" },
      {
        text: "Sil",
        style: "destructive",
        onPress: async () => {
          if (status.playing) player.pause();
          await deleteEntry(entry.id);
          await onChanged();
        },
      },
    ]);
  }

  const position = formatClock(status.currentTime);
  const total = formatClock(status.duration || entry.duration_ms / 1000);

  return (
    <Card style={styles.card}>
      <View style={styles.head}>
        <View style={styles.headText}>
          <Text style={styles.date}>
            {recordedAt.toLocaleDateString("tr-TR", {
              weekday: "long",
              day: "numeric",
              month: "long",
              year: "numeric",
            })}
          </Text>
          <Text style={styles.time}>
            {recordedAt.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}
          </Text>
        </View>
      </View>

      {hasAudio ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={status.playing ? "Sesi durdur" : "Sesi dinle"}
          onPress={toggle}
          style={({ pressed }) => [styles.player, pressed && styles.playerPressed]}
        >
          <View style={styles.playIcon}>
            <Ionicons
              name={status.playing ? "pause" : "play"}
              size={30}
              color={colors.accentInk}
            />
          </View>
          <View style={styles.playerText}>
            <Text style={styles.playerLabel}>
              {status.playing ? "Çalıyor" : "Sesini dinle"}
            </Text>
            <Text style={styles.playerTime}>
              {position} / {total}
            </Text>
          </View>
        </Pressable>
      ) : null}

      <TextInput
        value={text}
        onChangeText={setText}
        multiline
        accessibilityLabel={hasAudio ? "Yazıya çevrilmiş hâli" : "Yazdıkların"}
        placeholder={
          hasAudio ? "Bu kayıt için yazı yok. İstersen buraya kendin yazabilirsin." : "Bu not boş."
        }
        placeholderTextColor={colors.faint}
        style={styles.input}
      />

      <View style={styles.actions}>
        <Button disabled={!changed || saving} onPress={save} style={styles.actionButton}>
          Kaydet
        </Button>
        <Button
          disabled={saving}
          variant="ghost"
          onPress={() => shareEntry(entry).catch(() => {})}
          style={styles.actionButton}
        >
          Paylaş
        </Button>
        <Button disabled={saving} variant="danger" onPress={confirmDelete} style={styles.actionButton}>
          Sil
        </Button>
      </View>
    </Card>
  );
}

function formatClock(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

const styles = StyleSheet.create({
  list: {
    gap: spacing.md,
    paddingBottom: spacing.lg,
  },
  emptyList: {
    flexGrow: 1,
    justifyContent: "center",
  },
  card: {
    gap: spacing.sm,
  },
  head: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: spacing.sm,
    flexWrap: "wrap",
  },
  headText: {
    flexShrink: 1,
  },
  date: {
    color: colors.text,
    fontSize: type.lg,
    fontWeight: "700",
    textTransform: "capitalize",
  },
  time: {
    color: colors.muted,
    fontSize: type.sm,
  },
  player: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    backgroundColor: colors.accentSoft,
    borderColor: colors.accent,
    borderWidth: 2,
    borderRadius: radius.sm,
    padding: spacing.sm,
    minHeight: 72,
  },
  playerPressed: {
    transform: [{ translateY: 1 }],
  },
  playIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  playerText: {
    flex: 1,
  },
  playerLabel: {
    color: colors.accentDeep,
    fontSize: type.md,
    fontWeight: "700",
  },
  playerTime: {
    color: colors.muted,
    fontSize: type.sm,
    fontVariant: ["tabular-nums"],
  },
  input: {
    minHeight: 150,
    color: colors.text,
    fontSize: type.md,
    lineHeight: type.md * 1.6,
    backgroundColor: colors.surface2,
    borderColor: colors.lineStrong,
    borderWidth: 2,
    borderRadius: radius.sm,
    padding: spacing.md,
    textAlignVertical: "top",
  },
  actions: {
    flexDirection: "row",
    gap: spacing.sm,
    flexWrap: "wrap",
  },
  actionButton: {
    flexGrow: 1,
    flexBasis: 110,
  },
});
