// Günlüğüm — every entry, newest first, with the text it turned into.
// No analysis, no badges, no pipeline vocabulary.

import React, { useCallback, useEffect, useState } from "react";
import { FlatList, RefreshControl, StyleSheet, Text, TextInput, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { api } from "../services/api";
import { nlpEnabled } from "../services/features";
import { Button, Card, EmptyState, Loading, Screen } from "../ui/Primitives";
import { colors, radius, spacing, type } from "../theme";

export function EntriesScreen({ locked }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    if (locked) return;
    const data = await api.listSessions();
    setSessions(Array.isArray(data) ? data : data.results || []);
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
  }, [load]);

  async function refresh() {
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <Screen
      title="Günlüğüm"
      subtitle="Bugüne kadar kaydettiğin her şey burada. Yazıya çevrilmiş hâllerini okuyup düzeltebilirsin."
    >
      {loading ? (
        <Loading label="Günlüğün yükleniyor…" />
      ) : (
        <FlatList
          data={sessions}
          keyExtractor={(item) => String(item.id)}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} />}
          contentContainerStyle={sessions.length ? styles.list : styles.emptyList}
          renderItem={({ item }) => <EntryCard session={item} onChanged={refresh} />}
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

function EntryCard({ session, onChanged }) {
  const [text, setText] = useState(session.transcript || "");
  const [saving, setSaving] = useState(false);
  const hasAudio = Boolean(session.audio_url);
  const changed = text !== (session.transcript || "");
  const recordedAt = new Date(session.recorded_at);
  const busy = ["queued", "transcribing", "parsing"].includes(session.status);
  const failed = session.status === "failed";

  async function save() {
    setSaving(true);
    try {
      await api.updateSession(session.id, { transcript: text });
      await onChanged();
    } finally {
      setSaving(false);
    }
  }

  async function reprocess() {
    setSaving(true);
    try {
      if (changed) await api.updateSession(session.id, { transcript: text });
      await api.reprocess(session.id);
      await onChanged();
    } finally {
      setSaving(false);
    }
  }

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
        <View style={[styles.kind, hasAudio ? styles.kindVoice : styles.kindText]}>
          <Ionicons
            name={hasAudio ? "mic" : "create-outline"}
            size={18}
            color={hasAudio ? colors.accentDeep : colors.muted}
          />
          <Text style={[styles.kindLabel, hasAudio && styles.kindLabelVoice]}>
            {hasAudio ? "Sesli kayıt" : "Yazılı not"}
          </Text>
        </View>
      </View>

      {busy || failed ? (
        <View style={[styles.progress, failed && styles.progressFailed]}>
          <Text style={styles.progressText}>
            {failed
              ? "Bu kayıt yazıya çevrilemedi."
              : session.status === "queued"
              ? "Kaydın sırada bekliyor."
              : "Sesin yazıya çevriliyor."}
          </Text>
        </View>
      ) : null}

      <Text style={styles.label}>
        {hasAudio ? "Yazıya çevrilmiş hâli — düzeltebilirsin" : "Yazdıkların — düzeltebilirsin"}
      </Text>
      <TextInput
        value={text}
        onChangeText={setText}
        multiline
        placeholder={
          hasAudio
            ? busy
              ? "Sesin yazıya çevriliyor, birazdan burada olacak…"
              : "Bu kayıt için henüz yazı yok."
            : "Bu not boş."
        }
        placeholderTextColor={colors.faint}
        style={styles.input}
      />

      <View style={styles.actions}>
        <Button disabled={!changed || saving} onPress={save} style={styles.actionButton}>
          Kaydet
        </Button>
        {nlpEnabled() ? (
          <Button disabled={saving} variant="ghost" onPress={reprocess} style={styles.actionButton}>
            Yeniden işle
          </Button>
        ) : null}
      </View>
    </Card>
  );
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
  kind: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  kindVoice: {
    backgroundColor: colors.accentSoft,
    borderColor: colors.accent,
  },
  kindText: {
    backgroundColor: colors.surface2,
    borderColor: colors.lineStrong,
  },
  kindLabel: {
    color: colors.muted,
    fontSize: type.sm,
    fontWeight: "600",
  },
  kindLabelVoice: {
    color: colors.accentDeep,
  },
  progress: {
    backgroundColor: colors.goldSoft,
    borderColor: colors.gold,
    borderWidth: 1,
    borderRadius: radius.sm,
    padding: spacing.md,
  },
  progressFailed: {
    backgroundColor: colors.claySoft,
    borderColor: colors.clay,
  },
  progressText: {
    color: colors.text,
    fontSize: type.base,
    fontWeight: "600",
  },
  label: {
    color: colors.muted,
    fontSize: type.sm,
    fontWeight: "600",
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
    flexBasis: 150,
  },
});
