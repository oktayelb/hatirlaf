import React, { useCallback, useEffect, useState } from "react";
import { FlatList, RefreshControl, StyleSheet, Text, TextInput, View } from "react-native";
import { api } from "../services/api";
import { Button, Card, EmptyState, Loading, Screen } from "../ui/Primitives";
import { colors } from "../theme";

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
    <Screen title="Girişler" subtitle="Kayıtları gözden geçir, transkripti düzelt ve yeniden işle.">
      {loading ? (
        <Loading />
      ) : (
        <FlatList
          data={sessions}
          keyExtractor={(item) => String(item.id)}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refresh} />}
          contentContainerStyle={sessions.length ? styles.list : styles.emptyList}
          renderItem={({ item }) => <EntryCard session={item} onChanged={refresh} />}
          ListEmptyComponent={<EmptyState title="Henüz giriş yok" text="Ana ekrandan kayıt ekleyebilirsin." />}
        />
      )}
    </Screen>
  );
}

function EntryCard({ session, onChanged }) {
  const [text, setText] = useState(session.transcript || "");
  const [saving, setSaving] = useState(false);
  const changed = text !== (session.transcript || "");

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
      await api.reprocess(session.id);
      await onChanged();
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card style={styles.card}>
      <View style={styles.metaRow}>
        <Text style={styles.badge}>{session.audio_url ? "Sesli" : "Yazılı"}</Text>
        <Text style={styles.status}>{session.status_display || session.status}</Text>
      </View>
      <Text style={styles.date}>{new Date(session.recorded_at).toLocaleString("tr-TR")}</Text>
      <TextInput
        value={text}
        onChangeText={setText}
        multiline
        placeholder="Transkript yok"
        placeholderTextColor={colors.muted}
        style={styles.input}
      />
      <View style={styles.actions}>
        <Button disabled={!changed || saving} onPress={save} style={styles.actionButton}>
          Kaydet
        </Button>
        <Button disabled={saving} variant="ghost" onPress={reprocess} style={styles.actionButton}>
          Yeniden işle
        </Button>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  list: {
    gap: 12,
    paddingBottom: 20,
  },
  emptyList: {
    flexGrow: 1,
    justifyContent: "center",
  },
  card: {
    gap: 10,
  },
  metaRow: {
    flexDirection: "row",
    gap: 8,
    alignItems: "center",
  },
  badge: {
    color: colors.accent2,
    fontWeight: "800",
    fontSize: 12,
  },
  status: {
    color: colors.muted,
    fontSize: 12,
  },
  date: {
    color: colors.muted,
    fontSize: 12,
  },
  input: {
    minHeight: 92,
    color: colors.text,
    backgroundColor: "#101722",
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 8,
    padding: 10,
    textAlignVertical: "top",
  },
  actions: {
    flexDirection: "row",
    gap: 8,
  },
  actionButton: {
    flex: 1,
  },
});
