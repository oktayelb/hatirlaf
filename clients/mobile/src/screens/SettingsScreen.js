import React, { useCallback, useEffect, useRef, useState } from "react";
import { Alert, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import {
  audioFootprint,
  countEntries,
  countUntranscribed,
  formatBytes,
} from "../services/entries";
import { exportAll } from "../services/backup";
import { backfillTranscripts, explainBackfill } from "../services/backfill";
import { clearPassword, lockNow, lockStatus, setPassword, verify } from "../services/lock";
import { explainDownload, explainMode, needsManualInstall } from "../services/speech";
import { useTurkishModel } from "../services/useTurkishModel";
import { Button, Card, Help, Screen } from "../ui/Primitives";
import { colors, radius, spacing, type } from "../theme";

export function SettingsScreen({ privacy, onPrivacyChanged, onEntriesChanged }) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [stats, setStats] = useState({ entries: 0, bytes: 0, untranscribed: 0 });
  const [busy, setBusy] = useState(false);

  // The backfill: `null` when idle, `{ done, total, written }` while running,
  // and the finished tally afterwards.
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  // Set when the screen goes away, and polled between entries — that is how
  // leaving Ayarlar cancels a long run without killing it mid-file.
  const abandonedRef = useRef(false);

  const {
    capabilities,
    download,
    install: installModel,
    openSettings,
    refresh: refreshModel,
  } = useTurkishModel();

  const loadStats = useCallback(async () => {
    const [entries, bytes, untranscribed] = await Promise.all([
      countEntries(),
      audioFootprint(),
      countUntranscribed(),
    ]);
    setStats({ entries, bytes, untranscribed });
  }, []);

  useEffect(() => {
    loadStats().catch(() => {});
  }, [loadStats]);

  useEffect(() => {
    abandonedRef.current = false;
    return () => {
      abandonedRef.current = true;
    };
  }, []);

  // Give words to the recordings that never got any. Only offered once the
  // phone can actually transcribe — running it without the pack would walk the
  // whole backlog, fail on every entry, and look like the feature is broken.
  const runBackfill = useCallback(async () => {
    setResult(null);
    setProgress({ done: 0, total: stats.untranscribed, written: 0 });
    try {
      const tally = await backfillTranscripts({
        onProgress: (update) => {
          if (!abandonedRef.current) setProgress(update);
        },
        shouldStop: () => abandonedRef.current,
      });
      if (abandonedRef.current) return;
      setResult(tally);
      await loadStats();
      // Günlüğüm is rendered from its own query and would otherwise keep
      // showing the entries as wordless until something else reloaded it.
      if (tally.written) onEntriesChanged?.();
    } catch (err) {
      if (!abandonedRef.current) {
        Alert.alert("Çevrilemedi", String(err?.message || err));
      }
    } finally {
      if (!abandonedRef.current) setProgress(null);
    }
  }, [loadStats, onEntriesChanged, stats.untranscribed]);

  const modelNote = capabilities ? explainMode(capabilities) : "";
  const downloadNote = explainDownload(download);
  const downloading = download?.status === "working";
  const speechReady = capabilities?.mode === "speech";
  const backfilling = Boolean(progress);

  function clearFields() {
    setCurrentPassword("");
    setNewPassword("");
    setConfirmPassword("");
  }

  async function savePassword() {
    if (newPassword.length < 6) {
      Alert.alert("Parola kısa", "Parola en az 6 karakter olmalı.");
      return;
    }
    if (newPassword !== confirmPassword) {
      Alert.alert("Eşleşmiyor", "Yeni parolalar eşleşmiyor.");
      return;
    }
    setBusy(true);
    try {
      if (privacy?.password_enabled && !(await verify(currentPassword))) {
        Alert.alert("Parola yanlış", "Mevcut parolanı doğru yazman gerekiyor.");
        return;
      }
      await setPassword(newPassword);
      clearFields();
      onPrivacyChanged?.(await lockStatus());
      Alert.alert("Kaydedildi", "Uygulama parolası güncellendi.");
    } finally {
      setBusy(false);
    }
  }

  async function removePassword() {
    setBusy(true);
    try {
      if (!(await clearPassword(currentPassword))) {
        Alert.alert("Parola yanlış", "Parolayı kaldırmak için mevcut parolanı yaz.");
        return;
      }
      clearFields();
      onPrivacyChanged?.(await lockStatus());
      Alert.alert("Kaldırıldı", "Uygulama parolası kaldırıldı.");
    } finally {
      setBusy(false);
    }
  }

  async function lock() {
    lockNow();
    onPrivacyChanged?.(await lockStatus());
  }

  async function runExport() {
    setBusy(true);
    try {
      const result = await exportAll();
      if (!result.entries) {
        Alert.alert("Boş günlük", "Dışa aktarılacak kayıt yok.");
        return;
      }
      if (!result.shared) {
        Alert.alert("Paylaşım yok", "Bu telefonda paylaşma ekranı açılamıyor.");
        return;
      }
      if (result.withAudio) {
        Alert.alert(
          "Yazılar dışa aktarıldı",
          `${result.entries} kaydın yazısı bir dosyaya kondu. ${result.withAudio} kaydın sesi hâlâ yalnızca bu telefonda — sesleri tek tek Günlüğüm'deki Paylaş düğmesiyle gönderebilirsin.`
        );
      }
    } catch (err) {
      Alert.alert("Dışa aktarılamadı", String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen title="Ayarlar">
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Card style={styles.card}>
          <Text style={styles.sectionTitle}>Günlüğün</Text>
          <Text style={styles.copy}>
            {stats.entries} kayıt, {formatBytes(stats.bytes)} ses.
          </Text>
          <Help>
            Her şey bu telefonda duruyor. Hiçbir kayıt internete gönderilmiyor.
          </Help>
        </Card>

        <Card style={styles.card}>
          <Text style={styles.sectionTitle}>Türkçe Konuşma</Text>
          <Text style={styles.copy}>
            Durum: {speechReady ? "Hazır" : "Dil paketi kurulu değil"}
          </Text>
          {modelNote ? <Help>{modelNote}</Help> : null}
          {downloadNote ? <Help>{downloadNote}</Help> : null}

          {capabilities?.canDownloadModel ? (
            <Button
              variant="ghost"
              disabled={busy || downloading || backfilling}
              onPress={() => installModel({ userAsked: true })}
            >
              {downloading ? "İsteniyor…" : "Türkçe Dil Paketini İndir"}
            </Button>
          ) : null}

          {needsManualInstall(download) ? (
            <Button variant="ghost" disabled={busy || backfilling} onPress={openSettings}>
              Telefon Ayarlarından Kur
            </Button>
          ) : null}

          <Button
            variant="ghost"
            disabled={busy || downloading || backfilling}
            onPress={() => {
              refreshModel();
              loadStats().catch(() => {});
            }}
          >
            Durumu Yenile
          </Button>

          <Help>
            Türkçe yazıya çevirme telefonun kendi dil paketiyle yapılır. Sesin hiçbir
            zaman internete gönderilmez.
          </Help>
        </Card>

        <Card style={styles.card}>
          <Text style={styles.sectionTitle}>Eski Kayıtları Yazıya Çevir</Text>
          <Text style={styles.copy}>
            {stats.untranscribed
              ? `${stats.untranscribed} kaydın yazısı yok.`
              : "Bütün kayıtların yazısı var."}
          </Text>

          {backfilling ? (
            <Text style={styles.copy}>
              Çevriliyor: {progress.done} / {progress.total} — {progress.written} tamam
            </Text>
          ) : null}

          {result ? <Help>{explainBackfill(result)}</Help> : null}

          <Button
            disabled={busy || backfilling || !stats.untranscribed || !speechReady}
            onPress={runBackfill}
          >
            {backfilling ? "Çevriliyor…" : "Şimdi Çevir"}
          </Button>

          <Help>
            {speechReady
              ? "Dil paketi kurulduktan sonra, sesi olup yazısı olmayan kayıtları tek tek dinleyip yazıya çevirir. Uzun sürebilir; bu ekranda kal."
              : "Önce yukarıdan Türkçe dil paketini kur. Kurulmadan eski kayıtlar yazıya çevrilemez."}
          </Help>
        </Card>

        <Card style={styles.card}>
          <Text style={styles.sectionTitle}>Yedek Al</Text>
          <Help>
            Telefonun kaybolursa günlüğün de kaybolur. Ara sıra yedek almak iyi olur.
          </Help>
          <Button disabled={busy} onPress={runExport}>
            Günlüğümü Dışa Aktar
          </Button>
        </Card>

        <Card style={styles.card}>
          <Text style={styles.sectionTitle}>Parola</Text>
          <Text style={styles.copy}>
            Durum: {privacy?.password_enabled ? "Parola aktif" : "Parola yok"}
          </Text>
          {privacy?.password_enabled ? (
            <TextInput
              value={currentPassword}
              onChangeText={setCurrentPassword}
              secureTextEntry
              placeholder="Mevcut parola"
              placeholderTextColor={colors.faint}
              style={styles.input}
            />
          ) : null}
          <TextInput
            value={newPassword}
            onChangeText={setNewPassword}
            secureTextEntry
            placeholder="Yeni parola"
            placeholderTextColor={colors.faint}
            style={styles.input}
          />
          <TextInput
            value={confirmPassword}
            onChangeText={setConfirmPassword}
            secureTextEntry
            placeholder="Yeni parolayı tekrar yaz"
            placeholderTextColor={colors.faint}
            style={styles.input}
          />
          <View style={styles.row}>
            <Button disabled={busy} onPress={savePassword} style={styles.flex}>
              Kaydet
            </Button>
            {privacy?.password_enabled ? (
              <Button disabled={busy} variant="ghost" onPress={removePassword} style={styles.flex}>
                Kaldır
              </Button>
            ) : null}
          </View>
          {privacy?.password_enabled ? (
            <Button disabled={busy} variant="danger" onPress={lock}>
              Şimdi Kilitle
            </Button>
          ) : null}
          <Help>
            Parola, telefonu eline alan birinin günlüğünü açmasını engeller. Kayıtların
            kendisi şifrelenmez.
          </Help>
        </Card>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: {
    gap: spacing.md,
    paddingBottom: spacing.xl,
  },
  card: {
    gap: spacing.sm,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: type.lg,
    fontWeight: "700",
  },
  copy: {
    color: colors.muted,
    fontSize: type.base,
    lineHeight: type.base * 1.5,
  },
  input: {
    color: colors.text,
    fontSize: type.base,
    backgroundColor: colors.surface2,
    borderColor: colors.lineStrong,
    borderWidth: 2,
    borderRadius: radius.sm,
    minHeight: 56,
    paddingHorizontal: spacing.md,
  },
  row: {
    flexDirection: "row",
    gap: spacing.sm,
    flexWrap: "wrap",
  },
  flex: {
    flexGrow: 1,
    flexBasis: 150,
  },
});
