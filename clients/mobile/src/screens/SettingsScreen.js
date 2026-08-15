import React, { useCallback, useEffect, useState } from "react";
import { Alert, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { audioFootprint, countEntries, formatBytes } from "../services/entries";
import { exportAll } from "../services/backup";
import { clearPassword, lockNow, lockStatus, setPassword, verify } from "../services/lock";
import { Button, Card, Help, Screen } from "../ui/Primitives";
import { colors, radius, spacing, type } from "../theme";

export function SettingsScreen({ privacy, onPrivacyChanged }) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [stats, setStats] = useState({ entries: 0, bytes: 0 });
  const [busy, setBusy] = useState(false);

  const loadStats = useCallback(async () => {
    const [entries, bytes] = await Promise.all([countEntries(), audioFootprint()]);
    setStats({ entries, bytes });
  }, []);

  useEffect(() => {
    loadStats().catch(() => {});
  }, [loadStats]);

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
