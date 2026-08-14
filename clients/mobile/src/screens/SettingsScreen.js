import React, { useEffect, useState } from "react";
import { Alert, StyleSheet, Text, TextInput, View } from "react-native";
import { api } from "../services/api";
import { clearSessionCookie, getApiBase, setApiBase } from "../services/config";
import { flushQueue, queueCount } from "../services/queue";
import { Button, Card, Screen } from "../ui/Primitives";
import { colors, radius, spacing, type } from "../theme";

export function SettingsScreen({ privacy, onPrivacyChanged, queueSize, onQueueChanged }) {
  const [apiBase, setApiBaseInput] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getApiBase().then(setApiBaseInput);
  }, []);

  async function saveApiBase() {
    await setApiBase(apiBase);
    await clearSessionCookie();
    Alert.alert("Kaydedildi", "Sunucu adresi güncellendi.");
    onPrivacyChanged?.();
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
      await api.setPrivacyPassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      onPrivacyChanged?.();
      Alert.alert("Kaydedildi", "Uygulama parolası güncellendi.");
    } finally {
      setBusy(false);
    }
  }

  async function clearPassword() {
    setBusy(true);
    try {
      await api.clearPrivacyPassword(currentPassword);
      setCurrentPassword("");
      onPrivacyChanged?.();
      Alert.alert("Kaldırıldı", "Uygulama parolası kaldırıldı.");
    } finally {
      setBusy(false);
    }
  }

  async function lockNow() {
    await api.lockPrivacy();
    await clearSessionCookie();
    onPrivacyChanged?.({ password_enabled: true, unlocked: false });
  }

  async function syncNow() {
    setBusy(true);
    try {
      await flushQueue();
      const count = await queueCount();
      onQueueChanged?.(count);
      Alert.alert("Senkronizasyon", count ? `${count} kayıt hala sırada.` : "Kuyruk boş.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen title="Ayarlar" subtitle="Sunucu, gizlilik ve çevrimdışı senkronizasyon.">
      <Card style={styles.card}>
        <Text style={styles.sectionTitle}>Sunucu</Text>
        <TextInput
          value={apiBase}
          onChangeText={setApiBaseInput}
          autoCapitalize="none"
          keyboardType="url"
          placeholder="http://192.168.1.10:8001/api"
          placeholderTextColor={colors.faint}
          style={styles.input}
        />
        <Text style={styles.copy}>
          Fiziksel telefonda `127.0.0.1` bilgisayarını göstermez. Aynı ağdaki bilgisayar IP adresini kullan.
        </Text>
        <Button onPress={saveApiBase}>Sunucuyu Kaydet</Button>
      </Card>

      <Card style={styles.card}>
        <Text style={styles.sectionTitle}>Gizlilik</Text>
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
            <Button disabled={busy} variant="ghost" onPress={clearPassword} style={styles.flex}>
              Kaldır
            </Button>
          ) : null}
        </View>
        {privacy?.password_enabled ? (
          <Button disabled={busy} variant="danger" onPress={lockNow}>
            Şimdi Kilitle
          </Button>
        ) : null}
      </Card>

      <Card style={styles.card}>
        <Text style={styles.sectionTitle}>Çevrimdışı Kuyruk</Text>
        <Text style={styles.copy}>{queueSize} kayıt gönderilmeyi bekliyor.</Text>
        <Button disabled={busy} onPress={syncNow}>Şimdi Eşitle</Button>
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
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
