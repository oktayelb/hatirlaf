import React, { useState } from "react";
import { Alert, StyleSheet, Text, TextInput } from "react-native";
import { api } from "../services/api";
import { Button, Card, Screen } from "../ui/Primitives";
import { colors } from "../theme";

export function LockScreen({ onUnlocked }) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function unlock() {
    setBusy(true);
    try {
      const status = await api.unlockPrivacy(password);
      onUnlocked?.(status);
    } catch (err) {
      Alert.alert("Kilit açılamadı", "Parola yanlış ya da sunucuya ulaşılamıyor.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen title="Hatırlaf kilitli" subtitle="Günlük verilerini görmek için uygulama parolasını gir.">
      <Card style={styles.card}>
        <Text style={styles.mark}>H</Text>
        <TextInput
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          autoFocus
          placeholder="Parola"
          placeholderTextColor={colors.muted}
          style={styles.input}
        />
        <Button disabled={busy || !password} onPress={unlock}>
          Kilidi Aç
        </Button>
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
  card: {
    gap: 16,
    alignItems: "stretch",
  },
  mark: {
    width: 52,
    height: 52,
    borderRadius: 10,
    backgroundColor: colors.accent,
    color: "#fff",
    textAlign: "center",
    textAlignVertical: "center",
    lineHeight: 52,
    fontSize: 26,
    fontWeight: "900",
  },
  input: {
    color: colors.text,
    backgroundColor: "#101722",
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 8,
    minHeight: 46,
    paddingHorizontal: 12,
  },
});
