import React, { useState } from "react";
import { Alert, StyleSheet, Text, TextInput } from "react-native";
import { lockStatus, unlock } from "../services/lock";
import { Button, Card, Screen } from "../ui/Primitives";
import { colors, radius, spacing, type } from "../theme";

export function LockScreen({ onUnlocked }) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    try {
      if (await unlock(password)) {
        setPassword("");
        onUnlocked?.(await lockStatus());
      } else {
        Alert.alert("Kilit açılamadı", "Parola yanlış. Bir daha dener misin?");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen title="Günlüğün kilitli" subtitle="Kayıtlarını görmek için parolanı yaz.">
      <Card style={styles.card}>
        <Text style={styles.mark}>H</Text>
        <TextInput
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          autoFocus
          onSubmitEditing={password && !busy ? submit : undefined}
          placeholder="Parola"
          placeholderTextColor={colors.faint}
          style={styles.input}
        />
        <Button disabled={busy || !password} onPress={submit}>
          Kilidi Aç
        </Button>
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
  card: {
    gap: spacing.md,
    alignItems: "stretch",
  },
  mark: {
    width: 68,
    height: 68,
    borderRadius: 34,
    alignSelf: "center",
    backgroundColor: colors.accentSoft,
    borderColor: colors.accent,
    borderWidth: 2,
    color: colors.accentDeep,
    textAlign: "center",
    textAlignVertical: "center",
    lineHeight: 68,
    fontSize: type.xl,
    fontWeight: "700",
  },
  input: {
    color: colors.text,
    fontSize: type.md,
    backgroundColor: colors.surface2,
    borderColor: colors.lineStrong,
    borderWidth: 2,
    borderRadius: radius.sm,
    minHeight: 56,
    paddingHorizontal: spacing.md,
  },
});
