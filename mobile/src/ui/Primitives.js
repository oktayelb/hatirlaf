import React from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";
import { colors } from "../theme";

export function Screen({ children, title, subtitle, action }) {
  return (
    <View style={styles.screen}>
      <View style={styles.header}>
        <View style={styles.headerText}>
          <Text style={styles.title}>{title}</Text>
          {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
        </View>
        {action}
      </View>
      {children}
    </View>
  );
}

export function Card({ children, style }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Button({ children, onPress, variant = "primary", disabled = false, style }) {
  return (
    <Pressable
      onPress={disabled ? undefined : onPress}
      style={({ pressed }) => [
        styles.button,
        styles[variant],
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
        style,
      ]}
    >
      <Text style={[styles.buttonText, variant === "ghost" && styles.ghostText]}>{children}</Text>
    </Pressable>
  );
}

export function EmptyState({ title, text }) {
  return (
    <Card style={styles.empty}>
      <Text style={styles.emptyTitle}>{title}</Text>
      {text ? <Text style={styles.emptyText}>{text}</Text> : null}
    </Card>
  );
}

export function Loading({ label = "Yükleniyor..." }) {
  return (
    <View style={styles.loading}>
      <ActivityIndicator color={colors.accent2} />
      <Text style={styles.loadingText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    padding: 16,
    gap: 14,
  },
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 12,
  },
  headerText: {
    flex: 1,
  },
  title: {
    color: colors.text,
    fontSize: 25,
    fontWeight: "800",
    letterSpacing: 0,
  },
  subtitle: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 3,
  },
  card: {
    backgroundColor: colors.panel,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 10,
    padding: 14,
  },
  button: {
    minHeight: 44,
    borderRadius: 8,
    paddingHorizontal: 16,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
  },
  primary: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  ghost: {
    backgroundColor: "transparent",
    borderColor: colors.border,
  },
  danger: {
    backgroundColor: "rgba(239,101,104,0.14)",
    borderColor: "rgba(239,101,104,0.35)",
  },
  disabled: {
    opacity: 0.45,
  },
  pressed: {
    transform: [{ translateY: 1 }],
  },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 14,
  },
  ghostText: {
    color: colors.accent2,
  },
  empty: {
    alignItems: "center",
    paddingVertical: 28,
  },
  emptyTitle: {
    color: colors.text,
    fontWeight: "800",
    fontSize: 16,
  },
  emptyText: {
    color: colors.muted,
    textAlign: "center",
    lineHeight: 20,
    marginTop: 6,
  },
  loading: {
    padding: 24,
    alignItems: "center",
    gap: 10,
  },
  loadingText: {
    color: colors.muted,
  },
});
