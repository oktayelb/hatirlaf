import React from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";
import { TAP, colors, radius, spacing, type } from "../theme";

export function Screen({ children, title, subtitle, action }) {
  return (
    <View style={styles.screen}>
      {title ? (
        <View style={styles.header}>
          <View style={styles.headerText}>
            <Text style={styles.title}>{title}</Text>
            {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
          </View>
          {action}
        </View>
      ) : null}
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
      accessibilityRole="button"
      accessibilityState={{ disabled }}
      onPress={disabled ? undefined : onPress}
      style={({ pressed }) => [
        styles.button,
        styles[variant],
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
        style,
      ]}
    >
      <Text style={[styles.buttonText, styles[`${variant}Text`]]}>{children}</Text>
    </Pressable>
  );
}

/** The plain-language line that sits under a control and explains it. */
export function Help({ children }) {
  return <Text style={styles.help}>{children}</Text>;
}

export function EmptyState({ title, text }) {
  return (
    <Card style={styles.empty}>
      <Text style={styles.emptyTitle}>{title}</Text>
      {text ? <Text style={styles.emptyText}>{text}</Text> : null}
    </Card>
  );
}

export function Loading({ label = "Yükleniyor…" }) {
  return (
    <View style={styles.loading}>
      <ActivityIndicator color={colors.accent} />
      <Text style={styles.loadingText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    padding: spacing.md,
    gap: spacing.md,
  },
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: spacing.sm,
  },
  headerText: {
    flex: 1,
  },
  title: {
    color: colors.text,
    fontSize: type.xl,
    fontWeight: "700",
  },
  subtitle: {
    color: colors.muted,
    fontSize: type.base,
    lineHeight: type.base * 1.5,
    marginTop: spacing.xs,
  },
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.line,
    borderWidth: 1,
    borderRadius: radius.md,
    padding: spacing.lg,
  },
  button: {
    minHeight: TAP,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 2,
  },
  primary: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  primaryText: {
    color: colors.accentInk,
  },
  ghost: {
    backgroundColor: "transparent",
    borderColor: colors.lineStrong,
  },
  ghostText: {
    color: colors.accentDeep,
  },
  danger: {
    backgroundColor: colors.claySoft,
    borderColor: colors.clay,
  },
  dangerText: {
    color: colors.clay,
  },
  disabled: {
    opacity: 0.45,
  },
  pressed: {
    transform: [{ translateY: 1 }],
  },
  buttonText: {
    fontWeight: "700",
    fontSize: type.md,
    textAlign: "center",
  },
  help: {
    color: colors.muted,
    fontSize: type.sm,
    lineHeight: type.sm * 1.5,
  },
  empty: {
    alignItems: "center",
    paddingVertical: spacing.xl,
    borderStyle: "dashed",
    borderWidth: 2,
    borderColor: colors.lineStrong,
  },
  emptyTitle: {
    color: colors.text,
    fontWeight: "700",
    fontSize: type.lg,
    textAlign: "center",
  },
  emptyText: {
    color: colors.muted,
    fontSize: type.base,
    textAlign: "center",
    lineHeight: type.base * 1.5,
    marginTop: spacing.xs,
  },
  loading: {
    padding: spacing.xl,
    alignItems: "center",
    gap: spacing.sm,
  },
  loadingText: {
    color: colors.muted,
    fontSize: type.base,
  },
});
