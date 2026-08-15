import React, { useCallback, useEffect, useMemo, useState } from "react";
import { AppState, Pressable, StatusBar, StyleSheet, Text, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { pruneOrphanAudio } from "./src/services/entries";
import { lockNow, lockStatus } from "./src/services/lock";
import { EntriesScreen } from "./src/screens/EntriesScreen";
import { HomeScreen } from "./src/screens/HomeScreen";
import { LockScreen } from "./src/screens/LockScreen";
import { SettingsScreen } from "./src/screens/SettingsScreen";
import { TAP, colors, radius, spacing, type } from "./src/theme";

// Two tabs, and that is the whole app. Ayarlar lives behind the gear in the
// header rather than taking a third one.
const TABS = [
  { key: "home", label: "Ana", icon: "home-outline", activeIcon: "home" },
  { key: "entries", label: "Günlüğüm", icon: "book-outline", activeIcon: "book" },
];

export default function App() {
  const [tab, setTab] = useState("home");
  const [privacy, setPrivacy] = useState({ password_enabled: false, unlocked: true });
  // Bumped whenever an entry is written, so Günlüğüm reloads without the two
  // screens having to know about each other.
  const [reloadKey, setReloadKey] = useState(0);
  const locked = privacy.password_enabled && !privacy.unlocked;

  const refreshLock = useCallback(async () => {
    setPrivacy(await lockStatus());
  }, []);

  useEffect(() => {
    (async () => {
      await refreshLock();
      // A crash between moving a recording and inserting its row would leave
      // a file nothing points at. Cheap to check, and it only grows otherwise.
      await pruneOrphanAudio().catch(() => {});
    })();
  }, [refreshLock]);

  // Leaving the app relocks it. A diary that stays open in the task switcher
  // is not protected by a password at all.
  useEffect(() => {
    const sub = AppState.addEventListener("change", async (state) => {
      if (state === "background" && (await lockStatus()).password_enabled) {
        lockNow();
        await refreshLock();
        setTab("home");
      }
    });
    return () => sub.remove();
  }, [refreshLock]);

  const content = useMemo(() => {
    if (locked && tab !== "settings") {
      return <LockScreen onUnlocked={(status) => setPrivacy(status)} />;
    }
    switch (tab) {
      case "entries":
        return <EntriesScreen locked={locked} reloadKey={reloadKey} />;
      case "settings":
        return (
          <SettingsScreen
            privacy={privacy}
            onPrivacyChanged={(next) => (next ? setPrivacy(next) : refreshLock())}
            onEntriesChanged={() => setReloadKey((n) => n + 1)}
          />
        );
      case "home":
      default:
        return <HomeScreen onSaved={() => setReloadKey((n) => n + 1)} />;
    }
  }, [locked, privacy, refreshLock, reloadKey, tab]);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe}>
        <StatusBar barStyle="dark-content" backgroundColor={colors.paper2} />

        <View style={styles.header}>
          <Text style={styles.brand}>Hatırlaf</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Ayarlar"
            onPress={() => setTab(tab === "settings" ? "home" : "settings")}
            style={[styles.gear, tab === "settings" && styles.gearActive]}
          >
            <Ionicons
              name="settings-outline"
              size={24}
              color={tab === "settings" ? colors.accentDeep : colors.muted}
            />
          </Pressable>
        </View>

        <View style={styles.app}>{content}</View>

        <View style={styles.tabs}>
          {TABS.map(({ key, label, icon, activeIcon }) => {
            const active = tab === key;
            return (
              <Pressable
                key={key}
                accessibilityRole="tab"
                accessibilityState={{ selected: active }}
                onPress={() => setTab(key)}
                style={[styles.tab, active && styles.tabActive]}
              >
                <Ionicons
                  name={active ? activeIcon : icon}
                  size={26}
                  color={active ? colors.accentDeep : colors.muted}
                />
                <Text style={[styles.tabText, active && styles.tabTextActive]}>{label}</Text>
              </Pressable>
            );
          })}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.paper,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.paper2,
    borderBottomColor: colors.line,
    borderBottomWidth: 1,
  },
  brand: {
    color: colors.text,
    fontSize: type.lg,
    fontWeight: "700",
  },
  gear: {
    width: 48,
    height: 48,
    borderRadius: radius.sm,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.surface,
    borderColor: colors.lineStrong,
    borderWidth: 1,
  },
  gearActive: {
    backgroundColor: colors.accentSoft,
    borderColor: colors.accent,
  },
  app: {
    flex: 1,
  },
  tabs: {
    flexDirection: "row",
    gap: spacing.xs,
    padding: spacing.xs,
    backgroundColor: colors.surface2,
    borderTopColor: colors.line,
    borderTopWidth: 1,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 2,
    minHeight: TAP,
    borderRadius: radius.sm,
    borderWidth: 2,
    borderColor: "transparent",
  },
  tabActive: {
    backgroundColor: colors.surface,
    borderColor: colors.accent,
  },
  tabText: {
    color: colors.muted,
    fontSize: type.sm,
    fontWeight: "700",
  },
  tabTextActive: {
    color: colors.accentDeep,
  },
});
