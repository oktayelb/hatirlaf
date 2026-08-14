import React, { useCallback, useEffect, useMemo, useState } from "react";
import { AppState, Pressable, StatusBar, StyleSheet, Text, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { api, LockedError } from "./src/services/api";
import { registerBackgroundSync } from "./src/services/background";
import { loadFeatures, nlpEnabled } from "./src/services/features";
import { configureNotifications } from "./src/services/reminders";
import { flushQueue, queueCount } from "./src/services/queue";
import { CalendarScreen } from "./src/screens/CalendarScreen";
import { EntriesScreen } from "./src/screens/EntriesScreen";
import { HomeScreen } from "./src/screens/HomeScreen";
import { LockScreen } from "./src/screens/LockScreen";
import { RecapScreen } from "./src/screens/RecapScreen";
import { SettingsScreen } from "./src/screens/SettingsScreen";
import { TAP, colors, radius, spacing, type } from "./src/theme";

// Tabs carrying a `feature` only appear when the server says that feature is
// on. With the NLP pipeline off, the app is Ana + Günlüğüm and nothing else;
// Ayarlar lives behind the gear in the header rather than taking a third tab.
const TABS = [
  { key: "home", label: "Ana", icon: "home-outline", activeIcon: "home" },
  { key: "entries", label: "Günlüğüm", icon: "book-outline", activeIcon: "book" },
  {
    key: "calendar",
    label: "Takvim",
    icon: "calendar-outline",
    activeIcon: "calendar",
    feature: "nlp",
  },
  {
    key: "recap",
    label: "Özet",
    icon: "sparkles-outline",
    activeIcon: "sparkles",
    feature: "nlp",
  },
];

export default function App() {
  const [tab, setTab] = useState("home");
  const [privacy, setPrivacy] = useState({ password_enabled: false, unlocked: true });
  const [queueSize, setQueueSize] = useState(0);
  const [featuresReady, setFeaturesReady] = useState(false);
  const locked = privacy.password_enabled && !privacy.unlocked;

  const syncAndRefresh = useCallback(async () => {
    const count = await queueCount().catch(() => 0);
    setQueueSize(count);
    try {
      const status = await api.privacyStatus();
      setPrivacy(status);
      if (!status.password_enabled || status.unlocked) {
        await flushQueue().catch((err) => {
          if (!(err instanceof LockedError)) console.warn(err);
        });
        setQueueSize(await queueCount());
      }
    } catch (err) {
      if (err instanceof LockedError) setPrivacy({ password_enabled: true, unlocked: false });
      else console.warn(err);
    }
  }, []);

  useEffect(() => {
    (async () => {
      await loadFeatures();
      setFeaturesReady(true);
      // Reminders are built from extracted calendar events, so they belong
      // to the NLP layer. A capture-only app must not schedule them.
      await Promise.all([
        nlpEnabled() ? configureNotifications() : Promise.resolve(),
        registerBackgroundSync(),
      ]);
      await syncAndRefresh();
    })();

    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active") syncAndRefresh();
    });
    return () => sub.remove();
  }, [syncAndRefresh]);

  const tabs = useMemo(
    () => TABS.filter((item) => !item.feature || nlpEnabled()),
    [featuresReady]
  );

  // A flag flip can retire the tab the user is standing on.
  useEffect(() => {
    if (tab !== "settings" && !tabs.some((item) => item.key === tab)) setTab("home");
  }, [tab, tabs]);

  const content = useMemo(() => {
    if (locked && tab !== "settings") {
      return <LockScreen onUnlocked={(status) => setPrivacy(status)} />;
    }
    switch (tab) {
      case "entries":
        return <EntriesScreen locked={locked} />;
      case "calendar":
        return <CalendarScreen />;
      case "recap":
        return <RecapScreen />;
      case "settings":
        return (
          <SettingsScreen
            privacy={privacy}
            onPrivacyChanged={(next) => (next ? setPrivacy(next) : syncAndRefresh())}
            queueSize={queueSize}
            onQueueChanged={setQueueSize}
          />
        );
      case "home":
      default:
        return <HomeScreen onQueueChanged={setQueueSize} />;
    }
  }, [locked, privacy, queueSize, syncAndRefresh, tab]);

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
          {tabs.map(({ key, label, icon, activeIcon }) => {
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
                {key === "home" && queueSize ? (
                  <Text style={styles.queueDot}>{queueSize}</Text>
                ) : null}
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
  queueDot: {
    position: "absolute",
    top: 2,
    right: 12,
    minWidth: 22,
    height: 22,
    borderRadius: 11,
    overflow: "hidden",
    color: colors.accentInk,
    backgroundColor: colors.gold,
    textAlign: "center",
    fontSize: type.xs,
    fontWeight: "700",
    lineHeight: 22,
  },
});
