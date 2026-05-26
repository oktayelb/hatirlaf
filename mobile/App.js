import React, { useEffect, useMemo, useState } from "react";
import { AppState, Pressable, StatusBar, StyleSheet, Text, View } from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { api, LockedError } from "./src/services/api";
import { registerBackgroundSync } from "./src/services/background";
import { configureNotifications } from "./src/services/reminders";
import { flushQueue, queueCount } from "./src/services/queue";
import { CalendarScreen } from "./src/screens/CalendarScreen";
import { EntriesScreen } from "./src/screens/EntriesScreen";
import { LockScreen } from "./src/screens/LockScreen";
import { RecapScreen } from "./src/screens/RecapScreen";
import { RecordScreen } from "./src/screens/RecordScreen";
import { SettingsScreen } from "./src/screens/SettingsScreen";
import { colors } from "./src/theme";

const TABS = [
  { key: "record", label: "Kaydet", icon: "mic-outline", activeIcon: "mic" },
  { key: "entries", label: "Girişler", icon: "list-outline", activeIcon: "list" },
  { key: "calendar", label: "Takvim", icon: "calendar-outline", activeIcon: "calendar" },
  { key: "recap", label: "Özet", icon: "stats-chart-outline", activeIcon: "stats-chart" },
  { key: "settings", label: "Ayarlar", icon: "settings-outline", activeIcon: "settings" },
];

export default function App() {
  const [tab, setTab] = useState("record");
  const [privacy, setPrivacy] = useState({ password_enabled: false, unlocked: true });
  const [queueSize, setQueueSize] = useState(0);
  const locked = privacy.password_enabled && !privacy.unlocked;

  useEffect(() => {
    bootstrap();
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active") syncAndRefresh();
    });
    return () => sub.remove();
  }, []);

  async function bootstrap() {
    await Promise.all([configureNotifications(), registerBackgroundSync()]);
    await syncAndRefresh();
  }

  async function syncAndRefresh() {
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
  }

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
      case "record":
      default:
        return <RecordScreen onQueueChanged={setQueueSize} />;
    }
  }, [locked, privacy, queueSize, tab]);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe}>
        <StatusBar barStyle="light-content" />
        <View style={styles.app}>{content}</View>
        <View style={styles.tabs}>
          {TABS.map(({ key, label, icon, activeIcon }) => (
            <Pressable key={key} onPress={() => setTab(key)} style={styles.tab}>
              <Ionicons
                name={tab === key ? activeIcon : icon}
                size={21}
                color={tab === key ? colors.accent2 : colors.muted}
              />
              <Text style={[styles.tabText, tab === key && styles.activeTabText]}>{label}</Text>
              {key === "record" && queueSize ? <Text style={styles.queueDot}>{queueSize}</Text> : null}
            </Pressable>
          ))}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  app: {
    flex: 1,
  },
  tabs: {
    flexDirection: "row",
    borderTopColor: colors.border,
    borderTopWidth: 1,
    backgroundColor: "#111822",
    paddingTop: 6,
    paddingBottom: 8,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 46,
  },
  tabText: {
    color: colors.muted,
    fontSize: 11,
    fontWeight: "800",
  },
  activeTabText: {
    color: colors.accent2,
  },
  queueDot: {
    position: "absolute",
    top: 2,
    right: 14,
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    overflow: "hidden",
    color: "#fff",
    backgroundColor: colors.warn,
    textAlign: "center",
    fontSize: 11,
    fontWeight: "900",
    lineHeight: 18,
  },
});
