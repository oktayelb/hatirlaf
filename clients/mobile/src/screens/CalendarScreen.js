import React, { useEffect, useMemo, useState } from "react";
import { Alert, ScrollView, StyleSheet, Text, View } from "react-native";
import { api } from "../services/api";
import {
  cancelReminderForEvent,
  scheduleReminderForEvent,
  scheduledReminderIds,
} from "../services/reminders";
import { Button, Card, EmptyState, Loading, Screen } from "../ui/Primitives";
import { colors } from "../theme";

const MONTHS = [
  "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
  "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
];

export function CalendarScreen() {
  const now = new Date();
  const [cursor, setCursor] = useState({ year: now.getFullYear(), month: now.getMonth() });
  const [days, setDays] = useState({});
  const [scheduled, setScheduled] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const monthKey = `${cursor.year}-${String(cursor.month + 1).padStart(2, "0")}`;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.calendar(monthKey)
      .then((data) => {
        if (cancelled) return;
        setDays(data.days || {});
        scheduledReminderIds().then((ids) => !cancelled && setScheduled(ids));
      })
      .catch(console.warn)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [monthKey]);

  const agenda = useMemo(() => {
    return Object.entries(days)
      .flatMap(([date, events]) => (events || []).map((event) => ({ ...event, date })))
      .sort((a, b) => `${a.date}${a.saat || ""}`.localeCompare(`${b.date}${b.saat || ""}`));
  }, [days]);

  function changeMonth(delta) {
    const next = new Date(cursor.year, cursor.month + delta, 1);
    setCursor({ year: next.getFullYear(), month: next.getMonth() });
  }

  async function toggleReminder(event) {
    if (!event.reminder?.eligible) return;
    const isScheduled = scheduled.has(event.reminder.id);
    if (isScheduled) {
      await cancelReminderForEvent(event);
      const ids = await scheduledReminderIds();
      setScheduled(ids);
      Alert.alert("Hatırlatma kaldırıldı");
      return;
    }
    const result = await scheduleReminderForEvent(event);
    if (!result?.scheduled) {
      Alert.alert("Hatırlatma planlanamadı", "Bildirim izni kapalı olabilir ya da zaman geçmiş olabilir.");
      return;
    }
    const ids = await scheduledReminderIds();
    setScheduled(ids);
    Alert.alert("Hatırlatma planlandı", "Olaydan 30 dakika önce bildirim gelecek.");
  }

  return (
    <Screen
      title="Takvim"
      subtitle={`${MONTHS[cursor.month]} ${cursor.year}`}
      action={
        <View style={styles.nav}>
          <Button variant="ghost" onPress={() => changeMonth(-1)} style={styles.navButton}>‹</Button>
          <Button variant="ghost" onPress={() => changeMonth(1)} style={styles.navButton}>›</Button>
        </View>
      }
    >
      {loading ? (
        <Loading label="Takvim yükleniyor..." />
      ) : agenda.length ? (
        <ScrollView contentContainerStyle={styles.list}>
          {agenda.map((event, index) => (
            <Card key={`${event.date}-${index}`} style={styles.event}>
              <View style={styles.eventTop}>
                <Text style={styles.eventDate}>{formatDate(event.date)}</Text>
                <Text style={[styles.bucket, bucketStyle(event.zaman_dilimi)]}>
                  {event.zaman_dilimi || "Olay"}
                </Text>
              </View>
              <Text style={styles.eventText}>{event.olay || "Olay"}</Text>
              <Text style={styles.eventMeta}>
                {[event.saat, event.lokasyon, (event.kisiler || []).filter(Boolean).join(", ")]
                  .filter(Boolean)
                  .join(" · ")}
              </Text>
              {event.reminder?.eligible ? (
                <Button variant={scheduled.has(event.reminder.id) ? "ghost" : "primary"} onPress={() => toggleReminder(event)}>
                  {scheduled.has(event.reminder.id) ? "Hatırlatma açık" : "30 dk önce hatırlat"}
                </Button>
              ) : null}
            </Card>
          ))}
        </ScrollView>
      ) : (
        <EmptyState title="Bu ay kayıt yok" text="Kayıt ekledikçe olaylar burada görünür." />
      )}
    </Screen>
  );
}

function formatDate(date) {
  return new Date(`${date}T00:00:00`).toLocaleDateString("tr-TR", {
    weekday: "short",
    day: "numeric",
    month: "long",
  });
}

function bucketStyle(bucket) {
  if (bucket === "Gelecek") return { color: colors.warn };
  if (bucket === "Şu An") return { color: colors.ok };
  return { color: colors.accent2 };
}

const styles = StyleSheet.create({
  nav: {
    flexDirection: "row",
    gap: 6,
  },
  navButton: {
    width: 44,
    paddingHorizontal: 0,
  },
  list: {
    gap: 10,
    paddingBottom: 20,
  },
  event: {
    gap: 8,
  },
  eventTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 10,
  },
  eventDate: {
    color: colors.muted,
    fontWeight: "700",
  },
  bucket: {
    fontWeight: "800",
    fontSize: 12,
  },
  eventText: {
    color: colors.text,
    fontSize: 15,
    lineHeight: 22,
  },
  eventMeta: {
    color: colors.muted,
    fontSize: 12,
  },
});
