import React, { useEffect, useState } from "react";
import { ScrollView, StyleSheet, Text, View } from "react-native";
import { api } from "../services/api";
import { Button, Card, EmptyState, Loading, Screen } from "../ui/Primitives";
import { colors } from "../theme";

const MONTHS = [
  "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
  "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
];

export function RecapScreen() {
  const now = new Date();
  const [cursor, setCursor] = useState({ year: now.getFullYear(), month: now.getMonth() });
  const [recap, setRecap] = useState(null);
  const [loading, setLoading] = useState(true);
  const monthKey = `${cursor.year}-${String(cursor.month + 1).padStart(2, "0")}`;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api.recap(monthKey)
      .then((data) => !cancelled && setRecap(data))
      .catch(console.warn)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [monthKey]);

  function changeMonth(delta) {
    const next = new Date(cursor.year, cursor.month + delta, 1);
    setCursor({ year: next.getFullYear(), month: next.getMonth() });
  }

  return (
    <Screen
      title="Özet"
      subtitle={`${MONTHS[cursor.month]} ${cursor.year}`}
      action={
        <View style={styles.nav}>
          <Button variant="ghost" onPress={() => changeMonth(-1)} style={styles.navButton}>‹</Button>
          <Button variant="ghost" onPress={() => changeMonth(1)} style={styles.navButton}>›</Button>
        </View>
      }
    >
      {loading ? (
        <Loading label="Özet hazırlanıyor..." />
      ) : recap ? (
        <ScrollView contentContainerStyle={styles.list}>
          <Card style={styles.hero}>
            <Text style={styles.kicker}>Bu ayın hikayesi</Text>
            <Text style={styles.title}>{recap.title}</Text>
            <Text style={styles.summary}>{recap.summary}</Text>
          </Card>
          <View style={styles.stats}>
            {stat("Kayıt", recap.stats?.sessions)}
            {stat("Olay", recap.stats?.events)}
            {stat("Aktif gün", recap.stats?.active_days)}
            {stat("Kişi", recap.stats?.people)}
          </View>
          <Ranked title="İnsanlar" items={recap.top_people || []} />
          <Ranked title="Yerler" items={recap.top_places || []} />
        </ScrollView>
      ) : (
        <EmptyState title="Özet yok" text="Bu ay için veri bulunamadı." />
      )}
    </Screen>
  );
}

function stat(label, value = 0) {
  return (
    <Card style={styles.stat} key={label}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </Card>
  );
}

function Ranked({ title, items }) {
  return (
    <Card style={styles.ranked}>
      <Text style={styles.rankedTitle}>{title}</Text>
      {items.length ? (
        items.slice(0, 5).map((item, index) => (
          <View key={`${item.label}-${index}`} style={styles.rankRow}>
            <Text style={styles.rankNumber}>{index + 1}</Text>
            <Text style={styles.rankLabel}>{item.display_label || item.label}</Text>
            <Text style={styles.rankCount}>{item.count}</Text>
          </View>
        ))
      ) : (
        <Text style={styles.emptyText}>Henüz veri yok.</Text>
      )}
    </Card>
  );
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
    gap: 12,
    paddingBottom: 20,
  },
  hero: {
    gap: 8,
  },
  kicker: {
    color: colors.accent2,
    fontWeight: "900",
    fontSize: 12,
    textTransform: "uppercase",
  },
  title: {
    color: colors.text,
    fontSize: 22,
    fontWeight: "900",
  },
  summary: {
    color: colors.muted,
    lineHeight: 21,
  },
  stats: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  stat: {
    width: "47%",
  },
  statValue: {
    color: colors.text,
    fontSize: 24,
    fontWeight: "900",
  },
  statLabel: {
    color: colors.muted,
    fontSize: 12,
    marginTop: 4,
  },
  ranked: {
    gap: 10,
  },
  rankedTitle: {
    color: colors.muted,
    fontWeight: "800",
    fontSize: 12,
    textTransform: "uppercase",
  },
  rankRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  rankNumber: {
    color: colors.muted,
    width: 22,
    fontWeight: "800",
  },
  rankLabel: {
    color: colors.text,
    flex: 1,
  },
  rankCount: {
    color: colors.accent2,
    fontWeight: "900",
  },
  emptyText: {
    color: colors.muted,
  },
});
