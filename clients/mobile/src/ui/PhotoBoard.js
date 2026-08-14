import React, { useCallback, useEffect, useState } from "react";
import { Alert, Image, Pressable, StyleSheet, Text, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { SLOTS, loadPhotos, pickPhoto, removePhoto } from "../services/photos";
import { Button } from "./Primitives";
import { colors, radius, spacing, type } from "../theme";

const CAPTIONS = ["1. Fotoğraf", "2. Fotoğraf"];

export function PhotoBoard() {
  const [photos, setPhotos] = useState({});

  useEffect(() => {
    loadPhotos().then(setPhotos).catch(() => {});
  }, []);

  const choose = useCallback(async (slot) => {
    try {
      const next = await pickPhoto(slot);
      if (next) setPhotos({ ...next });
    } catch (err) {
      Alert.alert("Fotoğraf eklenemedi", err.message);
    }
  }, []);

  const drop = useCallback(async (slot) => {
    const next = await removePhoto(slot);
    setPhotos({ ...next });
  }, []);

  const hasAny = SLOTS.some((slot) => photos[slot]);
  // The second frame only appears once the first is filled, so a first-time
  // user sees one clear invitation rather than two identical empty boxes.
  const visible = hasAny ? SLOTS : [0];

  return (
    <View style={styles.board}>
      <View style={[styles.frames, visible.length === 2 && styles.framesPair]}>
        {visible.map((slot) =>
          photos[slot] ? (
            <FilledFrame key={slot} uri={photos[slot]} slot={slot} onPress={() => choose(slot)} />
          ) : (
            <EmptyFrame key={slot} slot={slot} onPress={() => choose(slot)} />
          )
        )}
      </View>

      {hasAny ? (
        <View style={styles.actions}>
          {SLOTS.filter((slot) => photos[slot]).map((slot) => (
            <Button key={slot} variant="ghost" onPress={() => drop(slot)} style={styles.actionButton}>
              {`${CAPTIONS[slot]}ı kaldır`}
            </Button>
          ))}
        </View>
      ) : null}
    </View>
  );
}

function FilledFrame({ uri, slot, onPress }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={`${CAPTIONS[slot]}ı değiştir`}
      onPress={onPress}
      style={styles.frame}
    >
      <View style={styles.mat}>
        <Image source={{ uri }} style={styles.image} resizeMode="cover" />
      </View>
      <Text style={styles.caption}>Değiştirmek için dokun</Text>
    </Pressable>
  );
}

function EmptyFrame({ slot, onPress }) {
  // A slim strip while empty — no mat, no 4:3 box — so the invitation takes
  // far less room than the picture it will hold.
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={`${CAPTIONS[slot]}ı ekle`}
      onPress={onPress}
      style={[styles.frame, styles.frameEmpty]}
    >
      <View style={styles.emptyInner}>
        <Ionicons name="image-outline" size={28} color={colors.accentDeep} />
        <Text style={styles.emptyLabel}>Fotoğraf Ekle</Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  board: {
    gap: spacing.sm,
  },
  frames: {
    flexDirection: "row",
    gap: spacing.md,
    // An empty slot next to a filled one keeps its own small height.
    alignItems: "flex-start",
  },
  framesPair: {
    // Both frames share the row evenly; `flex: 1` on the frame does the work.
  },
  frame: {
    flex: 1,
    padding: 12,
    borderRadius: radius.md,
    backgroundColor: "#a9835a",
    alignItems: "center",
  },
  frameEmpty: {
    backgroundColor: colors.surface2,
    borderWidth: 2,
    borderColor: colors.lineStrong,
    borderStyle: "dashed",
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  mat: {
    width: "100%",
    backgroundColor: colors.surface,
    borderRadius: 6,
    padding: 12,
  },
  image: {
    width: "100%",
    aspectRatio: 4 / 3,
    borderRadius: 3,
    backgroundColor: colors.surface2,
  },
  caption: {
    marginTop: spacing.sm,
    fontSize: type.sm,
    fontWeight: "600",
    color: "#4a3720",
    backgroundColor: "rgba(255,253,248,0.88)",
    borderRadius: radius.pill,
    paddingHorizontal: 12,
    paddingVertical: 4,
    overflow: "hidden",
  },
  emptyInner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
    minHeight: 44,
  },
  emptyLabel: {
    fontSize: type.md,
    fontWeight: "700",
    color: colors.accentDeep,
  },
  actions: {
    flexDirection: "row",
    gap: spacing.sm,
    flexWrap: "wrap",
  },
  actionButton: {
    flexGrow: 1,
    flexBasis: 150,
  },
});
