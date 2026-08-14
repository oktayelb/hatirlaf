// The one or two pictures shown on the home screen while you talk.
//
// Picked images are copied into the app's own document directory, so they
// survive the OS clearing its picker cache. They are never uploaded and are
// never attached to a diary entry.

import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system/legacy";
import * as ImagePicker from "expo-image-picker";

const PHOTOS_KEY = "hatirlaf.photos";
const PHOTO_DIR = `${FileSystem.documentDirectory}hatirlaf-photos/`;

export const SLOTS = [0, 1];

async function ensureDir() {
  const info = await FileSystem.getInfoAsync(PHOTO_DIR);
  if (!info.exists) await FileSystem.makeDirectoryAsync(PHOTO_DIR, { intermediates: true });
}

export async function loadPhotos() {
  const raw = await AsyncStorage.getItem(PHOTOS_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (_) {
    return {};
  }
}

async function savePhotos(photos) {
  await AsyncStorage.setItem(PHOTOS_KEY, JSON.stringify(photos));
}

/** Open the OS picker and store the result in `slot`. Returns the new map. */
export async function pickPhoto(slot) {
  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) {
    const error = new Error("Fotoğraflara erişim izni verilmedi.");
    error.code = "permission";
    throw error;
  }

  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ["images"],
    quality: 0.85,
    allowsEditing: true,
  });
  if (result.canceled || !result.assets?.length) return null;

  await ensureDir();
  const source = result.assets[0].uri;
  const ext = (source.split(".").pop() || "jpg").split("?")[0].slice(0, 5);
  // Slot-stable name plus a stamp, so replacing a photo never collides with
  // the cached copy of the one it replaced.
  const target = `${PHOTO_DIR}slot-${slot}-${Date.now()}.${ext}`;
  await FileSystem.copyAsync({ from: source, to: target });

  const photos = await loadPhotos();
  const previous = photos[slot];
  photos[slot] = target;
  await savePhotos(photos);
  if (previous) await FileSystem.deleteAsync(previous, { idempotent: true }).catch(() => {});
  return photos;
}

export async function removePhoto(slot) {
  const photos = await loadPhotos();
  const uri = photos[slot];
  delete photos[slot];
  await savePhotos(photos);
  if (uri) await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
  return photos;
}
