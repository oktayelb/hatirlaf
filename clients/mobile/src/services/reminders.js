import * as Notifications from "expo-notifications";
import AsyncStorage from "@react-native-async-storage/async-storage";

const STORE_KEY = "hatirlaf.scheduledReminders";

export async function configureNotifications() {
  try {
    Notifications.setNotificationHandler({
      handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: false,
        shouldSetBadge: false,
      }),
    });
  } catch (err) {
    console.warn("Notification setup failed", err);
  }
}

export async function scheduledReminderIds() {
  const items = await readScheduled();
  return new Set(items.map((item) => item.reminderId));
}

export async function scheduleReminderForEvent(event) {
  try {
    if (!event?.reminder?.eligible) return { scheduled: false, reason: "not_eligible" };
    let permissions = await Notifications.getPermissionsAsync();
    if (!permissions.granted) {
      permissions = await Notifications.requestPermissionsAsync();
    }
    if (!permissions.granted) return { scheduled: false, reason: "permission" };

    const triggerDate = new Date(event.reminder.remind_at);
    if (!Number.isFinite(triggerDate.getTime()) || triggerDate.getTime() <= Date.now()) {
      return { scheduled: false, reason: "expired" };
    }

    const existing = await readScheduled();
    const old = existing.find((item) => item.reminderId === event.reminder.id);
    if (old?.notificationId) {
      await Notifications.cancelScheduledNotificationAsync(old.notificationId).catch(() => {});
    }

    const notificationId = await Notifications.scheduleNotificationAsync({
      identifier: event.reminder.id,
      content: {
        title: "Hatırlaf",
        body: event.olay || "Yaklaşan bir olayın var.",
        data: { date: event.tarih, session_id: event.session_id },
      },
      trigger: {
        type: Notifications.SchedulableTriggerInputTypes.DATE,
        date: triggerDate,
      },
    });

    await writeScheduled([
      ...existing.filter((item) => item.reminderId !== event.reminder.id),
      {
        reminderId: event.reminder.id,
        notificationId,
        remindAt: event.reminder.remind_at,
        eventAt: event.reminder.event_at,
        title: event.reminder.title || event.olay || "Yaklaşan olay",
      },
    ]);
    return { scheduled: true, notificationId };
  } catch (err) {
    console.warn("Reminder scheduling failed", err);
    return { scheduled: false, reason: "error" };
  }
}

export async function cancelReminderForEvent(event) {
  const reminderId = event?.reminder?.id;
  if (!reminderId) return;
  const existing = await readScheduled();
  const old = existing.find((item) => item.reminderId === reminderId);
  if (old?.notificationId) {
    await Notifications.cancelScheduledNotificationAsync(old.notificationId).catch(() => {});
  }
  await writeScheduled(existing.filter((item) => item.reminderId !== reminderId));
}

async function readScheduled() {
  const raw = await AsyncStorage.getItem(STORE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

async function writeScheduled(items) {
  await AsyncStorage.setItem(STORE_KEY, JSON.stringify(items));
}
