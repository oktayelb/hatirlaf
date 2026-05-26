import * as Notifications from "expo-notifications";

export async function configureNotifications() {
  try {
    await Notifications.requestPermissionsAsync();
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

export async function scheduleFutureEventReminders(days) {
  try {
    const permissions = await Notifications.getPermissionsAsync();
    if (!permissions.granted) return 0;
    await Notifications.cancelAllScheduledNotificationsAsync();
    let count = 0;
    for (const [date, events] of Object.entries(days || {})) {
      for (const event of events || []) {
        if (event.zaman_dilimi !== "Gelecek") continue;
        const triggerDate = reminderDate(date, event.saat);
        if (!triggerDate || triggerDate.getTime() <= Date.now()) continue;
        await Notifications.scheduleNotificationAsync({
          content: {
            title: "Hatırlaf",
            body: event.olay || "Yaklaşan bir olayın var.",
            data: { date, session_id: event.session_id },
          },
          trigger: {
            type: Notifications.SchedulableTriggerInputTypes.DATE,
            date: triggerDate,
          },
        });
        count += 1;
      }
    }
    return count;
  } catch (err) {
    console.warn("Reminder scheduling failed", err);
    return 0;
  }
}

function reminderDate(date, time) {
  const clock = String(time || "09:00").match(/^(\d{1,2}):(\d{2})/);
  const hour = clock ? Number(clock[1]) : 9;
  const minute = clock ? Number(clock[2]) : 0;
  const trigger = new Date(`${date}T00:00:00`);
  trigger.setHours(hour, minute, 0, 0);
  trigger.setMinutes(trigger.getMinutes() - 30);
  return Number.isNaN(trigger.getTime()) ? null : trigger;
}
