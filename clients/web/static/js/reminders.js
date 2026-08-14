const STORE_KEY = "hatirlaf-reminders";
const MAX_DELAY = 2_147_483_647;
const timers = new Map();

export function initReminderTimers() {
  for (const reminder of readReminders()) {
    armTimer(reminder);
  }
}

export function isReminderScheduled(event) {
  const id = event?.reminder?.id;
  return Boolean(id && readReminders().some((reminder) => reminder.id === id));
}

export async function scheduleReminderForEvent(event) {
  const reminder = event?.reminder;
  if (!reminder?.eligible) {
    throw new Error("Bu olay için hatırlatma zamanı geçmiş.");
  }
  if (!("Notification" in window)) {
    throw new Error("Bu tarayıcı bildirimleri desteklemiyor.");
  }

  let permission = Notification.permission;
  if (permission === "default") permission = await Notification.requestPermission();
  if (permission !== "granted") {
    throw new Error("Bildirim izni verilmedi.");
  }

  const payload = {
    id: reminder.id,
    title: reminder.title || event.olay || "Yaklaşan olay",
    body: event.olay || "Yaklaşan bir olayın var.",
    date: event.tarih,
    sessionId: event.session_id,
    eventAt: reminder.event_at,
    remindAt: reminder.remind_at,
    createdAt: new Date().toISOString(),
  };

  const reminders = readReminders().filter((item) => item.id !== payload.id);
  reminders.push(payload);
  writeReminders(reminders);
  armTimer(payload);
  return payload;
}

export function cancelReminderForEvent(event) {
  const id = event?.reminder?.id;
  if (!id) return;
  clearTimer(id);
  writeReminders(readReminders().filter((item) => item.id !== id));
}

function armTimer(reminder) {
  clearTimer(reminder.id);
  const remindAt = new Date(reminder.remindAt);
  const delay = remindAt.getTime() - Date.now();
  if (!Number.isFinite(delay)) return;
  if (delay <= 0) {
    fireReminder(reminder);
    return;
  }
  timers.set(
    reminder.id,
    setTimeout(() => armTimer(reminder), Math.min(delay, MAX_DELAY))
  );
}

function clearTimer(id) {
  const timer = timers.get(id);
  if (timer) clearTimeout(timer);
  timers.delete(id);
}

function fireReminder(reminder) {
  clearTimer(reminder.id);
  writeReminders(readReminders().filter((item) => item.id !== reminder.id));
  if (!("Notification" in window) || Notification.permission !== "granted") return;

  const notification = new Notification("Hatırlaf", {
    body: reminder.body,
    tag: reminder.id,
    data: {
      date: reminder.date,
      sessionId: reminder.sessionId,
    },
  });
  notification.onclick = () => {
    window.focus();
    if (reminder.sessionId) location.hash = `#/review/${reminder.sessionId}`;
  };
}

function readReminders() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function writeReminders(reminders) {
  localStorage.setItem(STORE_KEY, JSON.stringify(reminders));
}
