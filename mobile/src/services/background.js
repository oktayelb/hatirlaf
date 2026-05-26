import * as BackgroundTask from "expo-background-task";
import * as TaskManager from "expo-task-manager";
import { flushQueue } from "./queue";

const SYNC_TASK = "hatirlaf-background-sync";

TaskManager.defineTask(SYNC_TASK, async () => {
  try {
    await flushQueue();
    return BackgroundTask.BackgroundTaskResult.Success;
  } catch (_) {
    return BackgroundTask.BackgroundTaskResult.Failed;
  }
});

export async function registerBackgroundSync() {
  try {
    const registered = await TaskManager.isTaskRegisteredAsync(SYNC_TASK);
    if (registered) return true;
    await BackgroundTask.registerTaskAsync(SYNC_TASK, {
      minimumInterval: 15 * 60,
    });
    return true;
  } catch (err) {
    console.warn("Background sync registration failed", err);
    return false;
  }
}
