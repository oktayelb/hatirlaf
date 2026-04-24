// IndexedDB-based offline cache + sync queue for audio uploads.
// This mirrors the React Native spec: record → store locally → background
// sync when online. Works in any evergreen browser on Linux.

const DB_NAME = "hatirlaf";
const DB_VERSION = 1;
const STORE_PENDING = "pendingSessions"; // recorded but not yet uploaded
const STORE_CACHE = "sessionsCache";     // last server state, for offline view

let _dbPromise = null;

function openDB() {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_PENDING)) {
        const s = db.createObjectStore(STORE_PENDING, { keyPath: "clientUuid" });
        s.createIndex("recordedAt", "recordedAt");
      }
      if (!db.objectStoreNames.contains(STORE_CACHE)) {
        db.createObjectStore(STORE_CACHE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return _dbPromise;
}

async function tx(store, mode, fn) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const s = t.objectStore(store);
    let result;
    Promise.resolve(fn(s))
      .then((r) => (result = r))
      .catch(reject);
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export const pendingStore = {
  async put(record) {
    await tx(STORE_PENDING, "readwrite", (s) => s.put(record));
    return record;
  },
  async get(clientUuid) {
    return tx(STORE_PENDING, "readonly", (s) =>
      new Promise((res) => {
        const r = s.get(clientUuid);
        r.onsuccess = () => res(r.result || null);
      })
    );
  },
  async all() {
    return tx(STORE_PENDING, "readonly", (s) =>
      new Promise((res) => {
        const r = s.getAll();
        r.onsuccess = () => res(r.result || []);
      })
    );
  },
  async delete(clientUuid) {
    return tx(STORE_PENDING, "readwrite", (s) => s.delete(clientUuid));
  },
  async count() {
    return tx(STORE_PENDING, "readonly", (s) =>
      new Promise((res) => {
        const r = s.count();
        r.onsuccess = () => res(r.result || 0);
      })
    );
  },
};

export const cacheStore = {
  async putMany(sessions) {
    return tx(STORE_CACHE, "readwrite", (s) => {
      sessions.forEach((session) => s.put(session));
    });
  },
  async all() {
    return tx(STORE_CACHE, "readonly", (s) =>
      new Promise((res) => {
        const r = s.getAll();
        r.onsuccess = () => res(r.result || []);
      })
    );
  },
  async clear() {
    return tx(STORE_CACHE, "readwrite", (s) => s.clear());
  },
};

export function uuid() {
  // Prefer native crypto.randomUUID, fall back to a simple RFC4122 v4 shim.
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
