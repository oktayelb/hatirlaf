// The database. One SQLite file on the phone, and nothing anywhere else.
//
// Entry text lives in the table; audio lives on the filesystem next to it and
// is referenced by path. That split is deliberate: SQLite is the wrong place
// for multi-megabyte WAV blobs, and keeping audio as ordinary files means the
// export in backup.js can hand them to the OS share sheet untouched.

import * as SQLite from "expo-sqlite";

const DB_NAME = "hatirlaf.db";

let handle = null;

/** Opens the database once and applies the schema. Safe to call anywhere. */
export async function db() {
  if (handle) return handle;
  handle = await SQLite.openDatabaseAsync(DB_NAME);
  await migrate(handle);
  return handle;
}

async function migrate(database) {
  // WAL matters here: recording writes an entry row while the entries list may
  // be reading, and the default journal mode would make one of them fail.
  await database.execAsync(`
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS entries (
      id            TEXT PRIMARY KEY NOT NULL,
      recorded_at   TEXT NOT NULL,
      transcript    TEXT NOT NULL DEFAULT '',
      audio_path    TEXT,
      duration_ms   INTEGER NOT NULL DEFAULT 0,
      source        TEXT NOT NULL DEFAULT 'text',
      created_at    TEXT NOT NULL,
      updated_at    TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS entries_recorded_at
      ON entries (recorded_at DESC);
  `);

  await addColumns(database);
}

// Added after the first release, so they arrive by ALTER rather than in the
// CREATE above — a phone that already has the diary must keep it.
//
// The recogniser needs to be told the shape of any audio it is asked to read
// back from a file, and it has no way to work that out for itself. Recording
// it at capture time is what lets the Ayarlar backfill transcribe an old entry
// without guessing. Rows written before this carry 0, which entries.audioFormat
// reads as "whatever that era's recorder produced".
const ADDED_COLUMNS = [
  ["sample_rate", "INTEGER NOT NULL DEFAULT 0"],
  ["channels", "INTEGER NOT NULL DEFAULT 0"],
];

async function addColumns(database) {
  const existing = await database.getAllAsync(`PRAGMA table_info(entries)`);
  const have = new Set(existing.map((column) => column.name));
  for (const [name, definition] of ADDED_COLUMNS) {
    if (have.has(name)) continue;
    await database.execAsync(`ALTER TABLE entries ADD COLUMN ${name} ${definition}`);
  }
}

/** Drops the handle so the next db() call reopens. Used after a wipe. */
export function resetHandle() {
  handle = null;
}

export { DB_NAME };
