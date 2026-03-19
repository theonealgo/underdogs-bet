// node:sqlite is built into Node.js 22.5+ (enable with --experimental-sqlite)
// The dev/build/start scripts in package.json set NODE_OPTIONS automatically.
import { DatabaseSync } from 'node:sqlite';
import path from 'path';

const DB_PATH = path.join(process.cwd(), 'social_media.db');

// Module-level singleton — persists across requests in the same server process
let _db: DatabaseSync | null = null;

export function getDb(): DatabaseSync {
  if (!_db) {
    _db = new DatabaseSync(DB_PATH);
    // PRAGMA via exec (node:sqlite has no .pragma() helper)
    _db.exec('PRAGMA journal_mode = WAL');
    _db.exec('PRAGMA foreign_keys = ON');
    initializeSchema(_db);
  }
  return _db;
}

function initializeSchema(db: DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS social_jobs (
      id                   TEXT PRIMARY KEY,
      name                 TEXT NOT NULL,
      url                  TEXT NOT NULL,
      screenshot_selector  TEXT NOT NULL,
      text_selector        TEXT NOT NULL,
      platforms            TEXT NOT NULL,   -- JSON array e.g. ["twitter","instagram"]
      schedule_time        TEXT NOT NULL,   -- "HH:MM" daily
      is_active            INTEGER NOT NULL DEFAULT 1,
      created_at           TEXT NOT NULL,
      last_run             TEXT,
      last_status          TEXT            -- "success" | "partial" | "pending" | "error"
    );

    CREATE TABLE IF NOT EXISTS post_history (
      id               TEXT PRIMARY KEY,
      job_id           TEXT NOT NULL,
      job_name         TEXT NOT NULL,
      caption          TEXT,
      hashtags         TEXT,
      screenshot_path  TEXT,
      extracted_text   TEXT,
      platforms        TEXT NOT NULL,   -- JSON array
      status           TEXT NOT NULL,
      platform_results TEXT,            -- JSON array of PlatformResult
      created_at       TEXT NOT NULL,
      FOREIGN KEY (job_id) REFERENCES social_jobs(id) ON DELETE CASCADE
    );
  `);
}
