"""Tiny SQLite store. One connection, guarded by a lock, JSON in text columns."""
import json
import sqlite3
import threading
import time
import uuid

from .config import DB_FILE

_lock = threading.RLock()
_con = sqlite3.connect(str(DB_FILE), check_same_thread=False)
_con.row_factory = sqlite3.Row
_con.executescript("""
CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT, created_at REAL, sample_count INTEGER, manifest TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, project_name TEXT, dataset_id TEXT, model_id TEXT, state TEXT,
  created_at REAL, started_at REAL, finished_at REAL, current_step INTEGER DEFAULT 0, total_steps INTEGER DEFAULT 0,
  loss REAL, error_message TEXT, config TEXT, metrics TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, timestamp REAL, event_type TEXT, message TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS ev_job ON events(job_id, id);
CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, job_id TEXT, version_name TEXT, base_model_id TEXT, created_at REAL, manifest TEXT);
""")
for _sql in ("ALTER TABLE jobs ADD COLUMN progress TEXT",):      # lightweight migrations for older databases
    try:
        _con.execute(_sql)
    except sqlite3.OperationalError:
        pass
_con.commit()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def q(sql, args=()):
    with _lock:
        cur = _con.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        return rows


def run(sql, args=()):
    with _lock:
        cur = _con.execute(sql, args)
        _con.commit()
        return cur.lastrowid


def one(sql, args=()):
    r = q(sql, args)
    return r[0] if r else None


def jload(s, default=None):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


def add_event(job_id, event_type, message, data=None):
    run("INSERT INTO events(job_id,timestamp,event_type,message,data) VALUES(?,?,?,?,?)",
        (job_id, time.time(), event_type, message, json.dumps(data) if data is not None else None))
