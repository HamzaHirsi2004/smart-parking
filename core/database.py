import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "parking.db"


def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS spaces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            label       TEXT UNIQUE NOT NULL,
            polygon     TEXT,              -- JSON [[x,y],...] in video pixels
            map_x       REAL NOT NULL,     -- position on the customer map
            map_y       REAL NOT NULL,
            is_occupied INTEGER DEFAULT 0,
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS occupancy_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total     INTEGER NOT NULL,
            occupied  INTEGER NOT NULL,
            available INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- spaces ----------
def add_space(label, map_x, map_y, polygon=None, is_occupied=0):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO spaces (label, polygon, map_x, map_y, is_occupied, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (label, json.dumps(polygon) if polygon else None, map_x, map_y, is_occupied, now()),
        )


def get_spaces():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM spaces ORDER BY label").fetchall()
    return [dict(r) for r in rows]


def update_statuses(status_by_label: dict):
    """status_by_label = {"A1": 1, "A2": 0, ...}  (1 = occupied)"""
    with get_conn() as conn:
        for label, occ in status_by_label.items():
            conn.execute(
                "UPDATE spaces SET is_occupied=?, updated_at=? WHERE label=?",
                (int(occ), now(), label),
            )


def get_summary():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(is_occupied),0) AS occupied, "
            "MAX(updated_at) AS last_updated FROM spaces"
        ).fetchone()
    total, occupied = row["total"], row["occupied"]
    return {
        "total": total,
        "occupied": occupied,
        "available": total - occupied,
        "last_updated": row["last_updated"],
    }


# ---------- history ----------
def log_occupancy():
    s = get_summary()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO occupancy_log (timestamp, total, occupied, available) VALUES (?,?,?,?)",
            (now(), s["total"], s["occupied"], s["available"]),
        )


def get_history():
    import pandas as pd
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM occupancy_log ORDER BY timestamp", conn, parse_dates=["timestamp"]
        )


# ---------- settings (e.g. entrance location) ----------
def set_setting(key, value):
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, json.dumps(value)))


def get_setting(key, default=None):
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default