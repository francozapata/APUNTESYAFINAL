
"""
One-shot SQLite schema upgrade for admin features.
- Adds users.is_active, users.is_admin, users.deleted_at
- Adds notes.deleted_at
- Creates admin_actions table if missing
"""
import os, sqlite3, sys
from datetime import datetime

DB_URL = os.getenv("DATABASE_URL", "sqlite:///instance/apuntesya2.db")
if DB_URL.startswith("sqlite:///"):
    db_path = DB_URL.replace("sqlite:///", "", 1)
else:
    raise SystemExit("This upgrader supports SQLite only. Set DATABASE_URL=sqlite:///...")

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", db_path))
os.makedirs(os.path.dirname(db_path), exist_ok=True)

con = sqlite3.connect(db_path)
cur = con.cursor()

def has_column(table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def table_exists(table):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None

# users columns
if not has_column("users", "is_active"):
    cur.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
if not has_column("users", "is_admin"):
    cur.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
if not has_column("users", "deleted_at"):
    cur.execute("ALTER TABLE users ADD COLUMN deleted_at DATETIME NULL")

# notes deleted_at
if not has_column("notes", "deleted_at"):
    cur.execute("ALTER TABLE notes ADD COLUMN deleted_at DATETIME NULL")

# admin_actions
if not table_exists("admin_actions"):
    cur.execute("""
        CREATE TABLE admin_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            reason TEXT NULL,
            ip TEXT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

con.commit()
con.close()
print("Upgrade OK on", db_path)
