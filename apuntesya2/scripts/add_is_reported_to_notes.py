from sqlalchemy import create_engine, text
from datetime import datetime
import os

# Uses same DB URI as app.py (default sqlite file in instance/ or root)
DB_URL = os.environ.get("DATABASE_URL") or "sqlite:///instance/apuntesya2.db"

engine = create_engine(DB_URL, future=True)
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE notes ADD COLUMN is_reported BOOLEAN DEFAULT 0"))
        print("✔ Columna is_reported agregada")
    except Exception as e:
        print("ℹ️ Posible que ya exista la columna is_reported:", e)