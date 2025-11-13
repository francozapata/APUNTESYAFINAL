import sqlite3, sys
if len(sys.argv) < 2:
    print("Uso: python migrate_add_seller_contact_sqlite.py <ruta_db.sqlite>")
    raise SystemExit(1)
db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
cur = conn.cursor()
try:
    cur.execute("ALTER TABLE users ADD COLUMN seller_contact VARCHAR(255)")
    conn.commit()
    print("OK: columna seller_contact agregada")
except Exception as e:
    print("Aviso:", e)
finally:
    conn.close()
    print("✔ Listo.")
