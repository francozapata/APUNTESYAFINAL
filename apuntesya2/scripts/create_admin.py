
"""
Create or promote an admin user by email.
Usage: python scripts/create_admin.py you@example.com
"""
import os, sys, sqlite3, hashlib
from werkzeug.security import generate_password_hash

if len(sys.argv) < 2:
    print("Email requerido: python scripts/create_admin.py correo@ejemplo.com [password]")
    raise SystemExit(1)

email = sys.argv[1].strip().lower()
password = sys.argv[2] if len(sys.argv) > 2 else None

DB_URL = os.getenv("DATABASE_URL", "sqlite:///instance/apuntesya2.db")
if DB_URL.startswith("sqlite:///"):
    db_path = DB_URL.replace("sqlite:///", "", 1)
else:
    raise SystemExit("Only SQLite is supported by this script.")

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", db_path))
con = sqlite3.connect(db_path)
cur = con.cursor()

# ensure user exists
cur.execute("SELECT id FROM users WHERE email=?", (email,))
row = cur.fetchone()
if row is None:
    if not password:
        print("El usuario no existe. Proporcione una contraseña para crearlo: python scripts/create_admin.py email password")
        raise SystemExit(1)
    pwd_hash = generate_password_hash(password)
    cur.execute("INSERT INTO users (name, email, password_hash, university, faculty, career, is_active, is_admin) VALUES (?,?,?,?,?,?,1,1)",
                (email.split('@')[0], email, pwd_hash, "N/A", "N/A", "N/A"))
    print("Usuario creado y promovido a admin:", email)
else:
    cur.execute("UPDATE users SET is_admin=1 WHERE email=?", (email,))
    print("Usuario promovido a admin:", email)

con.commit()
con.close()
