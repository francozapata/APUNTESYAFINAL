# make_admin.py
# Uso:
#   python make_admin.py --email usuario@ejemplo.com [--revoke]
#
# - Por defecto asigna rol admin (is_admin=1)
# - Con --revoke lo quita (is_admin=0)

import argparse
from sqlalchemy import create_engine, text
import os

# NOTE: the app uses DATABASE_URL (SQLite local / Postgres Supabase).

def set_admin(email: str, make_admin: bool = True) -> None:
    uri = os.getenv("DATABASE_URL")
    if not uri:
        print("❌ No se encontró DATABASE_URL en variables de entorno.")
        return

    if uri.startswith("postgresql://"):
        uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
    elif uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg://", 1)

    engine = create_engine(uri, future=True)

    with engine.begin() as conn:
        # 1) Verifico que exista el usuario
        row = conn.execute(
            text("SELECT id, email, is_admin, COALESCE(role,'') AS role FROM users WHERE lower(email) = lower(:email)"),
            {"email": email},
        ).fetchone()

        if not row:
            print(f"❌ No se encontró un usuario con el correo: {email}")
            return

        # 2) Actualizo rol + is_admin (compat)
        new_val = 1 if make_admin else 0
        new_role = "admin" if make_admin else "user"
        try:
            conn.execute(
                text("UPDATE users SET is_admin = :val, role = :role WHERE lower(email) = lower(:email)"),
                {"val": new_val, "role": new_role, "email": email},
            )
        except Exception:
            conn.execute(
                text("UPDATE users SET is_admin = :val WHERE lower(email) = lower(:email)"),
                {"val": new_val, "email": email},
            )

        # 3) Confirmo
        updated = conn.execute(
            text("SELECT id, email, is_admin, COALESCE(role,'') AS role FROM users WHERE lower(email) = lower(:email)"),
            {"email": email},
        ).fetchone()

    if updated and updated.is_admin == new_val:
        if make_admin:
            print(f"✅ El usuario {updated.email} ahora es ADMIN (role={updated.role or 'admin'}).")
        else:
            print(f"✅ El usuario {updated.email} ya NO es admin (role={updated.role or 'user'}).")
    else:
        print("⚠️ No se pudo confirmar el cambio. Revisá la base de datos.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asignar o revocar rol de administrador por email.")
    parser.add_argument("--email", required=True, help="Correo del usuario")
    parser.add_argument("--revoke", action="store_true", help="Revoca el rol de admin (is_admin=0)")
    args = parser.parse_args()

    set_admin(args.email, make_admin=(not args.revoke))
