from app import db
from sqlalchemy import text

# Ejecutá una sola vez
db.session.execute(text("ALTER TABLE users ADD COLUMN contact_phone VARCHAR(32)"))
db.session.execute(text("ALTER TABLE users ADD COLUMN contact_phone_visible BOOLEAN DEFAULT 1 NOT NULL"))
db.session.commit()
print("✔ Columns contact_phone y contact_phone_visible agregadas")
