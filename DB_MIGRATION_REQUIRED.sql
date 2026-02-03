-- ApuntesYa: migrate users contact fields
-- Safe for Postgres (Supabase). Run once.

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS contact_phone VARCHAR(64),
  ADD COLUMN IF NOT EXISTS contact_website VARCHAR(255);

-- Foto de perfil (si no existiera en tu esquema)
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS imagen_de_perfil VARCHAR(255);

-- (Optional) if you still don't have structured contact fields from earlier versions:
ALTER TABLE users
  ADD COLUMN IF NOT EXISTS contact_email VARCHAR(255),
  ADD COLUMN IF NOT EXISTS contact_whatsapp VARCHAR(64),
  ADD COLUMN IF NOT EXISTS contact_instagram VARCHAR(80),
  ADD COLUMN IF NOT EXISTS contact_visible_public BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS contact_visible_buyers BOOLEAN DEFAULT TRUE;


-- -----------------------
--  EXTRA: download_logs supports combo downloads
--  (note_id must be nullable for combo-only logs)
-- -----------------------
ALTER TABLE download_logs
  ALTER COLUMN note_id DROP NOT NULL;

ALTER TABLE download_logs
  ADD COLUMN IF NOT EXISTS combo_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_download_logs_combo_id ON download_logs(combo_id);
