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
