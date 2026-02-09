-- ============================================================
-- APUNTESYA - MIGRACIÓN A7
-- Legal acceptance (TyC + Privacidad + Seguridad)
-- ============================================================

ALTER TABLE users
  ADD COLUMN IF NOT EXISTS legal_version_accepted VARCHAR(32),
  ADD COLUMN IF NOT EXISTS legal_accepted_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_users_legal_version_accepted ON users (legal_version_accepted);
