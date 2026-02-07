-- APUNTESYA - A4 (Estadísticas + Movimientos)
-- Seguro de ejecutar varias veces.

BEGIN;

-- 1) Analytics events: aseguramos esquema completo (por si la tabla existía incompleta)
CREATE TABLE IF NOT EXISTS analytics_events (
  id BIGSERIAL PRIMARY KEY,
  event VARCHAR(64) NOT NULL,
  user_id INTEGER NULL,
  note_id INTEGER NULL,
  combo_id INTEGER NULL,
  path VARCHAR(255) NULL,
  meta_json JSONB NULL,
  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now()
);

ALTER TABLE analytics_events
  ADD COLUMN IF NOT EXISTS event VARCHAR(64),
  ADD COLUMN IF NOT EXISTS user_id INTEGER,
  ADD COLUMN IF NOT EXISTS note_id INTEGER,
  ADD COLUMN IF NOT EXISTS combo_id INTEGER,
  ADD COLUMN IF NOT EXISTS path VARCHAR(255),
  ADD COLUMN IF NOT EXISTS meta_json JSONB,
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITHOUT TIME ZONE;

-- backfill created_at si fuese NULL
UPDATE analytics_events SET created_at = now() WHERE created_at IS NULL;

-- 2) Índices para performance
CREATE INDEX IF NOT EXISTS idx_analytics_events_created_at ON analytics_events(created_at);
CREATE INDEX IF NOT EXISTS idx_analytics_events_event ON analytics_events(event);
CREATE INDEX IF NOT EXISTS idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_note_id ON analytics_events(note_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_combo_id ON analytics_events(combo_id);

-- 3) Índices para movimientos
-- Purchases
CREATE INDEX IF NOT EXISTS idx_purchases_status_created_at ON purchases(status, created_at);
CREATE INDEX IF NOT EXISTS idx_purchases_seller_id_created_at ON purchases(seller_id, created_at);
CREATE INDEX IF NOT EXISTS idx_purchases_buyer_id_created_at ON purchases(buyer_id, created_at);

-- Download logs
CREATE INDEX IF NOT EXISTS idx_download_logs_created_at ON download_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_download_logs_user_id_created_at ON download_logs(user_id, created_at);

COMMIT;
