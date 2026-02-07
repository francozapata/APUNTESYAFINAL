-- ============================================================
-- APUNTESYA - MIGRACIÓN A3 (Estadísticas + Movimientos)
-- ============================================================
-- Crea tabla de eventos de analytics para estadísticas de uso.
-- Ejecutar en Supabase (Postgres) en el mismo schema donde están las tablas.

BEGIN;

-- 1) Tabla de eventos (sin datos sensibles)
CREATE TABLE IF NOT EXISTS analytics_events (
  id BIGSERIAL PRIMARY KEY,
  event VARCHAR(64) NOT NULL,
  user_id INTEGER NULL,
  path VARCHAR(255) NULL,
  note_id INTEGER NULL,
  combo_id INTEGER NULL,
  ip VARCHAR(64) NULL,
  user_agent VARCHAR(255) NULL,
  referrer VARCHAR(255) NULL,
  meta JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2) Índices para consultas típicas de dashboard
CREATE INDEX IF NOT EXISTS idx_analytics_events_created_at ON analytics_events(created_at);
CREATE INDEX IF NOT EXISTS idx_analytics_events_event ON analytics_events(event);
CREATE INDEX IF NOT EXISTS idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_note_id ON analytics_events(note_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_combo_id ON analytics_events(combo_id);

COMMIT;
