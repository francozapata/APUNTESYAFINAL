-- ============================================================
-- APUNTESYA - MIGRACION A5 (Stats diarios)
-- Compatible con Supabase/Postgres
-- ============================================================

-- 1) Purchases: flag para evitar doble conteo (webhooks pueden repetirse)
ALTER TABLE purchases
  ADD COLUMN IF NOT EXISTS stats_counted BOOLEAN DEFAULT FALSE;

-- 2) Tabla de agregados diarios
CREATE TABLE IF NOT EXISTS stats_daily (
  day DATE PRIMARY KEY,
  gross_income_cents INTEGER DEFAULT 0,
  ay_commission_cents INTEGER DEFAULT 0,
  mp_fee_cents INTEGER DEFAULT 0,
  seller_income_cents INTEGER DEFAULT 0,
  sales_count INTEGER DEFAULT 0,
  free_downloads INTEGER DEFAULT 0,
  paid_downloads INTEGER DEFAULT 0
);

-- Opcional: índice para orden
CREATE INDEX IF NOT EXISTS stats_daily_day_idx ON stats_daily(day);
