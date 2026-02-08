-- ============================================================
-- APUNTESYA - MIGRACION A5 (Stats diarios)
-- SQLite / local
-- ============================================================

-- Purchases: flag para evitar doble conteo
-- (SQLite no soporta IF NOT EXISTS en ADD COLUMN de forma estándar; si ya existe, ignorar el error.)
ALTER TABLE purchases ADD COLUMN stats_counted BOOLEAN DEFAULT 0;

-- Tabla de agregados diarios
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
