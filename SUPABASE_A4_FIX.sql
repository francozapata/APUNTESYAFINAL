-- ============================================================
-- APUNTESYA - FIX ADMIN STATS/MOVEMENTS (Purchases analytics fields)
-- Ejecutar en Supabase (SQL Editor) UNA SOLA VEZ.
-- Seguro para re-ejecutar (IF NOT EXISTS).
-- ============================================================

-- 1) purchases: columnas faltantes para admin
ALTER TABLE purchases
  ADD COLUMN IF NOT EXISTS buyer_email VARCHAR(255),
  ADD COLUMN IF NOT EXISTS seller_id INTEGER,
  ADD COLUMN IF NOT EXISTS gross_cents INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS platform_fee_cents INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS mp_fee_cents INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS seller_net_cents INTEGER DEFAULT 0;

-- 2) download_logs: compat "was_free" (alias histórico)
ALTER TABLE download_logs
  ADD COLUMN IF NOT EXISTS was_free BOOLEAN DEFAULT FALSE;

-- 3) Backfill (best effort)
UPDATE download_logs
  SET was_free = COALESCE(is_free, FALSE)
  WHERE was_free IS DISTINCT FROM COALESCE(is_free, FALSE);

UPDATE purchases p
  SET buyer_email = u.email
  FROM users u
  WHERE p.buyer_id = u.id
    AND (p.buyer_email IS NULL OR p.buyer_email = '');

UPDATE purchases p
  SET seller_id = n.seller_id
  FROM notes n
  WHERE p.note_id = n.id
    AND p.seller_id IS NULL;

UPDATE purchases p
  SET seller_id = c.seller_id
  FROM combos c
  WHERE p.combo_id = c.id
    AND p.seller_id IS NULL;

UPDATE purchases p
  SET seller_net_cents = COALESCE(n.seller_net_cents, 0)
  FROM notes n
  WHERE p.note_id = n.id
    AND (p.seller_net_cents IS NULL OR p.seller_net_cents = 0);

UPDATE purchases p
  SET seller_net_cents = COALESCE(c.seller_net_cents, 0)
  FROM combos c
  WHERE p.combo_id = c.id
    AND (p.seller_net_cents IS NULL OR p.seller_net_cents = 0);

UPDATE purchases
  SET gross_cents = COALESCE(amount_cents, 0)
  WHERE gross_cents IS NULL OR gross_cents = 0;

-- fees: dejamos 0 por compatibilidad; a futuro se completan desde la app.
UPDATE purchases
  SET platform_fee_cents = COALESCE(platform_fee_cents, 0),
      mp_fee_cents = COALESCE(mp_fee_cents, 0),
      seller_net_cents = COALESCE(seller_net_cents, 0),
      gross_cents = COALESCE(gross_cents, 0)
  WHERE platform_fee_cents IS NULL OR mp_fee_cents IS NULL OR seller_net_cents IS NULL OR gross_cents IS NULL;

-- 4) Índices recomendados (opcional, pero ayuda al admin)
CREATE INDEX IF NOT EXISTS idx_purchases_created_at ON purchases(created_at);
CREATE INDEX IF NOT EXISTS idx_download_logs_created_at ON download_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_purchases_seller_id ON purchases(seller_id);

