-- A10) Imprenta - Fase 1 (Supabase / PostgreSQL)
--
-- 1) Extiende 'notes' para permitir impresión (solo para usuarios role='imprenta')
-- 2) Crea tabla 'print_orders' para registrar pedidos de impresión pagos (Mercado Pago)
--
-- Nota: La descarga del PDF sigue siendo gratuita (price_cents=0). El pago es por el servicio físico.

-- 1) NOTES: campos de impresión
ALTER TABLE notes
  ADD COLUMN IF NOT EXISTS print_enabled BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE notes
  ADD COLUMN IF NOT EXISTS print_net_cents INTEGER NOT NULL DEFAULT 0;

ALTER TABLE notes
  ADD COLUMN IF NOT EXISTS print_binding_extra_cents INTEGER NOT NULL DEFAULT 0;

ALTER TABLE notes
  ADD COLUMN IF NOT EXISTS print_specs TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_notes_print_enabled ON notes(print_enabled);

-- 2) PRINT ORDERS
CREATE TABLE IF NOT EXISTS print_orders (
  id SERIAL PRIMARY KEY,
  buyer_id INTEGER NOT NULL,
  note_id INTEGER NOT NULL,
  imprenta_id INTEGER NOT NULL,
  with_binding BOOLEAN NOT NULL DEFAULT FALSE,
  payment_id VARCHAR(64) NULL,
  preference_id VARCHAR(64) NULL,
  status VARCHAR(32) NOT NULL DEFAULT 'pending',
  gross_cents INTEGER NOT NULL DEFAULT 0,
  platform_fee_cents INTEGER NOT NULL DEFAULT 0,
  mp_fee_cents INTEGER NOT NULL DEFAULT 0,
  imprenta_net_cents INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  CONSTRAINT fk_print_orders_buyer FOREIGN KEY(buyer_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT fk_print_orders_note FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE,
  CONSTRAINT fk_print_orders_imprenta FOREIGN KEY(imprenta_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_print_orders_buyer ON print_orders(buyer_id);
CREATE INDEX IF NOT EXISTS idx_print_orders_note ON print_orders(note_id);
CREATE INDEX IF NOT EXISTS idx_print_orders_imprenta ON print_orders(imprenta_id);
CREATE INDEX IF NOT EXISTS idx_print_orders_status ON print_orders(status);
CREATE INDEX IF NOT EXISTS idx_print_orders_created_at ON print_orders(created_at);
