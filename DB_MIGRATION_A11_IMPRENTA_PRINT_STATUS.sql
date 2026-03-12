-- A11) Imprenta Fase 1 - Estado de impresión (separado del estado de pago MP)
--
-- Agrega print_status a print_orders.
--
-- Valores sugeridos:
--   pendiente | en_proceso | listo | retirado | cancelado
--
-- IMPORTANTE: 'status' en print_orders sigue siendo el estado de pago (MP).

ALTER TABLE print_orders
  ADD COLUMN IF NOT EXISTS print_status VARCHAR(32) DEFAULT 'pendiente';

CREATE INDEX IF NOT EXISTS idx_print_orders_imprenta_print_status
  ON print_orders (imprenta_id, print_status);
