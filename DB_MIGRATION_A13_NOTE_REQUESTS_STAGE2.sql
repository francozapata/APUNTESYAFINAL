-- Etapa 2 de pedidos de apuntes: negociación básica de precio

ALTER TABLE note_request_offers ADD COLUMN IF NOT EXISTS seller_price INTEGER NOT NULL DEFAULT 0;
ALTER TABLE note_request_offers ADD COLUMN IF NOT EXISTS buyer_counter_price INTEGER;
ALTER TABLE note_request_offers ADD COLUMN IF NOT EXISTS agreed_price INTEGER;

CREATE INDEX IF NOT EXISTS ix_note_requests_status ON note_requests(status);
CREATE INDEX IF NOT EXISTS ix_note_request_offers_status ON note_request_offers(status);
