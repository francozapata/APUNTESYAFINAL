-- Pedidos de apuntes (marketplace inverso)

CREATE TABLE IF NOT EXISTS note_requests (
  id BIGSERIAL PRIMARY KEY,
  buyer_id BIGINT NOT NULL REFERENCES users(id),
  title VARCHAR(180) NOT NULL,
  career VARCHAR(120) NOT NULL,
  subject VARCHAR(120) NOT NULL,
  university VARCHAR(120),
  faculty VARCHAR(120),
  professor VARCHAR(120),
  material_type VARCHAR(40) NOT NULL DEFAULT 'resumen',
  exam_date_text VARCHAR(80),
  description TEXT NOT NULL,
  offered_price INTEGER NOT NULL DEFAULT 0,
  accept_similar BOOLEAN NOT NULL DEFAULT FALSE,
  status VARCHAR(24) NOT NULL DEFAULT 'open',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS note_request_offers (
  id BIGSERIAL PRIMARY KEY,
  request_id BIGINT NOT NULL REFERENCES note_requests(id) ON DELETE CASCADE,
  seller_id BIGINT NOT NULL REFERENCES users(id),
  title VARCHAR(180) NOT NULL,
  description TEXT NOT NULL,
  material_type VARCHAR(40) NOT NULL DEFAULT 'resumen',
  professor VARCHAR(120),
  year_text VARCHAR(80),
  page_count INTEGER,
  allow_publish_after_sale BOOLEAN NOT NULL DEFAULT TRUE,
  seller_price INTEGER NOT NULL DEFAULT 0,
  buyer_counter_price INTEGER,
  agreed_price INTEGER,
  status VARCHAR(24) NOT NULL DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  CONSTRAINT uq_request_offer_seller UNIQUE (request_id, seller_id)
);

CREATE INDEX IF NOT EXISTS ix_note_requests_buyer_id ON note_requests(buyer_id);
CREATE INDEX IF NOT EXISTS ix_note_requests_status ON note_requests(status);
CREATE INDEX IF NOT EXISTS ix_note_request_offers_request_id ON note_request_offers(request_id);
CREATE INDEX IF NOT EXISTS ix_note_request_offers_seller_id ON note_request_offers(seller_id);
CREATE INDEX IF NOT EXISTS ix_note_request_offers_status ON note_request_offers(status);

ALTER TABLE note_request_offers ADD COLUMN IF NOT EXISTS seller_price INTEGER NOT NULL DEFAULT 0;
ALTER TABLE note_request_offers ADD COLUMN IF NOT EXISTS buyer_counter_price INTEGER;
ALTER TABLE note_request_offers ADD COLUMN IF NOT EXISTS agreed_price INTEGER;
