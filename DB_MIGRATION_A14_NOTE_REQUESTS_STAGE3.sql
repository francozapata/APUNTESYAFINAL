CREATE TABLE IF NOT EXISTS note_request_purchases (
    id BIGSERIAL PRIMARY KEY,
    buyer_id BIGINT NOT NULL REFERENCES users(id),
    seller_id BIGINT NOT NULL REFERENCES users(id),
    request_id BIGINT NOT NULL REFERENCES note_requests(id) ON DELETE CASCADE,
    offer_id BIGINT NOT NULL REFERENCES note_request_offers(id) ON DELETE CASCADE,
    payment_id VARCHAR(64),
    preference_id VARCHAR(64),
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    amount_cents INTEGER NOT NULL DEFAULT 0,
    gross_cents INTEGER NOT NULL DEFAULT 0,
    platform_fee_cents INTEGER NOT NULL DEFAULT 0,
    mp_fee_cents INTEGER NOT NULL DEFAULT 0,
    seller_net_cents INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_note_request_purchases_buyer_id ON note_request_purchases(buyer_id);
CREATE INDEX IF NOT EXISTS idx_note_request_purchases_seller_id ON note_request_purchases(seller_id);
CREATE INDEX IF NOT EXISTS idx_note_request_purchases_request_id ON note_request_purchases(request_id);
CREATE INDEX IF NOT EXISTS idx_note_request_purchases_offer_id ON note_request_purchases(offer_id);
CREATE INDEX IF NOT EXISTS idx_note_request_purchases_status ON note_request_purchases(status);
