-- -----------------------
-- A8) Legal acceptance audit (historial)
-- -----------------------
CREATE TABLE IF NOT EXISTS legal_acceptance_audit (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  legal_version VARCHAR(32) NOT NULL,
  accepted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  ip VARCHAR(64),
  user_agent VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_legal_accept_audit_user ON legal_acceptance_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_legal_accept_audit_version ON legal_acceptance_audit(legal_version);
CREATE INDEX IF NOT EXISTS idx_legal_accept_audit_accepted_at ON legal_acceptance_audit(accepted_at);
