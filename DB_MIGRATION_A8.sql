-- A8) Legal acceptance audit (historial) - SQLite
CREATE TABLE IF NOT EXISTS legal_acceptance_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  legal_version VARCHAR(32) NOT NULL,
  accepted_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  ip VARCHAR(64),
  user_agent VARCHAR(255),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_legal_accept_audit_user ON legal_acceptance_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_legal_accept_audit_version ON legal_acceptance_audit(legal_version);
CREATE INDEX IF NOT EXISTS idx_legal_accept_audit_accepted_at ON legal_acceptance_audit(accepted_at);
