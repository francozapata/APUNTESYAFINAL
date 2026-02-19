-- A9) Tickets (reportes / reclamos) - Supabase (PostgreSQL)
--
-- Crea un sistema de tickets para reemplazar el "flag" de reportado.
-- Genera trazabilidad, estados y notificaciones.

CREATE TABLE IF NOT EXISTS tickets (
  id SERIAL PRIMARY KEY,
  code VARCHAR(32) UNIQUE NOT NULL,
  note_id INTEGER NOT NULL,
  reporter_user_id INTEGER NULL,
  seller_user_id INTEGER NULL,
  status VARCHAR(24) NOT NULL DEFAULT 'new',
  reason VARCHAR(80) NOT NULL DEFAULT 'other',
  details TEXT NULL,
  resolution TEXT NULL,
  admin_notes TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  resolved_at TIMESTAMP NULL,
  CONSTRAINT fk_tickets_note FOREIGN KEY(note_id) REFERENCES notes(id) ON DELETE CASCADE,
  CONSTRAINT fk_tickets_reporter FOREIGN KEY(reporter_user_id) REFERENCES users(id) ON DELETE SET NULL,
  CONSTRAINT fk_tickets_seller FOREIGN KEY(seller_user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_code ON tickets(code);
CREATE INDEX IF NOT EXISTS idx_tickets_note ON tickets(note_id);
CREATE INDEX IF NOT EXISTS idx_tickets_reporter ON tickets(reporter_user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_seller ON tickets(seller_user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at);


CREATE TABLE IF NOT EXISTS ticket_events (
  id SERIAL PRIMARY KEY,
  ticket_id INTEGER NOT NULL,
  actor_user_id INTEGER NULL,
  event VARCHAR(32) NOT NULL DEFAULT 'status_change',
  from_status VARCHAR(24) NULL,
  to_status VARCHAR(24) NULL,
  message TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  CONSTRAINT fk_ticket_events_ticket FOREIGN KEY(ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
  CONSTRAINT fk_ticket_events_actor FOREIGN KEY(actor_user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_ticket_events_ticket ON ticket_events(ticket_id);
CREATE INDEX IF NOT EXISTS idx_ticket_events_actor ON ticket_events(actor_user_id);
CREATE INDEX IF NOT EXISTS idx_ticket_events_created_at ON ticket_events(created_at);
