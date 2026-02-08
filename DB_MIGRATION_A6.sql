-- ============================================================
-- APUNTESYA - A6 (Analytics / eventos)
-- Pageviews + embudo (vista -> intento compra -> checkout -> aprobado)
-- ============================================================

-- Eventos livianos (NO guarda datos sensibles)
CREATE TABLE IF NOT EXISTS analytics_events (
  id BIGSERIAL PRIMARY KEY,
  event VARCHAR(64) NOT NULL,
  user_id BIGINT NULL,
  path VARCHAR(255) NULL,

  note_id BIGINT NULL,
  combo_id BIGINT NULL,

  ip VARCHAR(64) NULL,
  user_agent VARCHAR(255) NULL,
  referrer VARCHAR(255) NULL,

  meta JSONB NULL,
  created_at TIMESTAMP DEFAULT NOW() NOT NULL
);

-- Índices recomendados
CREATE INDEX IF NOT EXISTS idx_analytics_events_created_at ON analytics_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_event ON analytics_events(event);
CREATE INDEX IF NOT EXISTS idx_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_note_id ON analytics_events(note_id);
CREATE INDEX IF NOT EXISTS idx_analytics_events_combo_id ON analytics_events(combo_id);
