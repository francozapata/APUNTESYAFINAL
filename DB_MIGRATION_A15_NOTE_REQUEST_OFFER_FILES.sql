-- A15: archivos adjuntos para propuestas de pedidos
ALTER TABLE note_request_offers
    ADD COLUMN IF NOT EXISTS file_path TEXT,
    ADD COLUMN IF NOT EXISTS file_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS file_mime VARCHAR(120),
    ADD COLUMN IF NOT EXISTS file_size INTEGER;
