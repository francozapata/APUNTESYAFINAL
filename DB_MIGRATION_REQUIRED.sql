-- ApuntesYa - Migraciones necesarias para compatibilidad con combos + descargas
-- Ejecutar en tu Postgres (Render) con el mismo usuario/DB que usa la app.

BEGIN;

-- 1) download_logs: columna combo_id (para registrar descargas que vienen de combos)
ALTER TABLE IF EXISTS download_logs
  ADD COLUMN IF NOT EXISTS combo_id INTEGER NULL;

-- 2) FK + índice (opcional pero recomendado)
DO $$
BEGIN
  -- Crear FK solo si existe la tabla combos y la constraint aún no existe
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'combos') THEN
    IF NOT EXISTS (
      SELECT 1
      FROM information_schema.table_constraints
      WHERE constraint_name = 'fk_download_logs_combo'
        AND table_name = 'download_logs'
    ) THEN
      ALTER TABLE download_logs
        ADD CONSTRAINT fk_download_logs_combo
        FOREIGN KEY (combo_id) REFERENCES combos(id)
        ON DELETE SET NULL;
    END IF;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_download_logs_combo_id ON download_logs (combo_id);

COMMIT;

-- Nota:
-- Si tu DB estaba "atrás" respecto al código, esta migración soluciona el error:
-- "column combo_id of relation download_logs does not exist".
