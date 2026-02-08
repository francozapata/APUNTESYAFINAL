# ApuntesYa — Administradores (ZIP 1)

## 1) Roles

- `user`: usuario normal
- `admin`: accede al panel `/admin/hub` (operación diaria)
- `superadmin`: puede además gestionar administradores, activar mantenimiento y ver auditoría

> Nota: se mantiene `users.is_admin` por compatibilidad con templates y rutas viejas.

## 2) Convertir la “cuenta madre” en superadmin

### Opción recomendada (sin tocar la BD): variable de entorno

En Render (o donde deployes), agregá:

```
SUPERADMIN_EMAILS=tu_email@gmail.com,otro_superadmin@gmail.com
```

Luego, iniciá sesión con Google con ese email. En el primer request la app lo promueve automáticamente a `superadmin`.

### Opción manual (Supabase SQL)

Si preferís hacerlo directo en la BD:

```sql
UPDATE users
SET role='superadmin', is_admin=true
WHERE lower(email)=lower('tu_email@gmail.com');
```

## 3) Supabase: qué cambia

La app aplica migraciones livianas en runtime (safe) al levantar (SQLite/Postgres):

- agrega `users.role` si falta
- crea `site_settings` si falta (para `maintenance_mode`)

Si querés aplicarlo manualmente igual, usá:

```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'user';

CREATE TABLE IF NOT EXISTS site_settings (
  key VARCHAR(60) PRIMARY KEY,
  value TEXT
);

INSERT INTO site_settings(key,value)
VALUES ('maintenance_mode','0')
ON CONFLICT (key) DO NOTHING;
```

## 4) Nuevas pantallas (solo superadmin)

- `/admin/admins` — gestionar administradores por email
- `/admin/maintenance` — activar/desactivar mantenimiento
- `/admin/audit` — ver auditoría (últimos 100 eventos)

## 5) Auditoría

Se usa la tabla existente `audit_events`.

Acciones auditadas:
- promoción automática por `SUPERADMIN_EMAILS`
- cambios de rol
- toggle de mantenimiento
