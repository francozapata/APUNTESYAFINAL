
# Panel de Administración — ApuntesYa

## 1) Upgrade de base de datos (una sola vez)
```bash
# Activá tu venv y variables .env si corresponde
python scripts/upgrade_admin_schema.py
```

## 2) Crear/promocionar usuario admin
```bash
# Promover un usuario existente a admin
python scripts/create_admin.py tu@email.com

# O crear usuario + admin (requiere password)
python scripts/create_admin.py tu@email.com TuPassword!
```

## 3) Ingresar al panel
- Iniciá sesión con el usuario admin.
- Navegá a: `/admin/` para ver el dashboard, `/admin/users` para listar usuarios.

## 4) Acciones
- **Desactivar usuario**: botón en la tabla (soft disable).
- **Eliminar apunte (soft)**: vía POST a `/admin/notes/<id>/soft-delete` (se puede agregar botón desde la UI de cada apunte).
- Todas las acciones registran un **audit log** en `admin_actions`.

> Requisitos: tener sesión iniciada y `is_admin=1`. Recomendado activar 2FA a nivel de correo y restringir acceso al panel por IP si lo subís a producción.
