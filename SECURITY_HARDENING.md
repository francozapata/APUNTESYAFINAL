# Blindaje (seguridad) — ApuntesYa

Este proyecto incluye un **baseline** de seguridad pensado para producción en Render.

## 1) Límites anti-abuso (rate limiting)

Hay un rate-limit simple por IP (por worker) en `app.py`:

- Descargas:
  - `/download/<id>`: 60/min
  - `/combos/<id>/download`: 30/min
- POST sensibles:
  - `/login`: 20/5min
  - `/register`: 10/5min
  - `/upload`: 20/10min
  - `/profile/upload_image`: 20/10min
  - `/profile/change_password`: 10/10min

> Nota: al usar múltiples workers, este rate-limit es **por worker**. Aun así reduce scraping / abuso.

## 2) CSRF

Se usa `Flask-WTF CSRFProtect` a nivel aplicación.

Si agregás rutas JSON/WEBHOOK, recordá eximirlas explícitamente con `@csrf.exempt`.

## 3) Headers de seguridad

Se inyectan en `@app.after_request`:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Cross-Origin-Resource-Policy: same-origin`
- `Content-Security-Policy` (conservadora; permite inline por compatibilidad)
- `Strict-Transport-Security` (solo si HTTPS)

## 4) Subidas (PDF)

### Tamaño máximo

Se controla con `MAX_CONTENT_LENGTH`:

- ENV: `MAX_UPLOAD_MB` (por default 100)

### Validación PDF (defense-in-depth)

Al subir, se valida:

- Header `%PDF`
- Parseo con PyMuPDF
- Bloqueo de JavaScript embebido (básico)
- Máximo de páginas

ENV opcional:

- `MAX_PDF_PAGES` (por default 1200)

## 5) Descargas

Las descargas se sirven **desde el backend** (no se exponen URLs públicas del storage).
