# 🚀 Deploy de ApuntesYa (Flask) en Render — Pack listo

Este paquete contiene todo lo necesario para publicar tu app Flask (como **ApuntesYa**) en **Render.com**.

## Contenido
- `Procfile` — indica a Render cómo arrancar el servidor.
- `wsgi.py` — punto de entrada WSGI (si tu host necesita un archivo dedicado).
- `gunicorn.conf.py` — configuración de Gunicorn.
- `render.yaml` — infraestructura como código (Render Blueprint).
- `.env.example` — ejemplo de variables de entorno.
- `README_DEPLOY_RENDER.md` — este instructivo.

---

## ✅ Requisitos previos
- Tu proyecto debe tener:
  - `requirements.txt` (con Flask, gunicorn y tus dependencias)
  - Tu módulo **exportando** `app` (por ejemplo: `from apuntesya2.app import app`)
- Tu repositorio subido a **GitHub**.

> **Nota importante:** Si tu proyecto usa SQLite, en Render Free el sistema de archivos es **ephemeral** (se resetea en cada deploy). Para producción, usá **Render Postgres** y seteá `DATABASE_URL`.

---

## 🧩 Cómo usar este pack
1) **Copiá todos los archivos** de este pack en la **raíz** de tu proyecto (donde está `requirements.txt`).  
   Si ya tenías un `Procfile` o `gunicorn.conf.py`, compará y ajustá.

2) Verificá que tu aplicación exporta `app` desde el módulo correcto.  
   Por defecto acá usamos:
   ```bash
   gunicorn -c gunicorn.conf.py apuntesya2.app:app
   ```
   Si tu app está en otra ruta, ajustá el comando en `Procfile` y `render.yaml`.

3) **Asegurá `requirements.txt`** conteniendo al menos:
   ```
   Flask>=2.2
   gunicorn>=21.0
   python-dotenv>=1.0
   SQLAlchemy>=2.0
   psycopg2-binary>=2.9
   requests>=2.31
   ```
   (Agregá el resto de tus dependencias reales.)

4) **Subí o actualizá** tu proyecto en **GitHub** (commit + push).

5) **Creá el servicio en Render**  
   - Entrá a https://render.com  
   - **New → Web Service → Connect con GitHub** y elegí tu repo  
   - Build command: `pip install -r requirements.txt`  
   - Start command: `gunicorn -c gunicorn.conf.py apuntesya2.app:app`
   - Agregá las **Environment Variables** (copiá desde `.env.example`)  
     - Recomendado: agregá una base de datos **Postgres** en Render y mapeá `DATABASE_URL`.

6) **Deploy** → Render te dará una **URL pública** cuando termine.

---

## 🔥 Firebase (Google Login) — variables para PROD y STAGING

En **Render**, cada servicio (tu web **PROD** y tu web **STAGING**) tiene su set de Environment Variables.

### A) Firebase Admin SDK (backend) — recomendado
Elegí **una** de estas opciones:

**Opción A1 (recomendada en Render):**
- `FIREBASE_SERVICE_ACCOUNT_B64` = base64 del JSON de service account
- `FIREBASE_PROJECT_ID` = tu projectId (opcional si viene dentro del JSON)

**Opción A2:**
- `FIREBASE_SERVICE_ACCOUNT_JSON` = ruta al JSON (sólo si lo montás en disco; en Render suele ser incómodo)

> Tip: para crear el base64 en tu PC:
> - Linux/macOS: `base64 -w 0 serviceAccount.json`
> - Windows PowerShell: `[Convert]::ToBase64String([IO.File]::ReadAllBytes('serviceAccount.json'))`

### B) Firebase Web SDK (frontend) — necesario para el botón “Continuar con Google”
Podés configurarlo de 2 maneras (la app soporta ambas):

**Opción B1 (más cómoda): 1 sola variable JSON**
- `FIREBASE_WEB_CONFIG_JSON` = JSON completo del Web SDK (el que te da Firebase en “Configuración del proyecto → Tus apps → SDK web”).

**Opción B2: variables individuales**
- `FIREBASE_WEB_API_KEY`
- `FIREBASE_WEB_AUTH_DOMAIN`
- `FIREBASE_WEB_PROJECT_ID`
- `FIREBASE_WEB_STORAGE_BUCKET` (opcional)
- `FIREBASE_WEB_MESSAGING_SENDER_ID` (opcional)
- `FIREBASE_WEB_APP_ID`

> Si falta alguna clave obligatoria (apiKey/authDomain/projectId/appId), la pantalla de login muestra un aviso y deshabilita el botón.

### Importante (Firebase Console)
Para que funcione en ambos entornos:
- En Firebase → Authentication → **Settings → Authorized domains** agregá:
  - tu dominio de Render PROD (ej: `tu-app.onrender.com`)
  - tu dominio de Render STAGING
  - (y si usás dominio propio, también)

---

## 🧪 Probar localmente (opcional)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
set FLASK_ENV=production
set SECRET_KEY=change-me
set DATABASE_URL=sqlite:///app.db
# en Linux/Mac usar export en lugar de set

# ejecutar con gunicorn en local, cambiando el módulo si hace falta
gunicorn -c gunicorn.conf.py apuntesya2.app:app
```

---

## 🔧 Problemas frecuentes
- **ImportError / ModuleNotFoundError**: verificá que la ruta `apuntesya2.app:app` coincida con tu proyecto real.
- **Jinja2 UndefinedError (variables de comisión)**: asegurate de setear `PLATFORM_COMMISSION` y `MP_COMMISSION_RATE` en variables de entorno o de tener defaults en tu código.
- **DB no persiste**: usá Postgres (no SQLite) en Render Free para persistencia.
- **Tiempo de arranque excedido**: subí el `timeout` o reducí inicializaciones pesadas.

---

## 📌 Extra: Deploy con Blueprint (render.yaml)
Si preferís, podés crear el servicio con **Render Blueprints**:
1. `render.yaml` en la raíz del repo (este archivo).
2. En Render → **Blueprints** → **New Blueprint** → seleccioná el repo.
3. Render creará los servicios definidos automáticamente.

---

¡Listo! Con este pack deberías poder publicar tu app Flask en Render sin dolor 😄
