# apuntesya2/app.py

import os
import secrets
import math
import json
import base64
import re
from datetime import datetime, timedelta, timedelta
from urllib.parse import urlencode
from functools import wraps
from google.cloud import storage
from google.oauth2 import service_account

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, abort, jsonify, session
)
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, abort, send_file

from flask_login import (
    LoginManager, login_user, logout_user, current_user, login_required
)
from sqlalchemy import create_engine, select, or_, and_, func, text, desc
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, auth as fb_auth

# modelos
from apuntesya2.models import (
    Base, User, Note, Purchase, University, Faculty, Career, WebhookEvent, Review, DownloadLog, Notification, Combo, ComboNote, ComboPurchase
)

# helpers MP
from apuntesya2 import mp

# Pricing (single source of truth)
from apuntesya2.pricing import (
    published_from_net_cents,
    cents_to_amount,
    amount_to_cents,
    breakdown_from_net,
    breakdown_from_published,
    money_1_decimal,
)

load_dotenv()

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)

# --- SECRET KEY robusto: usa ENV si existe; si no, persiste uno en disco ---
def _load_or_create_secret_key(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    return key

# Usa ENV si está; si no, archivo compartido (Render con múltiples workers)
SECRET_KEY_ENV = os.getenv("SECRET_KEY")
SECRET_KEY_FILE = os.path.join("/tmp", "data", "flask_secret_key")
app.config["SECRET_KEY"] = SECRET_KEY_ENV or _load_or_create_secret_key(SECRET_KEY_FILE)

# Cookies seguras
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["ENV"] = os.getenv("FLASK_ENV", "production")

# -----------------------------------------------------------------------------
# Firebase Admin (única inicialización)
# -----------------------------------------------------------------------------
def _init_firebase_admin():
    if firebase_admin._apps:
        return

    fb_opts = {}
    project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        fb_opts["projectId"] = project_id.strip()

    cred_obj = None

    # 1) Credencial como base64 en ENV
    b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64", "").strip()
    if b64:
        try:
            raw = base64.b64decode(b64).decode("utf-8")
            data = json.loads(raw)
            cred_obj = credentials.Certificate(data)
        except Exception as e:
            print("[Firebase] WARNING: no pude decodificar FIREBASE_SERVICE_ACCOUNT_B64:", e)

    # 2) Ruta a JSON en disco
    if not cred_obj:
        cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if cred_path and os.path.exists(cred_path):
            try:
                cred_obj = credentials.Certificate(cred_path)
            except Exception as e:
                print("[Firebase] WARNING: credencial en ruta inválida:", e)

    # 3) Sin credencial: igual inicializamos con projectId
    try:
        if cred_obj:
            firebase_admin.initialize_app(cred_obj, fb_opts or None)
        else:
            firebase_admin.initialize_app(options=fb_opts or None)
        print("[Firebase] Admin SDK inicializado.", "projectId=", fb_opts.get("projectId"))
    except Exception as e:
        print("[Firebase] WARNING al inicializar:", e)

_init_firebase_admin()

def verify_firebase_id_token(id_token: str):
    """Verifica el ID token de Firebase y devuelve datos básicos del usuario."""
    decoded = fb_auth.verify_id_token(id_token)
    email   = decoded.get("email")
    name    = decoded.get("name") or decoded.get("firebase", {}).get("sign_in_provider", "Google user")
    picture = decoded.get("picture")
    uid     = decoded.get("uid")
    return {"uid": uid, "email": email, "name": name, "picture": picture}

# --- MP immediate fee estimate available in templates ---
try:
    MP_FEE_IMMEDIATE_TOTAL_PCT = float(app.config.get("MP_FEE_IMMEDIATE_TOTAL_PCT", 7.61))
except Exception:
    MP_FEE_IMMEDIATE_TOTAL_PCT = 8.0

@app.context_processor
def fees_ctx():
    def mp_fee_estimate(amount, pct=MP_FEE_IMMEDIATE_TOTAL_PCT):
        try:
            return round(float(amount) * (float(pct) / 100.0), 2)
        except Exception:
            return 0.0
    return dict(MP_FEE_IMMEDIATE_TOTAL_PCT=MP_FEE_IMMEDIATE_TOTAL_PCT, mp_fee_estimate=mp_fee_estimate)

# -----------------------------------------------------------------------------
# Paths (Render usa /tmp; local usa ./data)
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

if os.getenv("RENDER", "").strip() == "1":
    BASE_DATA = "/tmp/data"
else:
    BASE_DATA = os.path.join(PROJECT_ROOT, "data")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DATA, "uploads"))
os.makedirs(BASE_DATA, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 25MB

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
from sqlalchemy.pool import NullPool

DEFAULT_DB = f"sqlite:///{os.path.join(BASE_DATA, 'apuntesya.db')}"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB)

# Si la URL viene de Postgres (Supabase), usamos el driver psycopg (v3)
if DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg://", 1)

engine_kwargs = {"pool_pre_ping": True, "future": True}

# Si es SQLite (local), dejamos como estaba
if DB_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

# Si es Postgres (Supabase) → usar NullPool
else:
    engine_kwargs["poolclass"] = NullPool

engine = create_engine(DB_URL, **engine_kwargs)



# -----------------------------------------------------------------------------
# Google Cloud Storage
# -----------------------------------------------------------------------------
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

gcs_client = None
gcs_bucket = None

if GCS_BUCKET_NAME and GCS_CREDENTIALS_JSON:
    try:
        creds_info = json.loads(GCS_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        gcs_client = storage.Client(credentials=credentials)
        gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        print(f"[GCS] Bucket configurado: {GCS_BUCKET_NAME}")
    except Exception as e:
        print("[GCS] ERROR al inicializar:", e)
else:
    print("[GCS] GCS no configurado (faltan variables de entorno)")

# -----------------------------------------------------------------------------
# Modelos e inicio de sesión
# -----------------------------------------------------------------------------
Base.metadata.create_all(engine)


def _ensure_schema(engine):
    """Lightweight runtime migrations for SQLite/Postgres.

    This project historically relied on create_all() only. For V1 marketplace
    features we add a few columns and tables, and we keep backward compatibility
    by applying safe ALTERs when missing.
    """
    insp = inspect(engine)
    with engine.begin() as conn:
        # users: structured contacts + visibility flags
        if insp.has_table('users'):
            cols = {c['name'] for c in insp.get_columns('users')}
            add_cols = []
            if 'contact_email' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_email VARCHAR(255)")
            if 'contact_whatsapp' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_whatsapp VARCHAR(64)")
            if 'contact_phone' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_phone VARCHAR(64)")
            if 'contact_website' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_website VARCHAR(255)")
            if 'contact_instagram' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_instagram VARCHAR(80)")
            if 'contact_visible_public' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_visible_public BOOLEAN DEFAULT 1")
            if 'contact_visible_buyers' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN contact_visible_buyers BOOLEAN DEFAULT 1")
            for stmt in add_cols:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass

        # notes: moderation + preview metadata
        if insp.has_table('notes'):
            cols = {c['name'] for c in insp.get_columns('notes')}
            add_cols = []
            if 'moderation_status' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN moderation_status VARCHAR(32) DEFAULT 'pending_ai'")
            if 'moderation_reason' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN moderation_reason TEXT")
            # AI moderation payload
            if 'ai_decision' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_decision VARCHAR(16)")
            if 'ai_confidence' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_confidence INTEGER")
            if 'ai_score_quality' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_score_quality INTEGER")
            if 'ai_score_copyright' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_score_copyright INTEGER")
            if 'ai_score_mismatch' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_score_mismatch INTEGER")
            if 'ai_model' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_model VARCHAR(80)")
            if 'ai_summary' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_summary TEXT")
            if 'ai_raw' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN ai_raw JSON")
            if 'manual_review_due_at' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN manual_review_due_at TIMESTAMP")
            if 'moderated_by_admin_id' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN moderated_by_admin_id INTEGER")
            if 'moderated_at' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN moderated_at TIMESTAMP")
            if 'preview_pages' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN preview_pages JSON")
            if 'preview_images' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN preview_images JSON")
            for stmt in add_cols:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass

        # New tables: reviews unique/constraint already in model; download_logs
        # create_all already handled table creation, but some DBs might have been created earlier.


try:
    _ensure_schema(engine)
except Exception as e:
    print('[schema] WARNING:', e)

Session = scoped_session(sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))

login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    with Session() as s:
        return s.get(User, int(user_id))

# Decorador simple para admin (única definición)
def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login"))
        if not getattr(current_user, "is_admin", False) or getattr(current_user, "is_blocked", False):
            abort(403)
        return fn(*args, **kwargs)
    return wrapper

# -----------------------------------------------------------------------------
# Config MP / Comisiones / Contacto
# -----------------------------------------------------------------------------
app.config["MP_PUBLIC_KEY"] = os.getenv("MP_PUBLIC_KEY", "")
app.config["MP_ACCESS_TOKEN"] = os.getenv("MP_ACCESS_TOKEN", "")
app.config["MP_WEBHOOK_SECRET"] = os.getenv("MP_WEBHOOK_SECRET", "")
app.config["BASE_URL"] = os.getenv("BASE_URL", "")

"""Pricing rules (uniform)

- Seller inputs NET (what they want to receive).
- Published buyer price: P = ceil_up_1_decimal(N / 0.82)

Fee split inside P:
  - 10% ApuntesYa
  - 8% Mercado Pago
  - Total: 18%

Rounding:
  - Published prices are always rounded UP to 1 decimal.
"""

# Fee rates used for accounting/labels
app.config["MP_COMMISSION_RATE"] = float(os.getenv("MP_COMMISSION_RATE", "0.08"))
app.config["APY_COMMISSION_RATE"] = float(os.getenv("APY_COMMISSION_RATE", "0.10"))
app.config["TOTAL_FEE_RATE"] = float(os.getenv("TOTAL_FEE_RATE", "0.18"))

# Porcentaje que mostramos en el footer (total de comisiones)
app.config["PLATFORM_FEE_PERCENT"] = 18.0

app.config["IIBB_ENABLED"] = os.getenv("IIBB_ENABLED", "false").lower() in ("1", "true", "yes")
app.config["IIBB_RATE"] = float(os.getenv("IIBB_RATE", "0.0"))

MP_COMMISSION_RATE = app.config["MP_COMMISSION_RATE"]
APY_COMMISSION_RATE = app.config["APY_COMMISSION_RATE"]
TOTAL_FEE_RATE = app.config["TOTAL_FEE_RATE"]
IIBB_ENABLED = app.config["IIBB_ENABLED"]
IIBB_RATE = app.config["IIBB_RATE"]
# Legacy multiplier kept only for backwards compatibility in old code paths.
GROSS_MULTIPLIER = 1.0 / (1.0 - float(TOTAL_FEE_RATE))

app.config["MP_ACCESS_TOKEN_PLATFORM"] = os.getenv("MP_ACCESS_TOKEN", "")
app.config["MP_OAUTH_REDIRECT_URL"] = os.getenv("MP_OAUTH_REDIRECT_URL")

app.config.setdefault("SECURITY_PASSWORD_SALT", os.getenv("SECURITY_PASSWORD_SALT", "pw-reset"))
app.config.setdefault("PASSWORD_RESET_EXPIRATION", int(os.getenv("PASSWORD_RESET_EXPIRATION", "3600")))
app.config.setdefault("ENABLE_SMTP", os.getenv("ENABLE_SMTP", "false"))

app.config["CONTACT_EMAILS"] = os.getenv("CONTACT_EMAILS", "soporte.apuntesya@gmail.com")
app.config["CONTACT_WHATSAPP"] = os.getenv("CONTACT_WHATSAPP", "+543510000000")
app.config["SUGGESTIONS_URL"] = os.getenv("SUGGESTIONS_URL",
    "https://docs.google.com/forms/d/e/1FAIpQLScDEukn0sLtjOoWgmvTNaF_qG0iDHue9EOqCYxz_z6bGxzErg/viewform?usp=header"
)

@app.context_processor
def inject_contacts():
    emails = [e.strip() for e in str(app.config.get("CONTACT_EMAILS","")).split(",") if e.strip()]
    return dict(CONTACT_EMAILS=emails,
                CONTACT_WHATSAPP=app.config.get("CONTACT_WHATSAPP"),
                SUGGESTIONS_URL=app.config.get("SUGGESTIONS_URL"))

@app.context_processor
def pricing_ctx():
    """Template helpers for uniform pricing.

    Seller provides NET in cents (what they want to receive).
    Buyer published price is computed with pricing.published_from_net_cents()
    and is always rounded UP to 1 decimal.
    """

    def published_price(net_cents: int | float | None) -> float:
        # returns ARS with 1 decimal
        pub_cents = published_from_net_cents(int(net_cents or 0))
        return float(money_1_decimal(cents_to_amount(pub_cents)))

    def published_price_cents(net_cents: int | float | None) -> int:
        return int(published_from_net_cents(int(net_cents or 0)))

    def fee_breakdown_from_net(net_cents: int | float | None):
        # returns FeeBreakdown with 1-decimal values for UI
        net = cents_to_amount(int(net_cents or 0))
        return breakdown_from_net(net)

    def fee_breakdown_from_published(published_cents: int | float | None):
        pub = cents_to_amount(int(published_cents or 0))
        return breakdown_from_published(pub)

    return dict(
        published_price=published_price,
        published_price_cents=published_price_cents,
        fee_breakdown_from_net=fee_breakdown_from_net,
        fee_breakdown_from_published=fee_breakdown_from_published,
    )


def get_valid_seller_token(seller: User) -> str | None:
    return seller.mp_access_token if (seller and seller.mp_access_token) else None



@app.context_processor
def inject_nav_notifications():
    """Expose latest notifications + unread count for navbar dropdown."""
    if not getattr(current_user, "is_authenticated", False):
        return dict(nav_notifications=[], nav_notif_unread=0)

    try:
        with Session() as s:
            nav_notifications = s.execute(
                select(Notification)
                .where(Notification.user_id == current_user.id)
                .order_by(Notification.created_at.desc())
                .limit(8)
            ).scalars().all()

            nav_notif_unread = s.execute(
                select(func.count(Notification.id))
                .where(Notification.user_id == current_user.id, Notification.is_read == False)
            ).scalar_one()
    except Exception:
        nav_notifications = []
        nav_notif_unread = 0

    return dict(nav_notifications=nav_notifications, nav_notif_unread=int(nav_notif_unread or 0))

# -----------------------------------------------------------------------------
# Admin blueprint (si existe) + auth_reset (legacy)
# -----------------------------------------------------------------------------
try:
    from .admin.routes import admin_bp
except Exception:
    try:
        from admin.routes import admin_bp
    except Exception:
        admin_bp = None

try:
    from apuntesya2.auth_reset.routes import bp as auth_reset_bp
except Exception:
    auth_reset_bp = None

if admin_bp:
    app.register_blueprint(admin_bp)
if auth_reset_bp:
    app.register_blueprint(auth_reset_bp)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def allowed_pdf(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"

def ensure_dirs():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def gcs_upload_file(file_storage, blob_name: str) -> str:
    """
    Sube un archivo (FileStorage de Flask) a GCS.
    Devuelve el nombre del blob guardado.
    """
    if not gcs_bucket:
        raise RuntimeError("GCS no está configurado")

    blob = gcs_bucket.blob(blob_name)
    file_storage.stream.seek(0)
    blob.upload_from_file(
        file_storage.stream,
        content_type=file_storage.content_type or "application/pdf"
    )
    return blob_name


def gcs_generate_signed_url(blob_name: str, seconds: int = 600) -> str:
    """
    Genera un link firmado para descargar el archivo desde GCS.
    """
    if not gcs_bucket:
        raise RuntimeError("GCS no está configurado")

    blob = gcs_bucket.blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=seconds),
        method="GET",
    )
    return url


def gcs_delete_blob(blob_name: str) -> bool:
    """Elimina un blob de GCS (best-effort). Devuelve True si intenta borrar."""
    if not gcs_bucket:
        return False
    try:
        gcs_bucket.blob(blob_name).delete()
        return True
    except Exception:
        return False


def gcs_download_to_temp(blob_name: str) -> str:
    """Download a blob to a temporary file and return the local path."""
    if not gcs_bucket:
        raise RuntimeError("GCS no está configurado")
    tmp_dir = os.path.join(BASE_DATA, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{secrets.token_hex(8)}.bin")
    blob = gcs_bucket.blob(blob_name)
    blob.download_to_filename(tmp_path)
    return tmp_path


def gcs_download_bytes(blob_name: str) -> bytes:
    if not gcs_bucket:
        raise RuntimeError("GCS no está configurado")
    blob = gcs_bucket.blob(blob_name)
    return blob.download_as_bytes()


def gcs_upload_bytes(data: bytes, blob_name: str, content_type: str = "application/octet-stream") -> str:
    if not gcs_bucket:
        raise RuntimeError("GCS no está configurado")
    blob = gcs_bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return blob_name


def _watermark_image(img, text: str = "APUNTESYA"):
    """Apply a repeated diagonal watermark over a PIL image."""
    from PIL import Image, ImageDraw, ImageFont

    if img.mode != "RGBA":
        base = img.convert("RGBA")
    else:
        base = img.copy()

    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Font fallback
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(16, int(min(base.size) * 0.05)))
    except Exception:
        font = ImageFont.load_default()

    w, h = base.size
    step = max(140, int(min(w, h) * 0.25))
    angle = -30

    # Create rotated watermark tile
    tile = Image.new("RGBA", (step, step), (255, 255, 255, 0))
    td = ImageDraw.Draw(tile)
    td.text((10, step // 2 - 10), text, font=font, fill=(0, 0, 0, 60))
    tile = tile.rotate(angle, expand=1)

    # Paste across
    for y in range(-step, h + step, step):
        for x in range(-step, w + step, step):
            overlay.alpha_composite(tile, (x, y))

    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def generate_note_preview(note: Note, max_pages: int = 4) -> tuple[list[int], list[str]]:
    """Generate preview images for a note PDF (with watermark) and store paths.

    Returns (pages, image_paths).
    """
    import random
    import fitz  # PyMuPDF
    from PIL import Image
    from io import BytesIO

    # Get local path to PDF
    tmp_pdf = None
    local_pdf = None
    try:
        if gcs_bucket and note_file_path and "/" in note_file_path:
            tmp_pdf = gcs_download_to_temp(note.file_path)
            local_pdf = tmp_pdf
        else:
            local_pdf = os.path.join(app.config["UPLOAD_FOLDER"], note.file_path)

        doc = fitz.open(local_pdf)
        total = doc.page_count
        if total <= 0:
            return ([], [])

        pages = {0}
        pages.add(max(0, total // 2))
        # add a couple pseudo-random pages (excluding first)
        if total > 2:
            candidates = list(range(1, total))
            random.shuffle(candidates)
            for p in candidates:
                pages.add(p)
                if len(pages) >= max_pages:
                    break
        pages = sorted(pages)[:max_pages]

        image_paths: list[str] = []
        for idx, pno in enumerate(pages, start=1):
            page = doc.load_page(pno)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            img = Image.open(BytesIO(pix.tobytes("png")))
            img = _watermark_image(img, text="APUNTESYA")

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            data = buf.getvalue()

            if gcs_bucket:
                blob_name = f"previews/{note.id}/{idx}.jpg"
                gcs_upload_bytes(data, blob_name, content_type="image/jpeg")
                image_paths.append(blob_name)
            else:
                prev_dir = os.path.join(app.config["UPLOAD_FOLDER"], "previews", str(note.id))
                os.makedirs(prev_dir, exist_ok=True)
                out_path = os.path.join(prev_dir, f"{idx}.jpg")
                with open(out_path, "wb") as f:
                    f.write(data)
                # store relative path
                image_paths.append(f"previews/{note.id}/{idx}.jpg")

        return (pages, image_paths)
    finally:
        try:
            if tmp_pdf and os.path.exists(tmp_pdf):
                os.remove(tmp_pdf)
        except Exception:
            pass


# ------------------------------ AI moderation ------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()


def _extract_text_for_moderation(note: Note, max_pages: int = 2, max_chars: int = 9000) -> str:
    """Extract a small text sample from the PDF (for AI moderation).

    Uses PyMuPDF. If extraction fails, returns empty string.
    """
    import fitz
    tmp_pdf = None
    local_pdf = None
    try:
        if gcs_bucket and note_file_path and "/" in note_file_path:
            tmp_pdf = gcs_download_to_temp(note.file_path)
            local_pdf = tmp_pdf
        else:
            local_pdf = os.path.join(app.config["UPLOAD_FOLDER"], note.file_path)

        doc = fitz.open(local_pdf)
        parts = []
        for pno in range(min(max_pages, doc.page_count)):
            try:
                parts.append(doc.load_page(pno).get_text("text"))
            except Exception:
                continue
        text_sample = "\n\n".join(parts).strip()
        if len(text_sample) > max_chars:
            text_sample = text_sample[:max_chars]
        return text_sample
    except Exception:
        return ""
    finally:
        try:
            if tmp_pdf and os.path.exists(tmp_pdf):
                os.remove(tmp_pdf)
        except Exception:
            pass


def _gemini_moderate_note(text_sample: str, meta: dict) -> dict:
    """Call Gemini (best-effort) to classify a note.

    Returns a dict with keys:
      decision: approve|review|deny
      confidence: 0..1 float
      quality_score / copyright_risk / mismatch_risk: 0..1 floats (optional)
      summary: short string
      reasons: list[str]
    """
    if not GEMINI_API_KEY:
        return {
            "decision": "review",
            "confidence": 0.5,
            "summary": "Gemini no configurado: enviado a revisión manual (hasta 12hs).",
            "reasons": ["missing_gemini_api_key"],
            "quality_score": 0.5,
            "copyright_risk": 0.0,
            "mismatch_risk": 0.0,
        }

    try:
        from google import genai
        from pydantic import BaseModel, Field
        from typing import List, Literal
        import json as _json

        class _ModerationResult(BaseModel):
            decision: Literal["approve", "review", "deny"]
            confidence: int = Field(ge=0, le=100)
            summary: str
            reasons: List[str] = []
            quality_score: int = Field(ge=0, le=100, default=50)
            copyright_risk: int = Field(ge=0, le=100, default=0)
            mismatch_risk: int = Field(ge=0, le=100, default=0)

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
Sos moderador de un marketplace de apuntes (Argentina). Evaluá el contenido y devolvé una decisión:
- approve: apunte educativo válido
- review: dudoso / posible copyright / requiere revisión humana
- deny: claramente spam/no educativo/ilegal

Metadatos del apunte (no confiar al 100%, solo contexto):
{_json.dumps(meta, ensure_ascii=False)}

Texto extraído (muestra):
{text_sample}

Devolvé SOLO JSON con los campos: decision, confidence (0-100), summary, reasons, quality_score, copyright_risk, mismatch_risk.
"""

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _ModerationResult,
            },
        )
        parsed = resp.parsed
        # Convert to legacy dict format
        out = {
            "decision": parsed.decision,
            "confidence": parsed.confidence / 100.0,
            "summary": parsed.summary,
            "reasons": parsed.reasons,
            "quality_score": parsed.quality_score / 100.0,
            "copyright_risk": parsed.copyright_risk / 100.0,
            "mismatch_risk": parsed.mismatch_risk / 100.0,
        }
        return out

    except Exception as e:
        return {
            "decision": "review",
            "confidence": 0.5,
            "summary": f"Error Gemini: {type(e).__name__}. Enviado a revisión manual.",
            "reasons": ["gemini_error"],
            "quality_score": 0.5,
            "copyright_risk": 0.0,
            "mismatch_risk": 0.0,
        }

def _decision_to_status(ai: dict) -> tuple[str, str]:
    """Map AI JSON to (moderation_status, reason)."""
    decision = (ai.get("decision") or "review").lower().strip()
    conf = float(ai.get("confidence") or 0.0)
    # extra safety: if decision isn't recognized -> manual
    if decision not in ("approve", "review", "deny"):
        decision = "review"
    if decision == "approve" and conf >= 0.70:
        return ("approved", None)
    if decision == "deny" and conf >= 0.70:
        return ("rejected", (ai.get("summary") or "Rechazado por moderación automática").strip())
    # default
    return ("pending_manual", (ai.get("summary") or "Revisión manual requerida").strip())


def _notify_users(session_db, user_ids: list[int], title: str, body: str, kind: str = "info"):
    """Create in-app notifications for a list of users."""
    for uid in user_ids:
        try:
            session_db.add(Notification(user_id=uid, kind=kind, title=title, body=body))
        except Exception:
            pass



# ------------------------------ util contacto vendedor ------------------------------
def _build_contact_link(raw: str) -> tuple[str, str]:
    """
    Devuelve (url, etiqueta). Acepta:
      - Tel/WhatsApp -> wa.me/...
      - Email -> mailto:
      - URL -> directa
    """
    v = (raw or "").strip()
    if not v:
        return ("", "")

    if v.lower().startswith(("http://", "https://")):
        return (v, "Abrir enlace")

    if "@" in v and "." in v.split("@")[-1]:
        return (f"mailto:{v}", "Enviar correo")

    digits = re.sub(r"[^\d+]", "", v)
    wa = digits.replace("+", "") if digits.startswith("+") else digits
    if wa:
        text = "Hola, te escribo por tu apunte en ApuntesYa."
        return (f"https://wa.me/{wa}?text={text}", "WhatsApp")

    return (v, "Contacto")

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}, 200

@app.route("/healthz")
def healthz():
    try:
        return {"status":"ok","version": app.config.get("APP_VERSION","unknown")}, 200
    except Exception as e:
        return {"status":"degraded","error": str(e)}, 200

# -----------------------------------------------------------------------------
# PROMOTE ADMIN (habilitado sólo con ENVs)
# -----------------------------------------------------------------------------
@app.route("/_promote_admin_once", methods=["GET"])
def _promote_admin_once():
    if os.getenv("PROMOTE_ADMIN_ENABLED", "0") != "1":
        abort(404)

    secret_env = os.getenv("PROMOTE_ADMIN_SECRET", "")
    secret_arg = request.args.get("secret", "")
    email = (request.args.get("email") or "").strip().lower()

    if not secret_env or secret_arg != secret_env:
        abort(403)
    if not email:
        return "Falta ?email=", 400

    with Session() as s:
        user = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if not user:
            return "Usuario no encontrado", 404
        user.is_admin = True
        s.commit()

    app.logger.warning("Promovido a admin: %s", email)
    return f"OK. {email} ahora es admin."

# -----------------------------------------------------------------------------
# Rutas principales
# -----------------------------------------------------------------------------
from sqlalchemy import select, desc

@app.route("/")
def index():
    with Session() as s:
        # Notes (solo aprobados/activos si existen esos campos)
        q_notes = select(Note).order_by(desc(Note.created_at)).limit(30)

        if hasattr(Note, "is_active"):
            q_notes = q_notes.where(Note.is_active == True)

        if hasattr(Note, "moderation_status"):
            q_notes = q_notes.where(Note.moderation_status == "approved")

        if hasattr(Note, "deleted_at"):
            q_notes = q_notes.where(Note.deleted_at.is_(None))

        notes = s.execute(q_notes).scalars().all()

        # Combos (SIN filtros para que aparezcan sí o sí)
        combos = s.execute(
            select(Combo)
            .where(
                Combo.is_active == True,
                # si tenés moderación en combos:
                Combo.moderation_status == "approved",
                # si existiera deleted_at en Combo:
                # Combo.deleted_at.is_(None),
            )
            .order_by(Combo.created_at.desc())
        ).scalars().all()

        # Rankings (si tenés estos modelos)
                # Rankings (filtrados para no mostrar borrados/inactivos/no aprobados)
        most_downloaded = []
        best_rated = []
        try:
            q_most = (
                select(Note, func.count(DownloadLog.id).label("dl"))
                .join(DownloadLog, DownloadLog.note_id == Note.id)
            )

            if hasattr(Note, "is_active"):
                q_most = q_most.where(Note.is_active == True)
            if hasattr(Note, "moderation_status"):
                q_most = q_most.where(Note.moderation_status == "approved")
            if hasattr(Note, "deleted_at"):
                q_most = q_most.where(Note.deleted_at.is_(None))

            most_downloaded = s.execute(
                q_most.group_by(Note.id)
                .order_by(desc(func.count(DownloadLog.id)))
                .limit(10)
            ).all()
        except Exception:
            pass

        try:
            q_best = (
                select(Note, func.avg(Review.rating).label("avg"))
                .join(Review, Review.note_id == Note.id)
            )

            if hasattr(Note, "is_active"):
                q_best = q_best.where(Note.is_active == True)
            if hasattr(Note, "moderation_status"):
                q_best = q_best.where(Note.moderation_status == "approved")
            if hasattr(Note, "deleted_at"):
                q_best = q_best.where(Note.deleted_at.is_(None))

            best_rated = s.execute(
                q_best.group_by(Note.id)
                .order_by(desc(func.avg(Review.rating)))
                .limit(10)
            ).all()
        except Exception:
            pass


    return render_template(
        "index.html",
        notes=notes,
        combos=combos,  # <- ESTO ES CLAVE
        most_downloaded=most_downloaded,
        best_rated=best_rated,
        include_dynamic_selects=True,
        q="",
        filters={},
        show_tab="quick",
    )

# ------------------------------
# BÚSQUEDA
# ------------------------------

from sqlalchemy import select, desc, or_, distinct

@app.get("/search/quick", endpoint="search_quick")
def search_quick():
    q = (request.args.get("q") or "").strip()

    notes = []
    combos = []

    if q:
        with Session() as s:
            like = f"%{q}%"

            # Notes
            notes_stmt = (
                select(Note)
                .where(
                    Note.is_active == True,
                    Note.moderation_status == "approved",
                    Note.deleted_at.is_(None),
                    or_(Note.title.ilike(like), Note.description.ilike(like)),
                )
                .order_by(desc(Note.created_at))
                .limit(100)
            )
            notes = s.execute(notes_stmt).scalars().all()

            # Combos (match por titulo/descripcion O por notas dentro del combo)
            combos_stmt = (
                select(Combo)
                .distinct()
                .join(ComboNote, ComboNote.combo_id == Combo.id)
                .join(Note, Note.id == ComboNote.note_id)
                .where(
                    Combo.is_active == True,
                    Combo.moderation_status == "approved",
                    Note.is_active == True,
                    Note.moderation_status == "approved",
                    Note.deleted_at.is_(None),
                    or_(
                        Combo.title.ilike(like),
                        Combo.description.ilike(like),
                        Note.title.ilike(like),
                        Note.description.ilike(like),
                    ),
                )
                .order_by(desc(Combo.created_at))
                .limit(100)
            )
            combos = s.execute(combos_stmt).scalars().all()

    return render_template(
        "index.html",
        notes=notes,
        combos=combos,
        buyer_price=_combo_buyer_price_cents,
        show_tab="quick",
        q=q,
        filters={"university": "", "faculty": "", "career": "", "type": ""},
        include_dynamic_selects=True,
    )


@app.get("/search/advanced", endpoint="search_advanced")
def search_advanced():
    q          = (request.args.get("q") or "").strip()
    university = (request.args.get("university") or "").strip()
    faculty    = (request.args.get("faculty") or "").strip()
    career     = (request.args.get("career") or "").strip()
    note_type  = (request.args.get("type") or "").strip()  # "free" | "paid" | ""

    with Session() as s:
        # ---------------- Notes ----------------
        notes_stmt = select(Note).where(
            Note.is_active == True,
            Note.moderation_status == "approved",
            Note.deleted_at.is_(None),
        )

        if q:
            like = f"%{q}%"
            notes_stmt = notes_stmt.where(or_(Note.title.ilike(like), Note.description.ilike(like)))

        if university:
            notes_stmt = notes_stmt.where(Note.university.ilike(f"%{university}%"))
        if faculty:
            notes_stmt = notes_stmt.where(Note.faculty.ilike(f"%{faculty}%"))
        if career:
            notes_stmt = notes_stmt.where(Note.career.ilike(f"%{career}%"))

        if note_type == "free":
            notes_stmt = notes_stmt.where(Note.price_cents == 0)
        elif note_type == "paid":
            notes_stmt = notes_stmt.where(Note.price_cents > 0)

        notes = s.execute(notes_stmt.order_by(desc(Note.created_at)).limit(100)).scalars().all()

        # ---- combos (búsqueda por texto; sin filtros académicos)
        combos = []
        if q:
            combos_stmt = (
                select(Combo)
                .where(Combo.is_active == True)  # noqa: E712
                .where(Combo.moderation_status == 'approved')
                .where(or_(
                    Combo.title.ilike(f"%{q}%"),
                    Combo.description.ilike(f"%{q}%"),
                ))
            )
            combos = s.execute(combos_stmt.order_by(desc(Combo.created_at)).limit(100)).scalars().all()

        # ---------------- Combos ----------------
        combos_stmt = (
            select(Combo)
            .distinct()
            .join(ComboNote, ComboNote.combo_id == Combo.id)
            .join(Note, Note.id == ComboNote.note_id)
            .where(
                Combo.is_active == True,
                Combo.moderation_status == "approved",
                Note.is_active == True,
                Note.moderation_status == "approved",
                Note.deleted_at.is_(None),
            )
        )

        if q:
            like = f"%{q}%"
            combos_stmt = combos_stmt.where(
                or_(
                    Combo.title.ilike(like),
                    Combo.description.ilike(like),
                    Note.title.ilike(like),
                    Note.description.ilike(like),
                )
            )

        if university:
            combos_stmt = combos_stmt.where(Note.university.ilike(f"%{university}%"))
        if faculty:
            combos_stmt = combos_stmt.where(Note.faculty.ilike(f"%{faculty}%"))
        if career:
            combos_stmt = combos_stmt.where(Note.career.ilike(f"%{career}%"))

        if note_type == "free":
            combos_stmt = combos_stmt.where(Combo.seller_net_cents == 0)
        elif note_type == "paid":
            combos_stmt = combos_stmt.where(Combo.seller_net_cents > 0)

        combos = s.execute(combos_stmt.order_by(desc(Combo.created_at)).limit(100)).scalars().all()

    return render_template(
        "index.html",
        notes=notes,
        combos=combos,
        buyer_price=_combo_buyer_price_cents,
        show_tab="advanced",
        q=q,
        filters={"university": university, "faculty": faculty, "career": career, "type": note_type},
        include_dynamic_selects=True,
    )



# Ruta de compatibilidad (si querés mantener /search)
@app.route("/search", methods=["GET"])
def search():
    q = (request.args.get("q") or "").strip()
    show_tab = request.args.get("tab") or "quick"

    # mantener parámetros para que no rompa el front viejo
    uni = request.args.get("university", "")
    fac = request.args.get("faculty", "")
    car = request.args.get("career", "")
    t = request.args.get("type", "")

    with Session(engine) as s:
        # ---- notes
        notes_stmt = (
            select(Note)
            .where(Note.moderation_status == "approved")
            .where(or_(
                Note.title.ilike(f"%{q}%"),
                Note.description.ilike(f"%{q}%"),
                Note.subject.ilike(f"%{q}%"),
            ))
        )
        notes = s.execute(notes_stmt.order_by(desc(Note.created_at)).limit(100)).scalars().all()

        # ---- combos (búsqueda por texto; sin filtros académicos)
        combos = []
        if q:
            combos_stmt = (
                select(Combo)
                .where(Combo.is_active == True)  # noqa: E712
                .where(Combo.moderation_status == "approved")
                .where(or_(
                    Combo.title.ilike(f"%{q}%"),
                    Combo.description.ilike(f"%{q}%"),
                ))
            )
            combos = s.execute(combos_stmt.order_by(desc(Combo.created_at)).limit(100)).scalars().all()

    return render_template(
        "index.html",
        notes=notes,
        combos=combos,
        buyer_price=_combo_buyer_price_cents,
        show_tab=show_tab,
        q=q,
        filters={"university": uni, "faculty": fac, "career": car, "type": t},
        include_dynamic_selects=True,
    )

# -----------------------------------------------------------------------------
# Auth (sólo Google con Firebase)
# -----------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET"])
def login():
    return render_template("login_google.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.post("/auth/session_login")
def auth_session_login():
    """Valida el ID token y decide si loguea directo o completa perfil."""
    try:
        data = request.get_json(silent=True) or {}
        id_token = (data.get("id_token") or "").strip()
        if not id_token:
            return {"ok": False, "error": "missing id_token"}, 400

        info = verify_firebase_id_token(id_token)
        email = (info.get("email") or "").lower().strip()
        name  = (info.get("name") or "").strip()

        if not email:
            return {"ok": False, "error": "google_without_email"}, 400

        with Session() as s:
            u = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if u:
                # 🚫 Si está bloqueado o inactivo, no lo dejamos entrar
                if getattr(u, "is_blocked", False) or not getattr(u, "is_active", True):
                    return {
                        "ok": False,
                        "error": "account_blocked",
                        "message": "Tu cuenta está bloqueada. Escribinos a soporte.apuntesya@gmail.com"
                    }, 403

                login_user(u)
                return {"ok": True, "next": url_for("index")}, 200


        session["pending_google"] = {
            "email": email,
            "name": name,
            "uid": info.get("uid"),
            "picture": info.get("picture")
        }
        return {"ok": True, "next": url_for("complete_profile")}, 200

    except Exception as e:
        app.logger.exception("session_login error")
        return {"ok": False, "error": str(e)}, 500

@app.get("/complete_profile")
def complete_profile():
    if "pending_google" not in session:
        return redirect(url_for("login"))
    data = session["pending_google"]
    return render_template("complete_profile.html", name=data.get("name"))

@app.post("/complete_profile")
def complete_profile_post():
    if "pending_google" not in session:
        return redirect(url_for("login"))

    university = (request.form.get("university") or "").strip()
    faculty    = (request.form.get("faculty") or "").strip()
    career     = (request.form.get("career") or "").strip()
    seller_contact = (request.form.get("seller_contact") or "").strip()

    if not (university and faculty and career):
        flash("Completá Universidad, Facultad y Carrera.")
        return redirect(url_for("complete_profile"))

    data = session["pending_google"]
    email = data.get("email")
    name  = data.get("name")

    with Session() as s:
        exists = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if exists:
            login_user(exists)
            session.pop("pending_google", None)
            return redirect(url_for("index"))

        u = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(secrets.token_urlsafe(16)),
            university=university,
            faculty=faculty,
            career=career,
            seller_contact=seller_contact or None,
            is_active=True,
        )
        s.add(u)
        s.commit()
        login_user(u)

    session.pop("pending_google", None)
    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
# Perfil
# -----------------------------------------------------------------------------
@app.route("/profile")
@login_required
def profile():
    with Session() as s:
        me = s.get(User, current_user.id)
        seller_contact = getattr(me, "seller_contact", "") or ""
        contact_url, contact_label = _build_contact_link(seller_contact)

        # Structured contacts
        contact_email = getattr(me, "contact_email", "") or ""
        contact_whatsapp = getattr(me, "contact_whatsapp", "") or ""
        contact_phone = getattr(me, "contact_phone", "") or ""
        contact_website = getattr(me, "contact_website", "") or ""
        contact_instagram = getattr(me, "contact_instagram", "") or ""
        contact_visible_public = bool(getattr(me, "contact_visible_public", True))
        contact_visible_buyers = bool(getattr(me, "contact_visible_buyers", True))

        mp_connected = bool(getattr(me, "mp_access_token", None))

    return render_template(
        "profile.html",
        seller_contact=seller_contact,
        seller_contact_url=contact_url,
        seller_contact_label=contact_label,
        contact_email=contact_email,
        contact_whatsapp=contact_whatsapp,
        contact_phone=contact_phone,
        contact_website=contact_website,
        contact_instagram=contact_instagram,
        contact_visible_public=contact_visible_public,
        contact_visible_buyers=contact_visible_buyers,
        mp_connected=mp_connected,
    )



# -----------------------------------------------------------------------------
# Notifications
# -----------------------------------------------------------------------------
@app.post("/notifications/mark_read")
@login_required
def notifications_mark_read():
    """Mark current user's notifications as read."""
    try:
        with Session() as s:
            s.query(Notification).filter(
                Notification.user_id == current_user.id,
                Notification.is_read == False
            ).update({Notification.is_read: True}, synchronize_session=False)
            s.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/profile/update_contact")
@login_required
def profile_update_contact():
    contact = (request.form.get("seller_contact") or "").strip()
    contact_email = (request.form.get("contact_email") or "").strip()
    contact_whatsapp = (request.form.get("contact_whatsapp") or "").strip()
    contact_phone = (request.form.get("contact_phone") or "").strip()
    contact_website = (request.form.get("contact_website") or "").strip()
    contact_instagram = (request.form.get("contact_instagram") or "").strip()
    visible_public = (request.form.get("contact_visible_public") == "1")
    visible_buyers = (request.form.get("contact_visible_buyers") == "1")
    with Session() as s:
        u = s.get(User, current_user.id)
        setattr(u, "seller_contact", contact or None)
        if hasattr(u, "contact_email"):
            u.contact_email = contact_email or None
            u.contact_whatsapp = contact_whatsapp or None
            if hasattr(u, 'contact_phone'):
                u.contact_phone = contact_phone or None
            if hasattr(u, 'contact_website'):
                u.contact_website = contact_website or None
            u.contact_instagram = contact_instagram or None
            u.contact_visible_public = bool(visible_public)
            u.contact_visible_buyers = bool(visible_buyers)
        s.commit()
    flash("Datos de contacto actualizados.")
    return redirect(url_for("profile"))


# -----------------------------------------------------------------------------
# Mis apuntes (hub) + edición unificada de apuntes/combos
# -----------------------------------------------------------------------------
@app.get("/my-notes")
@login_required
def my_notes_hub():
    """Hub para creadores: nuevo apunte / nuevo combo / editar publicaciones."""
    return render_template("my_notes_hub.html")


@app.get("/my-content/edit")
@login_required
def my_content_edit():
    """Página unificada para ver/editar/borrar apuntes y combos del usuario."""
    with Session() as s:
        # Apuntes del usuario + cantidad de descargas (compras aprobadas)
        rows = s.execute(
            select(
                Note,
                func.count(Purchase.id).label("download_count"),
            )
            .outerjoin(
                Purchase,
                and_(
                    Purchase.note_id == Note.id,
                    Purchase.status == "approved",
                ),
            )
            .where(
                Note.seller_id == current_user.id,
                Note.deleted_at.is_(None),
            )
            .group_by(Note.id)
            .order_by(Note.created_at.desc())
        ).all()

        my_notes = []
        for note, download_count in rows:
            note.download_count = int(download_count or 0)
            my_notes.append(note)

        combos = s.execute(
            select(Combo)
            .where(
                Combo.seller_id == current_user.id,
                Combo.is_active == True,
            )
            .order_by(Combo.created_at.desc())
        ).scalars().all()

    return render_template("my_content_edit.html", my_notes=my_notes, combos=combos)


# -----------------------------------------------------------------------------
# Vendedor: editar / borrar apuntes
# -----------------------------------------------------------------------------

@app.get("/profile/notes/<int:note_id>/edit")
@login_required
def seller_edit_note_get(note_id: int):
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or note.seller_id != current_user.id:
            abort(404)
        return render_template("note_edit.html", note=note)


@app.post("/profile/notes/<int:note_id>/edit")
@login_required
def seller_edit_note_post(note_id: int):
    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    university = (request.form.get("university") or "").strip()
    faculty = (request.form.get("faculty") or "").strip()
    career = (request.form.get("career") or "").strip()
    price_raw = (request.form.get("price") or "").strip().replace(",", ".")
    try:
        price_cents = int(round(float(price_raw) * 100)) if price_raw else 0
    except Exception:
        price_cents = 0

    # Paid notes require Mercado Pago linked
    if price_cents > 0 and not getattr(current_user, "mp_access_token", None):
        flash("Para publicar apuntes pagos tenés que vincular tu cuenta de Mercado Pago primero.", "warning")
        return redirect(url_for("profile"))

    new_file = request.files.get("file")
    replace_file = bool(new_file and new_file.filename)
    if replace_file and not allowed_pdf(new_file.filename):
        flash("Sólo PDF.", "danger")
        return redirect(url_for("seller_edit_note_get", note_id=note_id))

    with Session() as s:
        note = s.get(Note, note_id)
        if not note or note.seller_id != current_user.id:
            abort(404)

        note.title = title or note.title
        note.description = description
        note.university = university
        note.faculty = faculty
        note.career = career
        # Uniform: seller enters NET (what they want to receive)
        note.price_cents = max(price_cents, 0)        # legacy field (kept)
        note.seller_net_cents = max(price_cents, 0)   # canonical

        # Reemplazar archivo (best-effort)
        if replace_file:
            old_path = getattr(note, "file_path", None)
            base_name = secure_filename(new_file.filename)
            unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{base_name}"

            if gcs_bucket:
                blob_name = f"notes/{current_user.id}/{unique_name}"
                gcs_upload_file(new_file, blob_name)
                note.file_path = blob_name
                # borrar anterior si era de GCS
                try:
                    if old_path:
                        gcs_delete_blob(old_path)
                except Exception:
                    pass
            else:
                ensure_dirs()
                fpath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                new_file.save(fpath)
                note.file_path = unique_name
                # borrar anterior local
                try:
                    if old_path:
                        old_local = os.path.join(app.config["UPLOAD_FOLDER"], old_path)
                        if os.path.exists(old_local):
                            os.remove(old_local)
                except Exception:
                    pass

            # si reemplaza archivo → volver a revisión (conservador)
            try:
                note.moderation_status = "pending_manual"
                note.moderation_reason = "Archivo actualizado por el vendedor (requiere revisión)."
                note.manual_review_due_at = datetime.utcnow() + timedelta(hours=12)
            except Exception:
                pass

        s.commit()

    flash("Apunte actualizado.", "success")
    return redirect(url_for("profile"))


@app.post("/profile/notes/<int:note_id>/delete")
@login_required
def seller_delete_note(note_id: int):
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or note.seller_id != current_user.id:
            abort(404)

        # Soft delete
        note.is_active = False
        if hasattr(note, "deleted_at"):
            note.deleted_at = datetime.utcnow()

        # borrar archivo (best-effort)
        try:
            fp = getattr(note, "file_path", None)
            if fp:
                if gcs_bucket:
                    gcs_delete_blob(fp)
                else:
                    local = os.path.join(app.config["UPLOAD_FOLDER"], fp)
                    if os.path.exists(local):
                        os.remove(local)
        except Exception:
            pass

        s.commit()
    flash("Apunte eliminado.", "success")
    return redirect(url_for("profile"))

@app.route("/profile/balance")
@login_required
def profile_balance():
    fmt = "%Y-%m-%d"
    today = datetime.utcnow().date()
    default_start = today.replace(day=1)
    start_str = request.args.get("start", default_start.strftime(fmt))
    end_str = request.args.get("end", today.strftime(fmt))

    try:
        start = datetime.strptime(start_str, fmt)
        end = datetime.strptime(end_str, fmt) + timedelta(days=1)
    except Exception:
        start = datetime(default_start.year, default_start.month, 1)
        end = datetime(today.year, today.month, today.day) + timedelta(days=1)

    with Session() as s:
        base_filter = and_(
            Note.seller_id == current_user.id,
            Purchase.status == "approved",
            Purchase.created_at >= start,
            Purchase.created_at < end
        )

        totals = s.execute(
            select(
                func.count(Purchase.id),
                func.coalesce(func.sum(Purchase.amount_cents), 0)
            ).join(Note, Note.id == Purchase.note_id).where(base_filter)
        ).one()

        sold_count = int(totals[0] or 0)

        # 💰 Lo que realmente recibe el vendedor (precio que cargó en el apunte)
        gross_cents = int(totals[1] or 0)

        # Estas comisiones son solo referencias / estimaciones.
        mp_commission_cents  = int(round(gross_cents * float(MP_COMMISSION_RATE)))
        apy_commission_cents = int(round(gross_cents * float(APY_COMMISSION_RATE)))

        # ✅ Neto para el vendedor = lo que pidió cobrar (sin restar comisiones)
        net_cents = gross_cents


        has_views = hasattr(Note, "views")
        if has_views:
            rows = s.execute(
                select(
                    Note.id, Note.title, Note.views,
                    func.count(Purchase.id).label("sold_count"),
                    func.coalesce(func.sum(Purchase.amount_cents), 0).label("gross_cents")
                )
                .join(Purchase, Purchase.note_id == Note.id, isouter=True)
                .where(Note.seller_id == current_user.id, Purchase.created_at >= start, Purchase.created_at < end)
                .group_by(Note.id, Note.title, Note.views)
                .order_by(func.count(Purchase.id).desc())
            ).all()
        else:
            rows = s.execute(
                select(
                    Note.id, Note.title,
                    func.count(Purchase.id).label("sold_count"),
                    func.coalesce(func.sum(Purchase.amount_cents), 0).label("gross_cents")
                )
                .join(Purchase, Purchase.note_id == Note.id, isouter=True)
                .where(Note.seller_id == current_user.id, Purchase.created_at >= start, Purchase.created_at < end)
                .group_by(Note.id, Note.title)
                .order_by(func.count(Purchase.id).desc())
            ).all()

        per_note = []
        for r in rows:
            if has_views:
                _id, _title, _views, _sold, _gross = r
                views = int(_views or 0)
                sold  = int(_sold or 0)
                gross = int(_gross or 0)
            else:
                _id, _title, _sold, _gross = r
                views = None
                sold  = int(_sold or 0)
                gross = int(_gross or 0)

            mp_c  = int(round(gross * float(MP_COMMISSION_RATE)))
            apy_c = int(round(gross * float(APY_COMMISSION_RATE)))
            per_note.append({
                "id": _id,
                "title": _title,
                "sold_count": sold,
                "gross_cents": gross,              # total configurado por vos
                "mp_commission_cents": mp_c,       # solo referencia
                "apy_commission_cents": apy_c,     # solo referencia
                "net_cents": gross,                # ✅ lo que vos recibís
                "views": views,
                "conversion": (sold / views * 100.0) if (views and views > 0) else None
            })


    return render_template(
        "profile_balance.html",
        IIBB_ENABLED=IIBB_ENABLED, IIBB_RATE=IIBB_RATE, sold_count=sold_count,
        total_cents=gross_cents,
        mp_commission_cents=mp_commission_cents,
        apy_commission_cents=apy_commission_cents,
        net_cents=net_cents,
        per_note=per_note,
        start=start_str,
        end=(end - timedelta(days=1)).strftime(fmt),
        MP_COMMISSION_RATE=MP_COMMISSION_RATE,
        APY_COMMISSION_RATE=APY_COMMISSION_RATE
    )

@app.route("/profile/purchases")
@login_required
def profile_purchases():
    with Session() as s:
        purchases = s.execute(
            select(Purchase, Note)
            .join(Note, Note.id == Purchase.note_id)
            .where(
                Purchase.buyer_id == current_user.id,
                Purchase.status == 'approved'
            )
            .order_by(Purchase.created_at.desc())
        ).all()

    items = []
    for p, n in purchases:
        # amount_cents stores what the buyer paid (published price)
        buyer_price_cents = int(p.amount_cents or 0)
        is_free = buyer_price_cents == 0

        items.append(dict(
            id=p.id,
            note_id=n.id,
            title=n.title,
            price_cents=buyer_price_cents,
            is_free=is_free,
            created_at=p.created_at.strftime("%Y-%m-%d %H:%M"),
        ))
    return render_template("profile_purchases.html", items=items)



# -----------------------------------------------------------------------------
# Upload / Detail / Download
# -----------------------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_note():
    if request.method == "POST":
        title = request.form["title"].strip()
        description = request.form["description"].strip()
        university = request.form["university"].strip()
        faculty = request.form["faculty"].strip()
        career = request.form["career"].strip()
        price = request.form.get("price", "").strip()
        # price == seller net (what they want to receive)
        price_cents = int(round(float(price) * 100)) if price else 0

        # Moderation acknowledgement (required)
        if request.form.get("moderation_ack") != "1":
            flash("Antes de publicar, tenés que aceptar la leyenda de moderación (IA + posible revisión manual hasta 12hs).", "warning")
            return redirect(url_for("upload_note"))

        # Paid notes require Mercado Pago linked
        if price_cents > 0 and not getattr(current_user, "mp_access_token", None):
            flash("Para publicar apuntes pagos tenés que vincular tu cuenta de Mercado Pago primero.", "warning")
            return redirect(url_for("profile"))

        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Seleccioná un PDF.")
            return redirect(url_for("upload_note"))
        if not allowed_pdf(file.filename):
            flash("Sólo PDF.")
            return redirect(url_for("upload_note"))

        base_name = secure_filename(file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{base_name}"

        if gcs_bucket:
            # Guardamos en GCS bajo notes/<seller_id>/<archivo>
            blob_name = f"notes/{current_user.id}/{unique_name}"
            gcs_upload_file(file, blob_name)
            stored_path = blob_name
        else:
            # Fallback local (por ejemplo, entorno de desarrollo sin GCS)
            ensure_dirs()
            fpath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(fpath)
            stored_path = unique_name

        with Session() as s:
            note = Note(
                title=title,
                description=description,
                university=university,
                faculty=faculty,
                career=career,
                price_cents=price_cents,       # legacy field (kept)
                seller_net_cents=price_cents,  # canonical
                file_path=stored_path,
                seller_id=current_user.id
            )

            # New uploads go through AI moderation
            note.moderation_status = "pending_ai"

            s.add(note)
            s.commit()

            # Generate preview images (best-effort)
            try:
                pages, imgs = generate_note_preview(note)
                note.preview_pages = {"pages": pages}
                note.preview_images = {"images": imgs}
                s.commit()
            except Exception as e:
                app.logger.warning(f"preview generation failed: {e}")

            # AI moderation (best-effort). If OpenAI isn't configured, it will fall back to manual review.
            try:
                text_sample = _extract_text_for_moderation(note)
                meta = {
                    "title": title,
                    "description": description,
                    "university": university,
                    "faculty": faculty,
                    "career": career,
                    "seller_id": current_user.id,
                    "price_net": price_cents / 100.0,
                }
                ai = _gemini_moderate_note(text_sample=text_sample, meta=meta)
                status, reason = _decision_to_status(ai)

                note.ai_decision = (ai.get("decision") or None)
                note.ai_model = GEMINI_MODEL if GEMINI_API_KEY else None
                note.ai_summary = (ai.get("summary") or None)
                note.ai_raw = ai

                # store float scores as 0..1000 integers for portability
                def _to_i(x):
                    try:
                        return int(round(float(x) * 1000))
                    except Exception:
                        return None
                note.ai_confidence = _to_i(ai.get("confidence"))
                note.ai_score_quality = _to_i(ai.get("quality_score"))
                note.ai_score_copyright = _to_i(ai.get("copyright_risk"))
                note.ai_score_mismatch = _to_i(ai.get("mismatch_risk"))

                note.moderation_status = status
                note.moderation_reason = reason
                if status == "pending_manual":
                    note.manual_review_due_at = datetime.utcnow() + timedelta(hours=12)

                # Notifications
                admin_ids = [u.id for u in s.execute(select(User).where(User.is_admin == True)).scalars().all()]
                if status == "approved":
                    _notify_users(s, [current_user.id], "Apunte aprobado", "Tu apunte fue aprobado automáticamente y ya está publicado.", kind="success")
                elif status == "pending_manual":
                    _notify_users(s, [current_user.id], "Apunte en revisión manual", "La IA marcó tu apunte para revisión manual. Puede demorar hasta 12hs.", kind="warning")
                    _notify_users(s, admin_ids, "Revisión manual requerida", f"Hay un apunte pendiente de revisión manual: #{note.id} — {note.title}", kind="warning")
                elif status == "rejected":
                    _notify_users(s, [current_user.id], "Apunte rechazado", f"Tu apunte fue rechazado por la moderación automática. Motivo: {reason or 'sin detalle'}", kind="danger")
                    _notify_users(s, admin_ids, "Apunte rechazado por IA", f"Rechazo IA: #{note.id} — {note.title}. Motivo: {reason or 'sin detalle'}", kind="danger")

                s.commit()
            except Exception as e:
                app.logger.warning(f"ai moderation failed: {e}")

        # UX message based on status
        msg = "Apunte subido."
        try:
            if note.moderation_status == "approved":
                msg += " Fue aprobado automáticamente y ya está publicado."
            elif note.moderation_status == "pending_manual":
                msg += " Quedó en revisión manual (puede demorar hasta 12hs)."
            elif note.moderation_status == "rejected":
                msg += " Fue rechazado. Revisá el motivo en tu perfil."
            else:
                msg += " Quedó pendiente de revisión."
        except Exception:
            msg += ""
        flash(msg)
        return redirect(url_for("note_detail", note_id=note.id))
    return render_template("upload.html")

@app.route("/note/<int:note_id>")
def note_detail(note_id):
    paid_param = request.args.get("paid", "0")  # "1" si viene de MP

    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)

        # Hide non-approved notes from the public (seller/admin can still view)
        if getattr(note, "moderation_status", "approved") != "approved":
            if not current_user.is_authenticated:
                abort(404)
            if current_user.id != note.seller_id and not getattr(current_user, "is_admin", False):
                abort(404)

        # ¿Puede descargar?
        can_download = False
        if current_user.is_authenticated:
            if note.price_cents == 0 or note.seller_id == current_user.id:
                can_download = True
            else:
                p = s.execute(
                    select(Purchase).where(
                        Purchase.buyer_id == current_user.id,
                        Purchase.note_id == note.id,
                        Purchase.status == 'approved'
                    )
                ).scalar_one_or_none()
                can_download = p is not None

        # Downloads metric
        try:
            dl = s.execute(
                select(func.count(DownloadLog.id)).where(DownloadLog.note_id == note.id)
            ).scalar_one()
            note.download_count = int(dl or 0)
        except Exception:
            pass

        # Reseñas y nombres de usuarios
        rows = s.execute(
            select(Review, User.name)
            .join(User, User.id == Review.buyer_id)
            .where(Review.note_id == note.id)
            .order_by(Review.created_at.desc())
        ).all()
        reviews = rows

        # Promedio de estrellas
        if reviews:
            avg_rating = round(sum(r.rating for r, _ in reviews) / len(reviews), 2)
        else:
            avg_rating = None

        # ¿Puede calificar este usuario?
        can_review = False
        already_reviewed = False
        if current_user.is_authenticated and current_user.id != note.seller_id:

            if note.price_cents > 0:
                has_purchase = s.execute(
                    select(Purchase).where(
                        Purchase.buyer_id == current_user.id,
                        Purchase.note_id == note.id,
                        Purchase.status == 'approved'
                    )
                ).scalar_one_or_none() is not None
            else:
                has_purchase = True  # Gratis → puede reseñar

            if has_purchase:
                already_reviewed = s.execute(
                    select(Review).where(
                        Review.note_id == note.id,
                        Review.buyer_id == current_user.id
                    )
                ).scalar_one_or_none() is not None

                can_review = not already_reviewed

        # Vendedor (para perfil + contacto)
        seller = s.get(User, note.seller_id) if note.seller_id else None

        # Contactos del vendedor
        seller_contacts = []
        if seller:
            # Backward compatible: if legacy seller_contact is present, show it too
            if getattr(seller, "seller_contact", None):
                url, label = _build_contact_link(getattr(seller, "seller_contact"))
                if url:
                    seller_contacts.append((url, "Contacto"))

            # Structured contacts
            if getattr(seller, "contact_visible_public", True) or (can_download and getattr(seller, "contact_visible_buyers", True)):
                if getattr(seller, "contact_email", None):
                    seller_contacts.append((f"mailto:{seller.contact_email}", "Email"))
                if getattr(seller, "contact_whatsapp", None):
                    url, _ = _build_contact_link(seller.contact_whatsapp)
                    if url:
                        seller_contacts.append((url, "WhatsApp"))
                if getattr(seller, "contact_phone", None):
                    url, _ = _build_contact_link(seller.contact_phone)
                    if url:
                        seller_contacts.append((url, "Teléfono"))
                if getattr(seller, "contact_website", None):
                    w = (seller.contact_website or '').strip()
                    if w:
                        if not (w.startswith('http://') or w.startswith('https://')):
                            w = 'https://' + w
                        seller_contacts.append((w, "Web"))
                if getattr(seller, "contact_instagram", None):
                    ig = seller.contact_instagram.strip().lstrip("@").strip()
                    if ig:
                        seller_contacts.append((f"https://instagram.com/{ig}", "Instagram"))

        # Verified seller badge (MP connected + at least 1 approved sale)
        seller_verified = False
        if seller and getattr(seller, "mp_access_token", None):
            sold = s.execute(
                select(func.count(Purchase.id))
                .join(Note, Note.id == Purchase.note_id)
                .where(Note.seller_id == seller.id, Purchase.status == "approved", Purchase.amount_cents > 0)
            ).scalar_one()
            seller_verified = (sold or 0) >= 1

        # -----------------------------------------------------
        # 👉 PRECIOS
        # -----------------------------------------------------
        # Lo que recibe el vendedor
        base_price = note.price_cents / 100.0 if note.price_cents else 0.0

        # Precio final al comprador (con comisiones)
        buyer_price = None
        if base_price > 0:
            buyer_price = round(base_price * GROSS_MULTIPLIER, 2)

    return render_template(
        "note_detail.html",
        note=note,
        can_download=can_download,
        reviews=reviews,
        avg_rating=avg_rating,
        can_review=can_review,
        already_reviewed=already_reviewed,
        seller=seller,
        seller_verified=seller_verified,
        seller_contacts=seller_contacts,
        base_price=base_price,
        buyer_price=buyer_price,
        paid=(paid_param == "1"),
    )


@app.route("/seller/<int:seller_id>")
def seller_profile(seller_id: int):
    """Public seller profile with catalog."""
    with Session() as s:
        seller = s.get(User, seller_id)
        if not seller or not getattr(seller, "is_active", True):
            abort(404)

        notes = s.execute(
            select(Note)
            .where(
                Note.seller_id == seller_id,
                Note.is_active == True,
                Note.moderation_status == "approved",
                Note.deleted_at.is_(None)
            )
            .order_by(Note.created_at.desc())
        ).scalars().all()

        # Seller KPIs (paid sales, downloads)
        paid_sales = s.execute(
            select(func.count(Purchase.id))
            .join(Note, Note.id == Purchase.note_id)
            .where(Note.seller_id == seller_id, Purchase.status == "approved", Purchase.amount_cents > 0)
        ).scalar_one()
        total_downloads = s.execute(
            select(func.count(DownloadLog.id))
            .join(Note, Note.id == DownloadLog.note_id)
            .where(Note.seller_id == seller_id)
        ).scalar_one()

        avg_rating = s.execute(
            select(func.avg(Review.rating))
            .join(Note, Note.id == Review.note_id)
            .where(Note.seller_id == seller_id)
        ).scalar_one()
        avg_rating = round(float(avg_rating), 2) if avg_rating is not None else None

        seller_verified = bool(getattr(seller, "mp_access_token", None)) and (paid_sales or 0) >= 1

        # Public contacts (respect visibility)
        contacts = []
        if getattr(seller, "contact_visible_public", True):
            if getattr(seller, "contact_email", None):
                contacts.append((f"mailto:{seller.contact_email}", "Email"))
            if getattr(seller, "contact_whatsapp", None):
                url, _ = _build_contact_link(seller.contact_whatsapp)
                if url:
                    contacts.append((url, "WhatsApp"))
            if getattr(seller, "contact_phone", None):
                url, _ = _build_contact_link(seller.contact_phone)
                if url:
                    contacts.append((url, "Teléfono"))
            if getattr(seller, "contact_website", None):
                w = (seller.contact_website or '').strip()
                if w:
                    if not (w.startswith('http://') or w.startswith('https://')):
                        w = 'https://' + w
                    contacts.append((w, "Web"))
            if getattr(seller, "contact_instagram", None):
                ig = seller.contact_instagram.strip().lstrip("@").strip()
                if ig:
                    contacts.append((f"https://instagram.com/{ig}", "Instagram"))
            # legacy fallback
            if getattr(seller, "seller_contact", None):
                url, _ = _build_contact_link(seller.seller_contact)
                if url:
                    contacts.append((url, "Contacto"))

    return render_template(
        "seller_profile.html",
        seller=seller,
        seller_verified=seller_verified,
        notes=notes,
        avg_rating=avg_rating,
        paid_sales=int(paid_sales or 0),
        total_downloads=int(total_downloads or 0),
        contacts=contacts,
    )


@app.route("/preview/<int:note_id>/<int:idx>.jpg")
def note_preview_image(note_id: int, idx: int):
    """Serve protected preview images (watermarked) without exposing storage URLs."""
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)
        if getattr(note, "moderation_status", "approved") != "approved":
            abort(404)
        meta = getattr(note, "preview_images", None) or {}
        imgs = (meta or {}).get("images") or []
        if idx < 1 or idx > len(imgs):
            abort(404)
        path = imgs[idx - 1]

        from io import BytesIO
        if gcs_bucket and path and "/" in path:
            data = gcs_download_bytes(path)
            return send_file(BytesIO(data), mimetype="image/jpeg", as_attachment=False, download_name=f"preview_{note_id}_{idx}.jpg")

        # local fallback
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], path)
        if not os.path.exists(fpath):
            abort(404)
        return send_file(fpath, mimetype="image/jpeg", as_attachment=False)


@app.post("/note/<int:note_id>/review")
@login_required
def submit_review(note_id):
    rating = int(request.form.get("rating", "0") or 0)
    comment = (request.form.get("comment") or "").strip()

    # Validar rango de puntuación
    if rating < 1 or rating > 5:
        flash("La puntuación debe estar entre 1 y 5.")
        return redirect(url_for("note_detail", note_id=note_id, _anchor="reviews"))

    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)

        # No puede calificar su propio apunte
        if note.seller_id == current_user.id:
            flash("No podés calificar tu propio apunte.")
            return redirect(url_for("note_detail", note_id=note_id, _anchor="reviews"))

        # Debe haber comprado el apunte si es pago
        if note.price_cents > 0:
            has_purchase = s.execute(
                select(Purchase).where(
                    Purchase.buyer_id == current_user.id,
                    Purchase.note_id == note.id,
                    Purchase.status == "approved"
                )
            ).scalar_one_or_none() is not None
        else:
            has_purchase = True

        if not has_purchase:
            flash("Necesitás haber comprado este apunte para calificarlo.")
            return redirect(url_for("note_detail", note_id=note_id, _anchor="reviews"))

        # Evitar reseñas duplicadas
        exists = s.execute(
            select(Review).where(
                Review.note_id == note.id,
                Review.buyer_id == current_user.id
            )
        ).scalar_one_or_none()

        if exists:
            flash("Ya enviaste una reseña para este apunte.")
            return redirect(url_for("note_detail", note_id=note_id, _anchor="reviews"))

        # Crear la reseña
        r = Review(
            note_id=note.id,
            buyer_id=current_user.id,
            rating=rating,
            comment=comment
        )
        s.add(r)
        s.commit()

    flash("¡Gracias por tu reseña!")
    # Si querés seguir mostrando el mensaje de pago aprobado, podrías agregar otro flash acá
    # flash("✅ Pago aprobado, ya podés descargar.")
    return redirect(url_for("note_detail", note_id=note_id, _anchor="reviews"))


@app.route("/download/<int:note_id>")
@login_required
def download_note(note_id):
    # NOTA: este proyecto usa SQLAlchemy "core" con Session(), no Flask-SQLAlchemy.
    # Por eso NO usamos Note.query ni db.session.
    with Session() as s:
        note = s.get(Note, note_id)
        if not note:
            abort(404)

        # Access control
        is_admin = bool(getattr(current_user, "is_admin", False))
        is_owner = bool(getattr(note, "seller_id", None) == getattr(current_user, "id", None))

        # Gratis si el precio (neto del vendedor) es 0
        is_free = int(getattr(note, "price_cents", 0) or 0) <= 0

        allowed = False

        # Admin / dueño siempre
        if is_admin or is_owner:
            allowed = True
        # Premium puede descargar
        elif bool(getattr(current_user, "is_premium", False)):
            allowed = True
        # Apunte gratuito
        elif is_free:
            allowed = True
        else:
            # Apunte pago: requiere compra aprobada
            has_purchase = s.execute(
                select(Purchase.id).where(
                    Purchase.buyer_id == current_user.id,
                    Purchase.note_id == note.id,
                    Purchase.status == "approved",
                )
            ).scalar_one_or_none() is not None
            allowed = bool(has_purchase)

        if not allowed:
            flash("No tenés acceso a este archivo.", "danger")
            return redirect(url_for("note_detail", note_id=note.id))

        note_file_path = getattr(note, "file_path", None)

        # Log download (best effort)
        try:
            s.add(
                DownloadLog(
                    user_id=current_user.id,
                    note_id=note.id,
                    combo_id=None,
                    is_free=is_free,
                )
            )
            s.commit()
        except Exception as e:
            try:
                s.rollback()
            except Exception:
                pass
            app.logger.warning("DownloadLog insert failed: %s", e)

    # Deliver file without exposing storage links
    from io import BytesIO

    if gcs_bucket and note_file_path and "/" in note_file_path:
        data = gcs_download_bytes(note_file_path)
        fname = os.path.basename(note_file_path) or f"apunte_{note_id}.pdf"
        return send_file(
            BytesIO(data),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=fname,
        )

    if not note_file_path:
        flash("No se encontró el archivo asociado a este apunte.", "danger")
        return redirect(url_for("note_detail", note_id=note_id))

    return send_from_directory(app.config["UPLOAD_FOLDER"], note_file_path, as_attachment=True)

@app.route("/mp/connect")
@login_required
def connect_mp():
    return redirect(mp.oauth_authorize_url())

@app.route("/mp/oauth/callback")
@login_required
def mp_oauth_callback():
    if not current_user.is_authenticated:
        flash("Necesitás iniciar sesión para vincular Mercado Pago.")
        return redirect(url_for("login"))

    code = request.args.get("code")
    if not code:
        flash("No se recibió 'code' de autorización.")
        return redirect(url_for("profile"))

    try:
        data = mp.oauth_exchange_code(code)
    except Exception as e:
        flash(f"Error al intercambiar código: {e}")
        return redirect(url_for("profile"))

    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    user_id = str(data.get("user_id"))
    expires_in = int(data.get("expires_in", 0))
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)

    with Session() as s:
        u = s.get(User, current_user.id)
        u.mp_user_id = user_id
        u.mp_access_token = access_token
        u.mp_refresh_token = refresh_token
        u.mp_token_expires_at = expires_at
        s.commit()

    flash("¡Cuenta de Mercado Pago conectada!")
    return redirect(url_for("profile"))

@app.route("/mp/disconnect")
@login_required
def disconnect_mp():
    with Session() as s:
        u = s.get(User, current_user.id)
        u.mp_user_id = None
        u.mp_access_token = None
        u.mp_refresh_token = None
        u.mp_token_expires_at = None
        s.commit()
    flash("Se desvinculó Mercado Pago.")
    return redirect(url_for("profile"))

# -----------------------------------------------------------------------------
# Comprar
# -----------------------------------------------------------------------------
@app.route("/buy/<int:note_id>")
@login_required
def buy_note(note_id):
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)
        if getattr(note, "moderation_status", "approved") != "approved":
            abort(404)
        if note.seller_id == current_user.id:
            flash("No podés comprar tu propio apunte.")
            return redirect(url_for("note_detail", note_id=note.id))
        net_cents = int(getattr(note, "seller_net_cents", 0) or getattr(note, "price_cents", 0) or 0)
        if net_cents == 0:
            flash("Este apunte es gratuito.")
            return redirect(url_for("download_note", note_id=note.id))

        seller = s.get(User, note.seller_id)

        # Buyer published price (rounded up to 1 decimal)
        buyer_price_cents = published_from_net_cents(net_cents)

        # Store the buyer-paid amount in the purchase
        p = Purchase(buyer_id=current_user.id, note_id=note.id, status="pending", amount_cents=buyer_price_cents)
        s.add(p)
        s.commit()

        price_ars = float(money_1_decimal(cents_to_amount(buyer_price_cents)))  # final comprador (P), 1 decimal
        platform_fee_percent = 0.10  # 10% de P (comisión ApuntesYa)
        back_urls = {
            "success": url_for("mp_return", note_id=note.id, _external=True) + f"?external_reference=purchase:{p.id}",
            "failure": url_for("mp_return", note_id=note.id, _external=True) + f"?external_reference=purchase:{p.id}",
            "pending": url_for("mp_return", note_id=note.id, _external=True) + f"?external_reference=purchase:{p.id}",
        }

        try:
            seller_token = get_valid_seller_token(seller)
            if seller_token is None:
                flash("El vendedor no tiene Mercado Pago vinculado. No se puede procesar la compra.", "warning")
                return redirect(url_for("note_detail", note_id=note.id))

            use_token = seller_token
            # Mercado Pago marketplace_fee = comisión de plataforma (MP cobra su fee aparte)
            marketplace_fee = float(money_1_decimal(price_ars * platform_fee_percent))

            pref = mp.create_preference_for_seller_token(
                seller_access_token=use_token,
                title=note.title,
                unit_price=price_ars,
                quantity=1,
                marketplace_fee=marketplace_fee,
                external_reference=f"purchase:{p.id}",
                back_urls=back_urls,
                notification_url=url_for("mp_webhook", _external=True)
            )

            with Session() as s2:
                p2 = s2.get(Purchase, p.id)
                if p2:
                    p2.preference_id = pref.get("id") or pref.get("preference_id")
                    s2.commit()
            init_point = pref.get("init_point") or pref.get("sandbox_init_point")
            return redirect(init_point)
        except Exception as e:
            flash(f"Error al crear preferencia en Mercado Pago: {e}")
            return redirect(url_for("note_detail", note_id=note.id))

# -----------------------------------------------------------------------------
# MP return
# -----------------------------------------------------------------------------
@app.route("/mp/return/<int:note_id>")
def mp_return(note_id):
    """
    Callback de retorno desde Mercado Pago.
    - Busca el pago (por payment_id o external_reference).
    - Actualiza la Purchase (status, payment_id).
    - Si está aprobado => redirige directo a la descarga.
    - Si no se puede verificar => vuelve al detalle con mensaje.
    """
    # Parámetros que puede mandar MP
    payment_id = (
        request.args.get("payment_id")
        or request.args.get("collection_id")
        or request.args.get("id")
    )
    ext_ref = request.args.get("external_reference", "") or ""
    pref_id = request.args.get("preference_id", "") or ""

    # A veces MP manda el status como query (por las dudas lo usamos de fallback)
    status_query = (
        request.args.get("status")
        or request.args.get("collection_status")
        or ""
    )

    token = app.config["MP_ACCESS_TOKEN_PLATFORM"]
    pay = None

    # 1) Si viene payment_id, intentamos leer el pago directo
    if payment_id:
        try:
            pay = mp.get_payment(token, str(payment_id))
        except Exception as e:
            app.logger.warning(f"mp_return: error get_payment({payment_id}): {e}")

    # 2) Si no tenemos pago aún, probamos con external_reference
    if not pay and ext_ref:
        try:
            res = mp.search_payments_by_external_reference(token, ext_ref)
            results = (res or {}).get("results") or []
            if results:
                pay = results[0].get("payment") or results[0]
        except Exception as e:
            app.logger.warning(f"mp_return: error search by ext_ref {ext_ref}: {e}")

    # 3) Último intento: buscar el último Purchase de este note_id
    if not pay:
        try:
            with Session() as s:
                p_last = s.execute(
                    select(Purchase)
                    .where(Purchase.note_id == note_id)
                    .order_by(Purchase.created_at.desc())
                ).scalars().first()
                if p_last:
                    ext_ref_fallback = f"purchase:{p_last.id}"
                    res = mp.search_payments_by_external_reference(token, ext_ref_fallback)
                    results = (res or {}).get("results") or []
                    if results:
                        pay = results[0].get("payment") or results[0]
                        ext_ref = ext_ref_fallback
        except Exception as e:
            app.logger.warning(f"mp_return: fallback search error: {e}")

    # ------------------------------------------------------------------
    # Interpretar el resultado
    # ------------------------------------------------------------------
    status = None
    external_reference = ext_ref

    if isinstance(pay, dict):
        status = (pay.get("status") or "").lower()
        external_reference = (
            pay.get("external_reference")
            or external_reference
            or ""
        )
        payment_id = str(pay.get("id") or payment_id or "")
    else:
        # Si no pudimos obtener el pago desde la API, usamos lo que venga en la URL
        status = (status_query or "").lower()

    # Identificar purchase_id desde external_reference
    purchase_id = None
    if external_reference and external_reference.startswith("purchase:"):
        try:
            purchase_id = int(external_reference.split(":", 1)[1])
        except Exception:
            purchase_id = None

    # ------------------------------------------------------------------
    # Actualizar la Purchase en la base
    # ------------------------------------------------------------------
    with Session() as s:
        p = None
        if purchase_id:
            p = s.get(Purchase, purchase_id)
        else:
            # Fallback: última compra de ese apunte
            p = s.execute(
                select(Purchase)
                .where(Purchase.note_id == note_id)
                .order_by(Purchase.created_at.desc())
            ).scalars().first()

        if p:
            if payment_id:
                p.payment_id = str(payment_id)
            if status:
                p.status = status
            s.commit()

        # Si está aprobado: vamos directo a descargar
        if status == "approved":
            flash("✅ Pago aprobado, ya podés descargar el apunte.")
            # Volvemos al detalle, marcando que viene de un pago
            return redirect(
                url_for("note_detail", note_id=note_id, paid=1, _anchor="download")
            )


    # Si llegamos acá, no pudimos confirmar “approved”
    flash("Registramos tu intento de pago. Si ya figura aprobado en Mercado Pago, el botón de descarga se habilitará en unos instantes.")
    return redirect(url_for("note_detail", note_id=note_id, _anchor="download"))


# -----------------------------------------------------------------------------
# Webhook único
# -----------------------------------------------------------------------------
def _upsert_purchase_from_payment(pay: dict):
    try:
        status = (pay or {}).get("status")
        external_reference = (pay or {}).get("external_reference") or ""
        payment_id = str(pay.get("id") or "")

        # =========================
        # APUNTES (LO ACTUAL)
        # =========================
        if external_reference.startswith("purchase:"):
            pid = int(external_reference.split(":", 1)[1])
            with Session() as s:
                p = s.get(Purchase, pid)
                if p:
                    p.payment_id = payment_id
                    if status:
                        p.status = status
                    s.commit()
            return

        # =========================
        # COMBOS (NUEVO)
        # =========================
        if external_reference.startswith("combo_purchase:"):
            cp_id = int(external_reference.split(":", 1)[1])
            with Session() as s:
                cp = s.get(ComboPurchase, cp_id)
                if cp:
                    cp.payment_id = payment_id
                    if status:
                        cp.status = status
                    s.commit()
            return

    except Exception as e:
        try:
            app.logger.exception("upsert purchase error")
        except Exception:
            pass

def mp_webhook():
    if request.method == "GET":
        return ("ok", 200)

    try:
        configured_secret = (app.config.get("MP_WEBHOOK_SECRET") or "").strip()
        incoming_secret = (request.args.get("secret") or "").strip()
        if configured_secret and configured_secret != incoming_secret:
            return {"ok": False, "error": "unauthorized"}, 401

        payload = request.get_json(silent=True) or {}
        topic = payload.get("type") or payload.get("topic")
        action = payload.get("action") or (payload.get("data", {}) or {}).get("action")

        provider_id = str(
            payload.get("id")
            or (payload.get("data") or {}).get("id")
            or request.headers.get("X-Idempotency-Key")
            or ""
        ).strip()
        if not provider_id:
            provider_id = "no-id-" + str(abs(hash(request.data)))

        with Session() as sx:
            exists = sx.execute(
                text("SELECT 1 FROM webhook_events WHERE provider_id = :pid"),
                {"pid": provider_id}
            ).first()
            if not exists:
                evt = WebhookEvent(
                    provider="mercadopago",
                    provider_id=provider_id,
                    topic=topic,
                    action=action,
                    payload=payload
                )
                sx.add(evt)
                sx.commit()

        payment_id = (
            request.args.get("id")
            or (payload.get("data", {}) or {}).get("id")
            or payload.get("id")
        )
        if payment_id:
            try:
                token = app.config["MP_ACCESS_TOKEN_PLATFORM"]
                pay = mp.get_payment(token, str(payment_id))
                if isinstance(pay, dict):
                    _upsert_purchase_from_payment(pay)
            except Exception:
                pass

        return {"ok": True}, 200
    except Exception as e:
        try:
            app.logger.exception("mp_webhook error")
        except Exception:
            pass
        return {"ok": False, "error": str(e)}, 200

app.add_url_rule("/webhooks/mercadopago", view_func=mp_webhook, methods=["POST"], endpoint="mp_webhook")
app.add_url_rule("/mp/webhook",            view_func=mp_webhook, methods=["POST", "GET"], endpoint="mp_webhook_legacy")

# -----------------------------------------------------------------------------
# Términos
# -----------------------------------------------------------------------------
@app.route("/terms")
def terms():
    return render_template("terms.html")

# -----------------------------------------------------------------------------
# Reportar apunte
# -----------------------------------------------------------------------------
@app.route("/note/<int:note_id>/report", methods=["POST"])
@login_required
def report_note(note_id):
    with Session() as s:
        n = s.get(Note, note_id)
        if not n:
            abort(404)
        if hasattr(n, "is_reported"):
            n.is_reported = True
            s.commit()
    flash("Gracias por tu reporte. Un administrador lo revisará.")
    flash("✅ Pago aprobado, ya podés descargar.")
    return redirect(url_for("note_detail", note_id=note_id, _anchor='download', paid=1))

# -----------------------------------------------------------------------------
# Taxonomías académicas (dropdowns) + creación "aprendida"
# -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return (s or "").strip()

@app.get("/api/academics/universities")
def api_list_universities():
    with Session() as s:
        rows = s.execute(select(University).order_by(University.name)).scalars().all()
        return jsonify([{"id": u.id, "name": u.name} for u in rows])

@app.get("/api/academics/faculties")
def api_list_faculties():
    uid = request.args.get("university_id", type=int)
    with Session() as s:
        q = select(Faculty)
        if uid:
            q = q.where(Faculty.university_id == uid)
        rows = s.execute(q.order_by(Faculty.name)).scalars().all()
        return jsonify([{"id": f.id, "name": f.name, "university_id": f.university_id} for f in rows])

@app.get("/api/academics/careers")
def api_list_careers():
    fid = request.args.get("faculty_id", type=int)
    with Session() as s:
        q = select(Career)
        if fid:
            q = q.where(Career.faculty_id == fid)
        rows = s.execute(q.order_by(Career.name)).scalars().all()
        return jsonify([{"id": c.id, "name": c.name, "faculty_id": c.faculty_id} for c in rows])

@app.post("/api/academics/universities")
def api_create_university():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error":"name required"}), 400
    with Session() as s:
        exists = s.execute(
            select(University).where(func.lower(University.name) == name.lower())
        ).scalar_one_or_none()
        if exists:
            return jsonify({"id": exists.id, "name": exists.name}), 200
        u = University(name=name)
        s.add(u); s.commit()
        return jsonify({"id": u.id, "name": u.name}), 201

@app.post("/api/academics/faculties")
def api_create_faculty():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    university_id = data.get("university_id")
    if not name or not university_id:
        return jsonify({"error":"name and university_id required"}), 400
    with Session() as s:
        f = Faculty(name=name, university_id=int(university_id))
        s.add(f); s.commit()
        return jsonify({"id": f.id, "name": f.name, "university_id": f.university_id}), 201

@app.post("/api/academics/careers")
def api_create_career():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    faculty_id = data.get("faculty_id")
    if not name or not faculty_id:
        return jsonify({"error":"name and faculty_id required"}), 400
    with Session() as s:
        c = Career(name=name, faculty_id=int(faculty_id))
        s.add(c); s.commit()
        return jsonify({"id": c.id, "name": c.name, "faculty_id": c.faculty_id}), 201

# -----------------------------------------------------------------------------
# Foto de perfil
# -----------------------------------------------------------------------------
@app.route("/profile/upload_image", methods=["POST"])
@login_required
def upload_profile_image():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        flash("Formato no permitido. Usá PNG o JPG.")
        return redirect(url_for("profile"))

    dest_dir = os.path.join(app.static_folder, "uploads", "profile_images")
    os.makedirs(dest_dir, exist_ok=True)

    ext = ".jpg"
    if file.filename.lower().endswith(".png"):
        ext = ".png"

    filename = f"user_{current_user.id}{ext}"
    file.save(os.path.join(dest_dir, filename))

    with Session() as s:
        u = s.get(User, current_user.id)
        if hasattr(u, "imagen_de_perfil"):
            u.imagen_de_perfil = filename
        else:
            u.profile_image = filename
        s.commit()

    flash("📸 Foto actualizada con éxito")
    return redirect(url_for("profile"))

# -----------------------------------------------------------------------------
# Cambio de contraseña MANUAL (sólo si corresponde)
# -----------------------------------------------------------------------------
@app.route("/profile/change_password", methods=["POST"])
@login_required
def change_password():
    current_pw = request.form.get("current_password", "")
    new_pw = request.form.get("new_password", "")
    confirm_pw = request.form.get("confirm_password", "")

    if len(new_pw) < 8:
        flash("La nueva contraseña debe tener al menos 8 caracteres.", "danger")
        return redirect(url_for("profile"))

    if new_pw != confirm_pw:
        flash("La confirmación no coincide.", "danger")
        return redirect(url_for("profile"))

    try:
        with Session() as s:
            user_obj = s.execute(select(User).where(User.id == current_user.id)).scalar_one()
            user_obj.password_hash = generate_password_hash(new_pw)
            s.commit()
    except Exception as e:
        flash("Error al actualizar la contraseña: {}".format(e), "danger")
        return redirect(url_for("profile"))

    flash("¡Contraseña actualizada correctamente!", "success")
    return redirect(url_for("profile"))

# -----------------------------------------------------------------------------
# Ayuda
# -----------------------------------------------------------------------------
@app.route("/help/mercadopago")
def help_mp():
    return render_template("help/mp_linking.html")

@app.route("/help/comisiones")
def help_commissions():
    return render_template("help/commissions.html")

# -----------------------------------------------------------------------------
# HUB DE ADMIN + MINI APIs
# -----------------------------------------------------------------------------
@app.get("/admin/hub")
@login_required
@admin_required
def admin_hub():
    return render_template("admin/hub.html")

# Admin HUB - usuarios
@app.route("/admin/api/users", methods=["GET", "POST"], endpoint="admin_api_users_list")
@login_required
@admin_required
def admin_api_users_list():
    q = (request.args.get("q") or "").strip()
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        q = (payload.get("q") or q or "").strip()

    limit = request.args.get("limit", type=int) or 100

    with Session() as s:
        stmt = select(User).order_by(desc(User.id)).limit(limit)
        if q:
            like = f"%{q}%"
            stmt = select(User).where(
                or_(User.name.ilike(like), User.email.ilike(like))
            ).order_by(desc(User.created_at)).limit(limit)

        rows = s.execute(stmt).scalars().all()

        items = []
        for u in rows:
            items.append({
                "id": u.id,
                "name": u.name or "",
                "email": u.email or "",
                "created_at": (u.created_at.isoformat() if getattr(u, "created_at", None) else None),
                "university": getattr(u, "university", "") or "",
                "faculty": getattr(u, "faculty", "") or "",
                "career": getattr(u, "career", "") or "",
                "is_active": bool(getattr(u, "is_active", True)),
                "is_admin": bool(getattr(u, "is_admin", False)),
                "is_blocked": bool(getattr(u, "is_blocked", False)),
            })

    return jsonify({"items": items, "list": items})


# ----------------------------------------------------------------------
# Admin HUB - detalle de usuario (panel con movimientos)
# ----------------------------------------------------------------------
@app.get("/admin/user/<int:user_id>")
@login_required
@admin_required
def admin_user_detail_page(user_id):
    """
    Renderiza la página de detalle de un usuario en el panel admin.
    El frontend llama luego a /admin/api/users/<id>/detail para traer datos.
    """
    return render_template("admin/user_detail.html", user_id=user_id)


@app.get("/admin/api/users/<int:user_id>/detail")
@login_required
@admin_required
def admin_api_user_detail(user_id):
    """
    Devuelve resumen de movimientos del usuario:
    - compras pagas (cantidad y total)
    - descargas gratuitas
    - si es vendedor y cuántos apuntes subió
    - ventas realizadas y distribución de comisiones
    """
    with Session() as s:
        u = s.get(User, user_id)
        if not u:
            return jsonify({"ok": False, "error": "not_found"}), 404

        # ----------------------------------------
        # Compras pagas aprobadas como comprador
        # ----------------------------------------
        paid_count, paid_cents = s.execute(
            select(
                func.count(Purchase.id),
                func.coalesce(func.sum(Purchase.amount_cents), 0)
            ).where(
                Purchase.buyer_id == user_id,
                Purchase.status == "approved",
                Purchase.amount_cents > 0
            )
        ).one()
        paid_count = int(paid_count or 0)
        paid_cents = int(paid_cents or 0)

        # ----------------------------------------
        # Descargas gratuitas (compras en 0 aprobadas)
        # ----------------------------------------
        free_count = s.execute(
            select(func.count(Purchase.id)).where(
                Purchase.buyer_id == user_id,
                Purchase.status == "approved",
                Purchase.amount_cents == 0
            )
        ).scalar_one_or_none() or 0
        free_count = int(free_count)

        # ----------------------------------------
        # Ventas como vendedor (compras aprobadas de sus apuntes)
        # ----------------------------------------
        sold_count, sold_gross_cents = s.execute(
            select(
                func.count(Purchase.id),
                func.coalesce(func.sum(Purchase.amount_cents), 0)
            ).join(Note, Note.id == Purchase.note_id).where(
                Note.seller_id == user_id,
                Purchase.status == "approved"
            )
        ).one()
        sold_count = int(sold_count or 0)
        sold_gross_cents = int(sold_gross_cents or 0)

        # ----------------------------------------
        # Apuntes subidos (para saber cuántos tiene como vendedor)
        # ----------------------------------------
        notes_uploaded_count = s.execute(
            select(func.count(Note.id)).where(Note.seller_id == user_id)
        ).scalar_one_or_none() or 0
        notes_uploaded_count = int(notes_uploaded_count)
        is_seller = notes_uploaded_count > 0

        # ----------------------------------------
        # Distribución de comisiones sobre lo vendido
        # ----------------------------------------
        mp_commission_cents  = int(round(sold_gross_cents * float(MP_COMMISSION_RATE)))
        apy_commission_cents = int(round(sold_gross_cents * float(APY_COMMISSION_RATE)))
        net_cents_for_seller = sold_gross_cents - mp_commission_cents - apy_commission_cents

        return jsonify({
            "ok": True,
            "user": {
                "id": u.id,
                "name": u.name or "",
                "email": u.email or "",
                "university": getattr(u, "university", "") or "",
                "faculty": getattr(u, "faculty", "") or "",
                "career": getattr(u, "career", "") or "",
                "created_at": (u.created_at.isoformat() if getattr(u, "created_at", None) else None),
                "is_active": bool(getattr(u, "is_active", True)),
                "is_admin": bool(getattr(u, "is_admin", False)),
                "is_blocked": bool(getattr(u, "is_blocked", False)),
            },
            "stats": {
                "paid_purchases_count": paid_count,
                "paid_purchases_cents": paid_cents,
                "free_downloads_count": free_count,
                "sold_notes_count": sold_count,
                "sold_gross_cents": sold_gross_cents,
                "notes_uploaded_count": notes_uploaded_count,  # 👈 nuevo
                "is_seller": is_seller,
                "mp_commission_cents": mp_commission_cents,
                "apy_commission_cents": apy_commission_cents,
                "net_cents_for_seller": net_cents_for_seller,
            }
        })


# ----------------------------------------------------------------------
# Admin HUB - gestión de usuarios (bloquear / desbloquear / eliminar)
# ----------------------------------------------------------------------
@app.post("/admin/api/users/<int:user_id>/block")
@login_required
@admin_required
def admin_api_users_block(user_id):
    with Session() as s:
        u = s.get(User, user_id)
        if not u:
            return jsonify({"ok": False, "error": "not_found"}), 404

        # No permitir bloquearse a uno mismo
        if u.id == current_user.id:
            return jsonify({"ok": False, "error": "cannot_block_self"}), 400

        u.is_active = False
        if hasattr(u, "is_blocked"):
            u.is_blocked = True
        s.commit()

    return jsonify({"ok": True, "status": "blocked"})


@app.post("/admin/api/users/<int:user_id>/unblock")
@login_required
@admin_required
def admin_api_users_unblock(user_id):
    with Session() as s:
        u = s.get(User, user_id)
        if not u:
            return jsonify({"ok": False, "error": "not_found"}), 404

        u.is_active = True
        if hasattr(u, "is_blocked"):
            u.is_blocked = False
        s.commit()

    return jsonify({"ok": True, "status": "unblocked"})


@app.post("/admin/api/users/<int:user_id>/delete")
@login_required
@admin_required
def admin_api_users_delete(user_id):
    with Session() as s:
        u = s.get(User, user_id)
        if not u:
            return jsonify({"ok": False, "error": "not_found"}), 404

        # No permitir borrarse a uno mismo ni borrar admins
        if u.id == current_user.id:
            return jsonify({"ok": False, "error": "cannot_delete_self"}), 400
        if getattr(u, "is_admin", False):
            return jsonify({"ok": False, "error": "cannot_delete_admin"}), 400

        # Si tiene apuntes o compras asociadas, mejor bloquear en vez de borrar
        has_notes = bool(getattr(u, "notes", []) or [])
        has_purchases = bool(getattr(u, "purchases", []) or [])
        if has_notes or has_purchases:
            return jsonify({"ok": False, "error": "has_related_data"}), 400

        s.delete(u)
        s.commit()

    return jsonify({"ok": True, "status": "deleted"})


# Admin HUB - apuntes (info general)
@app.get("/admin/api/notes")
@login_required
@admin_required
def admin_api_notes():
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", type=int) or 100

    with Session() as s:
        stmt = select(Note).where(Note.is_active == True)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(or_(
                Note.title.ilike(like),
                Note.description.ilike(like),
                Note.university.ilike(like),
                Note.faculty.ilike(like),
                Note.career.ilike(like),
            ))
        stmt = stmt.order_by(desc(Note.created_at)).limit(limit)

        notes = s.execute(stmt).scalars().all()

        seller_ids = list({n.seller_id for n in notes if n.seller_id})
        sellers = {}
        if seller_ids:
            sellers_rows = s.execute(select(User.id, User.name).where(User.id.in_(seller_ids))).all()
            sellers = {i: n for i, n in sellers_rows}

        data = []
        for n in notes:
            data.append({
                "id": n.id,
                "title": n.title,
                "price_cents": n.price_cents,
                "seller_name": sellers.get(n.seller_id, ""),
                "university": n.university,
                "faculty": n.faculty,
                "career": n.career,
                "created_at": (n.created_at.isoformat() if getattr(n, "created_at", None) else None),
            })
        return jsonify({"items": data})

@app.post("/admin/api/notes/<int:note_id>/delete")
@login_required
@admin_required
def admin_api_notes_delete(note_id):
    with Session() as s:
        n = s.get(Note, note_id)
        if not n:
            return jsonify({"ok": False, "error": "not_found"}), 404
        n.is_active = False
        s.commit()
    return jsonify({"ok": True})

# Admin HUB - archivos (descarga PDFs)
@app.get("/admin/api/files")
@login_required
@admin_required
def admin_api_files():
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", type=int) or 100

    with Session() as s:
        stmt = select(Note).where(Note.is_active == True)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(or_(
                Note.title.ilike(like),
                Note.description.ilike(like),
                Note.university.ilike(like),
                Note.faculty.ilike(like),
                Note.career.ilike(like)
            ))
        stmt = stmt.order_by(desc(Note.created_at)).limit(limit)
        notes = s.execute(stmt).scalars().all()

        data = []
        for n in notes:
            data.append({
                "id": n.id,
                "title": n.title,
                "file_path": n.file_path,
                "download_url": url_for("admin_download_note", note_id=n.id),
                "created_at": (n.created_at.isoformat() if getattr(n, "created_at", None) else None),
                "university": n.university,
                "faculty": n.faculty,
                "career": n.career,
                "price_cents": n.price_cents
            })
    return jsonify({"items": data})


# Admin HUB - contenido unificado (apunte + archivo)
@app.get("/admin/api/content")
@login_required
@admin_required
def admin_api_content():
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", type=int) or 120

    with Session() as s:
        stmt = select(Note).where(Note.is_active == True)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(or_(
                Note.title.ilike(like),
                Note.description.ilike(like),
                Note.university.ilike(like),
                Note.faculty.ilike(like),
                Note.career.ilike(like),
            ))

        stmt = stmt.order_by(desc(Note.created_at)).limit(limit)
        notes = s.execute(stmt).scalars().all()

        seller_ids = list({n.seller_id for n in notes if n.seller_id})
        sellers = {}
        if seller_ids:
            sellers_rows = s.execute(select(User.id, User.name).where(User.id.in_(seller_ids))).all()
            sellers = {i: n for i, n in sellers_rows}

        items = []
        for n in notes:
            items.append({
                "id": n.id,
                "title": n.title,
                "price_cents": int(getattr(n, "price_cents", 0) or 0),
                "seller_name": sellers.get(n.seller_id, ""),
                "university": n.university,
                "faculty": n.faculty,
                "career": n.career,
                "file_path": getattr(n, "file_path", None),
                "download_url": url_for("admin_download_note", note_id=n.id),
                "created_at": (n.created_at.isoformat() if getattr(n, "created_at", None) else None),
            })

    return jsonify({"items": items})


# Admin HUB - moderación (apuntes + combos)
@app.get("/admin/api/moderation")
@login_required
@admin_required
def admin_api_moderation():
    """Devuelve apuntes y combos por estado para mostrar en el Hub (sin salir de la pantalla)."""
    status = (request.args.get("status") or "pending_manual").strip()
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", type=int) or 200

    with Session() as s:
        # ------- Notes -------
        # Solo mostramos no borrados si existe deleted_at
        notes_stmt = select(Note)
        if hasattr(Note, "deleted_at"):
            notes_stmt = notes_stmt.where(Note.deleted_at.is_(None))
        if hasattr(Note, "moderation_status"):
            notes_stmt = notes_stmt.where(Note.moderation_status == status)
        else:
            notes_stmt = notes_stmt.where(False)

        if q:
            like = f"%{q}%"
            notes_stmt = notes_stmt.where(or_(
                Note.title.ilike(like),
                Note.description.ilike(like),
                Note.university.ilike(like),
                Note.faculty.ilike(like),
                Note.career.ilike(like),
            ))

        if hasattr(Note, "created_at"):
            notes_stmt = notes_stmt.order_by(desc(Note.created_at))
        else:
            notes_stmt = notes_stmt.order_by(desc(Note.id))
        notes_stmt = notes_stmt.limit(limit)

        notes = s.execute(notes_stmt).scalars().all()

        seller_ids = list({n.seller_id for n in notes if getattr(n, "seller_id", None)})
        sellers = {}
        if seller_ids:
            sellers_rows = s.execute(select(User.id, User.name).where(User.id.in_(seller_ids))).all()
            sellers = {i: n for i, n in sellers_rows}

        notes_out = []
        for n in notes:
            notes_out.append({
                "id": n.id,
                "title": getattr(n, "title", "") or "",
                "price_cents": int(getattr(n, "price_cents", 0) or 0),
                "seller_name": sellers.get(getattr(n, "seller_id", None), ""),
                "university": getattr(n, "university", "") or "",
                "faculty": getattr(n, "faculty", "") or "",
                "career": getattr(n, "career", "") or "",
                "moderation_status": getattr(n, "moderation_status", "") or "",
                "moderation_reason": getattr(n, "moderation_reason", None),
                "ai_decision": getattr(n, "ai_decision", None),
                "ai_confidence": getattr(n, "ai_confidence", None),
            })

        # ------- Combos -------
        combos_stmt = select(Combo)
        if hasattr(Combo, "moderation_status"):
            combos_stmt = combos_stmt.where(Combo.moderation_status == status)
        else:
            combos_stmt = combos_stmt.where(False)

        if q:
            like = f"%{q}%"
            if hasattr(Combo, "title"):
                combos_stmt = combos_stmt.where(Combo.title.ilike(like))

        if hasattr(Combo, "created_at"):
            combos_stmt = combos_stmt.order_by(desc(Combo.created_at))
        else:
            combos_stmt = combos_stmt.order_by(desc(Combo.id))
        combos_stmt = combos_stmt.limit(limit)

        combos = s.execute(combos_stmt).scalars().all()

        combo_seller_ids = list({c.seller_id for c in combos if getattr(c, "seller_id", None)})
        combo_sellers = {}
        if combo_seller_ids:
            rows = s.execute(select(User.id, User.name).where(User.id.in_(combo_seller_ids))).all()
            combo_sellers = {i: n for i, n in rows}

        combos_out = []
        for c in combos:
            combos_out.append({
                "id": c.id,
                "title": getattr(c, "title", "") or "",
                "price_cents": int(getattr(c, "price_cents", 0) or 0),
                "seller_name": combo_sellers.get(getattr(c, "seller_id", None), ""),
                "moderation_status": getattr(c, "moderation_status", "") or "",
                "moderation_reason": getattr(c, "moderation_reason", None),
                "ai_decision": getattr(c, "ai_decision", None),
                "ai_confidence": getattr(c, "ai_confidence", None),
            })

    return jsonify({"ok": True, "status": status, "notes": notes_out, "combos": combos_out})


@app.post("/admin/api/content/<int:note_id>/delete")
@login_required
@admin_required
def admin_api_content_delete(note_id: int):
    """Borra (soft) el apunte y hace best-effort de borrar el archivo asociado."""
    with Session() as s:
        n = s.get(Note, note_id)
        if not n:
            return jsonify({"ok": False, "error": "not_found"}), 404

        n.is_active = False
        if hasattr(n, "deleted_at"):
            n.deleted_at = datetime.utcnow()

        # borrar archivo (best-effort)
        try:
            fp = getattr(n, "file_path", None)
            if fp:
                if gcs_bucket:
                    gcs_delete_blob(fp)
                else:
                    local = os.path.join(app.config["UPLOAD_FOLDER"], fp)
                    if os.path.exists(local):
                        os.remove(local)
        except Exception:
            pass

        s.commit()

    return jsonify({"ok": True})


@app.get("/admin/download/<int:note_id>")
@login_required
@admin_required
def admin_download_note(note_id):
    with Session() as s:
        n = s.get(Note, note_id)
        if not n or not n.is_active:
            abort(404)

        # Si el archivo está en GCS (nuevo esquema)
        if gcs_bucket and n.file_path and "/" in n.file_path:
            signed_url = gcs_generate_signed_url(n.file_path, seconds=600)
            return redirect(signed_url)

        # Fallback: archivo local (viejos apuntes, entorno dev sin GCS, etc.)
        return send_from_directory(app.config["UPLOAD_FOLDER"], n.file_path, as_attachment=True)



# -----------------------------------------------------------------------------
# Actualizar datos académicos (redirige a form estilo complete_profile)
# -----------------------------------------------------------------------------
@app.post("/profile/update_academics")
@login_required
def update_academics():
    university = (request.form.get("university") or "").strip()
    faculty    = (request.form.get("faculty") or "").strip()
    career     = (request.form.get("career") or "").strip()
    seller_contact = (request.form.get("seller_contact") or "").strip()

    if not (university and faculty and career):
        flash("Completá todos los campos para actualizar tus datos académicos.", "warning")
        return redirect(url_for("profile"))

    with Session() as s:
        u = s.get(User, current_user.id)
        u.university = university
        u.faculty    = faculty
        u.career     = career
        if seller_contact:
            u.seller_contact = seller_contact
        s.commit()

    flash("✅ Datos académicos actualizados correctamente.", "success")
    return redirect(url_for("profile"))


@app.get("/update_academics")
@login_required
def update_academics_get():
    """Muestra el mismo formulario que /complete_profile pero para actualizar."""
    return render_template(
        "complete_profile.html",
        name=current_user.name,
        mode="update"
    )

@app.post("/update_academics")
@login_required
def update_academics_post():
    university = (request.form.get("university") or "").strip()
    faculty    = (request.form.get("faculty") or "").strip()
    career     = (request.form.get("career") or "").strip()
    seller_contact = (request.form.get("seller_contact") or "").strip()

    if not (university and faculty and career):
        flash("Completá todos los campos antes de guardar.", "warning")
        return redirect(url_for("update_academics_get"))

    with Session() as s:
        u = s.get(User, current_user.id)
        u.university = university
        u.faculty    = faculty
        u.career     = career
        if seller_contact:
            u.seller_contact = seller_contact
        s.commit()

    flash("✅ Datos académicos actualizados.", "success")
    return redirect(url_for("profile"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# ------------------------------ Combos ------------------------------
from apuntesya2.models import ComboNote

def _combo_buyer_price_cents(combo: Combo) -> int:
    """Precio final al comprador para un combo.

    Regla: el vendedor ingresa el NETO (lo que quiere recibir) y al comprador se le
    suma el % de comisiones configurado (TOTAL_FEE_RATE). Si es gratis => 0.
    """
    if not combo:
        return 0
    net = int(getattr(combo, "seller_net_cents", 0) or 0)
    if net <= 0:
        return 0
    return int(published_from_net_cents(net))


@app.route("/profile/combos")
@login_required
def profile_combos():
    with Session() as s:
        combos = s.execute(
        select(Combo)
        .where(
            Combo.seller_id == current_user.id,
            Combo.is_active == True,
        )
        .order_by(Combo.created_at.desc())
        ).scalars().all()
    return render_template("profile_combos.html", combos=combos, buyer_price=_combo_buyer_price_cents)

import math
from datetime import datetime, timedelta
from flask import render_template, request, flash, redirect, url_for, abort
from flask_login import login_required, current_user
from sqlalchemy import select

# Asegurate de tener ComboNote importado correctamente:
# from apuntesya2.models import ComboNote
# (NO uses from ..models dentro de app.py si te rompe imports)
from apuntesya2.models import Note, Combo, ComboNote, Notification

@app.route("/combos/create", methods=["GET", "POST"], endpoint="create_combo")
@login_required
def combo_create():
    with Session() as s:
        # 1) Traer apuntes del usuario que realmente puedan ir en combo
        notes = s.execute(
            select(Note)
            .where(
                Note.seller_id == current_user.id,
                Note.deleted_at.is_(None),
                Note.is_active == True,
                Note.moderation_status == "approved",
            )
            .order_by(Note.created_at.desc())
        ).scalars().all()

        if request.method == "POST":
            title = (request.form.get("title") or "").strip()
            description = (request.form.get("description") or "").strip()

            # "price_net" llega en ARS (string)
            raw_price = (request.form.get("price_net") or "0").strip().replace(",", ".")
            try:
                price_net_cents = int(round(float(raw_price) * 100))
            except Exception:
                price_net_cents = 0

            note_ids = request.form.getlist("note_ids") or []
            note_ids = [int(x) for x in note_ids if str(x).isdigit()]

            if not title or not description or len(note_ids) < 2:
                flash("El combo debe tener título, descripción y al menos 2 apuntes.", "danger")
                return render_template("combo_create.html", notes=notes)

            # 2) Validar que esos apuntes sean del usuario y aprobados
            chosen = s.execute(
                select(Note).where(
                    Note.id.in_(note_ids),
                    Note.seller_id == current_user.id,
                    Note.deleted_at.is_(None),
                    Note.is_active == True,
                    Note.moderation_status == "approved",
                )
            ).scalars().all()

            if len(chosen) < 2:
                flash("Seleccioná al menos 2 apuntes aprobados.", "danger")
                return render_template("combo_create.html", notes=notes)

            # 3) Calcular precio final comprador (si neto = 0, es gratis)
            buyer_price_cents = published_from_net_cents(price_net_cents) if price_net_cents > 0 else 0

            # 4) Crear combo (IMPORTANTÍSIMO: setear seller_net_cents)
            combo = Combo(
                seller_id=current_user.id,
                title=title,
                description=description,
                seller_net_cents=max(price_net_cents, 0),
                price_cents=buyer_price_cents,
                is_active=True,
                moderation_status="approved",  # ya que solo deja elegir aprobados
                moderation_reason=None,
                created_at=datetime.utcnow(),
            )

            s.add(combo)
            s.flush()  # para obtener combo.id

            # 5) Crear relación combo_notes
            for n in chosen:
                s.add(ComboNote(combo_id=combo.id, note_id=n.id))

            # 6) Notificación opcional
            try:
                s.add(Notification(
                    user_id=current_user.id,
                    kind="success",
                    title="Combo creado",
                    body="Tu combo fue creado y ya está publicado."
                ))
            except Exception:
                pass

            s.commit()
            flash("Combo creado correctamente.", "success")
            return redirect(url_for("profile_combos"))

        return render_template("combo_create.html", notes=notes)


@app.route("/combos/<int:combo_id>/edit", methods=["GET", "POST"])
@login_required
def combo_edit(combo_id: int):
    with Session() as s:
        combo = s.get(Combo, combo_id)
        if not combo or combo.seller_id != current_user.id:
            abort(404)

        # Apuntes elegibles (mismos criterios que create)
        notes = s.execute(
            select(Note)
            .where(
                Note.seller_id == current_user.id,
                Note.deleted_at.is_(None),
                Note.is_active == True,
                Note.moderation_status == "approved",
            )
            .order_by(Note.created_at.desc())
        ).scalars().all()

        selected_ids = {cn.note_id for cn in getattr(combo, "combo_notes", [])}

        if request.method == "POST":
            title = (request.form.get("title") or "").strip()
            description = (request.form.get("description") or "").strip()
            raw_price = (request.form.get("price_net") or "0").strip().replace(",", ".")
            try:
                price_net_cents = int(round(float(raw_price) * 100))
            except Exception:
                price_net_cents = 0

            note_ids = request.form.getlist("note_ids") or []
            note_ids = [int(x) for x in note_ids if str(x).isdigit()]

            if not title or not description or len(note_ids) < 2:
                flash("El combo debe tener título, descripción y al menos 2 apuntes.", "danger")
                return render_template("combo_edit.html", combo=combo, notes=notes, selected_ids=selected_ids)

            chosen = s.execute(
                select(Note).where(
                    Note.id.in_(note_ids),
                    Note.seller_id == current_user.id,
                    Note.deleted_at.is_(None),
                    Note.is_active == True,
                    Note.moderation_status == "approved",
                )
            ).scalars().all()
            if len(chosen) < 2:
                flash("Seleccioná al menos 2 apuntes aprobados.", "danger")
                return render_template("combo_edit.html", combo=combo, notes=notes, selected_ids=selected_ids)

            # Precio final comprador
            buyer_price_cents = published_from_net_cents(price_net_cents) if price_net_cents > 0 else 0

            combo.title = title
            combo.description = description
            combo.seller_net_cents = max(price_net_cents, 0)
            combo.price_cents = buyer_price_cents

            # Reemplazar relaciones combo_notes
            try:
                combo.combo_notes.clear()
            except Exception:
                # fallback: delete explícito
                s.execute(text("DELETE FROM combo_notes WHERE combo_id = :cid"), {"cid": combo.id})

            for n in chosen:
                s.add(ComboNote(combo_id=combo.id, note_id=n.id))

            # Conservador: si se edita, vuelve a revisión manual
            try:
                combo.moderation_status = "pending_manual"
                combo.moderation_reason = "Combo actualizado por el vendedor (requiere revisión)."
                combo.manual_review_due_at = datetime.utcnow() + timedelta(hours=12)
            except Exception:
                pass

            s.commit()
            flash("Combo actualizado.", "success")
            return redirect(url_for("profile_combos"))

        return render_template("combo_edit.html", combo=combo, notes=notes, selected_ids=selected_ids)


@app.post("/combos/<int:combo_id>/delete")
@login_required
def combo_delete(combo_id: int):
    with Session() as s:
        combo = s.get(Combo, combo_id)
        if not combo or combo.seller_id != current_user.id:
            abort(404)
        combo.is_active = False
        s.commit()
    flash("Combo eliminado.", "success")
    return redirect(url_for("profile_combos"))


from flask import render_template, abort
from sqlalchemy import select
from sqlalchemy.orm import joinedload


@app.route("/combos/<int:combo_id>/buy", methods=["GET","POST"])
@login_required
def buy_combo(combo_id):
    with Session() as s:
        combo = s.get(Combo, combo_id)
        if not combo or (hasattr(combo, "is_active") and combo.is_active is False):
            abort(404)

        if getattr(combo, "moderation_status", "approved") != "approved":
            abort(404)

        if combo.seller_id == current_user.id:
            flash("No podés comprar tu propio combo.")
            return redirect(url_for("combo_detail", combo_id=combo.id))

        price_cents = _combo_buyer_price_cents(combo)

        # Mantener consistencia en DB (por si existen combos viejos)
        if (getattr(combo, "price_cents", 0) or 0) != price_cents:
            try:
                combo.price_cents = price_cents
                s.commit()
            except Exception:
                s.rollback()
        if price_cents <= 0:
            flash("Este combo es gratuito.")
            return redirect(url_for("combo_detail", combo_id=combo.id))

        seller = s.get(User, combo.seller_id)

        cp = ComboPurchase(
            buyer_id=current_user.id,
            combo_id=combo.id,
            status="pending",
            amount_cents=price_cents
        )
        s.add(cp)
        s.commit()

        price_ars = float(money_1_decimal(cents_to_amount(price_cents)))
        platform_fee_percent = 0.10
        marketplace_fee = float(money_1_decimal(price_ars * platform_fee_percent))

        back_urls = {
            "success": url_for("mp_return_combo", combo_id=combo.id, _external=True) + f"?external_reference=combo_purchase:{cp.id}",
            "failure": url_for("mp_return_combo", combo_id=combo.id, _external=True) + f"?external_reference=combo_purchase:{cp.id}",
            "pending": url_for("mp_return_combo", combo_id=combo.id, _external=True) + f"?external_reference=combo_purchase:{cp.id}",
        }

        try:
            seller_token = get_valid_seller_token(seller)
            if seller_token is None:
                flash("El vendedor no tiene Mercado Pago vinculado. No se puede procesar la compra.", "warning")
                return redirect(url_for("combo_detail", combo_id=combo.id))

            pref = mp.create_preference_for_seller_token(
                seller_access_token=seller_token,
                title=f"Combo: {combo.title}",
                unit_price=price_ars,
                quantity=1,
                marketplace_fee=marketplace_fee,
                external_reference=f"combo_purchase:{cp.id}",
                back_urls=back_urls,
                notification_url=url_for("mp_webhook", _external=True)
            )

            with Session() as s2:
                cp2 = s2.get(ComboPurchase, cp.id)
                if cp2:
                    cp2.preference_id = pref.get("id") or pref.get("preference_id")
                    s2.commit()

            init_point = pref.get("init_point") or pref.get("sandbox_init_point")
            return redirect(init_point)

        except Exception as e:
            flash(f"Error al crear preferencia en Mercado Pago: {e}")
            return redirect(url_for("combo_detail", combo_id=combo.id))


@app.route("/mp/return/combo/<int:combo_id>")
@login_required
def mp_return_combo(combo_id):
    flash("Pago en proceso. Si fue aprobado, el combo quedará disponible.", "info")
    return redirect(url_for("combo_detail", combo_id=combo_id))



@app.route("/combos/<int:combo_id>", endpoint="combo_detail")
def combo_detail(combo_id: int):
    with Session() as s:
        combo = (
            s.execute(
                select(Combo)
                .options(joinedload(Combo.seller))
                .where(Combo.id == combo_id)
            )
            .scalars()
            .first()
        )

        if not combo:
            abort(404)

        # is_active puede ser NULL en tu DB -> tratamos NULL como activo
        if hasattr(combo, "is_active") and (combo.is_active is False):
            abort(404)

        is_owner = getattr(current_user, "is_authenticated", False) and current_user.id == combo.seller_id
        is_admin = getattr(current_user, "is_authenticated", False) and getattr(current_user, "is_admin", False)

        # Público: solo approved. Dueño/admin pueden ver pending_review.
        if hasattr(combo, "moderation_status"):
            if combo.moderation_status != "approved" and not (is_owner or is_admin):
                abort(404)

        note_ids = (
            s.execute(select(ComboNote.note_id).where(ComboNote.combo_id == combo.id))
            .scalars()
            .all()
        )

        notes = []
        if note_ids:
            notes = (
                s.execute(select(Note).where(Note.id.in_(note_ids)))
                .scalars()
                .all()
            )

        seller = combo.seller

        buyer_price_cents = _combo_buyer_price_cents(combo)

        # Si el combo es viejo y el price_cents no coincide, lo actualizamos (best-effort)
        if (getattr(combo, "price_cents", 0) or 0) != buyer_price_cents:
            try:
                combo.price_cents = buyer_price_cents
                s.commit()
            except Exception:
                s.rollback()
        buyer_price = buyer_price_cents / 100.0  # <- precio final real (sin gross_price)

    return render_template(
        "combo_detail.html",
        combo=combo,
        seller=seller,
        notes=notes,
        buyer_price_cents=buyer_price_cents,
        buyer_price=buyer_price,
    )


if __name__ == "__main__":
    app.run(debug=True)

