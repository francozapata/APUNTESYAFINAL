# apuntesya2/app.py

import os
import uuid
import secrets
import math
import json
import base64
import re
import warnings
import smtplib
import ssl
import threading

from email.message import EmailMessage
from datetime import datetime, timedelta, timedelta
from urllib.parse import urlencode, urlparse
from functools import wraps
import boto3
from botocore.client import Config

# --- Log hygiene -------------------------------------------------------------
# Pydantic may emit a noisy warning in some environments when 3rd-party
# libraries pass builtins like `any` where a type is expected. This warning is
# harmless for our app (it only affects schema generation/validation in those
# libraries) but it clutters Render logs.
warnings.filterwarnings(
    "ignore",
    message=r".*<built-in function any> is not a Python type.*",
    category=UserWarning,
)

from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect, CSRFError
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, abort, jsonify, session
)
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, abort, send_file


# Optional: response compression (gzip)
try:
    from flask_compress import Compress
except Exception:
    Compress = None

from flask_login import (
    LoginManager, login_user, logout_user, current_user, login_required
)
from sqlalchemy import create_engine, select, or_, and_, func, text, desc, cast, Date
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, auth as fb_auth

# modelos
from apuntesya2.models import (
    Base,
    User,
    SiteSetting,
    DailyStat,
    Note,
    Purchase,
    University,
    Faculty,
    Career,
    AcademicSuggestion,
    WebhookEvent,
    Review,
    DownloadLog,
    Notification,
    Combo,
    ComboNote,
    ComboPurchase,
    AuditEvent,
    LegalAcceptanceAudit,
    AnalyticsEvent,
    Ticket,
    TicketEvent,
)

from apuntesya2.seed_unc_academics import seed_unc

# helpers MP
from apuntesya2 import mp

import random
# Pricing (single source of truth)
from apuntesya2.pricing import (
    published_from_net_cents,
    cents_to_amount,
    amount_to_cents,
    APY_RATE,
    breakdown_from_net,
    breakdown_from_published,
    money_1_decimal,
)

load_dotenv()

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)

# Enable gzip compression if available
if Compress:
    Compress(app)


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

# Static caching (browser) - 30 days by default
try:
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = int(os.getenv("STATIC_MAX_AGE_SEC", str(60*60*24*30)))
except Exception:
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 60*60*24*30


# -----------------------------------------------------------------------------
# Legal versions (TyC + Privacidad + Seguridad)
#
# Bump LEGAL_VERSION whenever you update any of the legal documents.
# Users will be prompted to accept again.
# -----------------------------------------------------------------------------
app.config["LEGAL_VERSION"] = os.getenv("LEGAL_VERSION", "2026-02-09")

# --- Security: upload size limit (100 MB) -------------------------------------
# Prevent abuse / accidental huge uploads. Adjust via MAX_UPLOAD_MB env.
try:
    _max_mb = int(os.getenv("MAX_UPLOAD_MB", "100"))
except Exception:
    _max_mb = 25
app.config["MAX_CONTENT_LENGTH"] = _max_mb * 1024 * 1024

# --- Security: CSRF protection ----------------------------------------------
csrf = CSRFProtect(app)


# --- Performance: tiny in-memory TTL cache (per-process) --------------------
_TTL_CACHE = {}  # key -> (expires_ts, value)

def _cache_get(key: str):
    try:
        exp, val = _TTL_CACHE.get(key, (0, None))
        if exp and exp > datetime.utcnow().timestamp():
            return val
    except Exception:
        pass
    return None

def _cache_set(key: str, value, ttl_sec: int):
    try:
        _TTL_CACHE[key] = (datetime.utcnow().timestamp() + int(ttl_sec), value)
    except Exception:
        pass


def _cache_invalidate_prefix(prefix: str):
    try:
        for k in list(_TTL_CACHE.keys()):
            if str(k).startswith(prefix):
                _TTL_CACHE.pop(k, None)
    except Exception:
        pass

# --- Security: simple rate limiting (per-process) ---------------------------
# NOTE: With multiple gunicorn workers this is per-worker, still useful as a first layer.
_RATE_BUCKET = {}  # (key)-> [timestamps]
def _rate_limit(key: str, limit: int, window_sec: int) -> bool:
    """Return True if allowed, False if rate limited."""
    now = datetime.utcnow().timestamp()
    bucket = _RATE_BUCKET.get(key, [])
    # keep only within window
    bucket = [t for t in bucket if now - t < window_sec]
    if len(bucket) >= limit:
        _RATE_BUCKET[key] = bucket
        return False
    bucket.append(now)
    _RATE_BUCKET[key] = bucket
    return True


from flask import send_from_directory

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon-32.png", mimetype="image/png")


@app.before_request
def _security_rate_limits():
    # Rate limit sensitive endpoints (per-process)
    path = request.path or ""

    # Allow GET by default, but limit heavy / enumeratable endpoints
    method = request.method

    ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
          or request.headers.get("X-Real-IP")
          or request.remote_addr
          or "unknown")

    # Downloads: limit GET to reduce scraping / enumeration
    if method == "GET":
        if path.startswith("/download/"):
            if not _rate_limit(f"dl:{ip}", 60, 60):
                return ("Too Many Requests", 429)
        if path.startswith("/combos/") and path.endswith("/download"):
            if not _rate_limit(f"dlc:{ip}", 30, 60):
                return ("Too Many Requests", 429)
        return None

    # Only limit POSTs to sensitive endpoints
    if method != "POST":
        return None

    path = request.path or ""
    ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
          or request.headers.get("X-Real-IP")
          or request.remote_addr
          or "unknown")
    # Tight limits for auth / uploads
    rules = [
        ("/login", 20, 300),                 # 20 per 5 min
        ("/register", 10, 300),              # 10 per 5 min
        ("/upload", 20, 600),                # 20 per 10 min
        ("/profile/upload_image", 20, 600),  # 20 per 10 min
        ("/profile/change_password", 10, 600),
        ("/api/academics/", 60, 300),         # 60 per 5 min (anti-spam suggestions)
    ]
    for prefix, limit, window in rules:
        if path.startswith(prefix):
            if not _rate_limit(f"{prefix}:{ip}", limit, window):
                return ("Demasiadas solicitudes. Probá de nuevo en unos minutos.", 429)
            break
    return None

@app.after_request
def _security_headers(resp):
    # Basic hardening headers (safe defaults that shouldn't break templates)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    # Reduce cross-origin leakage
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    # CSP: keep a stricter default for the whole site, and a slightly more permissive
    # policy only for /login where Google/Firebase scripts are required.
    # We keep 'unsafe-inline' because templates currently use inline scripts/styles.
    path = (getattr(request, "path", "") or "")

    # Firebase/Google sign-in may iframe/redirect via the project's auth domain
    # (e.g. https://<project>.firebaseapp.com and/or https://<project>.web.app).
    # Allow only for /login (route-specific CSP).
    firebase_domain_urls: list[str] = []
    try:
        pid = (os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
        if pid:
            firebase_domain_urls.extend([
                f"https://{pid}.firebaseapp.com",
                f"https://{pid}.web.app",
            ])
        extra_domains = [d.strip() for d in str(os.getenv("FIREBASE_AUTH_DOMAINS", "")).split(",") if d.strip()]
        for d in extra_domains:
            # Normalize to https://
            if d.startswith("http://") or d.startswith("https://"):
                firebase_domain_urls.append(d)
            else:
                firebase_domain_urls.append(f"https://{d}")
        # De-dup while preserving order
        seen = set()
        firebase_domain_urls = [u for u in firebase_domain_urls if not (u in seen or seen.add(u))]
    except Exception:
        firebase_domain_urls = []
    firebase_domains_csp = " ".join(firebase_domain_urls)

    # NOTE: Allow Google profile photos (Firebase/Google Sign-In commonly returns *.googleusercontent.com)
    csp_default = (
        "default-src 'self'; "
        "img-src 'self' data: blob: https://*.r2.cloudflarestorage.com https://r2.cloudflarestorage.com https://*.googleusercontent.com https://lh3.googleusercontent.com; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "frame-src 'self'; "
        "frame-ancestors 'none'"
    )

    # IMPORTANT: Google login uses ES-module imports from https://www.gstatic.com/firebasejs/...
    # and may load https://apis.google.com/js/api.js. Firebase Auth calls Google APIs.
    csp_login = (
        "default-src 'self'; "
        "img-src 'self' data: blob: https://www.gstatic.com https://*.r2.cloudflarestorage.com https://r2.cloudflarestorage.com https://*.googleusercontent.com https://lh3.googleusercontent.com; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline' https://www.gstatic.com https://apis.google.com https://accounts.google.com; "
        "font-src 'self' data: https://www.gstatic.com; "
        "connect-src 'self' https://www.googleapis.com https://*.googleapis.com "
        "https://identitytoolkit.googleapis.com https://securetoken.googleapis.com "
        f"https://accounts.google.com https://www.gstatic.com https://apis.google.com {firebase_domains_csp}; "
        f"frame-src 'self' https://accounts.google.com {firebase_domains_csp}; "
        "frame-ancestors 'none'"
    )

    resp.headers.setdefault(
        "Content-Security-Policy",
        csp_login if path.startswith("/login") else csp_default
    )
    # HSTS (Render serves HTTPS). Only set if request is https to avoid local dev issues.
    try:
        if request.is_secure:
            resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    except Exception:
        pass
    return resp


# --- Security: PDF validation helpers ----------------------------------------
def _validate_uploaded_pdf(path: str, max_pages: int = 1200) -> tuple[bool, str]:
    """Best-effort validation for uploaded PDFs.

    - Validates header magic
    - Opens with PyMuPDF to ensure it's parseable
    - Blocks PDFs with embedded JavaScript actions (basic)
    - Caps page count to avoid DoS files
    """
    try:
        with open(path, "rb") as fp:
            if fp.read(4) != b"%PDF":
                return (False, "PDF inválido (header).")
    except Exception:
        return (False, "No se pudo leer el archivo.")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)

        # Page cap (avoid pathological files)
        try:
            if doc.page_count and int(doc.page_count) > int(max_pages):
                try:
                    doc.close()
                except Exception:
                    pass
                return (False, f"El PDF tiene demasiadas páginas (máx {max_pages}).")
        except Exception:
            pass

        # Basic JS detection (not perfect, but blocks common embedded scripts)
        try:
            js = None
            try:
                js = doc.get_javascript()
            except Exception:
                js = None
            if js:
                try:
                    doc.close()
                except Exception:
                    pass
                return (False, "El PDF contiene JavaScript embebido (bloqueado).")
        except Exception:
            pass

        try:
            doc.close()
        except Exception:
            pass
    except Exception:
        return (False, "PDF dañado o no compatible.")

    return (True, "ok")


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


@app.context_processor
def legal_ctx():
    """Expose legal acceptance state to templates (navbar banner, etc.)."""
    try:
        ver = (app.config.get("LEGAL_VERSION") or "").strip()
    except Exception:
        ver = ""
    needs = False
    try:
        if getattr(current_user, "is_authenticated", False):
            needs = _user_needs_legal_accept(int(current_user.id))
    except Exception:
        needs = False
    return dict(LEGAL_VERSION=ver, legal_version=ver, legal_needs_accept=needs)

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
from sqlalchemy.pool import NullPool, QueuePool

DEFAULT_DB = f"sqlite:///{os.path.join(BASE_DATA, 'apuntesya.db')}"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB)

# Si la URL viene de Postgres (Supabase), usamos el driver psycopg (v3)
if DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg://", 1)

engine_kwargs = {"pool_pre_ping": True, "future": True}

# DB pooling:
# - SQLite local: keep check_same_thread off (single process dev)
# - Postgres/Supabase: use a small QueuePool to avoid per-request connect latency
if DB_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # If you use Supabase's pooler (PgBouncer), pooling is still fine here.
    engine_kwargs.update({
        "poolclass": QueuePool,
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE_SEC", "300")),
    })

engine = create_engine(DB_URL, **engine_kwargs)



# -----------------------------------------------------------------------------
# Object storage (Cloudflare R2 - S3 compatible)
#
# ⚠️ Backwards-compat: the codebase historically used GCS and checks
# `gcs_bucket` to decide whether storage is remote. To keep changes minimal,
# we keep the same variable names but they now point to R2.
# -----------------------------------------------------------------------------
R2_ACCOUNT_ID = (os.getenv("R2_ACCOUNT_ID") or "").strip()
R2_ACCESS_KEY_ID = (os.getenv("R2_ACCESS_KEY_ID") or "").strip()
R2_SECRET_ACCESS_KEY = (os.getenv("R2_SECRET_ACCESS_KEY") or "").strip()
R2_BUCKET_NAME = (os.getenv("R2_BUCKET_NAME") or "").strip()

# Optional fallback if you kept old env var names around
if not R2_BUCKET_NAME:
    R2_BUCKET_NAME = (os.getenv("GCS_BUCKET_NAME") or "").strip()

gcs_client = None  # now: boto3 S3 client (R2)
gcs_bucket = None  # now: bucket name string (R2)

if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME:
    try:
        gcs_client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        gcs_bucket = R2_BUCKET_NAME
        print(f"[R2] Bucket configurado: {R2_BUCKET_NAME}")
    except Exception as e:
        print("[R2] ERROR al inicializar:", e)
else:
    print("[R2] No configurado (faltan variables de entorno)")


import os, tempfile
from io import BytesIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ----------------------------
# R2 configuration (S3 compatible)
# ----------------------------
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "").strip()

r2_client = None
r2_bucket_enabled = False

def _init_r2():
    global r2_client, r2_bucket_enabled

    if not (R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME):
        r2_client = None
        r2_bucket_enabled = False
        return

    # Cloudflare R2: región puede ser "auto". Importante: signature_version="s3v4"
    r2_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name=os.getenv("R2_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )
    r2_bucket_enabled = True

_init_r2()


def r2_upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    """Upload raw bytes to R2 under a given key."""
    if not r2_bucket_enabled or r2_client is None:
        raise RuntimeError("R2 not configured (missing env vars).")

    r2_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
        # Cache control: previews se pueden cachear un rato
        CacheControl="public, max-age=300",
    )


def r2_download_bytes(key: str) -> bytes:
    """Download object bytes from R2."""
    if not r2_bucket_enabled or r2_client is None:
        raise RuntimeError("R2 not configured (missing env vars).")

    obj = r2_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
    return obj["Body"].read()


def r2_download_to_temp(key: str, suffix: str = ".pdf") -> str:
    """Download object from R2 to a temporary local file and return its path."""
    if not r2_bucket_enabled or r2_client is None:
        raise RuntimeError("R2 not configured (missing env vars).")

    fd, path = tempfile.mkstemp(prefix="ay_", suffix=suffix)
    os.close(fd)

    try:
        r2_client.download_file(Bucket=R2_BUCKET_NAME, Key=key, Filename=path)
    except Exception:
        # Limpieza si falló
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        raise

    return path


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
            # roles (user/admin/superadmin)
            if 'role' not in cols:
                add_cols.append("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'")
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

            # Backfill role from legacy is_admin flag
            try:
                cols2 = {c['name'] for c in insp.get_columns('users')}
                if 'role' in cols2:
                    conn.execute(text("UPDATE users SET role='admin' WHERE (role IS NULL OR role='' OR role='user') AND is_admin=1"))
            except Exception:
                pass

        # notes: moderation + preview metadata
        if insp.has_table('notes'):
            cols = {c['name'] for c in insp.get_columns('notes')}
            add_cols = []
            if 'moderation_status' not in cols:
                add_cols.append("ALTER TABLE notes ADD COLUMN moderation_status VARCHAR(32) DEFAULT 'auto_published'")
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

        # site_settings: maintenance mode toggle (superadmin only)
        try:
            if not insp.has_table('site_settings'):
                conn.execute(text("CREATE TABLE IF NOT EXISTS site_settings (key VARCHAR(60) PRIMARY KEY, value TEXT)"))
            conn.execute(text("INSERT INTO site_settings(key,value) VALUES ('maintenance_mode','0') ON CONFLICT (key) DO NOTHING"))
            conn.execute(text("INSERT INTO site_settings(key,value) VALUES ('sales_enabled','1') ON CONFLICT (key) DO NOTHING"))
        except Exception:
            # SQLite doesn't support ON CONFLICT in older versions the same way; ignore.
            try:
                conn.execute(text("INSERT OR IGNORE INTO site_settings(key,value) VALUES ('maintenance_mode','0')"))
                conn.execute(text("INSERT OR IGNORE INTO site_settings(key,value) VALUES ('sales_enabled','1')"))
            except Exception:
                pass


try:
    _ensure_schema(engine)
except Exception as e:
    print('[schema] WARNING:', e)


# -----------------------------------------------------------------------------
# Performance: create helpful indexes (safe to run multiple times)
# -----------------------------------------------------------------------------
def _ensure_indexes(engine):
    dialect = (engine.dialect.name or "").lower()
    with engine.begin() as conn:
        if dialect.startswith("postgres"):
            stmts = [
                # Public listing filters
                """CREATE INDEX IF NOT EXISTS idx_notes_public_list
                    ON notes (is_active, moderation_status, is_archived, deleted_at, created_at DESC)""",
                """CREATE INDEX IF NOT EXISTS idx_notes_academics
                    ON notes (university, faculty, career)""",
            ]
            # Optional trigram indexes for ILIKE searches (if extension is available)
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                stmts += [
                    """CREATE INDEX IF NOT EXISTS idx_notes_title_trgm
                        ON notes USING GIN (title gin_trgm_ops)""",
                    """CREATE INDEX IF NOT EXISTS idx_notes_desc_trgm
                        ON notes USING GIN (description gin_trgm_ops)""",
                ]
            except Exception:
                pass

            for st in stmts:
                try:
                    conn.execute(text(st))
                except Exception:
                    pass

        elif dialect.startswith("sqlite"):
            stmts = [
                "CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_notes_public_flags ON notes(is_active, moderation_status, is_archived)",
                "CREATE INDEX IF NOT EXISTS idx_notes_deleted ON notes(deleted_at)",
                "CREATE INDEX IF NOT EXISTS idx_notes_academics ON notes(university, faculty, career)",
            ]
            for st in stmts:
                try:
                    conn.execute(text(st))
                except Exception:
                    pass

try:
    _ensure_indexes(engine)
except Exception as e:
    print('[indexes] WARNING:', e)



# -----------------------------------------------------------------------------
# A6) Analytics: pageviews + funnel events
# -----------------------------------------------------------------------------


def _should_track_page_view() -> bool:
    """Return True if this request should be logged as a page view.

    We only track *public* HTML GET requests (no static, no admin, no webhooks).
    """
    try:
        if request.method != "GET":
            return False
        path = (request.path or "/").strip() or "/"
        # Ignore noisy paths
        if path.startswith("/static"):
            return False
        if path.startswith("/admin"):
            return False
        if path.startswith("/api"):
            return False
        if path.startswith("/health"):
            return False
        if path.startswith("/mp/") or path.startswith("/webhooks/"):
            return False
        # Ignore assets
        if re.search(r"\.(css|js|png|jpg|jpeg|gif|svg|webp|ico|woff2?)$", path, re.I):
            return False
        return True
    except Exception:
        return False


@app.after_request
def _track_page_view(resp):
    """Best-effort page view logging.

    This runs after the response to avoid slowing down requests.
    """
    try:
        if not resp:
            return resp
        if resp.status_code >= 400:
            return resp
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype:
            return resp
        if not _should_track_page_view():
            return resp

        uid = int(current_user.id) if getattr(current_user, "is_authenticated", False) else None
        log_analytics_event(event="page_view", user_id=uid, path=(request.path or None))
    except Exception:
        pass
    return resp

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Session = scoped_session(sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))
login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    with Session() as s:
        return s.get(User, int(user_id))


@app.before_request
def _enforce_user_suspension():
    """Si el usuario suspendió su cuenta, limitamos el acceso solo al perfil.

    - Puede ver /profile
    - Puede reactivar /account/reactivate
    - Puede eliminar /account/delete
    - Puede cerrar sesión
    """
    try:
        if not getattr(current_user, "is_authenticated", False):
            return None
        if not getattr(current_user, "is_suspended", False):
            return None

        # endpoints permitidos mientras está suspendido
        allowed = {
            "profile",
            "logout",
            "disconnect_mp",
            "disconnect_mp_post",
            "account_reactivate",
            "account_suspend",
            "account_delete",
            "static",
        }
        if (request.endpoint or "") in allowed:
            return None

        # permitir cualquier endpoint de auth/session_login para poder entrar
        if (request.endpoint or "").startswith("auth_"):
            return None

        flash("Tu cuenta está suspendida. Solo podés acceder a tu perfil hasta reactivarla.", "warning")
        return redirect(url_for("profile"))
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Legal acceptance enforcement
# -----------------------------------------------------------------------------
def _user_needs_legal_accept(user_id: int) -> bool:
    """True if the user must accept the current LEGAL_VERSION."""
    try:
        cur_ver = (app.config.get("LEGAL_VERSION") or "").strip()
        if not cur_ver:
            return False
        with Session() as s:
            u = s.get(User, int(user_id))
            if not u:
                return False
            accepted = (getattr(u, "legal_version_accepted", None) or "").strip()
            return accepted != cur_ver
    except Exception:
        return False


@app.before_request
def _enforce_legal_acceptance():
    """If legal docs were updated, require the user to re-accept."""
    try:
        if not getattr(current_user, "is_authenticated", False):
            return None

        if not _user_needs_legal_accept(int(current_user.id)):
            return None

        ep = (request.endpoint or "")
        allowed = {
            # legal flow
            "legal_accept",
            "legal_accept_post",
            # legal docs
            "terms",
            "terminos_redirect",
            "privacidad",
            "seguridad",
            "privacy",
            "security",
            # account basics
            "logout",
            "static",
        }

        # allow auth/session endpoints so they can log in
        if ep.startswith("auth_"):
            return None

        if ep in allowed:
            return None

        # Keep "next" so they return to where they were.
        return redirect(url_for("legal_accept", next=request.full_path))
    except Exception:
        return None


def _get_setting(key: str, default: str = "") -> str:
    try:
        with Session() as s:
            row = s.get(SiteSetting, key)
            if row and getattr(row, "value", None) is not None:
                return str(row.value)
    except Exception:
        pass
    return default


def _set_setting(key: str, value: str):
    with Session() as s:
        row = s.get(SiteSetting, key)
        if not row:
            row = SiteSetting(key=key, value=value)
            s.add(row)
        else:
            row.value = value
        s.commit()


@app.before_request
def _bootstrap_superadmin_and_maintenance():
    """1) Bootstrap SUPERADMIN_EMAILS; 2) Enforce maintenance mode."""
    try:
        # ---- bootstrap roles (best-effort) ----
        if getattr(current_user, "is_authenticated", False):
            emails = _parse_superadmin_emails()
            try:
                cu_email = (getattr(current_user, "email", "") or "").lower().strip()
            except Exception:
                cu_email = ""

            with Session() as s:
                dbu = s.get(User, int(current_user.id))
                if dbu:
                    # Promote to superadmin if email matches env
                    if cu_email and cu_email in emails and getattr(dbu, "role", "user") != "superadmin":
                        dbu.role = "superadmin"
                        dbu.is_admin = True
                        s.commit()
                        _audit("promote_superadmin_env", target_type="user", target_id=dbu.id, meta={"email": cu_email})
                    # Keep legacy is_admin in sync for staff
                    if _user_role(dbu) in ("admin", "superadmin") and not bool(getattr(dbu, "is_admin", False)):
                        dbu.is_admin = True
                        s.commit()

        # ---- maintenance mode ----
        mm = (_get_setting("maintenance_mode", "0") or "0").strip()
        maintenance_on = mm in ("1", "true", "True", "yes", "on")
        if not maintenance_on:
            return None

        # superadmin bypass
        if getattr(current_user, "is_authenticated", False) and _is_superadmin(current_user):
            return None

        # allowlist essential endpoints
        allowed = {
            "login",
            "logout",
            "auth_session_login",
            
            "complete_profile_post",
            "static",
            "health",
        }
        if (request.endpoint or "") in allowed:
            return None

        # show maintenance page
        return render_template("maintenance.html"), 503
    except Exception:
        return None

def _parse_superadmin_emails() -> set[str]:
    raw = (os.getenv("SUPERADMIN_EMAILS") or "").strip()
    if not raw:
        return set()
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _user_role(u) -> str:
    try:
        r = (getattr(u, "role", None) or "").strip().lower()
        if r:
            return r
    except Exception:
        pass
    # Backward compat
    if bool(getattr(u, "is_admin", False)):
        return "admin"
    return "user"


def _is_superadmin(u) -> bool:
    return _user_role(u) == "superadmin"


def _is_staff(u) -> bool:
    return _user_role(u) in ("admin", "superadmin")


def _audit(action: str, target_type: str | None = None, target_id: int | None = None, meta: dict | None = None):
    """Write an AuditEvent (best-effort, never breaks request)."""
    try:
        # Code format: AY-YYYYMMDD-000123 (id is the sequence)
        with Session() as s:
            ae = AuditEvent(
                code="PENDING",
                actor_user_id=(current_user.id if getattr(current_user, "is_authenticated", False) else None),
                action=action,
                target_type=target_type,
                target_id=target_id,
                meta=meta or {},
            )
            s.add(ae)
            s.flush()
            ae.code = f"AY-{datetime.utcnow().strftime('%Y%m%d')}-{ae.id:06d}"
            s.commit()
    except Exception:
        pass


def _ticket_code_for_id(ticket_id: int) -> str:
    """Código amigable de ticket.

    Formato: AYT-YYYYMMDD-000123
    """
    return f"AYT-{datetime.utcnow().strftime('%Y%m%d')}-{int(ticket_id):06d}"


def _notify_users(s, user_ids: list[int], *, kind: str, title: str, body: str | None = None):
    """Create notifications (best-effort)."""
    try:
        for uid in sorted({int(x) for x in user_ids if x}):
            try:
                s.add(Notification(user_id=uid, kind=kind, title=title, body=body))
            except Exception:
                continue
    except Exception:
        pass


def _admin_user_ids(s) -> list[int]:
    """Return user ids of staff (admin/superadmin)."""
    try:
        # role preferred
        q = s.query(User.id)
        if hasattr(User, "role"):
            q = q.filter(User.role.in_(["admin", "superadmin"]))
        else:
            q = q.filter(User.is_admin == True)
        return [int(r[0]) for r in q.all()]
    except Exception:
        return []


def staff_required(fn):
    """Admin OR Superadmin."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login"))
        if getattr(current_user, "is_blocked", False):
            abort(403)
        if not _is_staff(current_user):
            abort(403)
        return fn(*args, **kwargs)
    return wrapper


def superadmin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login"))
        if getattr(current_user, "is_blocked", False):
            abort(403)
        if not _is_superadmin(current_user):
            abort(403)
        return fn(*args, **kwargs)
    return wrapper


# Backward-compatible name used across the codebase
admin_required = staff_required

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
app.config["CONTACT_WHATSAPP"] = os.getenv("CONTACT_WHATSAPP", "+543516788775")
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

    # --- Fees globales para UI (se leen desde ENV) ---
    import os

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)).strip())
        except Exception:
            return default

    def _pct(x: float) -> float:
        # 0.2054 -> 20.54
        return round(x * 100.0, 2)

    # Defaults coherentes si no están en ENV
    apy_rate = _env_float("APY_COMMISSION_RATE", 0.10)
    mp_rate = _env_float("MP_COMMISSION_RATE", 0.08)
    total_rate = _env_float("TOTAL_FEE_RATE", apy_rate + mp_rate)

    return dict(
        published_price=published_price,
        published_price_cents=published_price_cents,
        fee_breakdown_from_net=fee_breakdown_from_net,
        fee_breakdown_from_published=fee_breakdown_from_published,

        # 🔥 NUEVO: variables globales para templates
        APY_FEE_RATE=apy_rate,
        MP_FEE_RATE=mp_rate,
        TOTAL_FEE_RATE=total_rate,

        APY_FEE_PERCENT=_pct(apy_rate),
        MP_FEE_PERCENT=_pct(mp_rate),
        TOTAL_FEE_PERCENT=_pct(total_rate),
    )


# -----------------------------------------------------------------------------
# Combo pricing helper (buyer price)
#
# Some templates/routes expect a callable named `_combo_buyer_price_cents`.
# If it's missing, /search/* crashes with NameError.
# We compute the buyer price using the stored `price_cents` when available,
# otherwise we fallback to seller_net_cents -> published_from_net_cents.
# -----------------------------------------------------------------------------
def _combo_buyer_price_cents(combo) -> int:
    try:
        if not combo:
            return 0
        pc = int(getattr(combo, "price_cents", 0) or 0)
        if pc > 0:
            return pc
        net = int(getattr(combo, "seller_net_cents", 0) or 0)
        return int(published_from_net_cents(net)) if net > 0 else 0
    except Exception:
        return 0


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

# Helpcenter (FAQ/Ayuda) + Admin FAQ (si existe)
try:
    from .blueprints.helpcenter import helpcenter_bp
except Exception:
    try:
        from blueprints.helpcenter import helpcenter_bp
    except Exception:
        helpcenter_bp = None

try:
    from .blueprints.admin_faq import admin_faq_bp
except Exception:
    try:
        from blueprints.admin_faq import admin_faq_bp
    except Exception:
        admin_faq_bp = None

if admin_bp:
    app.register_blueprint(admin_bp)
if auth_reset_bp:
    app.register_blueprint(auth_reset_bp)
if helpcenter_bp:
    app.register_blueprint(helpcenter_bp)
if admin_faq_bp:
    app.register_blueprint(admin_faq_bp)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def allowed_pdf(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"

def ensure_dirs():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def gcs_upload_file(file_storage, blob_name: str) -> str:
    """Upload a Flask FileStorage to object storage.

    Backwards-compat name: historically this uploaded to Google Cloud Storage.
    Now it uploads to Cloudflare R2 (S3 compatible) using boto3.
    """
    if not (gcs_client and gcs_bucket):
        raise RuntimeError("R2 no está configurado")
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass
    gcs_client.put_object(
        Bucket=gcs_bucket,
        Key=blob_name,
        Body=file_storage.stream,
        ContentType=file_storage.content_type or "application/pdf",
    )
    return blob_name


def gcs_upload_path(local_path: str, blob_name: str, content_type: str = "application/pdf") -> str:
    """Upload a local file path to object storage (R2)."""
    if not (gcs_client and gcs_bucket):
        raise RuntimeError("R2 no está configurado")
    extra = {"ContentType": content_type} if content_type else None
    if extra:
        gcs_client.upload_file(local_path, gcs_bucket, blob_name, ExtraArgs=extra)
    else:
        gcs_client.upload_file(local_path, gcs_bucket, blob_name)
    return blob_name


def gcs_generate_signed_url(blob_name: str, seconds: int = 600) -> str:
    """Generate a short-lived signed URL for downloads (R2)."""
    if not (gcs_client and gcs_bucket):
        raise RuntimeError("R2 no está configurado")
    return gcs_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": gcs_bucket, "Key": blob_name},
        ExpiresIn=int(seconds or 600),
    )


def gcs_delete_blob(blob_name: str) -> bool:
    """Best-effort delete from object storage."""
    if not (gcs_client and gcs_bucket):
        return False
    try:
        gcs_client.delete_object(Bucket=gcs_bucket, Key=blob_name)
        return True
    except Exception:
        return False


def gcs_download_to_temp(blob_name: str) -> str:
    """Download a blob to a temporary file and return the local path."""
    if not (gcs_client and gcs_bucket):
        raise RuntimeError("R2 no está configurado")
    tmp_dir = os.path.join(BASE_DATA, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{secrets.token_hex(8)}.bin")
    gcs_client.download_file(gcs_bucket, blob_name, tmp_path)
    return tmp_path


def gcs_download_bytes(blob_name: str) -> bytes:
    if not (gcs_client and gcs_bucket):
        raise RuntimeError("R2 no está configurado")
    obj = gcs_client.get_object(Bucket=gcs_bucket, Key=blob_name)
    return obj["Body"].read()


def gcs_upload_bytes(data: bytes, blob_name: str, content_type: str = "application/octet-stream") -> str:
    if not (gcs_client and gcs_bucket):
        raise RuntimeError("R2 no está configurado")
    gcs_client.put_object(
        Bucket=gcs_bucket,
        Key=blob_name,
        Body=data,
        ContentType=content_type or "application/octet-stream",
    )
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


def generate_note_preview(
    note: Note,
    max_pages: int = 4,
    local_pdf_override: str | None = None
) -> tuple[list[int], list[str]]:
    """
    Generate preview images for a note PDF (with watermark) and store paths.

    ✅ Usa el MISMO storage que tus PDFs: gcs_* (que en tu proyecto es R2 por Account ID).
    ✅ Ajustable por ENV sin redeploy:
      - PREVIEW_SCALE (default 1.0)
      - PREVIEW_JPEG_QUALITY (default 78)
      - PREVIEW_MAX_PAGES (default 4)
    """
    import fitz  # PyMuPDF
    from PIL import Image
    from io import BytesIO
    import os, random

    tmp_pdf = None
    local_pdf = None
    doc = None

    scale = float(os.getenv("PREVIEW_SCALE", "1.0"))               # 1.0 liviano
    jpeg_quality = int(os.getenv("PREVIEW_JPEG_QUALITY", "78"))    # 70-82 OK
    max_pages = int(os.getenv("PREVIEW_MAX_PAGES", str(max_pages)))

    try:
        # Resolver PDF local
        if local_pdf_override:
            local_pdf = local_pdf_override
        elif gcs_bucket and note.file_path and "/" in (note.file_path or ""):
            # descarga temporal desde tu bucket (R2)
            tmp_pdf = gcs_download_to_temp(note.file_path)
            local_pdf = tmp_pdf
        else:
            local_pdf = os.path.join(app.config["UPLOAD_FOLDER"], note.file_path)

        doc = fitz.open(local_pdf)
        total = doc.page_count
        if total <= 0:
            return ([], [])

        # Elegir páginas random (evitar portada si hay muchas)
        candidates = list(range(total))
        if total > 6:
            candidates = list(range(2, total))
        random.shuffle(candidates)

        pages = []
        for p in candidates:
            pages.append(p)
            if len(pages) >= max_pages:
                break
        pages = sorted(pages)[:max_pages]

        image_paths: list[str] = []
        mat = fitz.Matrix(scale, scale)

        for idx, pno in enumerate(pages, start=1):
            page = doc.load_page(pno)

            # Pixmap (lo pesado) -> escala baja por default
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.open(BytesIO(pix.tobytes("png")))
            img = _watermark_image(img, text="APUNTESYA")

            buf = BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            data = buf.getvalue()

            # ✅ Subir previews al MISMO bucket (R2) que ya usás
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
                image_paths.append(f"previews/{note.id}/{idx}.jpg")

        return (pages, image_paths)

    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass
        try:
            if tmp_pdf and os.path.exists(tmp_pdf):
                os.remove(tmp_pdf)
        except Exception:
            pass

import threading
import time

def enqueue_preview_generation(note_id: int):
    """
    Opción 1: genera previews en background para evitar 502/OOM en el request.
    - Usa una NUEVA Session dentro del thread.
    - No depende del archivo local (si tenés R2/gcs_bucket, generate_note_preview descarga temp).
    """
    # Si querés poder apagarlo sin redeploy:
    if os.getenv("PREVIEW_BG_ON_UPLOAD", "1") != "1":
        return

    def _worker(nid: int):
        try:
            # (Opcional) pequeña pausa para liberar el request / IO
            time.sleep(float(os.getenv("PREVIEW_BG_DELAY_SEC", "0.2")))
        except Exception:
            pass

        try:
            # En Flask, para loggers/config en threads a veces conviene app_context
            with app.app_context():
                with Session() as s:
                    note = s.get(Note, int(nid))
                    if not note or not getattr(note, "is_active", True):
                        return

                    # Si ya tiene previews, no regenerar
                    try:
                        meta = getattr(note, "preview_images", None) or {}
                        imgs = meta.get("images") or []
                        if imgs:
                            return
                    except Exception:
                        pass

                    try:
                        pages, imgs = generate_note_preview(
                            note,
                            max_pages=int(os.getenv("PREVIEW_MAX_PAGES", "4")),
                            local_pdf_override=None,  # <- clave: que use R2/temp
                        )
                        if imgs:
                            note.preview_pages = {"pages": pages}
                            note.preview_images = {"images": imgs}
                            s.commit()
                    except Exception as e:
                        try:
                            app.logger.warning(f"[preview-bg] failed for note_id={nid}: {e}")
                        except Exception:
                            pass

        except Exception as e:
            try:
                app.logger.warning(f"[preview-bg] thread crashed note_id={nid}: {e}")
            except Exception:
                pass

    try:
        t = threading.Thread(target=_worker, args=(int(note_id),), daemon=True)
        t.start()
    except Exception:
        # nunca romper el upload por esto
        pass


def _generate_preview_background(note_id: int, local_pdf_path: str | None = None):
    """Genera preview en background (best-effort). No debe romper nada si falla."""
    try:
        with Session() as s:
            note = s.get(Note, int(note_id))
            if not note or not getattr(note, "is_active", False):
                return

            # Si ya tiene previews, no hacemos nada
            try:
                meta = getattr(note, "preview_images", None) or {}
                imgs = (meta.get("images") or []) if isinstance(meta, dict) else []
                if imgs:
                    return
            except Exception:
                pass

            # Generar (usamos local_pdf_override si lo tenés, ayuda mucho)
            pages, new_imgs = generate_note_preview(
                note,
                max_pages=int(os.getenv("PREVIEW_MAX_PAGES", "4")),
                local_pdf_override=local_pdf_path,
            )
            if new_imgs:
                note.preview_pages = {"pages": pages}
                note.preview_images = {"images": new_imgs}
                s.commit()

    except Exception as e:
        try:
            app.logger.warning(f"bg preview failed note_id={note_id}: {e}")
        except Exception:
            pass
    finally:
        # Si pasaste el pdf local, lo podés borrar acá (opcional)
        try:
            if local_pdf_path and os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
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
        if gcs_bucket and note.file_path and "/" in note.file_path:
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
        # Fallback: we prefer publishing with audit instead of blocking everything.
        return {
            "risk_level": "medium",
            "confidence": 0.5,
            "summary": "IA no configurada: publicado con auditoría automática.",
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
            risk_level: Literal["low", "medium", "high"]
            confidence: int = Field(ge=0, le=100)
            summary: str
            reasons: List[str] = []
            quality_score: int = Field(ge=0, le=100, default=50)
            copyright_risk: int = Field(ge=0, le=100, default=0)
            mismatch_risk: int = Field(ge=0, le=100, default=0)

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
Sos moderador de un marketplace de apuntes (Argentina). Tu objetivo es reducir la revisión manual.

Clasificá el apunte en un NIVEL DE RIESGO:
- low: contenido educativo normal. Publicación automática.
- medium: dudas leves (calidad baja, texto corto, posible desajuste). Se publica igual pero queda para auditoría.
- high: señales fuertes de spam/ilegal/copyright evidente/no educativo. Bloquear y requerir revisión humana.

Metadatos del apunte (no confiar al 100%, solo contexto):
{_json.dumps(meta, ensure_ascii=False)}

Texto extraído (muestra):
{text_sample}

Devolvé SOLO JSON con los campos: risk_level, confidence (0-100), summary, reasons, quality_score, copyright_risk, mismatch_risk.
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
            "risk_level": parsed.risk_level,
            "confidence": parsed.confidence / 100.0,
            "summary": parsed.summary,
            "reasons": parsed.reasons,
            "quality_score": parsed.quality_score / 100.0,
            "copyright_risk": parsed.copyright_risk / 100.0,
            "mismatch_risk": parsed.mismatch_risk / 100.0,
        }
        return out

    except Exception as e:
        # Fallback: publish with audit (medium risk) instead of forcing manual review.
        return {
            "risk_level": "medium",
            "confidence": 0.5,
            "summary": f"Error IA ({type(e).__name__}): publicado con auditoría automática.",
            "reasons": ["ai_error"],
            "quality_score": 0.5,
            "copyright_risk": 0.0,
            "mismatch_risk": 0.0,
        }

def _decision_to_status(ai: dict) -> tuple[str, str]:
    """Map AI JSON to (moderation_status, reason).

    New scheme:
      - auto_published: visible to everyone
      - published_flagged: visible, but marked for optional audit
      - blocked_review: not visible, requires human review

    Backward compatible with old keys/values (approve|review|deny).
    """
    risk = (ai.get("risk_level") or ai.get("decision") or "medium").lower().strip()
    conf = float(ai.get("confidence") or 0.0)

    # Backward-compat mapping
    if risk in ("approve", "approved", "ok"):
        risk = "low"
    elif risk in ("review", "pending", "maybe"):
        risk = "medium"
    elif risk in ("deny", "rejected", "block"):
        risk = "high"

    if risk not in ("low", "medium", "high"):
        risk = "medium"

    # Only HIGH blocks. Everything else publishes.
    if risk == "low":
        return ("auto_published", None)
    if risk == "high":
        return ("blocked_review", (ai.get("summary") or "Revisión humana requerida").strip())
    # medium
    return ("published_flagged", (ai.get("summary") or "Publicado con auditoría automática").strip())




# Moderation helpers (new + backward compatible)
PUBLIC_MODERATION_STATUSES = {"approved", "auto_published", "published_flagged"}
HUMAN_REVIEW_STATUSES = {"pending_manual", "blocked_review"}

def is_public_moderation_status(status: str) -> bool:
    return (status or "approved") in PUBLIC_MODERATION_STATUSES

def needs_human_review(status: str) -> bool:
    return (status or "") in HUMAN_REVIEW_STATUSES

def _notify_users(session_db, user_ids: list[int], title: str, body: str, kind: str = "info"):
    """Create in-app notifications for a list of users."""
    for uid in user_ids:
        try:
            session_db.add(Notification(user_id=uid, kind=kind, title=title, body=body))
        except Exception:
            pass





# ------------------------------ Email (SMTP) ------------------------------
def _smtp_config():
    """Read SMTP config from environment variables (zero-cost friendly).
    Set at least SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, and MAIL_FROM.
    """
    return {
        "host": (os.getenv("SMTP_HOST") or "").strip(),
        "port": int(os.getenv("SMTP_PORT") or "587"),
        "user": (os.getenv("SMTP_USER") or "").strip(),
        "pass": (os.getenv("SMTP_PASS") or "").strip(),
        "tls": (os.getenv("SMTP_TLS", "1").strip() != "0"),
        # Some hosting providers have no IPv6 egress; forcing IPv4 avoids
        # "[Errno 101] Network is unreachable" when DNS resolves to AAAA first.
        "force_ipv4": (os.getenv("SMTP_FORCE_IPV4", "1").strip() != "0"),
        "from": (os.getenv("MAIL_FROM") or os.getenv("SMTP_FROM") or "no-reply@apuntesya.local").strip(),
        "enabled": (os.getenv("SMTP_ENABLED", "1").strip() != "0"),
    }

def _smtp_connect(host: str, port: int, timeout: int = 15, force_ipv4: bool = True) -> smtplib.SMTP:
    """Create an SMTP connection.

    If force_ipv4 is True, resolves host to IPv4 and connects to the IP, while
    keeping the original host for TLS SNI/certificate validation.
    """
    if not force_ipv4:
        return smtplib.SMTP(host, port, timeout=timeout)

    import socket
    infos = socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    if not infos:
        # fallback to default behavior
        return smtplib.SMTP(host, port, timeout=timeout)

    ip = infos[0][4][0]
    smtp = smtplib.SMTP(timeout=timeout)
    smtp.connect(ip, port)
    return smtp

def _brevo_config() -> dict:
    return {
        "api_key": (os.getenv("BREVO_API_KEY") or os.getenv("SENDINBLUE_API_KEY") or "").strip(),
        "sender_email": (os.getenv("BREVO_SENDER_EMAIL") or os.getenv("MAIL_FROM_EMAIL") or "").strip(),
        "sender_name": (os.getenv("BREVO_SENDER_NAME") or "ApuntesYa").strip(),
        "enabled": (os.getenv("BREVO_ENABLED", "1").strip() != "0"),
    }

def _send_email_brevo(to_email: str, subject: str, text_body: str, html_body: str | None = None) -> bool:
    """Send email via Brevo Transactional Email API (HTTPS/443)."""
    cfg = _brevo_config()
    if not cfg["enabled"] or not cfg["api_key"] or not to_email:
        return False

    # Use sender_email if provided; otherwise fall back to SMTP_FROM / MAIL_FROM parsing is messy.
    sender_email = cfg["sender_email"] or (os.getenv("SMTP_USER") or "").strip() or "no-reply@apuntesya.local"
    payload = {
        "sender": {"email": sender_email, "name": cfg["sender_name"]},
        "to": [{"email": to_email}],
        "subject": subject or "",
        "textContent": text_body or "",
    }
    if html_body:
        payload["htmlContent"] = html_body

    try:
        import requests
        r = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": cfg["api_key"],
                "accept": "application/json",
                "content-type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        if 200 <= r.status_code < 300:
            return True
        try:
            app.logger.warning(f"brevo email send failed to {to_email}: {r.status_code} {r.text[:500]}")
        except Exception:
            pass
        return False
    except Exception as e:
        try:
            app.logger.warning(f"brevo email send exception to {to_email}: {e}")
        except Exception:
            pass
        return False

def send_email(to_email: str, subject: str, text_body: str, html_body: str | None = None) -> bool:
    """Send an email.

    NOTE: Render free web services block outbound SMTP ports (25/465/587). If you
    are on Render Free, configure BREVO_API_KEY to send via HTTPS instead.
    """
    # Prefer Brevo API when configured (works over HTTPS/443)
    if (os.getenv("BREVO_API_KEY") or os.getenv("SENDINBLUE_API_KEY")):
        if _send_email_brevo(to_email, subject, text_body, html_body=html_body):
            return True

    # Fall back to SMTP if available (paid Render or non-Render hosting)
    cfg = _smtp_config()
    if not cfg["enabled"] or not cfg["host"] or not to_email:
        return False


def _make_audit_code(now: datetime | None = None, seq: int | None = None) -> str:
    """Build a human-friendly audit code.

    Example: AY-20260204-000123
    """
    now = now or datetime.utcnow()
    return f"AY-{now.strftime('%Y%m%d')}-{(seq or 0):06d}"


def log_audit_event(
    *,
    actor_user_id: int | None,
    action: str,
    target_type: str | None = None,
    target_id: int | None = None,
    meta: dict | None = None,
) -> str:
    """Persist an audit event and return its code."""
    with Session() as s:
        ev = AuditEvent(
            code="PENDING",
            actor_user_id=actor_user_id,
            action=(action or "").strip()[:64],
            target_type=(target_type or None),
            target_id=target_id,
            meta=meta or None,
        )
        s.add(ev)
        s.flush()  # get id
        ev.code = _make_audit_code(seq=int(ev.id))
        s.commit()
        return ev.code


def log_analytics_event(
    *,
    event: str,
    user_id: int | None = None,
    path: str | None = None,
    note_id: int | None = None,
    combo_id: int | None = None,
    meta: dict | None = None,
) -> None:
    """Best-effort analytics logging. Never blocks main flow."""
    try:
        ev = AnalyticsEvent(
            event=(event or "").strip()[:64],
            user_id=user_id,
            path=(path or None),
            note_id=note_id,
            combo_id=combo_id,
            ip=(request.headers.get("X-Forwarded-For") or request.remote_addr or "")[:64] or None,
            user_agent=(request.headers.get("User-Agent") or "")[:255] or None,
            referrer=(request.headers.get("Referer") or "")[:255] or None,
            meta=meta or None,
        )
        with SessionLocal() as s:
            s.add(ev)
            s.commit()
    except Exception:
        pass


# -----------------------
# A5) Stats diarios (agregados)
# -----------------------

def _stats_daily_get_or_create(session_db, day_date):
    """Return DailyStat row for a given date, creating it if missing."""
    row = session_db.get(DailyStat, day_date)
    if row:
        return row
    row = DailyStat(day=day_date)
    session_db.add(row)
    # flush to ensure it exists for subsequent updates in same txn
    try:
        session_db.flush()
    except Exception:
        pass
    return row


def stats_daily_add_purchase(session_db, p: Purchase):
    """Add an approved purchase into daily stats once (idempotent via p.stats_counted)."""
    try:
        if not p or (p.status or "").lower() != "approved":
            return
        if bool(getattr(p, "stats_counted", False)):
            return

        # determine breakdown (fallback for older rows)
        gross = int(getattr(p, "gross_cents", 0) or 0)
        if gross <= 0:
            gross = int(getattr(p, "amount_cents", 0) or 0)

        ay_fee = int(getattr(p, "platform_fee_cents", 0) or 0)
        mp_fee = int(getattr(p, "mp_fee_cents", 0) or 0)
        seller_net = int(getattr(p, "seller_net_cents", 0) or 0)

        # Fallback if seller_net missing but note exists
        if seller_net <= 0 and getattr(p, "note_id", None):
            try:
                n = session_db.get(Note, int(p.note_id))
                seller_net = int(getattr(n, "seller_net_cents", 0) or 0)
            except Exception:
                seller_net = seller_net

        # If ay_fee missing, derive it as remainder after mp + seller
        if ay_fee <= 0 and gross > 0:
            ay_fee = max(0, gross - mp_fee - seller_net)

        day = (getattr(p, "created_at", None) or datetime.utcnow()).date()
        ds = _stats_daily_get_or_create(session_db, day)

        ds.gross_income_cents = int(ds.gross_income_cents or 0) + gross
        ds.ay_commission_cents = int(ds.ay_commission_cents or 0) + ay_fee
        ds.mp_fee_cents = int(ds.mp_fee_cents or 0) + mp_fee
        ds.seller_income_cents = int(ds.seller_income_cents or 0) + seller_net
        ds.sales_count = int(ds.sales_count or 0) + 1

        p.stats_counted = True
    except Exception:
        # never block purchase flow
        pass


def stats_daily_add_download(session_db, *, is_free: bool, created_at_dt=None):
    """Add one download into daily stats (free vs paid)."""
    try:
        day = (created_at_dt or datetime.utcnow()).date()
        ds = _stats_daily_get_or_create(session_db, day)
        if is_free:
            ds.free_downloads = int(ds.free_downloads or 0) + 1
        else:
            ds.paid_downloads = int(ds.paid_downloads or 0) + 1
    except Exception:
        pass


def notify_user(
    *,
    user_id: int,
    title: str,
    body: str,
    kind: str = "info",
    email: bool = False,
    email_subject: str | None = None,
) -> None:
    """Best-effort in-app notification + optional email."""
    try:
        with Session() as s:
            s.add(Notification(user_id=user_id, kind=kind, title=title, body=body))
            u = s.get(User, user_id)
            to_email = (getattr(u, "email", "") or "").strip().lower() if u else ""
            s.commit()
        if email and to_email:
            send_email(to_email, email_subject or title, body)
    except Exception:
        pass

def _create_notification_once(session_db, user_id: int, kind: str, title: str, body: str) -> bool:
    """Create a notification once (best-effort dedupe) without schema changes.

    We dedupe by (user_id, kind, title, body) in the last 7 days.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=7)
        exists = session_db.execute(
            select(Notification.id).where(
                Notification.user_id == user_id,
                Notification.kind == kind,
                Notification.title == title,
                Notification.body == (body or None),
                Notification.created_at >= cutoff,
            ).limit(1)
        ).first()
        if exists:
            return False
    except Exception:
        pass

    try:
        session_db.add(Notification(user_id=user_id, kind=kind, title=title, body=(body or None)))
        return True
    except Exception:
        return False

def notify_and_email_users(session_db, user_ids: list[int], kind: str, title: str, body: str, email_subject: str | None = None, email_body: str | None = None, dedupe_key_prefix: str = ""):
    """Create in-app notifications and (optionally) send email for each user."""
    if not user_ids:
        return

    # preload emails in one query
    try:
        users = session_db.execute(select(User).where(User.id.in_(user_ids))).scalars().all()
        id_to_email = {u.id: (u.email or "").strip() for u in users}
        id_to_name = {u.id: (u.name or "").strip() for u in users}
    except Exception:
        id_to_email = {}
        id_to_name = {}

    for uid in user_ids:
        created = _create_notification_once(session_db, uid, kind=kind, title=title, body=body)

        # Email: send even if notification existed? we keep it consistent: send only if created
        if created:
            to_em = id_to_email.get(uid, "")
            if to_em:
                subj = (email_subject or title).strip()
                txt = (email_body or body or "").strip()
                # small personalization
                nm = id_to_name.get(uid) or ""
                if nm:
                    txt = f"Hola {nm},\n\n" + txt + "\n\n— ApuntesYa"
                else:
                    txt = txt + "\n\n— ApuntesYa"
                send_email(to_em, subj, txt)

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
    # Analytics: página principal (best-effort)
    try:
        log_analytics_event(
            event="page_view",
            user_id=(current_user.id if current_user.is_authenticated else None),
            path=request.path,
            meta={"page": "home"},
        )
    except Exception:
        pass

    is_anon = not getattr(current_user, "is_authenticated", False)

    # We cache *data* (ids + metrics), never full HTML (CSRF tokens are per-session).
    cache_notes_ttl = int(os.getenv("CACHE_HOME_NOTES_TTL_SEC", "60"))
    cache_rank_ttl = int(os.getenv("CACHE_HOME_RANKINGS_TTL_SEC", "600"))

    cache_key_notes = "home:notes_v1"
    cache_key_rank = "home:ranks_v1"

    cached_notes = _cache_get(cache_key_notes) if is_anon else None
    cached_ranks = _cache_get(cache_key_rank) if is_anon else None

    with Session() as s:
        # ---------------- Notes + Combos (frequent) ----------------
        if cached_notes:
            note_ids = cached_notes.get("note_ids", [])
            combo_ids = cached_notes.get("combo_ids", [])
            notes = []
            combos = []
            if note_ids:
                notes = s.execute(
                    select(Note).where(Note.id.in_(note_ids))
                ).scalars().all()
                # preserve order
                notes_by = {n.id: n for n in notes}
                notes = [notes_by[i] for i in note_ids if i in notes_by]
            if combo_ids:
                combos = s.execute(
                    select(Combo).where(Combo.id.in_(combo_ids))
                ).scalars().all()
                combos_by = {c.id: c for c in combos}
                combos = [combos_by[i] for i in combo_ids if i in combos_by]
        else:
            q_notes = select(Note).order_by(desc(Note.created_at)).limit(30)

            if hasattr(Note, "is_active"):
                q_notes = q_notes.where(Note.is_active == True)

            if hasattr(Note, "moderation_status"):
                q_notes = q_notes.where(Note.moderation_status.in_(("approved","auto_published","published_flagged"))).where(Note.is_archived == False)

            if hasattr(Note, "deleted_at"):
                q_notes = q_notes.where(Note.deleted_at.is_(None))

            notes = s.execute(q_notes).scalars().all()

            combos = s.execute(
                select(Combo)
                .where(
                    Combo.is_active == True,
                    Combo.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Combo.is_archived == False,
                )
                .order_by(Combo.created_at.desc())
            ).scalars().all()

            if is_anon:
                _cache_set(cache_key_notes, {
                    "note_ids": [n.id for n in notes],
                    "combo_ids": [c.id for c in combos],
                }, cache_notes_ttl)

        # ---------------- Rankings (heavier) ----------------
        most_downloaded = []
        best_rated = []

        if cached_ranks:
            md = cached_ranks.get("most_downloaded", [])
            br = cached_ranks.get("best_rated", [])
            # md/br: list of (note_id, metric)
            if md:
                ids = [i for (i, _v) in md]
                rows = s.execute(select(Note).where(Note.id.in_(ids))).scalars().all()
                by = {n.id: n for n in rows}
                most_downloaded = [(by[i], v) for (i, v) in md if i in by]
            if br:
                ids = [i for (i, _v) in br]
                rows = s.execute(select(Note).where(Note.id.in_(ids))).scalars().all()
                by = {n.id: n for n in rows}
                best_rated = [(by[i], v) for (i, v) in br if i in by]
        else:
            try:
                q_most = (
                    select(Note, func.count(DownloadLog.id).label("dl"))
                    .join(DownloadLog, DownloadLog.note_id == Note.id)
                )

                if hasattr(Note, "is_active"):
                    q_most = q_most.where(Note.is_active == True)
                if hasattr(Note, "moderation_status"):
                    q_most = q_most.where(Note.moderation_status.in_(("approved","auto_published","published_flagged"))).where(Note.is_archived == False)
                if hasattr(Note, "deleted_at"):
                    q_most = q_most.where(Note.deleted_at.is_(None))

                most_downloaded = s.execute(
                    q_most.group_by(Note.id)
                    .order_by(desc(func.count(DownloadLog.id)))
                    .limit(10)
                ).all()
            except Exception:
                most_downloaded = []

            try:
                q_best = (
                    select(Note, func.avg(Review.rating).label("avg"))
                    .join(Review, Review.note_id == Note.id)
                )

                if hasattr(Note, "is_active"):
                    q_best = q_best.where(Note.is_active == True)
                if hasattr(Note, "moderation_status"):
                    q_best = q_best.where(Note.moderation_status.in_(("approved","auto_published","published_flagged"))).where(Note.is_archived == False)
                if hasattr(Note, "deleted_at"):
                    q_best = q_best.where(Note.deleted_at.is_(None))

                best_rated = s.execute(
                    q_best.group_by(Note.id)
                    .order_by(desc(func.avg(Review.rating)))
                    .limit(10)
                ).all()
            except Exception:
                best_rated = []

            if is_anon:
                try:
                    _cache_set(cache_key_rank, {
                        "most_downloaded": [(n.id, int(dl or 0)) for (n, dl) in most_downloaded],
                        "best_rated": [(n.id, float(avg or 0.0)) for (n, avg) in best_rated],
                    }, cache_rank_ttl)
                except Exception:
                    pass

    return render_template(
        "index.html",
        notes=notes,
        combos=combos,
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

from sqlalchemy import select, desc, or_, distinct, func
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
                    Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
                    Note.deleted_at.is_(None),
                    or_(Note.title.ilike(like), Note.description.ilike(like)),
                )
                .order_by(desc(Note.created_at))
                .limit(100)
            )
            notes = s.execute(notes_stmt).scalars().all()

            # Combos (match por titulo/descripcion O por notas dentro del combo)
            # IMPORTANTE:
            # - Evitamos DISTINCT sobre entidades con columnas JSON (Postgres no define operador de igualdad para json).
            # - Evitamos ORDER BY con SELECT DISTINCT (Postgres exige que el ORDER BY esté en el SELECT).
            # Solución: agrupar por Combo.id y ordenar por max(created_at).
            created_at_expr = func.max(Combo.created_at).label("created_at")
            combo_ids_stmt = (
                select(Combo.id, created_at_expr)
                .join(ComboNote, ComboNote.combo_id == Combo.id)
                .join(Note, Note.id == ComboNote.note_id)
                .where(
                    Combo.is_active == True,
                    Combo.moderation_status.in_(("approved", "auto_published", "published_flagged")),
                    Combo.is_archived == False,
                    Note.is_active == True,
                    Note.moderation_status.in_(("approved", "auto_published", "published_flagged")),
                    Note.is_archived == False,
                    Note.deleted_at.is_(None),
                    or_(
                        Combo.title.ilike(like),
                        Combo.description.ilike(like),
                        Note.title.ilike(like),
                        Note.description.ilike(like),
                    ),
                )
                .group_by(Combo.id)
                .order_by(desc(created_at_expr))
                .limit(100)
            )
            combo_ids = [r[0] for r in s.execute(combo_ids_stmt).all()]
            if combo_ids:
                combos = (
                    s.execute(
                        select(Combo)
                        .where(Combo.id.in_(combo_ids))
                        .order_by(desc(Combo.created_at))
                    )
                    .scalars()
                    .all()
                )
            else:
                combos = []

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
            Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
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

        # ---------------- Combos ----------------
        # Para combos hacemos búsqueda sobre Combo y/o sobre notas incluidas en el combo.
        # Evitamos DISTINCT sobre entidades con JSON usando GROUP BY + carga por IDs.
        combos = []
        created_at_expr_adv = func.max(Combo.created_at).label("created_at")
        combo_ids_stmt = (
            select(Combo.id, created_at_expr_adv)
            .join(ComboNote, ComboNote.combo_id == Combo.id)
            .join(Note, Note.id == ComboNote.note_id)
            .where(
                Combo.is_active == True,
                Combo.moderation_status.in_(("approved","auto_published","published_flagged")),
                Combo.is_archived == False,
                Note.is_active == True,
                Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                Note.is_archived == False,
                Note.deleted_at.is_(None),
            )
        )

        if q:
            like = f"%{q}%"
            combo_ids_stmt = combo_ids_stmt.where(
                or_(
                    Combo.title.ilike(like),
                    Combo.description.ilike(like),
                    Note.title.ilike(like),
                    Note.description.ilike(like),
                )
            )

        if university:
            combo_ids_stmt = combo_ids_stmt.where(Note.university.ilike(f"%{university}%"))
        if faculty:
            combo_ids_stmt = combo_ids_stmt.where(Note.faculty.ilike(f"%{faculty}%"))
        if career:
            combo_ids_stmt = combo_ids_stmt.where(Note.career.ilike(f"%{career}%"))

        if note_type == "free":
            combo_ids_stmt = combo_ids_stmt.where(Combo.seller_net_cents == 0)
        elif note_type == "paid":
            combo_ids_stmt = combo_ids_stmt.where(Combo.seller_net_cents > 0)

        combo_ids_stmt = combo_ids_stmt.group_by(Combo.id).order_by(desc(created_at_expr_adv)).limit(100)
        combo_ids = [r[0] for r in s.execute(combo_ids_stmt).all()]
        if combo_ids:
            combos = s.execute(
                select(Combo).where(Combo.id.in_(combo_ids)).order_by(desc(Combo.created_at))
            ).scalars().all()

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
            .where(Note.moderation_status.in_(("approved","auto_published","published_flagged"))).where(Note.is_archived == False)
            .where(or_(
                Note.title.ilike(f"%{q}%"),
                Note.description.ilike(f"%{q}%"),
                Note.subject.ilike(f"%{q}%"),
            ))
        )
        notes = s.execute(notes_stmt.order_by(desc(Note.created_at)).limit(100)).scalars().all()

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

import os

@app.route("/login", methods=["GET"])
def login():
    firebase_web_config = {
        "apiKey": os.getenv("FIREBASE_WEB_API_KEY"),
        "authDomain": os.getenv("FIREBASE_WEB_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_WEB_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_WEB_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_WEB_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_WEB_APP_ID"),
    }

    return render_template(
        "login_google.html",
        firebase_web_config=firebase_web_config
    )

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

@csrf.exempt
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
                    # Buscar último ticket de suspensión/bloqueo (best-effort)
                    ticket = None
                    try:
                        ticket = s.execute(
                            select(AuditEvent.code)
                            .where(AuditEvent.target_type == "user", AuditEvent.target_id == int(u.id),
                                   AuditEvent.action.in_(["user_suspend"]))
                            .order_by(desc(AuditEvent.created_at))
                            .limit(1)
                        ).scalar_one_or_none()
                    except Exception:
                        ticket = None

                    msg = "Tu cuenta está bloqueada. Escribinos a soporte.apuntesya@gmail.com"
                    if ticket:
                        msg += f" (Nº de gestión: {ticket})"

                    return {
                        "ok": False,
                        "error": "account_blocked",
                        "message": msg,
                        "ticket": ticket
                    }, 403

                # ✅ Actualizamos la foto (y nombre si viene) desde Google en cada login
                try:
                    pic = (info.get("picture") or "").strip()
                    nm = (info.get("name") or "").strip()
                    changed = False
                    if pic and getattr(u, "imagen_de_perfil", None) != pic:
                        u.imagen_de_perfil = pic
                        changed = True
                    # Si el usuario no tiene nombre o viene uno mejor desde Google
                    if nm and (not getattr(u, "name", None) or getattr(u, "name", "").strip() == "Usuario"):
                        u.name = nm
                        changed = True
                    if changed:
                        s.commit()
                except Exception:
                    # No bloqueamos el login por un fallo de actualización de foto/nombre
                    s.rollback()
                login_user(u)
                # If legal docs were updated, force re-acceptance
                if _user_needs_legal_accept(int(u.id)):
                    return {"ok": True, "next": url_for("legal_accept")}, 200
                return {"ok": True, "next": url_for("index")}, 200


        
        # Nuevo usuario: creamos un perfil mínimo y lo mandamos a /profile para completar datos.
        with Session() as s:
            dummy_hash = "google"
            u = User(
                name=name or (email.split("@")[0] if email else "Usuario"),
                email=email,
                password_hash=dummy_hash,
                imagen_de_perfil=info.get("picture"),
            )
            s.add(u)
            s.commit()
            s.refresh(u)

        login_user(u)
        return {"ok": True, "next": url_for("profile", first="1")}, 200


    except Exception as e:
        app.logger.exception("session_login error")
        return {"ok": False, "error": str(e)}, 500



# /complete_profile eliminado: ahora se completa en /profile



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
            if _user_needs_legal_accept(int(exists.id)):
                return redirect(url_for("legal_accept"))
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
    # First-time users must accept legal docs
    return redirect(url_for("legal_accept"))


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
        # Notificaciones ahora viven solo en el ícono de la navbar (y /notifications)
    )


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Notifications
# -----------------------------------------------------------------------------
@app.get("/notifications")
@login_required
def notifications_list():
    """Página simple para ver todas las notificaciones (solo lectura)."""
    with Session() as s:
        notifs = s.execute(
            select(Notification)
            .where(Notification.user_id == current_user.id)
            .order_by(Notification.created_at.desc())
            .limit(200)
        ).scalars().all()
    return render_template("notifications.html", notifications=notifs)


@app.get("/notifications/<int:notif_id>")
@login_required
def notification_open(notif_id: int):
    """Marca una notificación como leída y redirige a un lugar útil."""
    target = None
    with Session() as s:
        n = s.get(Notification, notif_id)
        if not n or n.user_id != current_user.id:
            return redirect(url_for("notifications_list"))

        # Mark read
        try:
            n.is_read = True
            s.commit()
        except Exception:
            pass

        kind = (n.kind or "").lower().strip()

    # Routing heurístico por tipo
    if kind in ("purchase_buyer",):
        target = url_for("profile_purchases")
    elif kind in ("sale_seller",):
        target = url_for("profile_balance")
    elif kind.startswith("note_") or kind.startswith("combo_"):
        # Lo más lógico para el usuario: gestionar sus apuntes/combos
        target = url_for("my_notes_hub")
    elif "admin" in kind and getattr(current_user, "is_admin", False):
        target = url_for("admin_hub")

    return redirect(target or url_for("notifications_list"))


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
    """Hub para creadores: nuevo contenido / editar publicaciones / (próximamente) estadísticas."""
    with Session() as s:
        # Apuntes del usuario + cantidad de descargas (compras aprobadas)
        rows = s.execute(
            select(
                Note,
                func.count(DownloadLog.id).label("download_count"),
            )
            .outerjoin(
                DownloadLog,
                DownloadLog.note_id == Note.id,
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

        # Apuntes disponibles para armar combos (aprobados/activos si existen esos campos)
        q_combo_notes = select(Note).where(
            Note.seller_id == current_user.id,
            Note.deleted_at.is_(None),
        )
        if hasattr(Note, "is_active"):
            q_combo_notes = q_combo_notes.where(Note.is_active == True)
        if hasattr(Note, "moderation_status"):
            q_combo_notes = q_combo_notes.where(Note.moderation_status.in_(("approved","auto_published","published_flagged"))).where(Note.is_archived == False)

        combo_notes = s.execute(q_combo_notes.order_by(Note.created_at.desc())).scalars().all()


    
        # -----------------------------------------------------------------
        # Estadísticas (totales + por apunte) - basado en logs reales
        # -----------------------------------------------------------------
        # Totales de apuntes
        total_downloads = s.execute(
            select(func.count(DownloadLog.id))
            .select_from(DownloadLog)
            .join(Note, Note.id == DownloadLog.note_id)
            .where(Note.seller_id == current_user.id)
        ).scalar_one() or 0

        total_purchases = s.execute(
            select(func.count(Purchase.id))
            .select_from(Purchase)
            .join(Note, Note.id == Purchase.note_id)
            .where(
                Purchase.status == "approved",
                Note.seller_id == current_user.id,
            )
        ).scalar_one() or 0

        total_earned_cents_notes = s.execute(
            select(func.coalesce(func.sum(Note.price_cents), 0))
            .select_from(Purchase)
            .join(Note, Note.id == Purchase.note_id)
            .where(
                Purchase.status == "approved",
                Note.seller_id == current_user.id,
            )
        ).scalar_one() or 0

        # Totales de combos
        total_combo_purchases = s.execute(
            select(func.count(ComboPurchase.id))
            .select_from(ComboPurchase)
            .join(Combo, Combo.id == ComboPurchase.combo_id)
            .where(
                ComboPurchase.status == "approved",
                Combo.seller_id == current_user.id,
            )
        ).scalar_one() or 0

        total_earned_cents_combos = s.execute(
            select(func.coalesce(func.sum(Combo.seller_net_cents), 0))
            .select_from(ComboPurchase)
            .join(Combo, Combo.id == ComboPurchase.combo_id)
            .where(
                ComboPurchase.status == "approved",
                Combo.seller_id == current_user.id,
            )
        ).scalar_one() or 0

        stats_totals = {
            "notes_count": int(len(my_notes)),
            "combos_count": int(len(combos)),
            "downloads_total": int(total_downloads),
            "purchases_total": int(int(total_purchases) + int(total_combo_purchases)),
            "earned_total_cents": int(int(total_earned_cents_notes) + int(total_earned_cents_combos)),
            "earned_notes_cents": int(total_earned_cents_notes),
            "earned_combos_cents": int(total_earned_cents_combos),
        }

        stats_totals["earned_total_ars"] = float(money_1_decimal(cents_to_amount(stats_totals["earned_total_cents"])))
        stats_totals["earned_notes_ars"] = float(money_1_decimal(cents_to_amount(stats_totals["earned_notes_cents"])))
        stats_totals["earned_combos_ars"] = float(money_1_decimal(cents_to_amount(stats_totals["earned_combos_cents"])))

        # Por apunte
        dl_sub = (
            select(
                DownloadLog.note_id.label("nid"),
                func.count(DownloadLog.id).label("downloads"),
            )
            .select_from(DownloadLog)
            .group_by(DownloadLog.note_id)
            .subquery()
        )

        pc_sub = (
            select(
                Purchase.note_id.label("nid"),
                func.count(Purchase.id).label("purchases"),
                func.coalesce(func.sum(Note.price_cents), 0).label("earned_cents"),
            )
            .select_from(Purchase)
            .join(Note, Note.id == Purchase.note_id)
            .where(Purchase.status == "approved")
            .group_by(Purchase.note_id)
            .subquery()
        )

        note_stats_rows = s.execute(
            select(
                Note,
                func.coalesce(pc_sub.c.purchases, 0).label("purchases"),
                func.coalesce(dl_sub.c.downloads, 0).label("downloads"),
                func.coalesce(pc_sub.c.earned_cents, 0).label("earned_cents"),
            )
            .outerjoin(pc_sub, pc_sub.c.nid == Note.id)
            .outerjoin(dl_sub, dl_sub.c.nid == Note.id)
            .where(
                Note.seller_id == current_user.id,
                Note.deleted_at.is_(None),
            )
            .order_by(Note.created_at.desc())
        ).all()

        note_stats = []
        for n, pcount, dcount, earned_cents in note_stats_rows:
            note_stats.append(
                {
                    "id": n.id,
                    "title": n.title,
                    "university": n.university,
                    "faculty": n.faculty,
                    "career": n.career,
                    "purchases": int(pcount or 0),
                    "downloads": int(dcount or 0),
                    "earned_cents": int(earned_cents or 0),
                    "earned_ars": float(money_1_decimal(cents_to_amount(int(earned_cents or 0)))),
                    "is_free": int(getattr(n, "price_cents", 0) or 0) <= 0,
                }
            )


    return render_template("my_notes_hub.html", my_notes=my_notes, combos=combos, notes=combo_notes, stats_totals=stats_totals, note_stats=note_stats)



@app.get("/my-content/edit")
@login_required
def my_content_edit():
    """Página unificada para ver/editar/borrar apuntes y combos del usuario."""
    initial_tab = (request.args.get("tab") or "notes").lower()
    if initial_tab not in ("notes","combos"):
        initial_tab = "notes"
    with Session() as s:
        # Apuntes del usuario + cantidad de descargas (compras aprobadas)
        rows = s.execute(
            select(
                Note,
                func.count(DownloadLog.id).label("download_count"),
            )
            .outerjoin(
                DownloadLog,
                DownloadLog.note_id == Note.id,
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

    return render_template("my_content_edit.html", my_notes=my_notes, combos=combos, initial_tab=initial_tab)


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
    return redirect(url_for("note_detail", note_id=note_id))


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
    return redirect(url_for("my_content_edit", tab="notes"))

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
@app.route("/my-purchases")
@login_required
def my_purchases():
    """Unified page: paid purchases + free downloads (notes + combos)."""
    with Session() as s:
        # ---- Paid purchases (notes + combos) ----
        rows = s.execute(
            select(Purchase, Note, Combo)
            .outerjoin(Note, Note.id == Purchase.note_id)
            .outerjoin(Combo, Combo.id == Purchase.combo_id)
            .where(
                Purchase.buyer_id == current_user.id,
                Purchase.status == "approved",
            )
            .order_by(Purchase.created_at.desc())
        ).all()

        paid_items = []
        for p, n, c in rows:
            buyer_price_cents = int(getattr(p, "amount_cents", 0) or 0)

            if getattr(p, "note_id", None) and n:
                paid_items.append(dict(
                    kind="note",
                    purchase_id=p.id,
                    content_id=n.id,
                    title=n.title,
                    price_cents=buyer_price_cents,
                    created_at=p.created_at.strftime("%d/%m/%Y %H:%M"),
                    view_url=url_for("note_detail", note_id=n.id),
                    download_url=url_for("download_note", note_id=n.id),
                ))
            elif getattr(p, "combo_id", None) and c:
                paid_items.append(dict(
                    kind="combo",
                    purchase_id=p.id,
                    content_id=c.id,
                    title=c.title,
                    price_cents=buyer_price_cents,
                    created_at=p.created_at.strftime("%d/%m/%Y %H:%M"),
                    view_url=url_for("combo_detail", combo_id=c.id),
                    download_url=url_for("download_combo", combo_id=c.id),
                ))

        # ---- Free downloads (notes + combos) ----
        # We show only downloads marked as free to avoid duplicates with paid purchases.
        dl_rows = s.execute(
            select(DownloadLog, Note, Combo)
            .outerjoin(Note, Note.id == DownloadLog.note_id)
            .outerjoin(Combo, Combo.id == DownloadLog.combo_id)
            .where(
                DownloadLog.user_id == current_user.id,
                DownloadLog.is_free == True,
            )
            .order_by(DownloadLog.created_at.desc())
        ).all()

        free_items = []
        for dl, n, c in dl_rows:
            if getattr(dl, "note_id", None) and n:
                free_items.append(dict(
                    kind="note",
                    content_id=n.id,
                    title=n.title,
                    created_at=dl.created_at.strftime("%d/%m/%Y %H:%M"),
                    view_url=url_for("note_detail", note_id=n.id),
                    download_url=url_for("download_note", note_id=n.id),
                ))
            elif getattr(dl, "combo_id", None) and c:
                free_items.append(dict(
                    kind="combo",
                    content_id=c.id,
                    title=c.title,
                    created_at=dl.created_at.strftime("%d/%m/%Y %H:%M"),
                    view_url=url_for("combo_detail", combo_id=c.id),
                    download_url=url_for("download_combo", combo_id=c.id),
                ))

    return render_template("my_purchases.html", paid_items=paid_items, free_items=free_items)





# -----------------------------------------------------------------------------
# Upload / Detail / Download
# -----------------------------------------------------------------------------
from datetime import datetime, timedelta
import os
from flask import request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename

# Si usás rq:
try:
    from rq import Queue
except Exception:
    Queue = None

def _get_preview_queue():
    """
    Returns an RQ Queue if REDIS_URL exists and rq is installed.
    """
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url or Queue is None:
        return None
    try:
        import redis
        conn = redis.from_url(redis_url)
        return Queue("previews", connection=conn, default_timeout=600)  # 10 min
    except Exception:
        return None

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_note():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        description = (request.form.get("description") or "").strip()

        university = (request.form.get("university") or "").strip()
        faculty = (request.form.get("faculty") or "").strip()
        career = (request.form.get("career") or "").strip()

        if not title:
            flash("Ingresá un título.", "warning")
            return redirect(url_for("upload_note"))

        if not university:
            flash("Seleccioná tu Universidad.", "warning")
            return redirect(url_for("upload_note"))
        if not faculty:
            flash("Seleccioná tu Facultad.", "warning")
            return redirect(url_for("upload_note"))
        if not career:
            flash("Seleccioná tu Carrera.", "warning")
            return redirect(url_for("upload_note"))

        price = (request.form.get("price") or "").strip()
        try:
            price_cents = int(round(float(price) * 100)) if price else 0
            if price_cents < 0:
                raise ValueError("negative")
        except Exception:
            flash("Precio inválido.", "warning")
            return redirect(url_for("upload_note"))

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
            flash("Seleccioná un PDF.", "warning")
            return redirect(url_for("upload_note"))
        if not allowed_pdf(file.filename):
            flash("Sólo PDF.", "warning")
            return redirect(url_for("upload_note"))

        # Enforce max file size (defense-in-depth)
        try:
            file.stream.seek(0, os.SEEK_END)
            size = int(file.stream.tell() or 0)
            file.stream.seek(0)
            max_len = int(app.config.get("MAX_CONTENT_LENGTH") or 0)
            if max_len and size and size > max_len:
                flash(f"El archivo es muy grande. Máximo permitido: {int(max_len/1024/1024)}MB.", "warning")
                return redirect(url_for("upload_note"))
        except Exception:
            pass

        base_name = secure_filename(file.filename)
        unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{base_name}"

        # Guardamos temporal local
        ensure_dirs()
        local_pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(local_pdf_path)

        # Security: validate PDF (best-effort)
        try:
            ok, msg = _validate_uploaded_pdf(local_pdf_path, max_pages=int(os.getenv("MAX_PDF_PAGES", "1200")))
            if not ok:
                try:
                    os.remove(local_pdf_path)
                except Exception:
                    pass
                flash(msg or "PDF inválido.", "danger")
                return redirect(url_for("upload_note"))
        except Exception:
            # best-effort: si falla inesperado, no bloqueamos
            pass

        # Subimos PDF a R2 (tu gcs_bucket)
        if gcs_bucket:
            blob_name = f"notes/{current_user.id}/{unique_name}"
            try:
                gcs_upload_path(local_pdf_path, blob_name, content_type="application/pdf")
            except Exception as e:
                # Si falla el upload al storage, frenamos acá (si no, queda inconsistente)
                try:
                    app.logger.warning(f"storage upload failed: {e}")
                except Exception:
                    pass
                flash("Error subiendo el PDF al almacenamiento. Probá de nuevo.", "danger")
                return redirect(url_for("upload_note"))
            stored_path = blob_name
        else:
            stored_path = unique_name

        note_id = None
        moderation_status_for_msg = "auto_published"

        with Session() as s:
            note = Note(
                title=title,
                description=description,
                university=university,
                faculty=faculty,
                career=career,
                price_cents=price_cents,       # legacy
                seller_net_cents=price_cents,  # canonical
                file_path=stored_path,
                seller_id=current_user.id
            )

            note.moderation_status = "auto_published"
            s.add(note)
            s.commit()

            note_id = int(note.id)

            # Auditoría
            try:
                log_audit_event(
                    actor_user_id=int(getattr(current_user, "id", 0) or 0) or None,
                    action="note_uploaded",
                    target_type="note",
                    target_id=int(note_id),
                    meta={
                        "note_id": int(note_id),
                        "title": title,
                        "seller_id": int(getattr(current_user, "id", 0) or 0) or None,
                        "seller_email": getattr(current_user, "email", None),
                        "university": university,
                        "faculty": faculty,
                        "career": career,
                        "seller_net_cents": int(price_cents or 0),
                    },
                )
            except Exception:
                pass

            # ✅ Opción 1: PREVIEW EN BACKGROUND (NO en el request)
            try:
                enqueue_preview_generation(note_id)
            except Exception:
                pass

            # Moderación IA (best-effort)
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

                note.ai_decision = (ai.get("risk_level") or ai.get("decision") or None)
                note.ai_model = GEMINI_MODEL if GEMINI_API_KEY else None
                note.ai_summary = (ai.get("summary") or None)
                note.ai_raw = ai

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

                # Notifications (tu lógica)
                admin_ids = [u.id for u in s.execute(select(User).where(User.is_admin == True)).scalars().all()]

                if status in ("auto_published", "approved"):
                    notify_and_email_users(
                        s, [current_user.id],
                        kind="note_published",
                        title="Apunte publicado",
                        body="Tu apunte ya está publicado 🎉",
                        email_subject="Tu apunte ya está publicado",
                        email_body="Tu apunte ya está publicado.",
                        dedupe_key_prefix=f"note:{note_id}:published"
                    )
                elif status == "published_flagged":
                    notify_and_email_users(
                        s, [current_user.id],
                        kind="note_published_flagged",
                        title="Apunte publicado",
                        body="Tu apunte ya está publicado. Puede ser revisado de forma aleatoria.",
                        email_subject="Tu apunte ya está publicado",
                        email_body="Tu apunte ya está publicado. Puede ser revisado de forma aleatoria.",
                        dedupe_key_prefix=f"note:{note_id}:flagged"
                    )
                elif status in ("blocked_review", "pending_manual"):
                    note.manual_review_due_at = datetime.utcnow() + timedelta(hours=12)
                    notify_and_email_users(
                        s, [current_user.id],
                        kind="note_manual_review",
                        title="Apunte en revisión",
                        body="Tu apunte quedó en revisión. Puede demorar hasta 12hs.",
                        email_subject="Tu apunte está en revisión",
                        email_body="Tu apunte quedó en revisión. Puede demorar hasta 12hs.",
                        dedupe_key_prefix=f"note:{note_id}:manual"
                    )
                    notify_and_email_users(
                        s, admin_ids,
                        kind="manual_review_admin",
                        title="Revisión requerida",
                        body=f"Hay un apunte para revisar: #{note_id} — {note.title}",
                        email_subject="Apunte para revisar",
                        email_body=f"Hay un apunte para revisar: #{note_id} — {note.title}",
                        dedupe_key_prefix=f"note:{note_id}:admin_manual"
                    )

                s.commit()
                moderation_status_for_msg = status

            except Exception as e:
                try:
                    app.logger.warning(f"ai moderation failed: {e}")
                except Exception:
                    pass

        # Mensaje UX (usar variables locales, no el objeto note “detached”)
        msg = "Apunte subido."
        try:
            if moderation_status_for_msg in ("auto_published", "approved"):
                msg += " Ya está publicado 🎉"
            elif moderation_status_for_msg == "published_flagged":
                msg += " Ya está publicado. Puede ser revisado de forma aleatoria."
            elif moderation_status_for_msg in ("blocked_review", "pending_manual"):
                msg += " Quedó en revisión (puede demorar hasta 12hs)."
            elif moderation_status_for_msg == "rejected":
                msg += " Fue rechazado. Revisá el motivo en tu perfil."
            else:
                msg += " Quedó pendiente de revisión."
        except Exception:
            pass
        flash(msg)

        # Cleanup local PDF (si subimos a bucket, lo podemos borrar sin afectar al thread)
        try:
            if gcs_bucket and local_pdf_path and os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
        except Exception:
            pass

        return redirect(url_for("note_detail", note_id=note_id))

    return render_template("upload.html")



@app.route("/note/<int:note_id>")
def note_detail(note_id):
    paid_param = request.args.get("paid", "0")
    gen_preview = request.args.get("gen_preview", "0") == "1"

    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)

        # Moderation visibility
        if not is_public_moderation_status(getattr(note, "moderation_status", "approved")):
            if not current_user.is_authenticated:
                abort(404)
            if current_user.id != note.seller_id and not getattr(current_user, "is_admin", False):
                abort(404)

        # Archived visibility (igual)
        if bool(getattr(note, "is_archived", False)):
            is_admin = bool(current_user.is_authenticated and getattr(current_user, "is_admin", False))
            is_owner = bool(current_user.is_authenticated and current_user.id == note.seller_id)
            if not (is_admin or is_owner):
                has_access = False
                if current_user.is_authenticated:
                    try:
                        if int(note.price_cents or 0) <= 0:
                            has_access = s.execute(
                                select(DownloadLog.id).where(
                                    DownloadLog.user_id == current_user.id,
                                    DownloadLog.note_id == note.id,
                                ).limit(1)
                            ).scalar_one_or_none() is not None
                        else:
                            has_access = s.execute(
                                select(Purchase.id).where(
                                    Purchase.buyer_id == current_user.id,
                                    Purchase.note_id == note.id,
                                    Purchase.status == 'approved'
                                ).limit(1)
                            ).scalar_one_or_none() is not None
                    except Exception:
                        has_access = False
                if not has_access:
                    abort(404)

        # Can download
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

        # Preview meta
        try:
            imgs_meta = (getattr(note, "preview_images", None) or {})
            imgs = (imgs_meta.get("images") or []) if isinstance(imgs_meta, dict) else []
        except Exception:
            imgs = []

        is_owner = bool(current_user.is_authenticated and current_user.id == note.seller_id)
        is_admin = bool(current_user.is_authenticated and getattr(current_user, "is_admin", False))
        can_generate_preview = bool(is_owner or is_admin)

        # ✅ Generación manual (ENGRANAJE)
        if gen_preview and can_generate_preview:
            try:
                pages, new_imgs = generate_note_preview(note, max_pages=int(os.getenv("PREVIEW_MAX_PAGES", "4")))
                if new_imgs:
                    note.preview_pages = {"pages": pages}
                    note.preview_images = {"images": new_imgs}
                    s.commit()
                    imgs = new_imgs
                    flash("Preview generada ✅", "success")
                else:
                    flash("No se pudo generar la preview (PDF vacío o inválido).", "warning")
            except Exception as e:
                try:
                    app.logger.warning(f"preview generation failed: {e}")
                except Exception:
                    pass
                flash("Falló la preview (memoria/archivo). Probá con menos páginas o bajando escala.", "warning")

        # Analytics best-effort
        try:
            uid = int(current_user.id) if getattr(current_user, "is_authenticated", False) else None
            log_analytics_event(event="note_view", user_id=uid, path=request.path, note_id=int(note_id))
        except Exception:
            pass

        # Downloads metric
        try:
            dl = s.execute(select(func.count(DownloadLog.id)).where(DownloadLog.note_id == note.id)).scalar_one()
            note.download_count = int(dl or 0)
        except Exception:
            pass

        # Reviews
        rows = s.execute(
            select(Review, User.name)
            .join(User, User.id == Review.buyer_id)
            .where(Review.note_id == note.id)
            .order_by(Review.created_at.desc())
        ).all()
        reviews = rows
        avg_rating = round(sum(r.rating for r, _ in reviews) / len(reviews), 2) if reviews else None

        # Can review
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
                has_purchase = True

            if has_purchase:
                already_reviewed = s.execute(
                    select(Review).where(Review.note_id == note.id, Review.buyer_id == current_user.id)
                ).scalar_one_or_none() is not None
                can_review = not already_reviewed

        seller = s.get(User, note.seller_id) if note.seller_id else None

        # Seller contacts (igual)
        seller_contacts = []
        if seller:
            if getattr(seller, "seller_contact", None):
                url, _ = _build_contact_link(getattr(seller, "seller_contact"))
                if url:
                    seller_contacts.append((url, "Contacto"))

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

        seller_verified = False
        if seller and getattr(seller, "mp_access_token", None):
            sold = s.execute(
                select(func.count(Purchase.id))
                .join(Note, Note.id == Purchase.note_id)
                .where(Note.seller_id == seller.id, Purchase.status == "approved", Purchase.amount_cents > 0)
            ).scalar_one()
            seller_verified = (sold or 0) >= 1

        base_price = note.price_cents / 100.0 if note.price_cents else 0.0
        buyer_price = round(base_price * GROSS_MULTIPLIER, 2) if base_price > 0 else None

        # Preview order
        try:
            _imgs = (getattr(note, "preview_images", None) or {}).get("images") or []
            preview_idxs = list(range(1, len(_imgs) + 1))
            random.shuffle(preview_idxs)
            preview_idxs = preview_idxs[:min(4, len(preview_idxs))]
        except Exception:
            preview_idxs = []

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
            preview_idxs=preview_idxs,
            can_generate_preview=can_generate_preview,
            preview_has_images=bool(imgs),
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
                Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
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
    from io import BytesIO

    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)

        # Public viewers can access previews for public notes
        status = getattr(note, "moderation_status", "approved")
        if not is_public_moderation_status(status):
            allowed = False
            try:
                if current_user.is_authenticated and (
                    current_user.id == note.seller_id or getattr(current_user, "is_admin", False)
                ):
                    allowed = True
            except Exception:
                allowed = False
            if not allowed:
                abort(404)

        meta = getattr(note, "preview_images", None) or {}
        imgs = (meta or {}).get("images") or []
        if idx < 1 or idx > len(imgs):
            abort(404)
        path = imgs[idx - 1]

        # ✅ Bucket (tu R2 real vía gcs_*)
        if gcs_bucket and path and "/" in path:
            data = gcs_download_bytes(path)
            return send_file(
                BytesIO(data),
                mimetype="image/jpeg",
                as_attachment=False,
                download_name=f"preview_{note_id}_{idx}.jpg",
                max_age=60 * 5,
            )

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
            # Anti-enumeration: do not reveal if a paid note exists
            if not (is_admin or is_owner):
                abort(404)
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
                    was_free=is_free,
                )
            )
            # A5 stats
            try:
                stats_daily_add_download(s, is_free=bool(is_free))
            except Exception:
                pass
            s.commit()

            # Gestión / auditoría (descarga)
            try:
                log_audit_event(
                    actor_user_id=int(getattr(current_user, "id", 0) or 0) or None,
                    action="note_downloaded",
                    target_type="note",
                    target_id=int(note.id),
                    meta={
                        "note_id": int(note.id),
                        "note_title": getattr(note, "title", None),
                        "seller_id": int(getattr(note, "seller_id", 0) or 0) or None,
                        "buyer_id": int(getattr(current_user, "id", 0) or 0) or None,
                        "buyer_email": getattr(current_user, "email", None),
                        "is_free": bool(is_free),
                    },
                )
            except Exception:
                pass

            # Analytics
            try:
                log_analytics_event(
                    event="download",
                    user_id=int(current_user.id),
                    path=request.path,
                    note_id=int(note.id),
                    combo_id=None,
                    meta={"is_free": bool(is_free)},
                )
            except Exception:
                pass
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

    # --- Path traversal hardening (local) ---
    if ("/" in note_file_path) or ("\\" in note_file_path) or (".." in note_file_path):
        app.logger.warning("Blocked suspicious file_path: %r", note_file_path)
        abort(404)

    # Ensure file exists before sending
    try:
        full_path = os.path.join(app.config["UPLOAD_FOLDER"], note_file_path)
        if not os.path.exists(full_path):
            flash("El archivo no está disponible en este momento.", "danger")
            return redirect(url_for("note_detail", note_id=note_id))
    except Exception:
        pass

    return send_from_directory(app.config["UPLOAD_FOLDER"], note_file_path, as_attachment=True)



@app.route("/combos/<int:combo_id>/download", methods=["GET","POST"])
@login_required
def download_combo(combo_id):
    """Download a combo as a .zip (contains the PDF files of its notes)."""
    with Session() as s:
        combo = s.get(Combo, combo_id)
        if not combo or (hasattr(combo, "is_active") and combo.is_active is False):
            abort(404)

        if not is_public_moderation_status(getattr(combo, "moderation_status", "approved")):
            # Owner/admin can still download
            is_owner = bool(combo.seller_id == getattr(current_user, "id", None))
            is_admin = bool(getattr(current_user, "is_admin", False))
            if not (is_owner or is_admin):
                abort(404)

        is_admin = bool(getattr(current_user, "is_admin", False))
        is_owner = bool(combo.seller_id == getattr(current_user, "id", None))

        buyer_price_cents = _combo_buyer_price_cents(combo)
        is_free = int(buyer_price_cents or 0) <= 0

        allowed = False
        if is_admin or is_owner:
            allowed = True
        elif bool(getattr(current_user, "is_premium", False)):
            allowed = True
        elif is_free:
            allowed = True
        else:
            # Paid combo: needs approved purchase
            has_cp = s.execute(
                select(ComboPurchase.id).where(
                    ComboPurchase.buyer_id == current_user.id,
                    ComboPurchase.combo_id == combo.id,
                    ComboPurchase.status == "approved",
                )
            ).scalar_one_or_none() is not None
            allowed = bool(has_cp)

        if not allowed:
            flash("No tenés acceso a este combo.", "danger")
            return redirect(url_for("combo_detail", combo_id=combo.id))

        note_ids = (
            s.execute(select(ComboNote.note_id).where(ComboNote.combo_id == combo.id))
            .scalars()
            .all()
        )

        note_files = []
        if note_ids:
            _notes = s.execute(select(Note).where(Note.id.in_(note_ids))).scalars().all()
            for n in _notes:
                fp = getattr(n, "file_path", None)
                title = getattr(n, "title", "") or f"apunte_{getattr(n,'id', '')}"
                nid = getattr(n, "id", None)
                if fp:
                    note_files.append((fp, title, nid))

        # Log combo download (best effort)
        try:
            s.add(
                DownloadLog(
                    user_id=current_user.id,
                    note_id=None,
                    combo_id=combo.id,
                    is_free=is_free,
                    was_free=is_free,
                )
            )
            # A5 stats
            try:
                stats_daily_add_download(s, is_free=bool(is_free))
            except Exception:
                pass
            s.commit()

            # Gestión / auditoría (descarga de combo)
            try:
                log_audit_event(
                    actor_user_id=int(getattr(current_user, "id", 0) or 0) or None,
                    action="combo_downloaded",
                    target_type="combo",
                    target_id=int(combo.id),
                    meta={
                        "combo_id": int(combo.id),
                        "combo_title": getattr(combo, "title", None),
                        "seller_id": int(getattr(combo, "seller_id", 0) or 0) or None,
                        "buyer_id": int(getattr(current_user, "id", 0) or 0) or None,
                        "buyer_email": getattr(current_user, "email", None),
                        "is_free": bool(is_free),
                        "note_ids": [int(nid) for (_fp, _t, nid) in note_files if nid is not None],
                    },
                )
            except Exception:
                pass

            # Analytics
            try:
                log_analytics_event(
                    event="download",
                    user_id=int(current_user.id),
                    path=request.path,
                    combo_id=int(combo.id),
                    meta={"is_free": bool(is_free)},
                )
            except Exception:
                pass
        except Exception as e:
            try:
                s.rollback()
            except Exception:
                pass
            app.logger.warning("Combo DownloadLog insert failed: %s", e)

    # Build zip in-memory
    from io import BytesIO
    import zipfile as _zipfile

    buf = BytesIO()
    with _zipfile.ZipFile(buf, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
        for fp, title, nid in note_files:
            # filename: keep original basename, fallback to title
            base = os.path.basename(fp) or ""
            if not base:
                safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (title or "").strip())[:80]
                base = f"{safe or ('apunte_' + str(nid or ''))}.pdf"

            try:
                data = None
                if gcs_bucket and "/" in fp:
                    data = gcs_download_bytes(fp)
                else:
                    # local file
                    # --- Combo path traversal hardening ---
                    if ("/" in fp) or ("\\" in fp) or (".." in fp):
                        app.logger.warning("Blocked suspicious combo file_path: %r", fp)
                        continue
                    local_path = os.path.join(app.config["UPLOAD_FOLDER"], fp)
                    if not os.path.exists(local_path):
                        continue
                    with open(local_path, "rb") as f:
                        data = f.read()
                if data:
                    zf.writestr(base, data)
            except Exception as e:
                app.logger.warning("Combo download: failed to include note %s: %s", nid or "?", e)

    buf.seek(0)

    combo_title = (getattr(combo, "title", "") or f"combo_{combo.id}").strip()
    safe_title = re.sub(r"[^a-zA-Z0-9._-]+", "_", combo_title)[:60] or f"combo_{combo.id}"
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_title}.zip",
    )

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

@app.route("/mp/disconnect", methods=["GET"])
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


@app.post("/mp/disconnect")
@login_required
def disconnect_mp_post():
    """POST version to avoid accidental clicks."""
    return disconnect_mp()


# -----------------------------------------------------------------------------
# Cuenta: suspender/reactivar/eliminar
# -----------------------------------------------------------------------------
@app.post("/account/suspend")
@login_required
def account_suspend():
    with Session() as s:
        u = s.get(User, current_user.id)
        if not u:
            abort(404)
        u.is_suspended = True
        u.suspended_at = datetime.utcnow()
        s.commit()

    code = log_audit_event(actor_user_id=current_user.id, action="user_suspend", target_type="user", target_id=current_user.id)
    flash(f"Tu cuenta quedó suspendida. Nº de gestión: {code}", "warning")
    return redirect(url_for("profile"))


@app.post("/account/reactivate")
@login_required
def account_reactivate():
    with Session() as s:
        u = s.get(User, current_user.id)
        if not u:
            abort(404)
        u.is_suspended = False
        u.suspended_at = None
        s.commit()

    code = log_audit_event(actor_user_id=current_user.id, action="user_reactivate", target_type="user", target_id=current_user.id)
    flash(f"Tu cuenta se reactivó correctamente. Nº de gestión: {code}", "success")
    return redirect(url_for("profile"))


@app.post("/account/delete")
@login_required
def account_delete():
    """Eliminación 'definitiva' (soft-delete + anonimización) del usuario.

    - Desvincula Mercado Pago
    - Elimina compras/descargas del comprador
    - Borra/anonimiza datos personales
    - Mantiene apuntes/combos subidos (para quienes ya los descargaron/compraron)
    """
    confirm = (request.form.get("confirm") or "").strip().lower()
    if confirm != "ELIMINAR".lower():
        flash("Para eliminar tu cuenta escribí ELIMINAR en la confirmación.", "warning")
        return redirect(url_for("profile"))

    with Session() as s:
        u = s.get(User, current_user.id)
        if not u:
            abort(404)

        # --- desvincular MP ---
        u.mp_user_id = None
        u.mp_access_token = None
        u.mp_refresh_token = None
        u.mp_token_expires_at = None

        # --- borrar compras y descargas (historial del comprador) ---
        try:
            s.execute(text("DELETE FROM purchases WHERE buyer_id = :uid"), {"uid": u.id})
        except Exception:
            # fallback ORM
            try:
                for p in s.execute(select(Purchase).where(Purchase.buyer_id == u.id)).scalars().all():
                    s.delete(p)
            except Exception:
                pass

        # Combos purchases (si existe la tabla)
        try:
            s.execute(text("DELETE FROM combo_purchases WHERE buyer_id = :uid"), {"uid": u.id})
        except Exception:
            try:
                for p in s.execute(select(ComboPurchase).where(ComboPurchase.buyer_id == u.id)).scalars().all():
                    s.delete(p)
            except Exception:
                pass

        try:
            s.execute(text("DELETE FROM download_logs WHERE user_id = :uid"), {"uid": u.id})
        except Exception:
            try:
                for d in s.execute(select(DownloadLog).where(DownloadLog.user_id == u.id)).scalars().all():
                    s.delete(d)
            except Exception:
                pass

        # --- borrar notificaciones del usuario (opcional, pero alineado con 'borra datos') ---
        try:
            s.execute(text("DELETE FROM notifications WHERE user_id = :uid"), {"uid": u.id})
        except Exception:
            pass

        # ACCOUNT_DELETE_ARCHIVE: ocultamos contenidos del vendedor para nuevos compradores,
        # pero mantenemos acceso a quienes ya lo descargaron/compraron.
        try:
            now = datetime.utcnow()
            s.execute(update(Note).where(Note.seller_id == u.id).values(
                is_archived=True,
                archived_at=now,
                archived_reason="Cuenta eliminada"
            ))
            s.execute(update(Combo).where(Combo.seller_id == u.id).values(
                is_archived=True,
                archived_at=now,
                archived_reason="Cuenta eliminada"
            ))
        except Exception:
            pass

        # --- anonimización ---
        u.deleted_at = datetime.utcnow()
        u.is_suspended = True
        u.suspended_at = datetime.utcnow()
        u.is_active = False  # hard disable to avoid re-login with same Google email

        # Guardamos un placeholder único para no romper unique(email)
        placeholder_email = f"deleted_{u.id}_{int(datetime.utcnow().timestamp())}@apuntesya.local"
        u.email = placeholder_email
        u.name = "Usuario eliminado"
        u.phone = None
        u.phone_verified = False
        u.phone_verified_at = None
        u.imagen_de_perfil = None

        # contactos
        u.seller_contact = None
        u.contact_email = None
        u.contact_whatsapp = None
        u.contact_phone = None
        u.contact_website = None
        u.contact_instagram = None
        u.contact_visible_public = False
        u.contact_visible_buyers = False

        s.commit()

    code = log_audit_event(actor_user_id=None, action="user_delete", target_type="user", target_id=current_user.id)
    logout_user()
    flash(f"Tu cuenta fue eliminada. Nº de gestión: {code}", "info")
    return redirect(url_for("index"))

# -----------------------------------------------------------------------------
# Comprar
# -----------------------------------------------------------------------------
@app.route("/buy/<int:note_id>", methods=["GET","POST"])
@login_required
def buy_note(note_id):
    # A6 analytics: intent to buy (best-effort)
    try:
        log_analytics_event(
            event="buy_intent",
            user_id=int(current_user.id) if getattr(current_user, "is_authenticated", False) else None,
            path=request.path,
            note_id=int(note_id),
            meta={"source": "buy_note_route"},
        )
    except Exception:
        pass

    # Kill-switch (ventas)
    se = (_get_setting("sales_enabled", "1") or "1").strip()
    sales_enabled = se in ("1", "true", "True", "yes", "on")
    if not sales_enabled and not (_is_superadmin(current_user) if getattr(current_user, "is_authenticated", False) else False):
        flash("⚠️ Las ventas están pausadas por mantenimiento. Probá de nuevo más tarde.", "warning")
        return redirect(url_for("note_detail", note_id=note_id))
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)
        if not is_public_moderation_status(getattr(note, "moderation_status", "approved")):
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
        # (y completamos campos extra para admin/estadísticas si existen)
        gross_cents = int(buyer_price_cents or 0)
        try:
            mp_fee_cents = int(round(gross_cents * (float(MP_FEE_IMMEDIATE_TOTAL_PCT) / 100.0)))
        except Exception:
            mp_fee_cents = 0

        # Checkout confirmation (UX + legal)
        if request.method == "GET":
            gross_ars = money_1_decimal(cents_to_amount(gross_cents))
            return render_template(
                "checkout_note.html",
                note=note,
                seller=seller,
                gross_cents=gross_cents,
                gross_ars=gross_ars,
                legal_version=(app.config.get("LEGAL_VERSION") or "").strip(),
            )

        # On POST we require explicit acknowledgement
        if request.method == "POST":
            if request.form.get("ack") != "1":
                flash("Antes de pagar, tenés que confirmar la compra y aceptar la política de reembolso.", "warning")
                return redirect(url_for("buy_note", note_id=note.id))
        # Platform fee = lo que queda para ApuntesYa (sin MP) luego de que el vendedor reciba su neto
        try:
            platform_fee_cents = max(0, gross_cents - mp_fee_cents - int(net_cents or 0))
        except Exception:
            platform_fee_cents = 0

        p = Purchase(
            buyer_id=current_user.id,
            note_id=note.id,
            status="pending",
            amount_cents=gross_cents,
            # admin fields (nullable in DB)
            buyer_email=getattr(current_user, "email", None),
            seller_id=int(getattr(note, "seller_id", 0) or 0) or None,
            gross_cents=gross_cents,
            platform_fee_cents=platform_fee_cents,
            mp_fee_cents=mp_fee_cents,
            seller_net_cents=int(net_cents or 0),
        )
        s.add(p)
        s.commit()

        # Gestión / auditoría (compra iniciada)
        try:
            code = log_audit_event(
                actor_user_id=current_user.id,
                action="purchase_created",
                target_type="purchase",
                target_id=int(p.id),
                meta={
                    "purchase_id": int(p.id),
                    "note_id": int(note.id),
                    "note_title": getattr(note, "title", None),
                    "buyer_id": int(current_user.id),
                    "buyer_email": getattr(current_user, "email", None),
                    "seller_id": int(getattr(note, "seller_id", 0) or 0) or None,
                    "amount_cents": int(gross_cents or 0),
                    "seller_net_cents": int(net_cents or 0),
                    "mp_fee_cents_est": int(mp_fee_cents or 0),
                },
            )
            # Nota: no mostramos el Nº de gestión al comprador para no confundir;
            # queda en Gestiones (admin) para auditoría.
        except Exception:
            pass

        price_ars = float(money_1_decimal(cents_to_amount(buyer_price_cents)))  # final comprador (P), 1 decimal
        # Comisión de la plataforma (se cobra automáticamente vía marketplace_fee)
        platform_fee_percent = float(APY_RATE)
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

            # Protect webhook endpoint with a shared secret (query param) so random callers
            # can't spam it. MP will call exactly this URL.
            wh_secret = (app.config.get("MP_WEBHOOK_SECRET") or "").strip()
            notification_url = (
                url_for("mp_webhook", _external=True, secret=wh_secret)
                if wh_secret else url_for("mp_webhook", _external=True)
            )

            pref = mp.create_preference_for_seller_token(
                seller_access_token=use_token,
                title=note.title,
                unit_price=price_ars,
                quantity=1,
                marketplace_fee=marketplace_fee,
                external_reference=f"purchase:{p.id}",
                back_urls=back_urls,
                notification_url=notification_url
            )

            with Session() as s2:
                p2 = s2.get(Purchase, p.id)
                if p2:
                    p2.preference_id = pref.get("id") or pref.get("preference_id")
                    s2.commit()
            init_point = pref.get("init_point") or pref.get("sandbox_init_point")

            # A6 analytics: checkout started (redirect to MP)
            try:
                log_analytics_event(
                    event="checkout_start",
                    user_id=int(current_user.id) if getattr(current_user, "is_authenticated", False) else None,
                    path=request.path,
                    note_id=int(note.id) if note else int(note_id),
                    meta={
                        "purchase_id": int(p.id),
                        "preference_id": (pref.get("id") or pref.get("preference_id")),
                    },
                )
            except Exception:
                pass
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
            # Emit notifications/emails (idempotent by dedupe key)
            try:
                if p and p.id:
                    _emit_note_purchase_notifications(p.id)
            except Exception:
                pass
            flash("✅ Pago aprobado, ya podés descargar el apunte.")
            # Volvemos al detalle, marcando que viene de un pago
            return redirect(
                url_for("note_detail", note_id=note_id, paid=1, _anchor="download")
            )


    # Si llegamos acá, no pudimos confirmar “approved”
    flash("Registramos tu intento de pago. Si ya figura aprobado en Mercado Pago, el botón de descarga se habilitará en unos instantes.")
    return redirect(url_for("note_detail", note_id=note_id, _anchor="download"))


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Purchase notifications (in-app + email)
# -----------------------------------------------------------------------------
def _emit_note_purchase_notifications(purchase_id: int):
    """Create buyer+seller notifications/emails when a note purchase is approved."""
    with Session() as s:
        p = s.get(Purchase, purchase_id)
        if not p or (p.status or "").lower() != "approved" or not p.note_id:
            return

        note = s.get(Note, p.note_id)
        if not note:
            return
        buyer = s.get(User, p.buyer_id) if p.buyer_id else None
        seller = s.get(User, note.seller_id) if note.seller_id else None

        title_buyer = "✅ Compra confirmada"
        body_buyer = f"Compraste “{note.title}”. Ya podés descargarlo desde tu perfil (Compras)."
        title_seller = "💰 ¡Vendiste un apunte!"
        buyer_name = (buyer.name if buyer else "Un comprador")
        body_seller = f"""{buyer_name} compró tu apunte “{note.title}”."""

        # Dedupe keys include purchase id
        notify_and_email_users(
            s,
            user_ids=[p.buyer_id] if p.buyer_id else [],
            kind="purchase_buyer",
            title=title_buyer,
            body=body_buyer,
            email_subject="Compra confirmada en ApuntesYa",
            email_body=body_buyer,
            dedupe_key_prefix=f"purchase:{p.id}:buyer",
        )

        notify_and_email_users(
            s,
            user_ids=[note.seller_id] if note.seller_id else [],
            kind="sale_seller",
            title=title_seller,
            body=body_seller,
            email_subject="¡Vendiste un apunte en ApuntesYa!",
            email_body=body_seller,
            dedupe_key_prefix=f"purchase:{p.id}:seller",
        )

        s.commit()


def _emit_combo_purchase_notifications(combo_purchase_id: int):
    """Create buyer+seller notifications/emails when a combo purchase is approved."""
    with Session() as s:
        cp = s.get(ComboPurchase, combo_purchase_id)
        if not cp or (cp.status or "").lower() != "approved" or not cp.combo_id:
            return

        combo = s.get(Combo, cp.combo_id)
        if not combo:
            return
        buyer = s.get(User, cp.buyer_id) if cp.buyer_id else None
        seller = s.get(User, combo.seller_id) if combo.seller_id else None

        title_buyer = "✅ Compra confirmada"
        body_buyer = f"Compraste el combo “{combo.title}”. Ya podés descargarlo desde tu perfil (Compras)."
        title_seller = "💰 ¡Vendiste un combo!"
        buyer_name = (buyer.name if buyer else "Un comprador")
        body_seller = f"""{buyer_name} compró tu combo “{combo.title}”."""

        notify_and_email_users(
            s,
            user_ids=[cp.buyer_id] if cp.buyer_id else [],
            kind="purchase_buyer",
            title=title_buyer,
            body=body_buyer,
            email_subject="Compra confirmada en ApuntesYa",
            email_body=body_buyer,
            dedupe_key_prefix=f"combo_purchase:{cp.id}:buyer",
        )

        notify_and_email_users(
            s,
            user_ids=[combo.seller_id] if combo.seller_id else [],
            kind="sale_seller",
            title=title_seller,
            body=body_seller,
            email_subject="¡Vendiste un combo en ApuntesYa!",
            email_body=body_seller,
            dedupe_key_prefix=f"combo_purchase:{cp.id}:seller",
        )

        s.commit()

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
                    old_status = (p.status or "").lower() if getattr(p, "status", None) else ""
                    p.payment_id = payment_id
                    if status:
                        p.status = status
                    new_status = (p.status or "").lower() if getattr(p, "status", None) else ""
                    # Gestión / auditoría (cambio de estado de pago)
                    try:
                        if (status is not None) and (new_status != old_status):
                            log_audit_event(
                                actor_user_id=None,
                                action="purchase_status",
                                target_type="purchase",
                                target_id=int(p.id),
                                meta={
                                    "purchase_id": int(p.id),
                                    "note_id": int(p.note_id) if p.note_id else None,
                                    "buyer_id": int(p.buyer_id) if p.buyer_id else None,
                                    "seller_id": int(getattr(p, "seller_id", 0) or 0) or None,
                                    "from": old_status,
                                    "to": new_status,
                                    "payment_id": payment_id,
                                },
                            )
                    except Exception:
                        pass

                    # A5 stats (idempotent)
                    try:
                        if (status or "").lower() == "approved":
                            stats_daily_add_purchase(s, p)
                    except Exception:
                        pass
                    s.commit()
                    # Analytics (best-effort)
                    try:
                        if (status or "").lower() == "approved":
                            log_analytics_event(
                                event="purchase_approved",
                                user_id=int(p.buyer_id) if p.buyer_id else None,
                                note_id=int(p.note_id) if p.note_id else None,
                                meta={"amount_cents": int(p.amount_cents or 0), "payment_id": payment_id},
                            )
                    except Exception:
                        pass
            # Emit notifications/emails if approved (idempotent)
            try:
                if (status or "").lower() == "approved":
                    _emit_note_purchase_notifications(pid)
            except Exception:
                pass
            return

        # =========================
        # COMBOS (NUEVO)
        # =========================
        if external_reference.startswith("combo_purchase:"):
            cp_id = int(external_reference.split(":", 1)[1])
            with Session() as s:
                cp = s.get(ComboPurchase, cp_id)
                if cp:
                    old_status = (cp.status or "").lower() if getattr(cp, "status", None) else ""
                    cp.payment_id = payment_id
                    if status:
                        cp.status = status
                    new_status = (cp.status or "").lower() if getattr(cp, "status", None) else ""
                    # Gestión / auditoría (cambio de estado de pago combo)
                    try:
                        if (status is not None) and (new_status != old_status):
                            log_audit_event(
                                actor_user_id=None,
                                action="combo_purchase_status",
                                target_type="combo_purchase",
                                target_id=int(cp.id),
                                meta={
                                    "combo_purchase_id": int(cp.id),
                                    "combo_id": int(cp.combo_id) if getattr(cp, "combo_id", None) else None,
                                    "buyer_id": int(cp.buyer_id) if getattr(cp, "buyer_id", None) else None,
                                    "from": old_status,
                                    "to": new_status,
                                    "payment_id": payment_id,
                                },
                            )
                    except Exception:
                        pass
                    s.commit()

                    # A6 analytics: combo purchase approved
                    try:
                        if (status or "").lower() == "approved":
                            log_analytics_event(
                                event="purchase_approved",
                                user_id=int(cp.buyer_id) if cp.buyer_id else None,
                                combo_id=int(cp.combo_id) if cp.combo_id else None,
                                meta={"amount_cents": int(getattr(cp, "amount_cents", 0) or 0), "payment_id": payment_id},
                            )
                    except Exception:
                        pass

                    # -------------------------------------------------
                    # Mirror en tabla purchases para que el admin (stats/movements)
                    # tenga una única fuente de movimientos.
                    # Idempotente: si ya existe un Purchase para este payment+combo+buyer,
                    # no vuelve a crearlo.
                    try:
                        if (status or "").lower() == "approved":
                            existing = s.execute(
                                select(Purchase).where(
                                    Purchase.payment_id == payment_id,
                                    Purchase.combo_id == cp.combo_id,
                                    Purchase.buyer_id == cp.buyer_id,
                                ).limit(1)
                            ).scalars().first()
                            if not existing:
                                combo = s.get(Combo, cp.combo_id) if cp.combo_id else None
                                buyer_u = s.get(User, cp.buyer_id) if cp.buyer_id else None
                                gross_cents = int(getattr(cp, "amount_cents", 0) or 0)
                                try:
                                    mp_fee_cents = int(round(gross_cents * (MP_FEE_IMMEDIATE_TOTAL_PCT / 100.0)))
                                except Exception:
                                    mp_fee_cents = 0
                                seller_net_cents = int(getattr(combo, "seller_net_cents", 0) or 0) if combo else 0
                                platform_fee_cents = max(0, gross_cents - mp_fee_cents - seller_net_cents)
                                s.add(
                                    Purchase(
                                        buyer_id=cp.buyer_id,
                                        note_id=None,
                                        combo_id=cp.combo_id,
                                        status="approved",
                                        amount_cents=gross_cents,
                                        payment_id=payment_id,
                                        # extra fields
                                        buyer_email=(getattr(buyer_u, "email", None) if buyer_u else None),
                                        seller_id=(getattr(combo, "seller_id", None) if combo else None),
                                        gross_cents=gross_cents,
                                        mp_fee_cents=mp_fee_cents,
                                        platform_fee_cents=platform_fee_cents,
                                        seller_net_cents=seller_net_cents,
                                    )
                                )
                                # A5 stats (best-effort, idempotent)
                                try:
                                    # flush to get object and mark counted
                                    s.flush()
                                    created_p = s.execute(
                                        select(Purchase).where(
                                            Purchase.payment_id == payment_id,
                                            Purchase.combo_id == cp.combo_id,
                                            Purchase.buyer_id == cp.buyer_id,
                                        ).limit(1)
                                    ).scalars().first()
                                    if created_p:
                                        stats_daily_add_purchase(s, created_p)
                                except Exception:
                                    pass
                                s.commit()
                    except Exception:
                        pass
                    # Analytics (best-effort)
                    try:
                        if (status or "").lower() == "approved":
                            log_analytics_event(
                                event="combo_purchase_approved",
                                user_id=int(cp.buyer_id) if cp.buyer_id else None,
                                combo_id=int(cp.combo_id) if cp.combo_id else None,
                                meta={"amount_cents": int(cp.amount_cents or 0), "payment_id": payment_id},
                            )
                    except Exception:
                        pass
            # Emit notifications/emails if approved (idempotent)
            try:
                if (status or "").lower() == "approved":
                    _emit_combo_purchase_notifications(cp_id)
            except Exception:
                pass
            return

    except Exception as e:
        try:
            app.logger.exception("upsert purchase error")
        except Exception:
            pass

def mp_webhook():
    # Webhook should be POST-only. If something hits this with GET, reject.
    if request.method == "GET":
        return ("Method Not Allowed", 405)

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
        # Return 5xx so the provider can retry instead of treating it as success.
        return {"ok": False, "error": str(e)}, 500

app.add_url_rule("/webhooks/mercadopago", view_func=mp_webhook, methods=["POST"], endpoint="mp_webhook")
app.add_url_rule("/mp/webhook",            view_func=mp_webhook, methods=["POST"], endpoint="mp_webhook_legacy")

# CSRF must be disabled for external callbacks
try:
    csrf.exempt(mp_webhook)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Términos y condiciones, politicas de privacidad y seguridad
# -----------------------------------------------------------------------------
@app.get("/legal/accept", endpoint="legal_accept")
@login_required
def legal_accept():
    return render_template("legal_accept.html", legal_version=(app.config.get("LEGAL_VERSION") or ""))


@app.post("/legal/accept", endpoint="legal_accept_post")
@login_required
def legal_accept_post():
    # checkbox required by template
    if request.form.get("accept") != "1":
        flash("Tenés que aceptar los términos y políticas para continuar.", "warning")
        return redirect(url_for("legal_accept"))

    next_url = (request.form.get("next") or request.args.get("next") or url_for("index"))

    # Prevent open-redirects (only allow same-site relative paths)
    try:
        parsed = urlparse(str(next_url or ""))
        if parsed.scheme or parsed.netloc or not str(next_url).startswith("/"):
            next_url = url_for("index")
    except Exception:
        next_url = url_for("index")
    try:
        with Session() as s:
            u = s.get(User, int(current_user.id))
            if u:
                u.legal_version_accepted = (app.config.get("LEGAL_VERSION") or "").strip() or None
                u.legal_accepted_at = datetime.utcnow()
                # A8: auditoría histórica (best-effort)
                try:
                    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip() or None
                except Exception:
                    ip = None
                try:
                    ua = (request.headers.get("User-Agent") or "").strip() or None
                    if ua and len(ua) > 255:
                        ua = ua[:255]
                except Exception:
                    ua = None
                try:
                    ver = (u.legal_version_accepted or "").strip() or (app.config.get("LEGAL_VERSION") or "").strip() or "unknown"
                    s.add(LegalAcceptanceAudit(user_id=int(u.id), legal_version=ver, accepted_at=datetime.utcnow(), ip=ip, user_agent=ua))
                except Exception:
                    pass
                s.commit()
                try:
                    _audit("legal_accept", target_type="user", target_id=int(u.id), meta={"legal_version": u.legal_version_accepted, "ip": ip, "ua": ua})
                except Exception:
                    pass
        flash("✅ Gracias. Términos y políticas aceptados.")
    except Exception:
        flash("No pudimos guardar tu aceptación. Probá de nuevo.", "warning")
        return redirect(url_for("legal_accept"))

    return redirect(next_url)


@app.route("/terms", endpoint="terms")
def terms():
    return render_template("terms.html")  # <-- tu TyC existente

# Alias en español (opcional, recomendado)
@app.route("/terminos")
def terminos_redirect():
    return redirect(url_for("terms"))


# Privacidad
@app.route("/privacidad", endpoint="privacidad")
def politica_privacidad():
    return render_template("politica_privacidad.html")

# Alias en inglés (para templates antiguos)
@app.route("/privacy", endpoint="privacy")
def privacy_redirect():
    return redirect(url_for("privacidad"))


# Seguridad
@app.route("/seguridad", endpoint="seguridad")
def politica_seguridad():
    return render_template("politica_seguridad.html")

# Alias en inglés (para templates antiguos)
@app.route("/security", endpoint="security")
def security_redirect():
    return redirect(url_for("seguridad"))

# -----------------------------------------------------------------------------
# Centro de ayuda / FAQ (hotfix estático)
# -----------------------------------------------------------------------------
# Nota: en este proyecto usamos SQLAlchemy core (Session) y no hay un modelo FAQ
# dinámico en DB. Para evitar que el sitio caiga por endpoints inexistentes,
# exponemos una página de ayuda simple y estable.
@app.route("/faq", endpoint="faq")
@app.route("/ayuda", endpoint="ayuda")
def faq_static():
    return render_template("help/faq_static.html")

# -----------------------------------------------------------------------------
# Reportar apunte
# -----------------------------------------------------------------------------
@app.route("/note/<int:note_id>/report", methods=["POST"])
@login_required
def report_note(note_id):
    reason = (request.form.get("reason") or "other").strip()[:80] or "other"
    details = (request.form.get("details") or "").strip() or None

    with Session() as s:
        n = s.get(Note, note_id)
        if not n:
            abort(404)

        # Marcar legacy flag (si existe) para visibilidad rápida
        try:
            if hasattr(n, "is_reported"):
                n.is_reported = True
        except Exception:
            pass

        seller_id = int(getattr(n, "seller_id", 0) or 0) or None

        # Crear ticket
        t = Ticket(
            code="PENDING",
            note_id=int(note_id),
            reporter_user_id=int(current_user.id),
            seller_user_id=seller_id,
            status="new",
            reason=reason,
            details=details,
        )
        s.add(t)
        s.flush()
        t.code = _ticket_code_for_id(t.id)
        # Evento inicial
        s.add(TicketEvent(ticket_id=t.id, actor_user_id=int(current_user.id), event="created", from_status=None, to_status="new", message=details))

        # Auditoría (gestión histórica)
        try:
            log_audit_event(
                actor_user_id=int(current_user.id),
                action="ticket_created",
                target_type="ticket",
                target_id=int(t.id),
                meta={
                    "ticket_code": t.code,
                    "note_id": int(note_id),
                    "reason": reason,
                    "reporter_user_id": int(current_user.id),
                    "reporter_email": getattr(current_user, "email", None),
                },
            )
        except Exception:
            pass

        # Notificaciones
        admin_ids = _admin_user_ids(s)
        _notify_users(
            s,
            [int(current_user.id)],
            kind="info",
            title=f"Reporte recibido (Ticket {t.code})",
            body="Recibimos tu reporte. Un administrador lo revisará y te avisaremos novedades.",
        )
        if seller_id:
            _notify_users(
                s,
                [seller_id],
                kind="warning",
                title=f"Tu apunte fue reportado (Ticket {t.code})",
                body=f"Se abrió un ticket por un reporte sobre tu apunte. Motivo: {reason}. Te avisaremos cuando haya novedades.",
            )
        if admin_ids:
            _notify_users(
                s,
                admin_ids,
                kind="warning",
                title=f"Nuevo ticket {t.code}",
                body=f"Apunte #{note_id} reportado. Motivo: {reason}.",
            )

        s.commit()

    flash("✅ Reporte recibido. Creamos un ticket para seguimiento.", "success")
    flash(f"Ticket: {t.code}", "info")
    return redirect(url_for("ticket_detail", ticket_code=t.code))

# -----------------------------------------------------------------------------
# Taxonomías académicas (dropdowns) + creación "aprendida"
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Tickets (Reportes / Reclamos)
# -----------------------------------------------------------------------------

def _can_view_ticket(u, t: Ticket) -> bool:
    try:
        if _is_staff(u):
            return True
        uid = int(getattr(u, "id", 0) or 0)
        return uid and (uid == int(t.reporter_user_id or 0) or uid == int(t.seller_user_id or 0))
    except Exception:
        return False


@app.get("/tickets")
@login_required
def tickets_list():
    """Listado de tickets del usuario (como reportante o vendedor)."""
    with Session() as s:
        uid = int(current_user.id)
        q = s.query(Ticket).filter(
            (Ticket.reporter_user_id == uid) | (Ticket.seller_user_id == uid)
        ).order_by(Ticket.updated_at.desc(), Ticket.created_at.desc())
        rows = q.limit(300).all()
        return render_template("tickets_list.html", tickets=rows)


@app.get("/tickets/<ticket_code>")
@login_required
def ticket_detail(ticket_code: str):
    ticket_code = (ticket_code or "").strip()
    with Session() as s:
        t = s.query(Ticket).filter(Ticket.code == ticket_code).first()
        if not t:
            abort(404)
        if not _can_view_ticket(current_user, t):
            abort(403)

        # note title (best-effort)
        note = None
        try:
            note = s.get(Note, int(t.note_id))
        except Exception:
            note = None

        events = s.query(TicketEvent).filter(TicketEvent.ticket_id == t.id).order_by(TicketEvent.created_at.asc()).all()

        return render_template("ticket_detail.html", ticket=t, note=note, events=events)

def _norm(s: str) -> str:
    return (s or "").strip()

@app.get("/refunds", endpoint="refund_policy")
def refund_policy():
    return render_template("refund_policy.html")


@app.get("/api/academics/universities")
def api_list_universities():
    cache_key = "academics:universities:v1"
    cached = _cache_get(cache_key)
    if cached is not None:
        resp = jsonify(cached)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    with Session() as s:
        rows = s.execute(select(University).order_by(University.name)).scalars().all()
        payload = [{"id": u.id, "name": u.name} for u in rows]
        _cache_set(cache_key, payload, 86400)
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

@app.get("/api/academics/faculties")
def api_list_faculties():
    uid = request.args.get("university_id", type=int)
    cache_key = f"academics:faculties:v1:{uid or 0}"
    cached = _cache_get(cache_key)
    if cached is not None:
        resp = jsonify(cached)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    with Session() as s:
        q = select(Faculty)
        if uid:
            q = q.where(Faculty.university_id == uid)
        rows = s.execute(q.order_by(Faculty.name)).scalars().all()
        payload = [{"id": f.id, "name": f.name, "university_id": f.university_id} for f in rows]
        _cache_set(cache_key, payload, 86400)
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

@app.get("/api/academics/careers")
def api_list_careers():
    fid = request.args.get("faculty_id", type=int)
    cache_key = f"academics:careers:v1:{fid or 0}"
    cached = _cache_get(cache_key)
    if cached is not None:
        resp = jsonify(cached)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp

    with Session() as s:
        q = select(Career)
        if fid:
            q = q.where(Career.faculty_id == fid)
        rows = s.execute(q.order_by(Career.name)).scalars().all()
        payload = [{"id": c.id, "name": c.name, "faculty_id": c.faculty_id} for c in rows]
        _cache_set(cache_key, payload, 86400)
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp


@app.post("/api/academics/universities")
@csrf.exempt
def api_create_university():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name_required"}), 400

    # Only admins/superadmins can create taxonomy entities directly.
    if not (current_user.is_authenticated and (getattr(current_user, "is_admin", False) or getattr(current_user, "is_superadmin", False))):
        with Session() as s:
            sug = AcademicSuggestion(
                user_id=int(current_user.id) if current_user.is_authenticated else None,
                kind="university",
                name=name,
            )
            s.add(sug)
            s.commit()
        return jsonify({"ok": True, "status": "suggested"}), 202

    with Session() as s:
        existing = s.execute(select(University).where(University.name == name)).scalar_one_or_none()
        if existing:
            return jsonify({"id": existing.id, "name": existing.name}), 200
        u = University(name=name)
        s.add(u)
        s.commit()
        _cache_invalidate_prefix('academics:universities')
        _cache_invalidate_prefix('academics:faculties')
        return jsonify({"id": u.id, "name": u.name}), 201


@app.post("/api/academics/faculties")
@csrf.exempt
def api_create_faculty():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    university_id = data.get("university_id")
    try:
        university_id = int(university_id)
    except Exception:
        university_id = None

    if not name or not university_id:
        return jsonify({"error": "name_and_university_id_required"}), 400

    if not (current_user.is_authenticated and (getattr(current_user, "is_admin", False) or getattr(current_user, "is_superadmin", False))):
        with Session() as s:
            u_name = s.execute(select(University.name).where(University.id == university_id)).scalar_one_or_none()
            sug = AcademicSuggestion(
                user_id=int(current_user.id) if current_user.is_authenticated else None,
                kind="faculty",
                name=name,
                university_id=university_id,
                university_name=u_name,
            )
            s.add(sug)
            s.commit()
        return jsonify({"ok": True, "status": "suggested"}), 202

    with Session() as s:
        existing = s.execute(
            select(Faculty).where(Faculty.name == name, Faculty.university_id == university_id)
        ).scalar_one_or_none()
        if existing:
            return jsonify({"id": existing.id, "name": existing.name, "university_id": existing.university_id}), 200
        f = Faculty(name=name, university_id=university_id)
        s.add(f)
        s.commit()
        _cache_invalidate_prefix('academics:faculties')
        _cache_invalidate_prefix('academics:careers')
        return jsonify({"id": f.id, "name": f.name, "university_id": f.university_id}), 201


@app.post("/api/academics/careers")
@csrf.exempt
def api_create_career():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    faculty_id = data.get("faculty_id")
    try:
        faculty_id = int(faculty_id)
    except Exception:
        faculty_id = None

    if not name or not faculty_id:
        return jsonify({"error": "name_and_faculty_id_required"}), 400

    if not (current_user.is_authenticated and (getattr(current_user, "is_admin", False) or getattr(current_user, "is_superadmin", False))):
        with Session() as s:
            f_row = s.execute(select(Faculty.name, Faculty.university_id).where(Faculty.id == faculty_id)).first()
            fac_name = f_row[0] if f_row else None
            uni_id = f_row[1] if f_row else None
            uni_name = None
            if uni_id:
                uni_name = s.execute(select(University.name).where(University.id == uni_id)).scalar_one_or_none()
            sug = AcademicSuggestion(
                user_id=int(current_user.id) if current_user.is_authenticated else None,
                kind="career",
                name=name,
                university_id=uni_id,
                faculty_id=faculty_id,
                university_name=uni_name,
                faculty_name=fac_name,
            )
            s.add(sug)
            s.commit()
        return jsonify({"ok": True, "status": "suggested"}), 202

    with Session() as s:
        existing = s.execute(
            select(Career).where(Career.name == name, Career.faculty_id == faculty_id)
        ).scalar_one_or_none()
        if existing:
            return jsonify({"id": existing.id, "name": existing.name, "faculty_id": existing.faculty_id}), 200
        c = Career(name=name, faculty_id=faculty_id)
        s.add(c)
        s.commit()
        _cache_invalidate_prefix('academics:careers')
        return jsonify({"id": c.id, "name": c.name, "faculty_id": c.faculty_id}), 201


@app.post("/api/academics/suggestions")
@csrf.exempt
def api_academics_suggestion():
    """Create a suggestion without modifying taxonomy."""
    data = request.get_json(silent=True) or {}
    kind = (data.get("kind") or "").strip().lower()
    name = (data.get("name") or "").strip()
    if kind not in ("university", "faculty", "career") or not name:
        return jsonify({"error": "invalid"}), 400

    university_id = data.get("university_id")
    faculty_id = data.get("faculty_id")
    try:
        university_id = int(university_id) if university_id is not None else None
    except Exception:
        university_id = None
    try:
        faculty_id = int(faculty_id) if faculty_id is not None else None
    except Exception:
        faculty_id = None

    uni_name = (data.get("university_name") or "").strip() or None
    fac_name = (data.get("faculty_name") or "").strip() or None

    with Session() as s:
        sug = AcademicSuggestion(
            user_id=int(current_user.id) if current_user.is_authenticated else None,
            kind=kind,
            name=name,
            university_id=university_id,
            faculty_id=faculty_id,
            university_name=uni_name,
            faculty_name=fac_name,
        )
        s.add(sug)
        s.commit()
        return jsonify({"ok": True, "id": sug.id}), 201




# (removed duplicate academics POST endpoints)



@app.route("/profile/upload_image", methods=["POST"])
@login_required
def upload_profile_image():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        flash("Formato no permitido. Usá PNG, JPG o WEBP.")
        return redirect(url_for("profile"))

    # preferimos GCS en producción; fallback a static local si no está configurado
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = ".jpg"

    ref_to_store = None

    if gcs_bucket:
        try:
            blob_name = f"profile_images/user_{current_user.id}{ext}"
            # subimos como imagen
            gcs_upload_file(file, blob_name)
            ref_to_store = blob_name
        except Exception:
            ref_to_store = None

    if not ref_to_store:
        dest_dir = os.path.join(app.static_folder, "uploads", "profile_images")
        os.makedirs(dest_dir, exist_ok=True)
        filename = f"user_{current_user.id}{ext}"
        file.save(os.path.join(dest_dir, filename))
        ref_to_store = filename

    with Session() as s:
        u = s.get(User, current_user.id)
        if u and hasattr(u, "imagen_de_perfil"):
            u.imagen_de_perfil = ref_to_store
            s.commit()

    flash("📸 Foto actualizada con éxito")
    return redirect(url_for("profile"))

# -----------------------------------------------------------------------------
# Cambio de contraseña MANUAL (sólo si corresponde)
# -----------------------------------------------------------------------------

from flask import redirect, url_for, send_from_directory, abort

@app.get("/user/<int:user_id>/avatar")
def user_avatar(user_id: int):
    """
    Return a user avatar image.

    - If user has imagen_de_perfil:
        * If it's external URL -> redirect
        * If it's R2 key -> redirect to signed URL (short-lived)
        * If it's local file -> serve file
    - Else -> serve default static image

    Cache:
      - For dynamic avatars we keep no-store (URL stable but content can change).
      - For default image we allow a small cache to reduce load.
    """

    def _nocache(resp):
        try:
            resp.headers["Cache-Control"] = "no-store, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        except Exception:
            pass
        return resp

    def _cache_default(resp):
        # Default can be cached safely (changes rarely).
        # You can change max-age if you want.
        try:
            resp.headers["Cache-Control"] = "public, max-age=86400"  # 24hs
        except Exception:
            pass
        return resp

    # --- Fetch user avatar ref ---
    with Session() as s:
        u = s.get(User, user_id)
        ref = getattr(u, "imagen_de_perfil", None) if u else None

    # --- Serve default if missing ---
    default_dir = os.path.join(app.static_folder, "img")
    default_name = "default_profile.png"
    default_path = os.path.join(default_dir, default_name)

    if not ref:
        # Serve default as file (no redirect)
        if os.path.exists(default_path):
            return _cache_default(send_from_directory(default_dir, default_name))
        # If missing, hard fallback to static URL redirect
        return _cache_default(redirect(url_for("static", filename="img/default_profile.png")))

    ref = str(ref).strip()

    # External URL (Google, etc.)
    if ref.startswith("http://") or ref.startswith("https://"):
        return _nocache(redirect(ref))

    # Object key in R2 (your "gcs_*" wrapper)
    if gcs_bucket and ("/" in ref or ref.startswith("profile_images/") or ref.startswith("uploads/")):
        try:
            signed = gcs_generate_signed_url(ref, seconds=600)
            return _nocache(redirect(signed))
        except Exception:
            # If signing fails, fall back to default
            if os.path.exists(default_path):
                return _cache_default(send_from_directory(default_dir, default_name))
            return _cache_default(redirect(url_for("static", filename="img/default_profile.png")))

    # Local file fallback
    local_dir = os.path.join(app.static_folder, "uploads", "profile_images")
    local_path = os.path.join(local_dir, ref)
    if os.path.exists(local_path):
        return _nocache(send_from_directory(local_dir, ref))

    # Last fallback: default
    if os.path.exists(default_path):
        return _cache_default(send_from_directory(default_dir, default_name))
    return _cache_default(redirect(url_for("static", filename="img/default_profile.png")))

    

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
@staff_required
def admin_hub():
    from datetime import datetime
    return render_template("admin/hub.html", today_str=datetime.utcnow().strftime('%Y-%m-%d'))


# -----------------------------------------------------------------------------
# SUPERADMIN: Administradores + mantenimiento + auditoría
# -----------------------------------------------------------------------------

@app.route("/admin/admins", methods=["GET", "POST"])
@login_required
@superadmin_required
def admin_manage_admins():
    """Promote/demote admins/superadmins by email."""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        role = (request.form.get("role") or "").strip().lower()
        if role not in ("user", "admin", "superadmin"):
            flash("Rol inválido.", "danger")
            return redirect(url_for("admin_manage_admins"))
        if not email:
            flash("Ingresá un email.", "warning")
            return redirect(url_for("admin_manage_admins"))

        with Session() as s:
            u = s.execute(select(User).where(func.lower(User.email) == email)).scalar_one_or_none()
            if not u:
                flash("No existe un usuario con ese email. Primero debe iniciar sesión al menos una vez.", "warning")
                return redirect(url_for("admin_manage_admins"))

            # Safety: never remove last superadmin
            if u.id == current_user.id and role != "superadmin":
                flash("No podés quitarte el rol de superadmin a vos mismo.", "danger")
                return redirect(url_for("admin_manage_admins"))

            if role != "superadmin":
                super_count = s.execute(select(func.count(User.id)).where(User.role == "superadmin")).scalar_one() or 0
                if _user_role(u) == "superadmin" and super_count <= 1:
                    flash("Debe existir al menos 1 superadmin.", "danger")
                    return redirect(url_for("admin_manage_admins"))

            old_role = _user_role(u)
            u.role = role
            u.is_admin = (role in ("admin", "superadmin"))
            s.commit()

            _audit("set_role", target_type="user", target_id=u.id, meta={"from": old_role, "to": role, "email": email})
            flash(f"Rol actualizado: {email} → {role}", "success")

        return redirect(url_for("admin_manage_admins"))

    with Session() as s:
        superadmins = s.execute(select(User).where(User.role == "superadmin").order_by(User.id.asc())).scalars().all()
        admins = s.execute(select(User).where(User.role == "admin").order_by(User.id.asc())).scalars().all()
    return render_template("admin/manage_admins.html", superadmins=superadmins, admins=admins)


@app.get("/admin/maintenance")
@login_required
@superadmin_required
def admin_maintenance():
    mm = (_get_setting("maintenance_mode", "0") or "0").strip()
    maintenance_on = mm in ("1", "true", "True", "yes", "on")

    se = (_get_setting("sales_enabled", "1") or "1").strip()
    sales_enabled = se in ("1", "true", "True", "yes", "on")

    return render_template("admin/maintenance.html", maintenance_on=maintenance_on, sales_enabled=sales_enabled)


@app.post("/admin/maintenance/toggle")
@login_required
@superadmin_required
def admin_toggle_maintenance():
    mode = (request.form.get("mode") or "0").strip()
    mode_norm = "1" if mode in ("1", "true", "True", "yes", "on") else "0"
    _set_setting("maintenance_mode", mode_norm)
    _audit("toggle_maintenance", target_type="site", target_id=None, meta={"mode": mode_norm})
    flash("Modo mantenimiento actualizado.", "success")
    return redirect(url_for("admin_maintenance"))


@app.post("/admin/sales/toggle")
@login_required
@superadmin_required
def admin_toggle_sales():
    mode = (request.form.get("mode") or "1").strip()
    mode_norm = "1" if mode in ("1", "true", "True", "yes", "on") else "0"
    _set_setting("sales_enabled", mode_norm)
    _audit("toggle_sales", target_type="site", target_id=None, meta={"sales_enabled": mode_norm})
    flash("Ventas actualizadas.", "success")
    return redirect(url_for("admin_maintenance"))


@app.get("/admin/audit")
@login_required
@superadmin_required
def admin_audit_log():
    with Session() as s:
        events = s.execute(select(AuditEvent).order_by(desc(AuditEvent.id)).limit(100)).scalars().all()
    return render_template("admin/audit_log.html", events=events)


@app.get("/admin/tools")
@login_required
@superadmin_required
def admin_tools():
    """Herramientas de mantenimiento (A7)."""
    mm = (_get_setting("maintenance_mode", "0") or "0").strip()
    maintenance_on = mm in ("1", "true", "True", "yes", "on")

    # Preview health (A11): count notes with missing/empty preview_images
    previews_total = 0
    previews_missing = 0
    previews_missing_samples = []
    try:
        with Session() as s:
            rows = (
                s.query(Note.id, Note.title, Note.preview_images, Note.deleted_at)
                .filter(Note.is_active == True)
                .all()
            )
            previews_total = len(rows)
            for nid, title, pimgs, deleted_at in rows:
                if deleted_at is not None:
                    continue
                imgs = []
                try:
                    imgs = (pimgs or {}).get("images") or []
                except Exception:
                    imgs = []
                if not imgs:
                    previews_missing += 1
                    if len(previews_missing_samples) < 20:
                        previews_missing_samples.append({"id": nid, "title": title or f"Apunte #{nid}"})
    except Exception:
        pass

    return render_template(
        "admin/tools.html",
        maintenance_on=maintenance_on,
        previews_total=previews_total,
        previews_missing=previews_missing,
        previews_missing_samples=previews_missing_samples,
    )

@app.post("/admin/previews/rebuild")
@login_required
@superadmin_required
def admin_previews_rebuild():
    """Regenera previews en lote (A11).

    - scope=missing (default): solo los que no tienen preview
    - scope=all: todos los apuntes
    - note_id: opcional (si querés uno puntual)
    - limit: opcional (por defecto 50)
    - max_pages: opcional (por defecto 4)
    - force=1: fuerza regeneración aunque ya exista
    """
    scope = (request.form.get("scope") or "missing").strip().lower()
    note_id_raw = (request.form.get("note_id") or "").strip()
    force = (request.form.get("force") or "").strip() in ("1", "true", "True", "yes", "on")
    try:
        limit = int(request.form.get("limit") or 50)
    except Exception:
        limit = 50
    limit = max(1, min(limit, 500))

    try:
        max_pages = int(request.form.get("max_pages") or 4)
    except Exception:
        max_pages = 4
    max_pages = max(1, min(max_pages, 10))

    rebuilt = 0
    skipped = 0
    failed = 0

    try:
        with Session() as s:
            q = s.query(Note).filter(Note.deleted_at.is_(None)).filter(Note.is_active == True)

            if note_id_raw:
                try:
                    nid = int(note_id_raw)
                    q = q.filter(Note.id == nid)
                except Exception:
                    flash("note_id inválido.", "danger")
                    return redirect(url_for("admin_tools"))

            notes = q.order_by(Note.id.desc()).limit(limit).all()

            for note in notes:
                try:
                    imgs = []
                    try:
                        imgs = (getattr(note, "preview_images", None) or {}).get("images") or []
                    except Exception:
                        imgs = []

                    if not force:
                        if scope == "missing" and imgs:
                            skipped += 1
                            continue

                    pages, image_paths = generate_note_preview(note, max_pages=max_pages)
                    note.preview_pages = {"pages": pages, "generated_at": datetime.utcnow().isoformat()}
                    note.preview_images = {"images": image_paths, "generated_at": datetime.utcnow().isoformat()}
                    s.add(note)
                    rebuilt += 1
                except Exception:
                    failed += 1
                    continue

            s.commit()
    except Exception as e:
        flash(f"Error al regenerar previews: {e}", "danger")
        return redirect(url_for("admin_tools"))

    flash(f"Previews: regenerados={rebuilt} | omitidos={skipped} | fallidos={failed}", "success" if failed == 0 else "warning")
    return redirect(url_for("admin_tools"))



@app.post("/admin/reset")
@login_required
@superadmin_required
def admin_reset_platform():
    """Reset de plataforma (borra histórico) sin eliminar superadmins."""
    confirm = (request.form.get("confirm") or "").strip().upper()
    if confirm != "RESET":
        flash("Para confirmar, escribí RESET.", "danger")
        return redirect(url_for("admin_tools"))

    tables_to_clear = [
        "purchases",
        "combo_purchases",
        "download_logs",
        "analytics_events",
        "stats_daily",
        "reviews",
        "notifications",
        "webhook_events",
        "admin_actions",
        "audit_events",
        "combo_notes",
        "combos",
        "notes",
    ]

    cleared = []
    with Session() as s:
        from sqlalchemy import text as sqltext
        for t in tables_to_clear:
            try:
                s.execute(sqltext(f"DELETE FROM {t}"))
                cleared.append(t)
            except Exception:
                # tabla puede no existir en algunos entornos / versiones
                s.rollback()
        # borrar usuarios NO superadmin
        try:
            s.execute(sqltext("DELETE FROM users WHERE COALESCE(role,'user') <> 'superadmin'"))
        except Exception:
            s.rollback()
        # asegurar is_admin para superadmins
        try:
            s.execute(sqltext("UPDATE users SET is_admin = TRUE WHERE COALESCE(role,'user') = 'superadmin'"))
        except Exception:
            s.rollback()
        s.commit()

    _audit("reset_platform", target_type="site", target_id=None, meta={"cleared": cleared})
    flash("Reset realizado. Se borró el histórico (se mantuvieron superadmins).", "success")
    return redirect(url_for("admin_tools"))


def _reset_db_complete(session):
    """Reset completo de DB (mantiene superadmins).

    Nota: trabajamos a nivel SQL para ser tolerantes a diferencias entre
    entornos (SQLite vs Postgres) y a tablas que puedan no existir.
    """
    from sqlalchemy import text as sqltext

    # Orden: primero tablas dependientes/child, luego parent.
    tables_to_clear = [
        "legal_acceptance_audit",
        "otps",
        "academic_suggestions",
        "reviews",
        "download_logs",
        "combo_purchases",
        "purchases",
        "notifications",
        "webhook_events",
        "analytics_events",
        "stats_daily",
        "admin_actions",
        "audit_events",
        "combo_notes",
        "combos",
        "notes",
        # Academic taxonomy
        "careers",
        "faculties",
        "universities",
    ]

    cleared = []
    for t in tables_to_clear:
        try:
            session.execute(sqltext(f"DELETE FROM {t}"))
            cleared.append(t)
        except Exception:
            session.rollback()

    # borrar usuarios NO superadmin
    try:
        session.execute(sqltext("DELETE FROM users WHERE COALESCE(role,'user') <> 'superadmin'"))
        cleared.append("users(non-superadmin)")
    except Exception:
        session.rollback()

    # asegurar flags correctos para superadmins
    try:
        session.execute(sqltext("UPDATE users SET is_admin = TRUE WHERE COALESCE(role,'user') = 'superadmin'"))
    except Exception:
        session.rollback()

    return cleared


def _r2_delete_prefixes(prefixes):
    """Delete objects in R2 under the given prefixes.

    Returns dict with counts and deleted keys.
    """
    if not gcs_client or not gcs_bucket:
        raise RuntimeError("R2 no está configurado")

    deleted_total = 0
    per_prefix = {}

    for prefix in prefixes:
        prefix_deleted = 0
        token = None
        while True:
            kwargs = {"Bucket": gcs_bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = gcs_client.list_objects_v2(**kwargs)
            contents = resp.get("Contents") or []
            if contents:
                # delete in batches (up to 1000)
                objs = [{"Key": obj["Key"]} for obj in contents if obj.get("Key")]
                if objs:
                    gcs_client.delete_objects(Bucket=gcs_bucket, Delete={"Objects": objs, "Quiet": True})
                    prefix_deleted += len(objs)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                continue
            break

        per_prefix[prefix] = prefix_deleted
        deleted_total += prefix_deleted

    return {"deleted_total": deleted_total, "per_prefix": per_prefix}


@app.post("/admin/reset-db")
@login_required
@superadmin_required
def admin_reset_db():
    """Reset DB completo (seguro): borra datos en DB, mantiene superadmins."""
    confirm = (request.form.get("confirm") or "").strip().upper()
    ack = (request.form.get("ack") or "").strip()
    if confirm != "RESET_DB" or ack != "1":
        flash("Para confirmar, escribí RESET_DB y marcá el checkbox.", "danger")
        return redirect(url_for("admin_tools"))

    cleared = []
    with Session() as s:
        cleared = _reset_db_complete(s)
        s.commit()

    _audit("reset_db_complete", target_type="site", target_id=None, meta={"cleared": cleared})
    flash("Reset DB completo realizado (superadmins conservados).", "success")
    return redirect(url_for("admin_tools"))


@app.post("/admin/reset-total")
@login_required
@superadmin_required
def admin_reset_total():
    """Reset TOTAL: DB completo + borrado de archivos en R2 (prefijos controlados)."""
    confirm = (request.form.get("confirm") or "").strip().upper()
    ack = (request.form.get("ack") or "").strip()
    if confirm != "RESET_TOTAL" or ack != "1":
        flash("Para confirmar, escribí RESET_TOTAL y marcá el checkbox.", "danger")
        return redirect(url_for("admin_tools"))

    cleared = []
    r2_result = None
    with Session() as s:
        cleared = _reset_db_complete(s)
        s.commit()

    # Borrado en R2 (prefijos seguros)
    try:
        r2_result = _r2_delete_prefixes(["notes/", "previews/"])
    except Exception as e:
        _audit(
            "reset_total_r2_failed",
            target_type="site",
            target_id=None,
            meta={"cleared": cleared, "r2_error": str(e)},
        )
        flash(f"Reset DB hecho, pero falló el borrado en R2: {e}", "warning")
        return redirect(url_for("admin_tools"))

    _audit(
        "reset_total",
        target_type="site",
        target_id=None,
        meta={"cleared": cleared, "r2": r2_result},
    )
    flash(
        f"Reset TOTAL realizado. R2 borrado: {r2_result.get('deleted_total', 0)} objetos.",
        "success",
    )
    return redirect(url_for("admin_tools"))


@app.post("/admin/academics/seed-unc")
@login_required
@superadmin_required
def admin_seed_unc():
    """Carga idempotente del seed de UNC (universidad + facultades + carreras)."""
    confirm = (request.form.get("confirm") or "").strip().upper()
    if confirm != "SEED_UNC":
        flash("Para confirmar, escribí SEED_UNC.", "danger")
        return redirect(url_for("admin_tools"))

    with Session() as s:
        result = seed_unc(s)
        s.commit()

    _cache_invalidate_prefix("academics:")
    _audit("seed_unc", target_type="site", target_id=None, meta=result)
    flash(
        f"Seed UNC listo. Facultades nuevas: {result.get('created_faculties', 0)} | Carreras nuevas: {result.get('created_careers', 0)}",
        "success",
    )
    return redirect(url_for("admin_tools"))




@app.route("/admin/tickets", methods=["GET"])
@login_required
@admin_required
def admin_tickets_list():
    status = (request.args.get("status") or "open").strip().lower()
    with Session() as s:
        q = s.query(Ticket).order_by(Ticket.updated_at.desc(), Ticket.created_at.desc())
        if status == "open":
            q = q.filter(Ticket.status.in_(["new", "in_review", "need_seller_action"]))
        elif status in ("resolved", "rejected"):
            q = q.filter(Ticket.status == status)
        rows = q.limit(500).all()
        return render_template("admin/tickets.html", tickets=rows, status=status)


@app.route("/admin/tickets/<int:ticket_id>", methods=["GET", "POST"])
@login_required
@admin_required
def admin_ticket_detail(ticket_id: int):
    with Session() as s:
        t = s.get(Ticket, int(ticket_id))
        if not t:
            abort(404)

        note = None
        try:
            note = s.get(Note, int(t.note_id))
        except Exception:
            note = None

        if request.method == "POST":
            new_status = (request.form.get("status") or t.status or "new").strip().lower()
            if new_status not in ("new", "in_review", "need_seller_action", "resolved", "rejected"):
                new_status = t.status or "new"

            resolution = (request.form.get("resolution") or "").strip() or None
            admin_notes = (request.form.get("admin_notes") or "").strip() or None

            changed = False
            if new_status != (t.status or "new"):
                t.status = new_status
                changed = True
            # always allow updating resolution/notes
            t.resolution = resolution
            t.admin_notes = admin_notes

            # timeline event
            ev = TicketEvent(
                ticket_id=int(t.id),
                actor_user_id=int(current_user.id),
                event_type="status_update",
                message=f"Estado → {t.status}" + (f" | Resolución: {resolution}" if resolution else ""),
            )
            s.add(ev)
            try:
                t.updated_at = datetime.utcnow()
            except Exception:
                pass

            s.commit()

            # Notificaciones (best-effort)
            try:
                title = f"Ticket {t.code}: actualización"
                body = f"Estado: {t.status}" + (f"\nResolución: {t.resolution}" if t.resolution else "")
                ids = []
                if t.reporter_user_id:
                    ids.append(int(t.reporter_user_id))
                if t.seller_user_id and int(t.seller_user_id) not in ids:
                    ids.append(int(t.seller_user_id))
                if ids:
                    _notify_users(s, ids, kind="info", title=title, body=body)
            except Exception:
                pass

            flash("Ticket actualizado.", "success")
            return redirect(f"/admin/tickets/{t.id}")

        events = s.query(TicketEvent).filter(TicketEvent.ticket_id == t.id).order_by(TicketEvent.created_at.asc()).all()
        return render_template("admin/ticket_detail.html", t=t, note=note, events=events)
@app.get("/admin/academics/suggestions")
@login_required
@admin_required
def admin_academics_suggestions():
    """Lista sugerencias académicas (admins y superadmins)."""
    status = (request.args.get("status") or "pending").strip().lower()
    if status not in ("pending", "approved", "rejected", "all"):
        status = "pending"

    with Session() as s:
        q = select(AcademicSuggestion).order_by(desc(AcademicSuggestion.created_at))
        if status != "all":
            q = q.where(AcademicSuggestion.status == status)
        items = s.execute(q.limit(200)).scalars().all()

    return render_template("admin/academics_suggestions.html", items=items, status=status)


@app.post("/admin/academics/suggestions/<int:sug_id>/action")
@login_required
@admin_required
def admin_academics_suggestions_action(sug_id: int):
    action = (request.form.get("action") or "").strip().lower()  # approve|reject
    note = (request.form.get("admin_note") or "").strip() or None

    with Session() as s:
        sug = s.get(AcademicSuggestion, sug_id)
        if not sug:
            flash("Sugerencia no encontrada.", "danger")
            return redirect(url_for("admin_academics_suggestions"))

        if action == "reject":
            sug.status = "rejected"
            sug.admin_note = note
            s.add(sug)
            s.commit()
            _audit("academics_suggestion_rejected", target_type="academic_suggestion", target_id=sug_id, meta={"note": note})
            flash("Sugerencia rechazada.", "success")
            return redirect(url_for("admin_academics_suggestions"))

        if action != "approve":
            flash("Acción inválida.", "danger")
            return redirect(url_for("admin_academics_suggestions"))

        # Approve: promote to taxonomy
        created = {}

        if sug.kind == "university":
            existing = s.execute(select(University).where(University.name == sug.name)).scalar_one_or_none()
            if not existing:
                u = University(name=sug.name)
                s.add(u)
                s.flush()
                created["university_id"] = u.id
            sug.status = "approved"

        elif sug.kind == "faculty":
            uid = sug.university_id
            if not uid and sug.university_name:
                u = s.execute(select(University).where(University.name == sug.university_name)).scalar_one_or_none()
                if u:
                    uid = u.id
            if not uid:
                flash("No se pudo aprobar: falta universidad asociada.", "danger")
                return redirect(url_for("admin_academics_suggestions"))
            existing = s.execute(select(Faculty).where(Faculty.name == sug.name, Faculty.university_id == uid)).scalar_one_or_none()
            if not existing:
                f = Faculty(name=sug.name, university_id=uid)
                s.add(f)
                s.flush()
                created["faculty_id"] = f.id
            sug.status = "approved"

        elif sug.kind == "career":
            fid = sug.faculty_id
            if not fid and sug.faculty_name and sug.university_id:
                f = s.execute(select(Faculty).where(Faculty.name == sug.faculty_name, Faculty.university_id == sug.university_id)).scalar_one_or_none()
                if f:
                    fid = f.id
            if not fid:
                flash("No se pudo aprobar: falta facultad asociada.", "danger")
                return redirect(url_for("admin_academics_suggestions"))
            existing = s.execute(select(Career).where(Career.name == sug.name, Career.faculty_id == fid)).scalar_one_or_none()
            if not existing:
                c = Career(name=sug.name, faculty_id=fid)
                s.add(c)
                s.flush()
                created["career_id"] = c.id
            sug.status = "approved"

        sug.admin_note = note
        s.add(sug)
        s.commit()

    _cache_invalidate_prefix("academics:")
    _audit("academics_suggestion_approved", target_type="academic_suggestion", target_id=sug_id, meta={"created": created, "note": note})
    flash("Sugerencia aprobada y cargada al listado.", "success")
    return redirect(url_for("admin_academics_suggestions"))

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
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip()

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

        # Gestión / auditoría
        ticket = None
        try:
            ticket = log_audit_event(
                actor_user_id=int(getattr(current_user, "id", 0) or 0) or None,
                action="user_suspend",
                target_type="user",
                target_id=int(u.id),
                meta={
                    "reason": reason or None,
                    "target_user_id": int(u.id),
                    "target_email": getattr(u, "email", None),
                },
            )
        except Exception:
            ticket = None

    return jsonify({"ok": True, "status": "blocked", "ticket": ticket})


@app.post("/admin/api/users/<int:user_id>/unblock")
@login_required
@admin_required
def admin_api_users_unblock(user_id):
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip()

    with Session() as s:
        u = s.get(User, user_id)
        if not u:
            return jsonify({"ok": False, "error": "not_found"}), 404

        u.is_active = True
        if hasattr(u, "is_blocked"):
            u.is_blocked = False
        s.commit()

        ticket = None
        try:
            ticket = log_audit_event(
                actor_user_id=int(getattr(current_user, "id", 0) or 0) or None,
                action="user_reactivate",
                target_type="user",
                target_id=int(u.id),
                meta={
                    "reason": reason or None,
                    "target_user_id": int(u.id),
                    "target_email": getattr(u, "email", None),
                },
            )
        except Exception:
            ticket = None

    return jsonify({"ok": True, "status": "unblocked", "ticket": ticket})


@app.post("/admin/api/users/<int:user_id>/delete")
@login_required
@admin_required
def admin_api_users_delete(user_id):
    data = request.get_json(silent=True) or {}
    reason = (data.get("reason") or "").strip()

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

        target_email = getattr(u, "email", None)
        s.delete(u)
        s.commit()

        ticket = None
        try:
            ticket = log_audit_event(
                actor_user_id=int(getattr(current_user, "id", 0) or 0) or None,
                action="user_delete",
                target_type="user",
                target_id=int(user_id),
                meta={
                    "reason": reason or None,
                    "target_user_id": int(user_id),
                    "target_email": target_email,
                },
            )
        except Exception:
            ticket = None

    return jsonify({"ok": True, "status": "deleted", "ticket": ticket})


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
                "title": getattr(n, "title", "") or "",
                "file_path": getattr(n, "file_path", None),
                "download_url": url_for("admin_download_note", note_id=n.id),
                "created_at": (n.created_at.isoformat() if getattr(n, "created_at", None) else None),
                "university": getattr(n, "university", "") or "",
                "faculty": getattr(n, "faculty", "") or "",
                "career": getattr(n, "career", "") or "",
                "price_cents": int(getattr(n, "price_cents", 0) or 0),
            })

    return jsonify({"items": data})



# Admin HUB - contenido unificado (apunte + archivo)
@app.get("/admin/api/content")
@login_required
@admin_required
def admin_api_content():
    """Contenido unificado para el Hub admin (apuntes + combos) con métricas."""
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", type=int) or 120

    with Session() as s:
        like = f"%{q}%" if q else None

        # ---------------- Notes ----------------
        notes_stmt = select(Note)
        if hasattr(Note, "deleted_at"):
            notes_stmt = notes_stmt.where(Note.deleted_at.is_(None))
        if hasattr(Note, "is_active"):
            notes_stmt = notes_stmt.where(Note.is_active == True)
        if like:
            notes_stmt = notes_stmt.where(or_(
                Note.title.ilike(like),
                Note.description.ilike(like),
                Note.university.ilike(like),
                Note.faculty.ilike(like),
                Note.career.ilike(like),
            ))
        notes_stmt = notes_stmt.order_by(desc(getattr(Note, "created_at", Note.id))).limit(limit)
        notes = s.execute(notes_stmt).scalars().all()

        note_ids = [n.id for n in notes]
        seller_ids = list({n.seller_id for n in notes if getattr(n, "seller_id", None)})

        sellers = {}
        if seller_ids:
            sellers_rows = s.execute(select(User.id, User.name).where(User.id.in_(seller_ids))).all()
            sellers = {i: n for i, n in sellers_rows}

        # purchases/downloads counts (notes)
        purchases_note = {}
        downloads_note = {}
        if note_ids:
            for nid, cnt in s.execute(
                select(Purchase.note_id, func.count(Purchase.id))
                .where(Purchase.note_id.in_(note_ids))
                .group_by(Purchase.note_id)
            ).all():
                purchases_note[int(nid)] = int(cnt or 0)
            for nid, cnt in s.execute(
                select(DownloadLog.note_id, func.count(DownloadLog.id))
                .where(DownloadLog.note_id.in_(note_ids))
                .group_by(DownloadLog.note_id)
            ).all():
                downloads_note[int(nid)] = int(cnt or 0)

        items = []
        for n in notes:
            items.append({
                "type": "note",
                "id": int(n.id),
                "title": getattr(n, "title", "") or "",
                "price_cents": int(getattr(n, "price_cents", 0) or 0),
                "seller_name": sellers.get(getattr(n, "seller_id", None), ""),
                "university": getattr(n, "university", "") or "",
                "faculty": getattr(n, "faculty", "") or "",
                "career": getattr(n, "career", "") or "",
                "file_path": getattr(n, "file_path", None),
                "download_url": url_for("admin_download_note", note_id=n.id),
                "purchases_count": int(purchases_note.get(int(n.id), 0)),
                "downloads_count": int(downloads_note.get(int(n.id), 0)),
                "created_at": (n.created_at.isoformat() if getattr(n, "created_at", None) else None),
            })

        # ---------------- Combos ----------------
        combos_stmt = select(Combo)
        if hasattr(Combo, "is_active"):
            combos_stmt = combos_stmt.where(Combo.is_active == True)
        if like:
            combos_stmt = combos_stmt.where(or_(
                Combo.title.ilike(like),
                Combo.description.ilike(like),
            ))
        combos_stmt = combos_stmt.order_by(desc(getattr(Combo, "created_at", Combo.id))).limit(limit)
        combos = s.execute(combos_stmt).scalars().all()

        combo_ids = [c.id for c in combos]
        combo_seller_ids = list({c.seller_id for c in combos if getattr(c, "seller_id", None)})
        combo_sellers = {}
        if combo_seller_ids:
            rows = s.execute(select(User.id, User.name).where(User.id.in_(combo_seller_ids))).all()
            combo_sellers = {i: n for i, n in rows}

        purchases_combo = {}
        downloads_combo = {}
        if combo_ids:
            try:
                for cid, cnt in s.execute(
                    select(ComboPurchase.combo_id, func.count(ComboPurchase.id))
                    .where(ComboPurchase.combo_id.in_(combo_ids))
                    .group_by(ComboPurchase.combo_id)
                ).all():
                    purchases_combo[int(cid)] = int(cnt or 0)
            except Exception:
                pass
            try:
                for cid, cnt in s.execute(
                    select(DownloadLog.combo_id, func.count(DownloadLog.id))
                    .where(DownloadLog.combo_id.in_(combo_ids))
                    .group_by(DownloadLog.combo_id)
                ).all():
                    downloads_combo[int(cid)] = int(cnt or 0)
            except Exception:
                pass

        for c in combos:
            items.append({
                "type": "combo",
                "id": int(c.id),
                "title": getattr(c, "title", "") or "",
                "price_cents": int(getattr(c, "price_cents", 0) or 0),
                "seller_name": combo_sellers.get(getattr(c, "seller_id", None), ""),
                "university": "",
                "faculty": "",
                "career": "",
                "file_path": None,
                "download_url": url_for("download_combo", combo_id=c.id),
                "purchases_count": int(purchases_combo.get(int(c.id), 0)),
                "downloads_count": int(downloads_combo.get(int(c.id), 0)),
                "created_at": (getattr(c, "created_at", None).isoformat() if getattr(c, "created_at", None) else None),
            })

    # Orden: más recientes primero (best-effort)
    items.sort(key=lambda x: (x.get("created_at") or ""), reverse=True)
    return jsonify({"items": items})



# Admin HUB - gestiones (tickets / auditoría)
@app.get("/admin/api/tickets")
@login_required
@admin_required
def admin_api_tickets():
    """Lista y búsqueda de tickets de gestión (audit_events).

    Devuelve información lista para UI: resumen, labels y links al objeto involucrado.
    """
    q = (request.args.get("q") or "").strip()
    date_from = (request.args.get("from") or "").strip()  # YYYY-MM-DD
    date_to = (request.args.get("to") or "").strip()      # YYYY-MM-DD
    category = (request.args.get("category") or "").strip()
    actor = (request.args.get("actor") or "").strip()     # user|admin|system
    critical_only = (request.args.get("critical") or "").strip() in ("1", "true", "True", "yes")

    # Limit hard cap (safety)
    limit = request.args.get("limit", type=int) or 200
    if limit < 1:
        limit = 50
    if limit > 2000:
        limit = 2000


    def _action_label(action: str) -> str:
        a = (action or "").strip()
        return {
            "purchase_created": "Compra iniciada",
            "purchase_status": "Estado de compra actualizado",
            "combo_purchase_created": "Compra de combo iniciada",
            "combo_purchase_status": "Estado de compra de combo actualizado",
            "note_reported": "Apunte reportado",
            "user_suspend": "Cuenta suspendida",
            "user_reactivate": "Cuenta reactivada",
            "user_delete": "Cuenta eliminada",
            "admin_delete_note": "Apunte eliminado por admin",
            "admin_delete_combo": "Combo eliminado por admin",
            "note_uploaded": "Apunte subido",
            "note_downloaded": "Descarga de apunte",
            "combo_downloaded": "Descarga de combo",
        }.get(a, a)

    with Session() as s:
        filters = []

        # Date filters (inclusive)
        try:
            if date_from:
                dt_from = datetime.strptime(date_from, "%Y-%m-%d")
                filters.append(AuditEvent.created_at >= dt_from)
        except Exception:
            pass
        try:
            if date_to:
                dt_to = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                filters.append(AuditEvent.created_at < dt_to)
        except Exception:
            pass

        # Category filters (maps to actions)
        CATEGORY_ACTIONS = {
            "transactions": {"purchase_created", "purchase_status", "combo_purchase_created", "combo_purchase_status"},
            "downloads": {"note_downloaded", "combo_downloaded"},
            "content": {"note_uploaded", "admin_delete_note", "admin_delete_combo", "user_delete_note", "user_delete_combo", "note_edited", "combo_edited"},
            "accounts": {"user_suspend", "user_reactivate", "user_delete"},
            "moderation": {"note_moderation", "combo_moderation", "note_reported"},
            "system": {"legal_accept"},
        }
        if category in CATEGORY_ACTIONS:
            filters.append(AuditEvent.action.in_(list(CATEGORY_ACTIONS[category])))

        # Critical-only filter
        CRITICAL_ACTIONS = {
            "note_reported",
            "user_suspend",
            "admin_delete_note",
            "admin_delete_combo",
            "user_delete",
            "note_moderation",
            "combo_moderation",
        }
        if critical_only:
            filters.append(AuditEvent.action.in_(list(CRITICAL_ACTIONS)))

        # Actor filter (user/admin/system)
        join_user = False
        if actor == "system":
            filters.append(AuditEvent.actor_user_id.is_(None))
        elif actor in ("admin", "user"):
            join_user = True  # need User.is_admin to distinguish

        # Search (code/action/meta/target)
        if q:
            like = f"%{q}%"
            search_filters = [
                AuditEvent.code.ilike(like),
                AuditEvent.action.ilike(like),
                cast(AuditEvent.meta, String).ilike(like),
                cast(AuditEvent.target_type, String).ilike(like),
            ]
            try:
                if q.isdigit():
                    search_filters.append(AuditEvent.target_id == int(q))
                    search_filters.append(AuditEvent.actor_user_id == int(q))
            except Exception:
                pass
            filters.append(or_(*search_filters))

        # Build stmt
        stmt = select(AuditEvent)
        if join_user:
            stmt = stmt.join(User, User.id == AuditEvent.actor_user_id)
            if actor == "admin":
                stmt = stmt.where(or_(User.is_admin == True, User.role == "superadmin"))
            elif actor == "user":
                stmt = stmt.where(and_(User.is_admin == False, or_(User.role.is_(None), User.role != "superadmin")))

        if filters:
            stmt = stmt.where(and_(*filters))

        stmt = stmt.order_by(desc(AuditEvent.created_at)).limit(limit)
        events = s.execute(stmt).scalars().all()

        # Actors
        actor_ids = list({e.actor_user_id for e in events if getattr(e, "actor_user_id", None)})
        actors = {}
        if actor_ids:
            rows = s.execute(select(User.id, User.name, User.email).where(User.id.in_(actor_ids))).all()
            actors = {int(r[0]): {"name": r[1], "email": r[2]} for r in rows}

        # Targets: notes / combos / users (best-effort)
        note_ids = [int(e.target_id) for e in events if (e.target_type == "note" and e.target_id is not None)]
        combo_ids = [int(e.target_id) for e in events if (e.target_type == "combo" and e.target_id is not None)]
        user_ids = [int(e.target_id) for e in events if (e.target_type == "user" and e.target_id is not None)]

        purchase_ids = [int(e.target_id) for e in events if (e.target_type == "purchase" and e.target_id is not None)]
        combo_purchase_ids = [int(e.target_id) for e in events if (e.target_type == "combo_purchase" and e.target_id is not None)]

        notes = {}
        if note_ids:
            rows = s.execute(select(Note.id, Note.title).where(Note.id.in_(note_ids))).all()
            notes = {int(r[0]): {"title": r[1]} for r in rows}

        combos = {}
        if combo_ids:
            rows = s.execute(select(Combo.id, Combo.title).where(Combo.id.in_(combo_ids))).all()
            combos = {int(r[0]): {"title": r[1]} for r in rows}

        users = {}
        if user_ids:
            rows = s.execute(select(User.id, User.name, User.email).where(User.id.in_(user_ids))).all()
            users = {int(r[0]): {"name": r[1], "email": r[2]} for r in rows}

        purchases = {}
        if purchase_ids:
            rows = s.execute(
                select(Purchase.id, Purchase.note_id, Purchase.combo_id, Purchase.buyer_id, Purchase.buyer_email, Purchase.seller_id,
                       Purchase.amount_cents, Purchase.status, Purchase.created_at)
                .where(Purchase.id.in_(purchase_ids))
            ).all()
            purchases = {
                int(r[0]): {
                    "note_id": (int(r[1]) if r[1] is not None else None),
                    "combo_id": (int(r[2]) if r[2] is not None else None),
                    "buyer_id": (int(r[3]) if r[3] is not None else None),
                    "buyer_email": r[4],
                    "seller_id": (int(r[5]) if r[5] is not None else None),
                    "amount_cents": int(r[6] or 0),
                    "status": r[7],
                    "created_at": (r[8].isoformat() if r[8] else None),
                }
                for r in rows
            }

        combo_purchases = {}
        if combo_purchase_ids:
            rows = s.execute(
                select(ComboPurchase.id, ComboPurchase.combo_id, ComboPurchase.buyer_id, ComboPurchase.amount_cents, ComboPurchase.status, ComboPurchase.created_at)
                .where(ComboPurchase.id.in_(combo_purchase_ids))
            ).all()
            combo_purchases = {
                int(r[0]): {
                    "combo_id": (int(r[1]) if r[1] is not None else None),
                    "buyer_id": (int(r[2]) if r[2] is not None else None),
                    "amount_cents": int(r[3] or 0),
                    "status": r[4],
                    "created_at": (r[5].isoformat() if r[5] else None),
                }
                for r in rows
            }


        items = []
        for ev in events:
            a = actors.get(int(ev.actor_user_id)) if ev.actor_user_id else None

            # Meta: superadmin ve todo; admin ve solo campos seguros (para evitar exponer datos sensibles)
            meta_pretty = None
            safe_meta = None
            try:
                if isinstance(ev.meta, dict) and ev.meta:
                    if getattr(current_user, "is_superadmin", False):
                        safe_meta = ev.meta
                    else:
                        SAFE_KEYS = {
                            "note_id","note_title","combo_id","combo_title","title",
                            "buyer_email","seller_email","buyer_id","seller_id",
                            "amount_cents","status","from","to","is_free",
                            "reason","reporter_email","reporter_user_id",
                            "university","faculty","career",
                        }
                        safe_meta = {k: ev.meta.get(k) for k in SAFE_KEYS if k in ev.meta}
                    meta_pretty = json.dumps(safe_meta, ensure_ascii=False, indent=2)
            except Exception:
                meta_pretty = None
                safe_meta = None

            # target label + url + compact info
            target_label = None
            target_url = None
            note_info = None
            combo_info = None
            user_info = None

            if ev.target_type == "note" and ev.target_id is not None:
                n = notes.get(int(ev.target_id))
                title = (n or {}).get("title") or ""
                target_label = f"Apunte #{int(ev.target_id)}" + (f" · {title}" if title else "")
                target_url = f"/note/{int(ev.target_id)}"
                note_info = target_label
            elif ev.target_type == "combo" and ev.target_id is not None:
                c = combos.get(int(ev.target_id))
                title = (c or {}).get("title") or ""
                target_label = f"Combo #{int(ev.target_id)}" + (f" · {title}" if title else "")
                target_url = f"/combos/{int(ev.target_id)}"
                combo_info = target_label
            elif ev.target_type == "user" and ev.target_id is not None:
                u = users.get(int(ev.target_id))
                nm = (u or {}).get("name") or f"Usuario #{int(ev.target_id)}"
                em = (u or {}).get("email")
                target_label = nm + (f" · {em}" if em else "")
                user_info = target_label

            elif ev.target_type == "purchase" and ev.target_id is not None:
                pinfo = purchases.get(int(ev.target_id))
                amt = None
                try:
                    amt = cents_to_amount(int((pinfo or {}).get("amount_cents") or 0))
                except Exception:
                    amt = None
                target_label = f"Compra #{int(ev.target_id)}" + (f" · ${money_1_decimal(amt)}" if amt is not None else "")
                target_url = None
            elif ev.target_type == "combo_purchase" and ev.target_id is not None:
                cpinfo = combo_purchases.get(int(ev.target_id))
                amt = None
                try:
                    amt = cents_to_amount(int((cpinfo or {}).get("amount_cents") or 0))
                except Exception:
                    amt = None
                target_label = f"Compra combo #{int(ev.target_id)}" + (f" · ${money_1_decimal(amt)}" if amt is not None else "")
                target_url = None

            action_label = _action_label(ev.action)

            # summary short, human-friendly
            summary = None
            if ev.action == "purchase_created":
                try:
                    m = ev.meta or {}
                    note_title = m.get("note_title") or ""
                    amt = cents_to_amount(int(m.get("amount_cents") or 0))
                    summary = f"Compra iniciada por {m.get('buyer_email') or 'comprador'} para el apunte #{m.get('note_id')}" + (f" · {note_title}" if note_title else "") + f" por ${money_1_decimal(amt)}."
                except Exception:
                    summary = "Compra iniciada."
            elif ev.action == "purchase_status":
                try:
                    m = ev.meta or {}
                    summary = f"Estado de compra actualizado: {m.get('from') or '-'} → {m.get('to') or '-'}."
                except Exception:
                    summary = "Estado de compra actualizado."
            elif ev.action == "combo_purchase_created":
                try:
                    m = ev.meta or {}
                    amt = cents_to_amount(int(m.get("amount_cents") or 0))
                    summary = f"Compra de combo iniciada (combo #{m.get('combo_id')}) por ${money_1_decimal(amt)}."
                except Exception:
                    summary = "Compra de combo iniciada."
            elif ev.action == "combo_purchase_status":
                try:
                    m = ev.meta or {}
                    summary = f"Estado de compra de combo actualizado: {m.get('from') or '-'} → {m.get('to') or '-'}."
                except Exception:
                    summary = "Estado de compra de combo actualizado."
            elif ev.action == "note_uploaded":
                try:
                    m = ev.meta or {}
                    summary = f"Se subió un apunte: #{m.get('note_id')} · {m.get('title') or ''}."
                except Exception:
                    summary = "Se subió un apunte."
            elif ev.action == "note_downloaded":
                try:
                    m = ev.meta or {}
                    summary = f"Descarga de apunte #{m.get('note_id')}" + (" (gratis)." if m.get('is_free') else ".")
                except Exception:
                    summary = "Descarga de apunte."
            elif ev.action == "combo_downloaded":
                try:
                    m = ev.meta or {}
                    summary = f"Descarga de combo #{m.get('combo_id')}" + (" (gratis)." if m.get('is_free') else ".")
                except Exception:
                    summary = "Descarga de combo."
            elif ev.action == "note_reported":
                summary = "Se recibió un reporte sobre un apunte. Queda en revisión."
            elif ev.action == "admin_delete_note":
                reason = None
                try:
                    reason = (ev.meta or {}).get("reason")
                except Exception:
                    reason = None
                summary = f"Se eliminó un apunte{' (' + reason + ')' if reason else ''}."
            elif ev.action == "admin_delete_combo":
                reason = None
                try:
                    reason = (ev.meta or {}).get("reason")
                except Exception:
                    reason = None
                summary = f"Se eliminó un combo{' (' + reason + ')' if reason else ''}."
            elif ev.action == "user_suspend":
                summary = "La cuenta quedó suspendida: solo acceso al perfil hasta reactivación."
            elif ev.action == "user_reactivate":
                summary = "La cuenta fue reactivada y recuperó el acceso total."
            elif ev.action == "user_delete":
                summary = "Se eliminó la cuenta (anonimización + limpieza de compras/descargas)."

            # inline meta (very short)
            meta_inline = None
            try:
                m = safe_meta if isinstance(safe_meta, dict) else (ev.meta if isinstance(ev.meta, dict) else {})
                parts = []
                if m.get("buyer_email"): parts.append(f"Comprador: {m.get('buyer_email')}")
                if m.get("seller_email"): parts.append(f"Vendedor: {m.get('seller_email')}")
                if m.get("reason"): parts.append(f"Motivo: {m.get('reason')}")
                if m.get("from") or m.get("to"):
                    parts.append(f"Estado: {m.get('from') or '-'} → {m.get('to') or '-'}")
                if parts:
                    meta_inline = " · ".join(parts[:3])
            except Exception:
                meta_inline = None

            items.append({
                "id": int(ev.id),
                "code": ev.code,
                "created_at": (ev.created_at.isoformat() if getattr(ev, "created_at", None) else None),
                "actor_user_id": (int(ev.actor_user_id) if ev.actor_user_id else None),
                "actor_name": (a.get("name") if a else None),
                "actor_email": (a.get("email") if a else None),
                "action": action_label,
                "action_raw": ev.action,
                "target_type": ev.target_type,
                "target_id": (int(ev.target_id) if ev.target_id is not None else None),
                "target_label": target_label,
                "target_url": target_url,
                "summary": summary,
                "note_info": note_info,
                "combo_info": combo_info,
                "user_info": user_info,
                "meta": ev.meta,
                "meta_pretty": meta_pretty,
                "meta_inline": meta_inline,
            })

    return jsonify({"items": items})


# Admin HUB - export gestiones (CSV)
@app.get("/admin/export/tickets.csv")
@login_required
@admin_required
def admin_export_tickets_csv():
    q = (request.args.get("q") or "").strip()
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()
    category = (request.args.get("category") or "").strip()
    actor = (request.args.get("actor") or "").strip()
    critical_only = (request.args.get("critical") or "").strip() in ("1", "true", "True", "yes")

    limit = request.args.get("limit", type=int) or 2000
    if limit < 1:
        limit = 200
    if limit > 20000:
        limit = 20000  # export cap

    # Reuse the API logic by calling the same query builder inline (simplified)
    with Session() as s:
        filters = []

        try:
            if date_from:
                dt_from = datetime.strptime(date_from, "%Y-%m-%d")
                filters.append(AuditEvent.created_at >= dt_from)
        except Exception:
            pass
        try:
            if date_to:
                dt_to = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
                filters.append(AuditEvent.created_at < dt_to)
        except Exception:
            pass

        CATEGORY_ACTIONS = {
            "transactions": {"purchase_created", "purchase_status", "combo_purchase_created", "combo_purchase_status"},
            "downloads": {"note_downloaded", "combo_downloaded"},
            "content": {"note_uploaded", "admin_delete_note", "admin_delete_combo", "user_delete_note", "user_delete_combo", "note_edited", "combo_edited"},
            "accounts": {"user_suspend", "user_reactivate", "user_delete"},
            "moderation": {"note_moderation", "combo_moderation", "note_reported"},
            "system": {"legal_accept"},
        }
        if category in CATEGORY_ACTIONS:
            filters.append(AuditEvent.action.in_(list(CATEGORY_ACTIONS[category])))

        CRITICAL_ACTIONS = {
            "note_reported",
            "user_suspend",
            "admin_delete_note",
            "admin_delete_combo",
            "user_delete",
            "note_moderation",
            "combo_moderation",
        }
        if critical_only:
            filters.append(AuditEvent.action.in_(list(CRITICAL_ACTIONS)))

        join_user = False
        if actor == "system":
            filters.append(AuditEvent.actor_user_id.is_(None))
        elif actor in ("admin", "user"):
            join_user = True

        if q:
            like = f"%{q}%"
            search_filters = [
                AuditEvent.code.ilike(like),
                AuditEvent.action.ilike(like),
                cast(AuditEvent.meta, String).ilike(like),
                cast(AuditEvent.target_type, String).ilike(like),
            ]
            try:
                if q.isdigit():
                    search_filters.append(AuditEvent.target_id == int(q))
                    search_filters.append(AuditEvent.actor_user_id == int(q))
            except Exception:
                pass
            filters.append(or_(*search_filters))

        stmt = select(AuditEvent)
        if join_user:
            stmt = stmt.join(User, User.id == AuditEvent.actor_user_id)
            if actor == "admin":
                stmt = stmt.where(or_(User.is_admin == True, User.role == "superadmin"))
            elif actor == "user":
                stmt = stmt.where(and_(User.is_admin == False, or_(User.role.is_(None), User.role != "superadmin")))

        if filters:
            stmt = stmt.where(and_(*filters))

        stmt = stmt.order_by(desc(AuditEvent.created_at)).limit(limit)
        events = s.execute(stmt).scalars().all()

    import io, csv as _csv
    output = io.StringIO()
    w = _csv.writer(output)
    w.writerow(["code", "created_at", "action", "actor_user_id", "target_type", "target_id", "meta_json"])
    for ev in events:
        meta = None
        try:
            if isinstance(ev.meta, dict):
                # admins get safe meta; superadmin gets full
                if getattr(current_user, "is_superadmin", False):
                    meta = ev.meta
                else:
                    SAFE_KEYS = {
                        "note_id","note_title","combo_id","combo_title","title",
                        "buyer_email","seller_email","buyer_id","seller_id",
                        "amount_cents","status","from","to","is_free",
                        "reason","reporter_email","reporter_user_id",
                        "university","faculty","career",
                    }
                    meta = {k: ev.meta.get(k) for k in SAFE_KEYS if k in ev.meta}
        except Exception:
            meta = None

        w.writerow([
            ev.code,
            (ev.created_at.isoformat() if ev.created_at else ""),
            ev.action,
            ev.actor_user_id or "",
            ev.target_type or "",
            ev.target_id or "",
            (json.dumps(meta, ensure_ascii=False) if meta else ""),
        ])

    csv_bytes = output.getvalue().encode("utf-8")
    resp = make_response(csv_bytes)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="gestiones_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv"'
    return resp


# -----------------------
# A4) Admin: Dashboard + movimientos
# -----------------------

@app.get("/admin/api/stats")
@login_required
@admin_required
def admin_api_stats():
    """KPIs + series para el dashboard admin.

    A5: preferimos `stats_daily` (rápido/estable). Si todavía no hay filas,
    hacemos fallback a consultas directas.
    """

    days = 30
    try:
        days = max(7, min(int(request.args.get("days", "30")), 365))
    except Exception:
        days = 30

    since_dt = datetime.utcnow() - timedelta(days=days)
    since_day = (datetime.utcnow() - timedelta(days=days - 1)).date()
    today_day = datetime.utcnow().date()

    with Session() as s:
        # Totales
        total_users = int(s.execute(select(func.count(User.id))).scalar_one() or 0)
        total_notes = int(s.execute(select(func.count(Note.id))).scalar_one() or 0)
        total_combos = int(s.execute(select(func.count(Combo.id))).scalar_one() or 0)
        users_range = 0
        try:
            users_range = int(s.execute(select(func.count(User.id)).where(User.created_at >= since_dt)).scalar_one() or 0)
        except Exception:
            users_range = 0

        # Days list
        days_list = []
        cur = since_day
        for _ in range(days):
            days_list.append(cur.isoformat())
            cur = cur + timedelta(days=1)

        # ---- A5: stats_daily ----
        stats_rows = s.execute(
            select(DailyStat).where(DailyStat.day >= since_day, DailyStat.day <= today_day).order_by(DailyStat.day)
        ).scalars().all()

        st_map = {r.day.isoformat(): r for r in stats_rows}

        # Sums over range
        gross_range = sum(int(getattr(r, "gross_income_cents", 0) or 0) for r in stats_rows)
        ay_range = sum(int(getattr(r, "ay_commission_cents", 0) or 0) for r in stats_rows)
        mp_range = sum(int(getattr(r, "mp_fee_cents", 0) or 0) for r in stats_rows)
        seller_range = sum(int(getattr(r, "seller_income_cents", 0) or 0) for r in stats_rows)
        sales_range = sum(int(getattr(r, "sales_count", 0) or 0) for r in stats_rows)
        free_dl_range = sum(int(getattr(r, "free_downloads", 0) or 0) for r in stats_rows)
        paid_dl_range = sum(int(getattr(r, "paid_downloads", 0) or 0) for r in stats_rows)

        # If stats table is empty (first deploy), fallback to historical queries
        if not stats_rows:
            try:
                sales_range = int(
                    s.execute(select(func.count(Purchase.id)).where(Purchase.status == "approved", Purchase.created_at >= since_dt)).scalar_one() or 0
                )
                gross_range = int(
                    s.execute(select(func.coalesce(func.sum(Purchase.gross_cents), 0)).where(Purchase.status == "approved", Purchase.created_at >= since_dt)).scalar_one() or 0
                )
                ay_range = int(
                    s.execute(select(func.coalesce(func.sum(Purchase.platform_fee_cents), 0)).where(Purchase.status == "approved", Purchase.created_at >= since_dt)).scalar_one() or 0
                )
                mp_range = int(
                    s.execute(select(func.coalesce(func.sum(Purchase.mp_fee_cents), 0)).where(Purchase.status == "approved", Purchase.created_at >= since_dt)).scalar_one() or 0
                )
                seller_range = int(
                    s.execute(select(func.coalesce(func.sum(Purchase.seller_net_cents), 0)).where(Purchase.status == "approved", Purchase.created_at >= since_dt)).scalar_one() or 0
                )
                free_dl_range = int(
                    s.execute(select(func.count(DownloadLog.id)).where(DownloadLog.created_at >= since_dt, DownloadLog.was_free == True)).scalar_one() or 0
                )
                paid_dl_range = int(
                    s.execute(select(func.count(DownloadLog.id)).where(DownloadLog.created_at >= since_dt, DownloadLog.was_free == False)).scalar_one() or 0
                )
            except Exception:
                pass

        # ---- Series (A5 stats) ----
        purchases_series = []
        free_dl_series = []
        paid_dl_series = []
        for d in days_list:
            r = st_map.get(d)
            purchases_series.append(int(getattr(r, "sales_count", 0) or 0) if r else 0)
            free_dl_series.append(int(getattr(r, "free_downloads", 0) or 0) if r else 0)
            paid_dl_series.append(int(getattr(r, "paid_downloads", 0) or 0) if r else 0)

        # Page views series (no A5 table yet; keep query)
        def _day_expr(col):
            return func.date_trunc("day", col)

        pv_rows = s.execute(
            select(_day_expr(AnalyticsEvent.created_at).label("d"), func.count(AnalyticsEvent.id).label("c"))
            .where(AnalyticsEvent.event == "page_view", AnalyticsEvent.created_at >= since_dt)
            .group_by("d").order_by("d")
        ).all()
        pv_map = {r.d.date().isoformat(): int(r.c) for r in pv_rows}
        page_views_series = [pv_map.get(d, 0) for d in days_list]

        series = {
            "days": days_list,
            "purchases": purchases_series,
            "free_downloads": free_dl_series,
            "paid_downloads": paid_dl_series,
            "page_views": page_views_series,
        }

        # Top sellers (approved purchases)
        top_sellers_rows = s.execute(
            select(Purchase.seller_id, func.coalesce(func.sum(Purchase.seller_net_cents), 0).label("net"), func.count(Purchase.id).label("cnt"))
            .where(Purchase.status == "approved")
            .group_by(Purchase.seller_id)
            .order_by(func.coalesce(func.sum(Purchase.seller_net_cents), 0).desc())
            .limit(10)
        ).all()
        seller_ids = [r.seller_id for r in top_sellers_rows if r.seller_id]
        seller_map = {}
        if seller_ids:
            for u in s.execute(select(User).where(User.id.in_(seller_ids))).scalars().all():
                seller_map[u.id] = {"name": u.name, "email": u.email}

        top_sellers = []
        for r in top_sellers_rows:
            u = seller_map.get(r.seller_id, {})
            top_sellers.append({
                "seller_id": r.seller_id,
                "seller_name": u.get("name") or f"Usuario #{r.seller_id}",
                "seller_email": u.get("email"),
                "sales_count": int(r.cnt),
                "seller_net_cents": int(r.net or 0),
            })

        # Top content by purchases (note + combo)
        top_notes = s.execute(
            select(Purchase.note_id, func.count(Purchase.id).label("cnt"))
            .where(Purchase.status == "approved", Purchase.note_id.isnot(None))
            .group_by(Purchase.note_id)
            .order_by(func.count(Purchase.id).desc())
            .limit(10)
        ).all()
        note_ids = [r.note_id for r in top_notes if r.note_id]
        note_title = {}
        if note_ids:
            for n in s.execute(select(Note.id, Note.title).where(Note.id.in_(note_ids))).all():
                note_title[n.id] = n.title

        top_combos = s.execute(
            select(Purchase.combo_id, func.count(Purchase.id).label("cnt"))
            .where(Purchase.status == "approved", Purchase.combo_id.isnot(None))
            .group_by(Purchase.combo_id)
            .order_by(func.count(Purchase.id).desc())
            .limit(10)
        ).all()
        combo_ids = [r.combo_id for r in top_combos if r.combo_id]
        combo_title = {}
        if combo_ids:
            for c in s.execute(select(Combo.id, Combo.title).where(Combo.id.in_(combo_ids))).all():
                combo_title[c.id] = c.title

        top_content = {
            "notes": [{"id": r.note_id, "title": note_title.get(r.note_id) or f"Apunte #{r.note_id}", "purchases": int(r.cnt)} for r in top_notes],
            "combos": [{"id": r.combo_id, "title": combo_title.get(r.combo_id) or f"Combo #{r.combo_id}", "purchases": int(r.cnt)} for r in top_combos],
        }

    # Backwards compatible response: keep old keys AND add `kpis` used by admin UI.
    return jsonify({
        "ok": True,
        "range_days": int(days),
        "kpis": {
            "users_total": int(total_users),
            "users_range": int(users_range),
            "notes_total": int(total_notes),
            "combos_total": int(total_combos),

            "purchases_range": int(sales_range),
            "revenue_gross_range_cents": int(gross_range),
            "revenue_ay_range_cents": int(ay_range),
            "revenue_mp_range_cents": int(mp_range),
            "revenue_seller_range_cents": int(seller_range),
            "free_downloads_range": int(free_dl_range),
            "paid_downloads_range": int(paid_dl_range),
            "pageviews_range": int(sum(page_views_series or [])),
            "conversion_rate_pct": int(round((sales_range / max(1, sum(page_views_series or []))) * 100.0)),
        },

        # legacy blocks (kept)
        "totals": {
            "users": int(total_users),
            "notes": int(total_notes),
            "combos": int(total_combos),
        },
        "money": {
            "gross_cents": int(gross_range),
            "platform_fee_cents": int(ay_range),
            "mp_fee_cents": int(mp_range),
            "seller_net_cents": int(seller_range),
        },
        "series": series,
        "top_sellers": top_sellers,
        "top_content": top_content,
    })


@app.get("/admin/api/analytics")
@login_required
@admin_required
def admin_api_analytics():
    """A6 analytics dashboard: pageviews + funnel (note + combo).

    Returns daily series and totals for the last N days.
    """
    days = 30
    try:
        days = max(7, min(int(request.args.get("days", "30")), 365))
    except Exception:
        days = 30

    since_dt = datetime.utcnow() - timedelta(days=days)
    since_day = (datetime.utcnow() - timedelta(days=days - 1)).date()
    today_day = datetime.utcnow().date()

    # Days list (ISO)
    days_list = []
    cur = since_day
    for _ in range(days):
        days_list.append(cur.isoformat())
        cur = cur + timedelta(days=1)

    with Session() as s:
        # Dialect-safe "day" extractor
        def _day_expr(col):
            try:
                if engine.dialect.name.startswith("postgres"):
                    return func.date_trunc("day", col)
            except Exception:
                pass
            return func.date(col)

        # Aggregate per day & event
        rows = s.execute(
            select(
                _day_expr(AnalyticsEvent.created_at).label("d"),
                AnalyticsEvent.event.label("e"),
                func.count(AnalyticsEvent.id).label("c"),
            )
            .where(AnalyticsEvent.created_at >= since_dt)
            .group_by("d", "e")
            .order_by("d")
        ).all()

        # Normalize map: day_iso -> event -> count
        m = {}
        for r in rows:
            try:
                d = r.d.date().isoformat() if hasattr(r.d, "date") else str(r.d)
            except Exception:
                d = str(r.d)
            m.setdefault(d, {})[str(r.e)] = int(r.c)

        def series_for(event_name: str):
            return [int(m.get(d, {}).get(event_name, 0)) for d in days_list]

        series = {
            "days": days_list,
            "page_view": series_for("page_view"),
            "note_view": series_for("note_view"),
            "combo_view": series_for("combo_view"),
            "buy_intent": series_for("buy_intent"),
            "checkout_start": series_for("checkout_start"),
            "purchase_approved": series_for("purchase_approved"),
        }

        totals = {k: int(sum(v)) for k, v in series.items() if k != "days"}

        # Funnel breakdown NOTE-only (by presence of note_id)
        note_view_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "note_view",
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )
        note_buy_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "buy_intent",
                    AnalyticsEvent.note_id.isnot(None),
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )
        note_checkout_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "checkout_start",
                    AnalyticsEvent.note_id.isnot(None),
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )
        note_approved_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "purchase_approved",
                    AnalyticsEvent.note_id.isnot(None),
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )

        # Funnel breakdown COMBO-only
        combo_view_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "combo_view",
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )
        combo_buy_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "buy_intent",
                    AnalyticsEvent.combo_id.isnot(None),
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )
        combo_checkout_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "checkout_start",
                    AnalyticsEvent.combo_id.isnot(None),
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )
        combo_approved_cnt = int(
            s.execute(
                select(func.count(AnalyticsEvent.id)).where(
                    AnalyticsEvent.event == "purchase_approved",
                    AnalyticsEvent.combo_id.isnot(None),
                    AnalyticsEvent.created_at >= since_dt,
                )
            ).scalar_one() or 0
        )

        def pct(a, b):
            try:
                return round((float(a) / float(b)) * 100.0, 1) if b else 0.0
            except Exception:
                return 0.0

        funnels = {
            "notes": {
                "views": note_view_cnt,
                "buy_intents": note_buy_cnt,
                "checkout_starts": note_checkout_cnt,
                "approved": note_approved_cnt,
                "view_to_buy_pct": pct(note_buy_cnt, note_view_cnt),
                "buy_to_checkout_pct": pct(note_checkout_cnt, note_buy_cnt),
                "checkout_to_approved_pct": pct(note_approved_cnt, note_checkout_cnt),
            },
            "combos": {
                "views": combo_view_cnt,
                "buy_intents": combo_buy_cnt,
                "checkout_starts": combo_checkout_cnt,
                "approved": combo_approved_cnt,
                "view_to_buy_pct": pct(combo_buy_cnt, combo_view_cnt),
                "buy_to_checkout_pct": pct(combo_checkout_cnt, combo_buy_cnt),
                "checkout_to_approved_pct": pct(combo_approved_cnt, combo_checkout_cnt),
            },
        }

    return jsonify({
        "ok": True,
        "range_days": int(days),
        "series": series,
        "totals": totals,
        "funnels": funnels,
    })


@app.get("/admin/api/movements")
@login_required
@admin_required
def admin_api_movements():
    q = (request.args.get("q") or "").strip().lower()
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()

    def _parse_date(s):
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    dt_from = _parse_date(date_from)
    dt_to = _parse_date(date_to)
    if dt_to:
        dt_to = dt_to + timedelta(days=1)  # inclusive

    items = []
    with Session() as s:
        # Purchases
        pq = select(Purchase).order_by(Purchase.created_at.desc()).limit(200)
        if dt_from:
            pq = pq.where(Purchase.created_at >= dt_from)
        if dt_to:
            pq = pq.where(Purchase.created_at < dt_to)
        if q:
            pq = pq.where(func.lower(Purchase.buyer_email).like(f"%{q}%"))

        purchases = s.execute(pq).scalars().all()

        # preload users and content titles
        buyer_emails = {p.buyer_email for p in purchases if p.buyer_email}
        seller_ids = {p.seller_id for p in purchases if p.seller_id}
        note_ids = {p.note_id for p in purchases if p.note_id}
        combo_ids = {p.combo_id for p in purchases if p.combo_id}

        users_by_email = {}
        if buyer_emails:
            for u in s.execute(select(User).where(func.lower(User.email).in_([e.lower() for e in buyer_emails]))).scalars().all():
                users_by_email[u.email.lower()] = u
        sellers = {}
        if seller_ids:
            for u in s.execute(select(User).where(User.id.in_(seller_ids))).scalars().all():
                sellers[u.id] = u
        notes = {}
        if note_ids:
            for n in s.execute(select(Note).where(Note.id.in_(note_ids))).scalars().all():
                notes[n.id] = n
        combos = {}
        if combo_ids:
            for c in s.execute(select(Combo).where(Combo.id.in_(combo_ids))).scalars().all():
                combos[c.id] = c

        for p in purchases:
            seller = sellers.get(p.seller_id)
            buyer = users_by_email.get((p.buyer_email or "").lower())
            obj = None
            obj_type = None
            if p.note_id:
                obj = notes.get(p.note_id)
                obj_type = "note"
            elif p.combo_id:
                obj = combos.get(p.combo_id)
                obj_type = "combo"

            title = (obj.title if obj else None)
            # resumen
            summary_parts = []
            if obj_type == "note":
                summary_parts.append("Compra de apunte")
            elif obj_type == "combo":
                summary_parts.append("Compra de combo")
            else:
                summary_parts.append("Compra")
            if title:
                summary_parts.append(f"— {title}")
            summary = " ".join(summary_parts)

            items.append({
                "kind": "purchase",
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "status": p.status,
                "buyer": {"email": p.buyer_email, "name": getattr(buyer, "name", None)},
                "seller": {"id": p.seller_id, "name": getattr(seller, "name", None), "email": getattr(seller, "email", None)},
                "object": {"type": obj_type, "id": (p.note_id or p.combo_id), "title": title},
                "money": {
                    "gross_cents": int(p.gross_cents or 0),
                    "platform_fee_cents": int(p.platform_fee_cents or 0),
                    "mp_fee_cents": int(p.mp_fee_cents or 0),
                    "seller_net_cents": int(p.seller_net_cents or 0),
                },
                "summary": summary,
            })

        # Free downloads
        dq = select(DownloadLog).where(DownloadLog.was_free == True).order_by(DownloadLog.created_at.desc()).limit(200)
        if dt_from:
            dq = dq.where(DownloadLog.created_at >= dt_from)
        if dt_to:
            dq = dq.where(DownloadLog.created_at < dt_to)
        dls = s.execute(dq).scalars().all()
        user_ids = {d.user_id for d in dls if d.user_id}
        note_ids2 = {d.note_id for d in dls if d.note_id}
        combo_ids2 = {d.combo_id for d in dls if d.combo_id}
        users = {}
        if user_ids:
            for u in s.execute(select(User).where(User.id.in_(user_ids))).scalars().all():
                users[u.id] = u
        notes2 = {}
        if note_ids2:
            for n in s.execute(select(Note).where(Note.id.in_(note_ids2))).scalars().all():
                notes2[n.id] = n
        combos2 = {}
        if combo_ids2:
            for c in s.execute(select(Combo).where(Combo.id.in_(combo_ids2))).scalars().all():
                combos2[c.id] = c

        for d in dls:
            u = users.get(d.user_id)
            obj = None
            obj_type = None
            seller = None
            if d.note_id:
                obj = notes2.get(d.note_id)
                obj_type = "note"
                seller = users.get(getattr(obj, "seller_id", None)) if obj else None
            elif d.combo_id:
                obj = combos2.get(d.combo_id)
                obj_type = "combo"
                seller = users.get(getattr(obj, "seller_id", None)) if obj else None
            title = (obj.title if obj else None)
            summary = f"Descarga gratuita" + (f" — {title}" if title else "")
            items.append({
                "kind": "free_download",
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "status": "free",
                "buyer": {"email": getattr(u, "email", None), "name": getattr(u, "name", None), "id": d.user_id},
                "seller": {"id": getattr(seller, "id", None), "name": getattr(seller, "name", None), "email": getattr(seller, "email", None)},
                "object": {"type": obj_type, "id": (d.note_id or d.combo_id), "title": title},
                "money": {"gross_cents": 0, "platform_fee_cents": 0, "mp_fee_cents": 0, "seller_net_cents": 0},
                "summary": summary,
            })

    # sort by created_at desc
    def _key(it):
        return it.get("created_at") or ""
    items.sort(key=_key, reverse=True)

    # additional text filter over titles / names (client side is ok, but we can do a bit here)
    if q:
        def match(it):
            hay = " ".join([
                (it.get("summary") or ""),
                (it.get("buyer") or {}).get("email") or "",
                (it.get("buyer") or {}).get("name") or "",
                (it.get("seller") or {}).get("email") or "",
                (it.get("seller") or {}).get("name") or "",
                (it.get("object") or {}).get("title") or "",
            ]).lower()
            return q in hay
        items = [it for it in items if match(it)]

    return jsonify({"items": items[:400]})


@app.get("/admin/export/movements.csv")
@login_required
@admin_required
def admin_export_movements_csv():
    """Exporta movimientos (compras + descargas gratis) a CSV.

    Respeta filtros q/from/to (mismo formato que /admin/api/movements).
    """
    import csv
    from io import StringIO

    q = (request.args.get("q") or "").strip().lower()
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()

    def _parse_date(s):
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    dt_from = _parse_date(date_from)
    dt_to = _parse_date(date_to)
    if dt_to:
        dt_to = dt_to + timedelta(days=1)

    rows = []
    with Session() as s:
        pq = select(Purchase).order_by(Purchase.created_at.desc()).limit(5000)
        if dt_from:
            pq = pq.where(Purchase.created_at >= dt_from)
        if dt_to:
            pq = pq.where(Purchase.created_at < dt_to)
        if q:
            pq = pq.where(func.lower(Purchase.buyer_email).like(f"%{q}%"))
        purchases = s.execute(pq).scalars().all()

        # preload sellers, notes, combos
        seller_ids = {p.seller_id for p in purchases if p.seller_id}
        note_ids = {p.note_id for p in purchases if p.note_id}
        combo_ids = {p.combo_id for p in purchases if p.combo_id}
        sellers = {}
        if seller_ids:
            for u in s.execute(select(User).where(User.id.in_(seller_ids))).scalars().all():
                sellers[u.id] = u
        notes = {}
        if note_ids:
            for n in s.execute(select(Note).where(Note.id.in_(note_ids))).scalars().all():
                notes[n.id] = n
        combos = {}
        if combo_ids:
            for c in s.execute(select(Combo).where(Combo.id.in_(combo_ids))).scalars().all():
                combos[c.id] = c

        for p in purchases:
            obj_type = 'note' if p.note_id else ('combo' if p.combo_id else '')
            title = None
            if p.note_id and p.note_id in notes:
                title = notes[p.note_id].title
            if p.combo_id and p.combo_id in combos:
                title = combos[p.combo_id].title
            seller = sellers.get(p.seller_id)
            rows.append({
                'kind': 'purchase',
                'created_at': p.created_at.isoformat() if p.created_at else '',
                'status': p.status or '',
                'buyer_email': p.buyer_email or '',
                'seller_email': getattr(seller, 'email', '') or '',
                'object_type': obj_type,
                'object_id': p.note_id or p.combo_id or '',
                'object_title': title or '',
                'gross_cents': int(p.gross_cents or 0),
                'ay_fee_cents': int(p.platform_fee_cents or 0),
                'mp_fee_cents': int(p.mp_fee_cents or 0),
                'seller_net_cents': int(p.seller_net_cents or 0),
                'payment_id': p.payment_id or '',
            })

        dq = select(DownloadLog).where(DownloadLog.was_free == True).order_by(DownloadLog.created_at.desc()).limit(5000)
        if dt_from:
            dq = dq.where(DownloadLog.created_at >= dt_from)
        if dt_to:
            dq = dq.where(DownloadLog.created_at < dt_to)
        dls = s.execute(dq).scalars().all()

        user_ids = {d.user_id for d in dls if d.user_id}
        note_ids2 = {d.note_id for d in dls if d.note_id}
        combo_ids2 = {d.combo_id for d in dls if d.combo_id}
        users = {}
        if user_ids:
            for u in s.execute(select(User).where(User.id.in_(user_ids))).scalars().all():
                users[u.id] = u
        notes2 = {}
        if note_ids2:
            for n in s.execute(select(Note).where(Note.id.in_(note_ids2))).scalars().all():
                notes2[n.id] = n
        combos2 = {}
        if combo_ids2:
            for c in s.execute(select(Combo).where(Combo.id.in_(combo_ids2))).scalars().all():
                combos2[c.id] = c

        for d in dls:
            u = users.get(d.user_id)
            obj_type = 'note' if d.note_id else ('combo' if d.combo_id else '')
            title = None
            if d.note_id and d.note_id in notes2:
                title = notes2[d.note_id].title
            if d.combo_id and d.combo_id in combos2:
                title = combos2[d.combo_id].title
            rows.append({
                'kind': 'free_download',
                'created_at': d.created_at.isoformat() if d.created_at else '',
                'status': 'free',
                'buyer_email': getattr(u, 'email', '') or '',
                'seller_email': '',
                'object_type': obj_type,
                'object_id': d.note_id or d.combo_id or '',
                'object_title': title or '',
                'gross_cents': 0,
                'ay_fee_cents': 0,
                'mp_fee_cents': 0,
                'seller_net_cents': 0,
                'payment_id': '',
            })

    # Text filter across some columns if q is provided
    if q:
        def match(r):
            hay = ' '.join([r.get('buyer_email',''), r.get('seller_email',''), r.get('object_title',''), r.get('object_type',''), r.get('status','')]).lower()
            return q in hay
        rows = [r for r in rows if match(r)]

    rows.sort(key=lambda r: r.get('created_at',''), reverse=True)

    out = StringIO()
    w = csv.writer(out)
    w.writerow(['kind','created_at','status','buyer_email','seller_email','object_type','object_id','object_title','gross','ay_fee','mp_fee','seller_net','payment_id'])
    for r in rows:
        w.writerow([
            r['kind'], r['created_at'], r['status'], r['buyer_email'], r['seller_email'],
            r['object_type'], r['object_id'], r['object_title'],
            int(r['gross_cents']), int(r['ay_fee_cents']), int(r['mp_fee_cents']), int(r['seller_net_cents']),
            r['payment_id'],
        ])

    resp = make_response(out.getvalue())
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename="apuntesya_movimientos.csv"'
    return resp


@app.get("/admin/export/events.csv")
@login_required
@admin_required
def admin_export_events_csv():
    """Exporta eventos analytics a CSV.

    Respeta filtros: days (rango), event (tipo)
    """
    import csv
    from io import StringIO

    days = request.args.get('days', '30')
    try:
        days_i = max(1, min(365, int(days)))
    except Exception:
        days_i = 30
    event = (request.args.get('event') or '').strip()
    since = datetime.utcnow() - timedelta(days=days_i)

    with Session() as s:
        q = select(AnalyticsEvent).where(AnalyticsEvent.created_at >= since)
        if event:
            q = q.where(AnalyticsEvent.event == event)
        q = q.order_by(AnalyticsEvent.created_at.desc()).limit(20000)
        evs = s.execute(q).scalars().all()

    out = StringIO()
    w = csv.writer(out)
    w.writerow(['created_at','event','user_id','note_id','combo_id','path','referrer','ip','user_agent','meta'])
    for e in evs:
        w.writerow([
            e.created_at.isoformat() if e.created_at else '',
            e.event or '',
            e.user_id or '',
            e.note_id or '',
            e.combo_id or '',
            e.path or '',
            e.referrer or '',
            e.ip or '',
            e.user_agent or '',
            (json.dumps(e.meta, ensure_ascii=False) if getattr(e,'meta',None) else ''),
        ])

    resp = make_response(out.getvalue())
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename="apuntesya_eventos.csv"'
    return resp


@app.get("/admin/api/stats_legacy")
@admin_required
def admin_api_stats_summary():
    """KPIs y series básicas para el dashboard de estadísticas."""
    days = request.args.get("days", "30")
    try:
        days_i = max(1, min(365, int(days)))
    except Exception:
        days_i = 30

    since = datetime.utcnow() - timedelta(days=days_i)

    with Session() as s:
        # Totales
        total_users = s.execute(select(func.count(User.id))).scalar_one() or 0
        total_notes = s.execute(select(func.count(Note.id))).scalar_one() or 0
        total_combos = s.execute(select(func.count(Combo.id))).scalar_one() or 0

        # Periodo
        new_users = s.execute(select(func.count(User.id)).where(User.created_at >= since)).scalar_one() or 0

        purchases_notes = s.execute(select(func.count(Purchase.id)).where(Purchase.created_at >= since, Purchase.status == "approved")).scalar_one() or 0
        purchases_combos = s.execute(select(func.count(ComboPurchase.id)).where(ComboPurchase.created_at >= since, ComboPurchase.status == "approved")).scalar_one() or 0

        amount_notes = s.execute(select(func.coalesce(func.sum(Purchase.amount_cents), 0)).where(Purchase.created_at >= since, Purchase.status == "approved")).scalar_one() or 0
        amount_combos = s.execute(select(func.coalesce(func.sum(ComboPurchase.amount_cents), 0)).where(ComboPurchase.created_at >= since, ComboPurchase.status == "approved")).scalar_one() or 0

        free_downloads = s.execute(select(func.count(DownloadLog.id)).where(DownloadLog.created_at >= since, DownloadLog.is_free == True)).scalar_one() or 0

        # Analytics events (page views)
        pageviews = s.execute(select(func.count(AnalyticsEvent.id)).where(AnalyticsEvent.created_at >= since, AnalyticsEvent.event == "page_view")).scalar_one() or 0

        # Top sellers (ventas)
        top_sellers = []
        try:
            rows = s.execute(
                select(Note.seller_id, func.count(Purchase.id).label("cnt"), func.coalesce(func.sum(Purchase.amount_cents), 0).label("sum"))
                .join(Purchase, Purchase.note_id == Note.id)
                .where(Purchase.status == "approved", Purchase.created_at >= since)
                .group_by(Note.seller_id)
                .order_by(func.count(Purchase.id).desc())
                .limit(8)
            ).all()
            # attach names
            seller_ids = [int(r[0]) for r in rows if r[0] is not None]
            sellers = {u.id: u for u in s.execute(select(User).where(User.id.in_(seller_ids))).scalars().all()}
            for sid, cnt, sm in rows:
                u = sellers.get(int(sid))
                top_sellers.append({
                    "seller_id": int(sid),
                    "seller_name": (getattr(u, "name", None) or f"Usuario #{sid}"),
                    "sales_count": int(cnt or 0),
                    "sales_amount_cents": int(sm or 0),
                })
        except Exception:
            top_sellers = []

    purchases_total = int(purchases_notes) + int(purchases_combos)
    amount_total_cents = int(amount_notes) + int(amount_combos)

    return jsonify({
        "range_days": days_i,
        "totals": {
            "users": int(total_users),
            "notes": int(total_notes),
            "combos": int(total_combos),
        },
        "period": {
            "since": since.isoformat(),
            "new_users": int(new_users),
            "pageviews": int(pageviews),
            "purchases": int(purchases_total),
            "sales_amount_cents": int(amount_total_cents),
            "free_downloads": int(free_downloads),
        },
        "top_sellers": top_sellers,
    })


### (el endpoint /admin/api/movements está definido más arriba; evitamos duplicados)


# Admin HUB - moderación (apuntes + combos)
@app.get("/admin/api/moderation")
@login_required
@admin_required
def admin_api_moderation():
    """Devuelve apuntes y combos por estado para mostrar en el Hub (sin salir de la pantalla)."""
    status = (request.args.get("status") or "pending_manual").strip()
    q = (request.args.get("q") or "").strip()
    date_from = (request.args.get("from") or "").strip()  # YYYY-MM-DD
    date_to = (request.args.get("to") or "").strip()      # YYYY-MM-DD
    category = (request.args.get("category") or "").strip()
    actor = (request.args.get("actor") or "").strip()     # user|admin|system
    critical_only = (request.args.get("critical") or "").strip() in ("1", "true", "True", "yes")

    # Limit hard cap (safety)
    limit = request.args.get("limit", type=int) or 200
    if limit < 1:
        limit = 50
    if limit > 2000:
        limit = 2000


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


def _admin_soft_delete_note(s, n: Note, reason: str | None = None) -> None:
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

    # notificar vendedor
    try:
        body = "Tu apunte fue eliminado por un administrador por incumplimiento con los Términos de la plataforma."
        if reason:
            body += f"\n\nMotivo: {reason}"
        notify_user(
            user_id=int(n.seller_id),
            kind="danger",
            title="Apunte eliminado",
            body=body,
            email=True,
            email_subject="ApuntesYa: apunte eliminado",
        )
    except Exception:
        pass


def _admin_soft_delete_combo(s, c: Combo, reason: str | None = None) -> None:
    try:
        c.is_active = False
    except Exception:
        pass
    if hasattr(c, "deleted_at"):
        try:
            c.deleted_at = datetime.utcnow()
        except Exception:
            pass

    # notificar vendedor
    try:
        body = "Tu combo fue eliminado por un administrador por incumplimiento con los Términos de la plataforma."
        if reason:
            body += f"\n\nMotivo: {reason}"
        notify_user(
            user_id=int(c.seller_id),
            kind="danger",
            title="Combo eliminado",
            body=body,
            email=True,
            email_subject="ApuntesYa: combo eliminado",
        )
    except Exception:
        pass


@app.post("/admin/api/content/<int:note_id>/delete")
@login_required
@admin_required
def admin_api_content_delete(note_id: int):
    """Compat: elimina un apunte por id (sin tipo)."""
    return admin_api_content_delete_typed("note", note_id)


@app.post("/admin/api/content/<string:ctype>/<int:item_id>/delete")
@login_required
@admin_required
def admin_api_content_delete_typed(ctype: str, item_id: int):
    reason = None
    if request.is_json:
        reason = (request.get_json(silent=True) or {}).get("reason")
    else:
        reason = request.form.get("reason")
    reason = (reason or "").strip() or None

    with Session() as s:
        if (ctype or "").lower() == "combo":
            c = s.get(Combo, item_id)
            if not c:
                return jsonify({"ok": False, "error": "not_found"}), 404
            _admin_soft_delete_combo(s, c, reason=reason)
            s.commit()
            code = log_audit_event(actor_user_id=current_user.id, action="admin_delete_combo", target_type="combo", target_id=int(c.id), meta={"reason": reason})
            return jsonify({"ok": True, "ticket": code})

        # default: note
        n = s.get(Note, item_id)
        if not n:
            return jsonify({"ok": False, "error": "not_found"}), 404
        _admin_soft_delete_note(s, n, reason=reason)
        s.commit()
        code = log_audit_event(actor_user_id=current_user.id, action="admin_delete_note", target_type="note", target_id=int(n.id), meta={"reason": reason})
        return jsonify({"ok": True, "ticket": code})


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
    # Página eliminada: la edición se hace en /profile
    return redirect(url_for("profile") + "#academics")


@app.post("/update_academics")
@login_required
def update_academics_post():
    # Compatibilidad: enviamos al handler nuevo en /profile
    return profile_update_academics()


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
                Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
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
                    Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
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
            # Ir al detalle del combo recién creado
            return redirect(url_for("combo_detail", combo_id=combo.id))

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
                Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
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
                    Note.moderation_status.in_(("approved","auto_published","published_flagged")),
                    Note.is_archived == False,
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
            return redirect(url_for("combo_detail", combo_id=combo.id))

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
    return redirect(url_for("my_content_edit", tab="combos"))


from flask import render_template, abort
from sqlalchemy import select
from sqlalchemy.orm import joinedload


@app.route("/combos/<int:combo_id>/buy", methods=["GET","POST"])
@login_required
def buy_combo(combo_id):
    # A6 analytics: intent to buy combo (best-effort)
    try:
        log_analytics_event(
            event="buy_intent",
            user_id=int(current_user.id) if getattr(current_user, "is_authenticated", False) else None,
            path=request.path,
            combo_id=int(combo_id),
            meta={"source": "buy_combo_route"},
        )
    except Exception:
        pass

    # Kill-switch (ventas)
    se = (_get_setting("sales_enabled", "1") or "1").strip()
    sales_enabled = se in ("1", "true", "True", "yes", "on")
    if not sales_enabled and not (_is_superadmin(current_user) if getattr(current_user, "is_authenticated", False) else False):
        flash("⚠️ Las ventas están pausadas por mantenimiento. Probá de nuevo más tarde.", "warning")
        return redirect(url_for("combo_detail", combo_id=combo_id))
    with Session() as s:
        combo = s.get(Combo, combo_id)
        if not combo or (hasattr(combo, "is_active") and combo.is_active is False):
            abort(404)

        if not is_public_moderation_status(getattr(combo, "moderation_status", "approved")):
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

        if request.method == "GET":
            gross_ars = money_1_decimal(cents_to_amount(price_cents))
            return render_template(
                "checkout_combo.html",
                combo=combo,
                seller=seller,
                gross_cents=int(price_cents),
                gross_ars=gross_ars,
                legal_version=(app.config.get("LEGAL_VERSION") or "").strip(),
            )

        if request.method == "POST":
            if request.form.get("ack") != "1":
                flash("Antes de pagar, tenés que confirmar la compra y aceptar la política de reembolso.", "warning")
                return redirect(url_for("buy_combo", combo_id=combo.id))

        cp = ComboPurchase(
            buyer_id=current_user.id,
            combo_id=combo.id,
            status="pending",
            amount_cents=price_cents
        )
        s.add(cp)
        s.commit()

        # Gestión / auditoría (compra de combo iniciada)
        try:
            log_audit_event(
                actor_user_id=current_user.id,
                action="combo_purchase_created",
                target_type="combo_purchase",
                target_id=int(cp.id),
                meta={
                    "combo_purchase_id": int(cp.id),
                    "combo_id": int(combo.id),
                    "combo_title": getattr(combo, "title", None),
                    "buyer_id": int(current_user.id),
                    "buyer_email": getattr(current_user, "email", None),
                    "amount_cents": int(price_cents or 0),
                },
            )
        except Exception:
            pass

        price_ars = float(money_1_decimal(cents_to_amount(price_cents)))
        platform_fee_percent = float(APY_RATE)
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

            # Protect webhook endpoint with a shared secret (query param).
            wh_secret = (app.config.get("MP_WEBHOOK_SECRET") or "").strip()
            notification_url = (
                url_for("mp_webhook", _external=True, secret=wh_secret)
                if wh_secret else url_for("mp_webhook", _external=True)
            )

            pref = mp.create_preference_for_seller_token(
                seller_access_token=seller_token,
                title=f"Combo: {combo.title}",
                unit_price=price_ars,
                quantity=1,
                marketplace_fee=marketplace_fee,
                external_reference=f"combo_purchase:{cp.id}",
                back_urls=back_urls,
                notification_url=notification_url
            )

            with Session() as s2:
                cp2 = s2.get(ComboPurchase, cp.id)
                if cp2:
                    cp2.preference_id = pref.get("id") or pref.get("preference_id")
                    s2.commit()

            init_point = pref.get("init_point") or pref.get("sandbox_init_point")

            # A6 analytics: checkout started (combo)
            try:
                log_analytics_event(
                    event="checkout_start",
                    user_id=int(current_user.id) if getattr(current_user, "is_authenticated", False) else None,
                    path=request.path,
                    combo_id=int(combo.id) if combo else int(combo_id),
                    meta={
                        "combo_purchase_id": int(cp.id),
                        "preference_id": (pref.get("id") or pref.get("preference_id")),
                    },
                )
            except Exception:
                pass
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
        # A6 analytics: combo view
        try:
            uid = int(current_user.id) if getattr(current_user, "is_authenticated", False) else None
            log_analytics_event(event="combo_view", user_id=uid, path=request.path, combo_id=int(combo_id))
        except Exception:
            pass

        if not combo:
            abort(404)

        # is_active puede ser NULL en tu DB -> tratamos NULL como activo
        if hasattr(combo, "is_active") and (combo.is_active is False):
            abort(404)

        is_owner = getattr(current_user, "is_authenticated", False) and current_user.id == combo.seller_id
        is_admin = getattr(current_user, "is_authenticated", False) and getattr(current_user, "is_admin", False)

        # Público: solo approved. Dueño/admin pueden ver pending_review.
        if hasattr(combo, "moderation_status"):
            if not is_public_moderation_status(getattr(combo, "moderation_status", "approved")) and not (is_owner or is_admin):
                abort(404)

        # ARCHIVED_VISIBILITY_COMBO: oculto para nuevos usuarios, pero accesible para quienes ya lo compraron/descargaron.
        if getattr(combo, 'is_archived', False) and not (is_owner or is_admin):
            has_access = False
            try:
                if getattr(current_user, 'is_authenticated', False):
                    if int(getattr(combo, 'price_cents', 0) or 0) <= 0:
                        has_access = s.execute(
                            select(DownloadLog.id).where(DownloadLog.user_id == current_user.id, DownloadLog.combo_id == combo.id)
                        ).scalar_one_or_none() is not None
                    else:
                        has_access = s.execute(
                            select(Purchase.id).where(Purchase.user_id == current_user.id, Purchase.combo_id == combo.id, Purchase.status == 'approved')
                        ).scalar_one_or_none() is not None
            except Exception:
                has_access = False
            if not has_access:
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

        can_download = False
        if getattr(current_user, "is_authenticated", False):
            is_owner2 = bool(current_user.id == combo.seller_id)
            is_admin2 = bool(getattr(current_user, "is_admin", False))
            is_free2 = int(buyer_price_cents or 0) <= 0
            if is_owner2 or is_admin2 or bool(getattr(current_user, "is_premium", False)) or is_free2:
                can_download = True
            else:
                try:
                    has_cp2 = s.execute(
                        select(ComboPurchase.id).where(
                            ComboPurchase.buyer_id == current_user.id,
                            ComboPurchase.combo_id == combo.id,
                            ComboPurchase.status == "approved",
                        )
                    ).scalar_one_or_none() is not None
                    can_download = bool(has_cp2)
                except Exception:
                    can_download = False

    # Analytics: vista de combo
    try:
        log_analytics_event(
            event="page_view",
            user_id=(current_user.id if current_user.is_authenticated else None),
            path=request.path,
            combo_id=int(combo_id),
            meta={"page": "combo_detail", "price_cents": int(getattr(combo, "price_cents", 0) or 0)},
        )
    except Exception:
        pass

    return render_template(
        "combo_detail.html",
        combo=combo,
        seller=seller,
        notes=notes,
        buyer_price_cents=buyer_price_cents,
        buyer_price=buyer_price,
    )

@app.before_request
def maintenance_mode():
    if os.getenv("MAINTENANCE_MODE", "false").lower() != "true":
        return None

    # Dejar health libre (Render)
    if request.path == "/health":
        return None

    # Dejar archivos estáticos libres (imagen, css)
    if request.path.startswith("/static/"):
        return None

    return render_template("maintenance.html"), 503


# =========================
# Error pages (404/403/429/500)
# =========================
from werkzeug.exceptions import HTTPException

@app.errorhandler(404)
def _err_404(e):
    return render_template("errors/404.html"), 404

@app.errorhandler(403)
def _err_403(e):
    return render_template("errors/403.html"), 403

@app.errorhandler(429)
def _err_429(e):
    # Flask-Limiter raises 429
    return render_template("errors/429.html"), 429

@app.errorhandler(500)
def _err_500(e):
    error_id = str(uuid.uuid4())[:8]
    try:
        app.logger.exception("500 error_id=%s path=%s", error_id, request.path)
    except Exception:
        pass
    return render_template("errors/500.html", error_id=error_id), 500


if __name__ == "__main__":
    app.run(debug=True)


# =========================
# UX: upload size + CSRF friendly errors
# =========================
from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash(f"El archivo supera el tamaño máximo permitido ({_max_mb} MB).", "danger")
    return redirect(request.referrer or url_for("upload_note"))

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    # CSRF failures should show a friendly message instead of "no pasa nada"
    flash("Tu sesión expiró o el formulario es inválido. Probá de nuevo.", "danger")
    return redirect(request.referrer or url_for("index"))