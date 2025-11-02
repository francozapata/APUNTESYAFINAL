import os
import os
import secrets
import math
from datetime import datetime, timedelta
from urllib.parse import urlencode
from sqlalchemy import select, func, and_
from apuntesya2.models import Review

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, abort, jsonify
)
from flask_login import (
    LoginManager, login_user, logout_user, current_user, login_required
)
from sqlalchemy import create_engine, select, or_, and_, func, text
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# modelos
from apuntesya2.models import Base, User, Note, Purchase, University, Faculty, Career, WebhookEvent

# helpers MP
from apuntesya2 import mp
from apuntesya2.models import Base, User, Note, Purchase, University, Faculty, Career, Review

load_dotenv()

# -----------------------------------------------------------------------------
## App
# -----------------------------------------------------------------------------
app = Flask(__name__, instance_relative_config=True)

# --- MP immediate fee estimate available in templates ---
try:
    MP_FEE_IMMEDIATE_TOTAL_PCT = float(app.config.get("MP_FEE_IMMEDIATE_TOTAL_PCT", 7.61))
except Exception:
    MP_FEE_IMMEDIATE_TOTAL_PCT = 7.61

def user_bought_note(s, buyer_id: int, note_id: int) -> bool:
    q = select(Purchase.id).where(
        Purchase.buyer_id == buyer_id,
        Purchase.note_id == note_id,
        Purchase.status == "approved"
    )
    return s.execute(q).first() is not None


@app.context_processor
def fees_ctx():
    def mp_fee_estimate(amount, pct=MP_FEE_IMMEDIATE_TOTAL_PCT):
        try:
            return round(float(amount) * (float(pct) / 100.0), 2)
        except Exception:
            return 0.0
    return dict(MP_FEE_IMMEDIATE_TOTAL_PCT=MP_FEE_IMMEDIATE_TOTAL_PCT, mp_fee_estimate=mp_fee_estimate)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(16))
app.config["ENV"] = os.getenv("FLASK_ENV", "production")

# -----------------------------------------------------------------------------
# Helpers de config
# -----------------------------------------------------------------------------
def _as_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "t", "yes", "y", "on")

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
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

# -----------------------------------------------------------------------------
# DB URL (SQLite por defecto)
# -----------------------------------------------------------------------------
DEFAULT_DB = f"sqlite:///{os.path.join(BASE_DATA, 'apuntesya.db')}"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB)

engine_kwargs = {"pool_pre_ping": True, "future": True}
if DB_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DB_URL, **engine_kwargs)

# -----------------------------------------------------------------------------
# Modelos e inicio de sesión
# -----------------------------------------------------------------------------
# Crear tablas
Base.metadata.create_all(engine)

# Sesión global (¡OJO! no importamos Session arriba para no ensombrecer)
Session = scoped_session(sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))

login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    with Session() as s:
        return s.get(User, int(user_id))

# -----------------------------------------------------------------------------
# Config MP / Comisiones / Mail / Contact
# -----------------------------------------------------------------------------
# Mercado Pago
app.config["MP_PUBLIC_KEY"] = os.getenv("MP_PUBLIC_KEY", "")
app.config["MP_ACCESS_TOKEN"] = os.getenv("MP_ACCESS_TOKEN", "")
app.config["MP_WEBHOOK_SECRET"] = os.getenv("MP_WEBHOOK_SECRET", "")

# BASE_URL para construir enlaces externos (emails, callbacks)
app.config["BASE_URL"] = os.getenv("BASE_URL", "").rstrip("/")

# Comisiones
app.config["PLATFORM_FEE_PERCENT"] = float(os.getenv("MP_PLATFORM_FEE_PERCENT", "5.0"))
app.config["MP_COMMISSION_RATE"] = float(os.getenv("MP_COMMISSION_RATE", "0.0774"))
app.config["APY_COMMISSION_RATE"] = float(os.getenv("APY_COMMISSION_RATE", "0.05"))
app.config["IIBB_ENABLED"] = _as_bool(os.getenv("IIBB_ENABLED", "false"))
app.config["IIBB_RATE"] = float(os.getenv("IIBB_RATE", "0.0"))

MP_COMMISSION_RATE = app.config["MP_COMMISSION_RATE"]
APY_COMMISSION_RATE = app.config["APY_COMMISSION_RATE"]
IIBB_ENABLED = app.config["IIBB_ENABLED"]
IIBB_RATE = app.config["IIBB_RATE"]

# Token plataforma (fallback si el vendedor no vinculó MP)
app.config["MP_ACCESS_TOKEN_PLATFORM"] = os.getenv("MP_ACCESS_TOKEN", "")
app.config["MP_OAUTH_REDIRECT_URL"] = os.getenv("MP_OAUTH_REDIRECT_URL")

# Password reset (si tu blueprint existe)
app.config["SECURITY_PASSWORD_SALT"] = os.getenv("SECURITY_PASSWORD_SALT", "pw-reset")
app.config["PASSWORD_RESET_EXPIRATION"] = int(os.getenv("PASSWORD_RESET_EXPIRATION", "3600"))

# SMTP config (SIEMPRE directo desde ENV)
app.config["ENABLE_SMTP"] = _as_bool(os.getenv("ENABLE_SMTP", "false"))
app.config["MAIL_SERVER"] = os.getenv("MAIL_SERVER", "").strip()
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"] = _as_bool(os.getenv("MAIL_USE_TLS", "true"))
app.config["MAIL_USE_SSL"] = _as_bool(os.getenv("MAIL_USE_SSL", "false"))
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "").strip()
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "").strip()
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_DEFAULT_SENDER", "").strip()

# Contacto
app.config["CONTACT_EMAILS"] = os.getenv("CONTACT_EMAILS", "soporte.apuntesya@gmail.com")
app.config["CONTACT_WHATSAPP"] = os.getenv("CONTACT_WHATSAPP", "+543510000000")
app.config["SUGGESTIONS_URL"] = os.getenv("SUGGESTIONS_URL",
    "https://docs.google.com/forms/d/e/1FAIpQLScDEukn0sLtjOoWgmvTNaF_qG0iDHue9EOqCYxz_z6bGxzErg/viewform?usp=header"
)

# -----------------------------------------------------------------------------
# Debug de SMTP al iniciar (imprime en logs de Render)
# -----------------------------------------------------------------------------
def _debug_log_smtp_config(app):
    def mask(v):
        if not v:
            return "<empty>"
        s = str(v)
        return s[:2] + "…" + s[-2:] if len(s) > 6 else "***"
    cfg = app.config
    print("[SMTP DEBUG] ENABLE_SMTP =", cfg.get("ENABLE_SMTP"))
    print("[SMTP DEBUG] MAIL_SERVER =", cfg.get("MAIL_SERVER"))
    print("[SMTP DEBUG] MAIL_PORT   =", cfg.get("MAIL_PORT"))
    print("[SMTP DEBUG] MAIL_USE_TLS=", cfg.get("MAIL_USE_TLS"))
    print("[SMTP DEBUG] MAIL_USE_SSL=", cfg.get("MAIL_USE_SSL"))
    print("[SMTP DEBUG] MAIL_USERNAME =", cfg.get("MAIL_USERNAME"))
    print("[SMTP DEBUG] MAIL_PASSWORD =", mask(cfg.get("MAIL_PASSWORD")))
    print("[SMTP DEBUG] MAIL_DEFAULT_SENDER =", cfg.get("MAIL_DEFAULT_SENDER"))
    print("[SMTP DEBUG] BASE_URL =", cfg.get("BASE_URL"))

_debug_log_smtp_config(app)

@app.route("/__debug/smtp")
def __debug_smtp():
    from flask import jsonify
    cfg = app.config
    def has(v): return bool(v and str(v).strip())
    return jsonify({
        "ENABLE_SMTP": cfg.get("ENABLE_SMTP"),
        "MAIL_SERVER": cfg.get("MAIL_SERVER"),
        "MAIL_PORT": cfg.get("MAIL_PORT"),
        "MAIL_USE_TLS": cfg.get("MAIL_USE_TLS"),
        "MAIL_USE_SSL": cfg.get("MAIL_USE_SSL"),
        "MAIL_USERNAME": cfg.get("MAIL_USERNAME"),
        "MAIL_PASSWORD_present": has(cfg.get("MAIL_PASSWORD")),
        "MAIL_DEFAULT_SENDER": cfg.get("MAIL_DEFAULT_SENDER"),
        "BASE_URL": cfg.get("BASE_URL"),
    })

@app.context_processor
def inject_contacts():
    emails = [e.strip() for e in str(app.config.get("CONTACT_EMAILS","")).split(",") if e.strip()]
    return dict(CONTACT_EMAILS=emails,
                CONTACT_WHATSAPP=app.config.get("CONTACT_WHATSAPP"),
                SUGGESTIONS_URL=app.config.get("SUGGESTIONS_URL"))

def get_valid_seller_token(seller: User) -> str | None:
    return seller.mp_access_token if (seller and seller.mp_access_token) else None

# -----------------------------------------------------------------------------
# Admin blueprint (si existe) + auth_reset
# -----------------------------------------------------------------------------
try:
    from .admin.routes import admin_bp
except Exception:
    try:
        from admin.routes import admin_bp
    except Exception:
        admin_bp = None

from apuntesya2.auth_reset.routes import bp as auth_reset_bp

if admin_bp:
    app.register_blueprint(admin_bp)
app.register_blueprint(auth_reset_bp)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def allowed_pdf(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"

def ensure_dirs():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

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
@app.route("/")
def index():
    with Session() as s:
        notes = s.execute(
            select(Note).where(Note.is_active == True).order_by(Note.created_at.desc()).limit(30)
        ).scalars().all()
    return render_template("index.html", notes=notes)

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    university = request.args.get("university", "").strip()
    faculty = request.args.get("faculty", "").strip()
    career = request.args.get("career", "").strip()
    t = request.args.get("type", "")

    with Session() as s:
        stmt = select(Note).where(Note.is_active == True)
        if q:
            stmt = stmt.where(or_(Note.title.ilike(f"%{q}%"), Note.description.ilike(f"%{q}%")))
        if university:
            stmt = stmt.where(Note.university.ilike(f"%{university}%"))
        if faculty:
            stmt = stmt.where(Note.faculty.ilike(f"%{faculty}%"))
        if career:
            stmt = stmt.where(Note.career.ilike(f"%{career}%"))
        if t == "free":
            stmt = stmt.where(Note.price_cents == 0)
        elif t == "paid":
            stmt = stmt.where(Note.price_cents > 0)
        notes = s.execute(stmt.order_by(Note.created_at.desc()).limit(100)).scalars().all()
    return render_template("index.html", notes=notes)

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        university = request.form["university"].strip()
        faculty = request.form["faculty"].strip()
        career = request.form["career"].strip()
        with Session() as s:
            exists = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if exists:
                flash("Ese email ya está registrado.")
                return redirect(url_for("register"))
            u = User(
                name=name, email=email, password_hash=generate_password_hash(password),
                university=university, faculty=faculty, career=career
            )
            s.add(u)
            s.commit()
            login_user(u)
            return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        with Session() as s:
            u = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
            if not u or not check_password_hash(u.password_hash, password):
                flash("Credenciales inválidas.")
                return redirect(url_for("login"))
            login_user(u)
            return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

# -----------------------------------------------------------------------------
# Perfil
# -----------------------------------------------------------------------------
@app.route("/profile")
@login_required
def profile():
    with Session() as s:
        my_notes = s.execute(
            select(Note).where(Note.seller_id == current_user.id).order_by(Note.created_at.desc())
        ).scalars().all()
    return render_template("profile.html", my_notes=my_notes)

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
        end = datetime.strptime(end_str, fmt) + timedelta(days=1)  # inclusivo
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
        gross_cents = int(totals[1] or 0)

        mp_commission_cents  = int(round(gross_cents * float(MP_COMMISSION_RATE)))
        apy_commission_cents = int(round(gross_cents * float(APY_COMMISSION_RATE)))
        net_cents = gross_cents - mp_commission_cents - apy_commission_cents

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
                "gross_cents": gross,
                "mp_commission_cents": mp_c,
                "apy_commission_cents": apy_c,
                "net_cents": gross - mp_c - apy_c,
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
            .where(Purchase.buyer_id == current_user.id, Purchase.status == 'approved')
            .order_by(Purchase.created_at.desc())
        ).all()

        items = []
        for p, n in purchases:
            items.append(dict(
                id=p.id,
                note_id=n.id,
                title=n.title,
                price_cents=p.amount_cents,
                created_at=p.created_at.strftime("%Y-%m-%d %H:%M")
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
        price_cents = int(round(float(price) * 100)) if price else 0

        file = request.files.get("file")
        if not file or file.filename == "":
            flash("Seleccioná un PDF.")
            return redirect(url_for("upload_note"))
        if not allowed_pdf(file.filename):
            flash("Sólo PDF.")
            return redirect(url_for("upload_note"))

        ensure_dirs()
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secure_filename(file.filename)}"
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(fpath)

        with Session() as s:
            note = Note(
                title=title, description=description, university=university, faculty=faculty, career=career,
                price_cents=price_cents, file_path=filename, seller_id=current_user.id
            )
            s.add(note)
            s.commit()
        flash("Apunte subido correctamente.")
        return redirect(url_for("note_detail", note_id=note.id))
    return render_template("upload.html")

@app.route("/note/<int:note_id>")
def note_detail(note_id):
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
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

        # Reseñas + promedio
        rows = s.execute(
            select(Review, User.name)
            .join(User, User.id == Review.buyer_id)
            .where(Review.note_id == note.id)
            .order_by(Review.created_at.desc())
        ).all()
        reviews = rows  # lista de tuplas (Review, nombre)

        if reviews:
            avg_rating = round(sum(r.rating for r, _ in reviews) / len(reviews), 2)
        else:
            avg_rating = None

        # ¿Puede calificar?
        can_review = False
        already_reviewed = False
        if current_user.is_authenticated and current_user.id != note.seller_id:
            # Si es pago: requiere compra aprobada; si es gratis: alcanza con estar logueado
            if note.price_cents > 0:
                has_purchase = s.execute(
                    select(Purchase).where(
                        Purchase.buyer_id == current_user.id,
                        Purchase.note_id == note.id,
                        Purchase.status == 'approved'
                    )
                ).scalar_one_or_none() is not None
            else:
                has_purchase = True  # gratis: permitimos reseñar

            if has_purchase:
                already_reviewed = s.execute(
                    select(Review).where(
                        Review.note_id == note.id,
                        Review.buyer_id == current_user.id
                    )
                ).scalar_one_or_none() is not None
                can_review = not already_reviewed

    return render_template(
        "note_detail.html",
        note=note,
        can_download=can_download,
        reviews=reviews,
        avg_rating=avg_rating,
        can_review=can_review,
        already_reviewed=already_reviewed
    )

@app.post("/note/<int:note_id>/review")
@login_required
def submit_review(note_id):
    rating = int(request.form.get("rating", "0") or 0)
    comment = (request.form.get("comment") or "").strip()

    if rating < 1 or rating > 5:
        flash("La puntuación debe estar entre 1 y 5.")
        return redirect(url_for("note_detail", note_id=note_id))

    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)

        # No puede reseñar su propio apunte
        if note.seller_id == current_user.id:
            flash("No podés calificar tu propio apunte.")
            return redirect(url_for("note_detail", note_id=note_id))

        # Elegibilidad (mismo criterio que en note_detail)
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

        if not has_purchase:
            flash("Necesitás haber comprado este apunte para calificarlo.")
            return redirect(url_for("note_detail", note_id=note_id))

        # Evitar duplicado
        exists = s.execute(
            select(Review).where(
                Review.note_id == note.id,
                Review.buyer_id == current_user.id
            )
        ).scalar_one_or_none()
        if exists:
            flash("Ya enviaste una reseña para este apunte.")
            return redirect(url_for("note_detail", note_id=note_id))

        r = Review(note_id=note.id, buyer_id=current_user.id, rating=rating, comment=comment)
        s.add(r)
        s.commit()

    flash("¡Gracias por tu reseña!")
    return redirect(url_for("note_detail", note_id=note_id))

@app.route("/download/<int:note_id>")
@login_required
def download_note(note_id):
    with Session() as s:
        note = s.get(Note, note_id)
        if not note or not note.is_active:
            abort(404)

        allowed = False
        if note.seller_id == current_user.id or note.price_cents == 0:
            allowed = True
        else:
            p = s.execute(
                select(Purchase).where(
                    Purchase.buyer_id == current_user.id,
                    Purchase.note_id == note.id,
                    Purchase.status == 'approved'
                )
            ).scalar_one_or_none()
            allowed = p is not None

        if not allowed:
            flash("Necesitás comprar este apunte para descargarlo.")
            return redirect(url_for("note_detail", note_id=note.id))

        return send_from_directory(app.config["UPLOAD_FOLDER"], note.file_path, as_attachment=True)

# -----------------------------------------------------------------------------
# MP OAuth
# -----------------------------------------------------------------------------
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
        if note.seller_id == current_user.id:
            flash("No podés comprar tu propio apunte.")
            return redirect(url_for("note_detail", note_id=note.id))
        if note.price_cents == 0:
            flash("Este apunte es gratuito.")
            return redirect(url_for("download_note", note_id=note.id))

        seller = s.get(User, note.seller_id)
        p = Purchase(buyer_id=current_user.id, note_id=note.id, status="pending", amount_cents=note.price_cents)
        s.add(p)
        s.commit()

        price_ars = round(note.price_cents / 100, 2)
        platform_fee_percent = (app.config["PLATFORM_FEE_PERCENT"] / 100.0)
        back_urls = {
            "success": url_for("mp_return", note_id=note.id, _external=True) + f"?external_reference=purchase:{p.id}",
            "failure": url_for("mp_return", note_id=note.id, _external=True) + f"?external_reference=purchase:{p.id}",
            "pending": url_for("mp_return", note_id=note.id, _external=True) + f"?external_reference=purchase:{p.id}",
        }

        try:
            seller_token = get_valid_seller_token(seller)
            if seller_token is None:
                use_token = app.config["MP_ACCESS_TOKEN_PLATFORM"]
                marketplace_fee = 0.0
                flash("El vendedor no tiene Mercado Pago vinculado. Se procesa con token de la plataforma y sin comisión.", "info")
            else:
                use_token = seller_token
                marketplace_fee = round(price_ars * platform_fee_percent, 2)

            pref = mp.create_preference_for_seller_token(
                seller_access_token=use_token,
                title=note.title,
                unit_price=price_ars,
                quantity=1,
                marketplace_fee=marketplace_fee,
                external_reference=f"purchase:{p.id}",
                back_urls=back_urls,
                notification_url=url_for("mp_webhook", _external=True)  # apunta al endpoint nuevo
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
    payment_id = request.args.get("payment_id") or request.args.get("collection_id") or request.args.get("id")
    ext_ref = request.args.get("external_reference", "")
    pref_id = request.args.get("preference_id", "")

    token = app.config["MP_ACCESS_TOKEN_PLATFORM"]
    pay = None

    if payment_id:
        try:
            pay = mp.get_payment(token, str(payment_id))
        except Exception as e:
            flash(f"No se pudo verificar el pago aún: {e}")
            return redirect(url_for("note_detail", note_id=note_id))
    elif ext_ref:
        try:
            res = mp.search_payments_by_external_reference(token, ext_ref)
            results = (res or {}).get("results") or []
            if results:
                pay = results[0].get("payment") or results[0]
                payment_id = str(pay.get("id")) if pay else None
        except Exception:
            pass

    if not pay:
        with Session() as s:
            p_last = s.execute(
                select(Purchase).where(Purchase.note_id == note_id).order_by(Purchase.created_at.desc())
            ).scalars().first()
            if p_last:
                try:
                    res = mp.search_payments_by_external_reference(token, f"purchase:{p_last.id}")
                    results = (res or {}).get("results") or []
                    if results:
                        pay = results[0].get("payment") or results[0]
                        payment_id = str(pay.get("id")) if pay else None
                        ext_ref = f"purchase:{p_last.id}"
                except Exception:
                    pass

    status = (pay or {}).get("status")
    external_reference = (pay or {}).get("external_reference") or ext_ref or ""
    purchase_id = None
    if external_reference and external_reference.startswith("purchase:"):
        try:
            purchase_id = int(external_reference.split(":")[1])
        except Exception:
            purchase_id = None

    with Session() as s:
        if purchase_id:
            p = s.get(Purchase, purchase_id)
        else:
            p = s.execute(
                select(Purchase).where(Purchase.note_id == note_id).order_by(Purchase.created_at.desc())
            ).scalars().first()

        if p:
            p.payment_id = str((pay or {}).get("id") or "")
            if status:
                p.status = status
            s.commit()

        if status == "approved":
            flash("¡Pago verificado! Descargando el apunte...")
            return redirect(url_for("download_note", note_id=note_id))

    flash("Pago registrado. Si ya figura aprobado, el botón de descarga estará disponible.")
    return redirect(url_for("note_detail", note_id=note_id))

# -----------------------------------------------------------------------------
# Webhook único (dos rutas válidas -> mismo handler, endpoints distintos)
# -----------------------------------------------------------------------------
def _upsert_purchase_from_payment(pay: dict):
    """Actualiza la compra si el payment trae external_reference=purchase:<id>"""
    try:
        status = (pay or {}).get("status")
        external_reference = (pay or {}).get("external_reference") or ""
        if external_reference.startswith("purchase:"):
            pid = int(external_reference.split(":")[1])
            with Session() as s:
                p = s.get(Purchase, pid)
                if p:
                    p.payment_id = str(pay.get("id") or "")
                    if status:
                        p.status = status
                    s.commit()
    except Exception:
        pass

def mp_webhook():
    """Webhook idempotente. Acepta POST (y GET legacy). Registra evento + actualiza Purchase si corresponde."""
    if request.method == "GET":
        # ping o verificación legacy
        return ("ok", 200)

    try:
        # Secret opcional (solo aplicable a la ruta nueva; si lo configuran, exigirlo en ambas)
        configured_secret = (app.config.get("MP_WEBHOOK_SECRET") or "").strip()
        incoming_secret = (request.args.get("secret") or "").strip()
        if configured_secret and configured_secret != incoming_secret:
            return {"ok": False, "error": "unauthorized"}, 401

        payload = request.get_json(silent=True) or {}
        topic = payload.get("type") or payload.get("topic")
        action = payload.get("action") or (payload.get("data", {}) or {}).get("action")

        # idempotency key
        provider_id = str(
            payload.get("id")
            or (payload.get("data") or {}).get("id")
            or request.headers.get("X-Idempotency-Key")
            or ""
        ).strip()
        if not provider_id:
            provider_id = "no-id-" + str(abs(hash(request.data)))

        # Guardar evento si no existe
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

        # Si viene payment id, intentar actualizar compra
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

# Registramos UNA sola vez, endpoints diferentes
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
    return redirect(url_for("note_detail", note_id=note_id))

# -----------------------------------------------------------------------------
# Taxonomías académicas (dropdowns que aprenden)
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
def api_add_university():
    data = request.get_json(silent=True) or {}
    name = _norm(data.get("name"))
    if not name:
        return jsonify({"error": "name required"}), 400
    with Session() as s:
        u = s.execute(select(University).where(func.lower(University.name) == name.lower())).scalar_one_or_none()
        if u:
            return jsonify({"id": u.id, "name": u.name})
        u = University(name=name)
        s.add(u)
        s.commit()
        return jsonify({"id": u.id, "name": u.name})

@app.post("/api/academics/faculties")
def api_add_faculty():
    data = request.get_json(silent=True) or {}
    name = _norm(data.get("name"))
    uid = data.get("university_id")
    if not (name and uid):
        return jsonify({"error": "name and university_id required"}), 400
    with Session() as s:
        f = s.execute(select(Faculty).where(
            func.lower(Faculty.name) == name.lower(),
            Faculty.university_id == uid
        )).scalar_one_or_none()
        if f:
            return jsonify({"id": f.id, "name": f.name, "university_id": f.university_id})
        f = Faculty(name=name, university_id=uid)
        s.add(f)
        s.commit()
        return jsonify({"id": f.id, "name": f.name, "university_id": f.university_id})

@app.post("/api/academics/careers")
def api_add_career():
    data = request.get_json(silent=True) or {}
    name = _norm(data.get("name"))
    fid = data.get("faculty_id")
    if not (name and fid):
        return jsonify({"error": "name and faculty_id required"}), 400
    with Session() as s:
        c = s.execute(select(Career).where(
            func.lower(Career.name) == name.lower(),
            Career.faculty_id == fid
        )).scalar_one_or_none()
        if c:
            return jsonify({"id": c.id, "name": c.name, "faculty_id": c.faculty_id})
        c = Career(name=name, faculty_id=fid)
        s.add(c)
        s.commit()
        return jsonify({"id": c.id, "name": c.name, "faculty_id": c.faculty_id})

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
# Cambio de contraseña (POST /profile/change_password)
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
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
