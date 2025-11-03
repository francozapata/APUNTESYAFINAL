import os
import secrets
from datetime import datetime, timedelta
from urllib.parse import urlencode
from flask import session
from firebase_admin import auth as fb_auth, credentials, initialize_app as fb_init
import os, secrets


from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, abort, jsonify, session
)
from flask_login import (
    LoginManager, login_user, logout_user, current_user, login_required
)
from sqlalchemy import create_engine, select, or_, and_, func, text
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------
from apuntesya2.models import Base, User, Note, Purchase, University, Faculty, Career, WebhookEvent, Review

# ---------------------------------------------------------------------
# Firebase Auth (nuevo)
# ---------------------------------------------------------------------
from apuntesya2.firebase_auth import verify_id_token, init_firebase_admin

# ---------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------
load_dotenv()
app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(16))
app.config["ENV"] = os.getenv("FLASK_ENV", "production")

# Firebase Admin (usar SERVICE_ACCOUNT_JSON o cred predeterminada)
if not getattr(app, "_fb_admin_inited", False):
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if cred_path and os.path.exists(cred_path):
        fb_init(credentials.Certificate(cred_path))
    else:
        # Para Render: cred predeterminada si la cuenta de servicio está en env
        try:
            fb_init()
        except Exception:
            pass
    app._fb_admin_inited = True
 

# Firebase Admin (usar SERVICE_ACCOUNT_JSON o cred predeterminada)
if not getattr(app, "_fb_admin_inited", False):
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if cred_path and os.path.exists(cred_path):
        fb_init(credentials.Certificate(cred_path))
    else:
        # Para Render: cred predeterminada si la cuenta de servicio está en env
        try:
            fb_init()
        except Exception:
            pass
    app._fb_admin_inited = True

# ---------------------------------------------------------------------
# Archivos
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Base de datos
# ---------------------------------------------------------------------
DEFAULT_DB = f"sqlite:///{os.path.join(BASE_DATA, 'apuntesya.db')}"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB)
engine_kwargs = {"pool_pre_ping": True, "future": True}
if DB_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DB_URL, **engine_kwargs)
Base.metadata.create_all(engine)

Session = scoped_session(sessionmaker(bind=engine, autoflush=False, expire_on_commit=False))

# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    with Session() as s:
        return s.get(User, int(user_id))

# ---------------------------------------------------------------------
# Inicializar Firebase Admin
# ---------------------------------------------------------------------
try:
    init_firebase_admin()
except Exception as e:
    print("[Firebase] No se pudo inicializar:", e)

# ---------------------------------------------------------------------
# Rutas principales
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Login con Google
# ---------------------------------------------------------------------
@app.route("/login", methods=["GET"])
def login():
    return render_template("login_google.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    # Registro solo vía Google
    return redirect(url_for("login"))

@app.post("/auth/session_login")
def auth_session_login():
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
                # Usuario existente -> loguear y a inicio
                login_user(u)
                return {"ok": True, "next": url_for("index")}, 200

        # No existe -> guardar datos básicos en la sesión y pedir perfil
        session["pending_google"] = {
            "email": email,
            "name": name,
            "uid": info.get("uid"),
            "picture": info.get("picture")
        }
        return {"ok": True, "next": url_for("complete_profile")}, 200

    except Exception as e:
        # Log interno y devolver 500 con texto legible
        app.logger.exception("session_login error")
        return {"ok": False, "error": str(e)}, 500

@app.get("/complete_profile")
def complete_profile():
    if "pending_google" not in session:
        return redirect(url_for("login"))
    data = session["pending_google"]
    # Podés pasar el nombre para saludar
    return render_template("complete_profile.html", name=data.get("name"))

@app.post("/complete_profile")
def complete_profile_post():
    if "pending_google" not in session:
        return redirect(url_for("login"))

    university = (request.form.get("university") or "").strip()
    faculty    = (request.form.get("faculty") or "").strip()
    career     = (request.form.get("career") or "").strip()

    if not (university and faculty and career):
        flash("Completá Universidad, Facultad y Carrera.")
        return redirect(url_for("complete_profile"))

    data = session["pending_google"]
    email = data.get("email")
    name  = data.get("name")

    with Session() as s:
        exists = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if exists:
            # Raro: si llegó acá y ya existe, solo logueamos
            login_user(exists)
            session.pop("pending_google", None)
            return redirect(url_for("index"))

        # Crear usuario completo (password opcional/random)
        u = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(secrets.token_urlsafe(16)),  # placeholder
            university=university,
            faculty=faculty,
            career=career,
            is_active=True,
        )
        s.add(u)
        s.commit()

        login_user(u)
        session.pop("pending_google", None)

    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

# ---------------------------------------------------------------------
# Perfil y notas (mantenemos las rutas antiguas)
# ---------------------------------------------------------------------
@app.route("/profile")
@login_required
def profile():
    with Session() as s:
        my_notes = s.execute(
            select(Note).where(Note.seller_id == current_user.id).order_by(Note.created_at.desc())
        ).scalars().all()
    return render_template("profile.html", my_notes=my_notes)

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
        if not file.filename.lower().endswith(".pdf"):
            flash("Solo se permiten archivos PDF.")
            return redirect(url_for("upload_note"))

        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{secure_filename(file.filename)}"
        fpath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(fpath)

        with Session() as s:
            note = Note(
                title=title, description=description, university=university,
                faculty=faculty, career=career, price_cents=price_cents,
                file_path=filename, seller_id=current_user.id
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
    return render_template("note_detail.html", note=note)



# --- API académica (dropdowns) ---
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
# ---------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True}, 200


# ---------------------------------------------------------------------
# Términos y condiciones
# ---------------------------------------------------------------------
@app.route("/terms")
def terms():
    return render_template("terms.html")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

