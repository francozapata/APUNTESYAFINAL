
from flask import Blueprint, request, jsonify, abort, render_template
from flask_login import login_required, current_user
from datetime import datetime
from ..models import User, Note, AdminAction, Base
from ..app import Session
from sqlalchemy.orm import joinedload
from flask_login import login_required, current_user
from sqlalchemy import select, func, desc
from apuntesya2.models import User, Note, Purchase
from ..app import Session, MP_COMMISSION_RATE, APY_COMMISSION_RATE



admin_bp = Blueprint("admin", __name__, url_prefix="/admin", template_folder="templates")

def _require_admin():
    if not (hasattr(current_user, "is_authenticated") and current_user.is_authenticated and getattr(current_user, "is_admin", False)):
        abort(403)

@admin_bp.route("/")
@login_required
def dashboard():
    _require_admin()
    with Session() as s:
        users_count = s.query(User).count()
        notes_count = s.query(Note).count()
        return render_template("admin/dashboard.html", users_count=users_count, notes_count=notes_count)

@admin_bp.route("/users")
@login_required
def users_list():
    _require_admin()
    with Session() as s:
        users = s.query(User).all()
        return render_template("admin/users.html", users=users)

@admin_bp.route("/users/<int:user_id>/deactivate", methods=["POST"])
@login_required
def deactivate_user(user_id):
    _require_admin()
    reason = request.json.get("reason") if request.is_json else request.form.get("reason")
    with Session() as s:
        u = s.get(User, user_id)
        if not u:
            abort(404)
        u.is_active = False
        s.add(AdminAction(admin_id=current_user.id, action="deactivate_user", target_type="user",
                          target_id=u.id, reason=reason, ip=request.remote_addr))
        s.commit()
    return jsonify(ok=True)

@admin_bp.route("/notes/<int:note_id>/soft-delete", methods=["POST"])
@login_required
def soft_delete_note(note_id):
    _require_admin()
    reason = request.json.get("reason") if request.is_json else request.form.get("reason")
    with Session() as s:
        n = s.get(Note, note_id)
        if not n:
            abort(404)
        n.deleted_at = datetime.utcnow()
        s.add(AdminAction(admin_id=current_user.id, action="soft_delete_note", target_type="note",
                          target_id=n.id, reason=reason, ip=request.remote_addr))
        s.commit()
    return jsonify(ok=True)

@admin_bp.route("/actions")
@login_required
def admin_actions():
    _require_admin()
    with Session() as s:
        actions = s.query(AdminAction).order_by(AdminAction.created_at.desc()).limit(200).all()
        return render_template("admin/actions.html", actions=actions)

@admin_bp.route("/users/archivos")
@login_required
def users_files():
    _require_admin()
    email = request.args.get('email', '').strip()
    reported = request.args.get('reported', '').strip()
    with Session() as s:
        q = s.query(Note).join(User, Note.seller_id == User.id)
        if email:
            q = q.filter(User.email.ilike(f"%{email}%"))
        if reported == '1':
            q = q.filter(getattr(Note, "is_reported", False) == True)  # works even if column missing
        notes = q.order_by(Note.id.desc()).limit(500).all()
        return render_template("admin/users_files.html", notes=notes, email=email, reported=reported)


@admin_bp.route("/users/archivos/<int:note_id>/delete", methods=["POST"])
@login_required
def delete_user_file(note_id):
    _require_admin()
    reason = request.form.get("reason") or "deleted_from_users_files"
    with Session() as s:
        n = s.get(Note, note_id)
        if not n:
            abort(404)
        # If the model has is_reported, optionally enforce it
        if hasattr(n, "is_reported") and not (n.is_reported or request.form.get("force") == "1"):
            # require reported or explicit force
            abort(400)
        n.deleted_at = datetime.utcnow()
        s.add(AdminAction(admin_id=current_user.id, action="soft_delete_note", target_type="note",
                          target_id=n.id, reason=reason, ip=request.remote_addr))
        s.commit()
    # Redirect back to list
    from flask import redirect, url_for
    return redirect(url_for("admin.files_index_admin"))


# --- Admin hard-delete extensions ---


from flask import current_app, url_for, redirect, flash
import os

@admin_bp.route("/files", endpoint="files_index_admin")
@login_required
def files_index_admin():
    """List all uploaded notes/files for admin with options to hard-delete."""
    _require_admin()
    with Session() as s:
        notes = (
            s.query(Note)
             .options(joinedload(Note.seller))          # <= carga seller en la misma query
             .order_by(getattr(Note, "created_at", Note.id).desc())
             .all()
        )
    return render_template("admin/files_list.html", notes=notes)


@admin_bp.route("/delete_file/<int:note_id>", methods=["POST"])
@login_required
def hard_delete_note(note_id):
    """Permanently delete a note and its file from disk and DB. Records AdminAction."""
    _require_admin()
    reason = request.form.get("reason", "admin_hard_delete")
    with Session() as s:
        note = s.get(Note, note_id)
        if not note:
            abort(404)
        # remove file from disk if exists
        upload_dir = current_app.config.get("UPLOAD_FOLDER")
        if upload_dir and note.file_path:
            fpath = os.path.join(upload_dir, note.file_path)
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception as e:
                # log but continue
                print("Failed to remove file:", fpath, e)
        # delete purchases referencing the note (if Purchase model exists)
        try:
            from ..models import Purchase
            purchases = s.query(Purchase).filter(Purchase.note_id==note.id).all()
            for p in purchases:
                s.delete(p)
        except Exception:
            pass
        # record action
        s.add(AdminAction(admin_id=current_user.id, action="hard_delete_note", target_type="note",
                          target_id=note.id, reason=reason, ip=request.remote_addr))
        # finally delete note record
        s.delete(note)
        s.commit()
    flash("El apunte y archivo fueron eliminados permanentemente.", "success")
    return redirect(url_for("admin.files_index_admin"))


@admin_bp.route("/summary")
@login_required
def admin_summary():
    if not getattr(current_user, "is_admin", False):
        abort(403)

    from datetime import datetime, timedelta, date
    from sqlalchemy import func, and_

    start_date = (datetime.utcnow().date() - timedelta(days=29))  # últimos 30 días inclusive

    with Session() as s:
        # KPI generales
        total_users = s.execute(select(func.count(User.id))).scalar_one()
        total_notes = s.execute(select(func.count(Note.id))).scalar_one()
        total_purchases = s.execute(select(func.count(Purchase.id))).scalar_one()
        total_approved  = s.execute(select(func.count(Purchase.id)).where(Purchase.status == "approved")).scalar_one()
        total_pending   = s.execute(select(func.count(Purchase.id)).where(Purchase.status == "pending")).scalar_one()
        total_rejected  = s.execute(select(func.count(Purchase.id)).where(Purchase.status == "rejected")).scalar_one()

        gross_cents = s.execute(
            select(func.coalesce(func.sum(Purchase.amount_cents), 0)).where(Purchase.status == "approved")
        ).scalar_one()

        mp_commission_cents  = int(round(gross_cents * float(MP_COMMISSION_RATE)))
        apy_commission_cents = int(round(gross_cents * float(APY_COMMISSION_RATE)))
        net_cents = gross_cents - mp_commission_cents - apy_commission_cents

        # Top 10 apuntes (por ventas)
        top_notes = s.execute(
    select(
        Note.id, Note.title,
        func.count(Purchase.id).label("sold_count"),
        func.coalesce(func.sum(Purchase.amount_cents), 0).label("gross_cents")
    )
    .join(Purchase, Purchase.note_id == Note.id)
    .where(Purchase.status == "approved")
    .group_by(Note.id, Note.title)
    .order_by(func.count(Purchase.id).desc())   # <- FIX
    .limit(10)
).all()

        # Reportados (si existe columna)
        reported_notes = []
        if hasattr(Note, "is_reported"):
            reported_notes = s.execute(
                select(Note.id, Note.title).where(Note.is_reported == True).order_by(Note.id.desc()).limit(20)
            ).all()

        # Últimas compras
        last_purchases = s.execute(
            select(Purchase.id, Purchase.status, Purchase.amount_cents, Purchase.created_at, Note.title)
            .join(Note, Note.id == Purchase.note_id)
            .order_by(Purchase.created_at.desc())
            .limit(15)
        ).all()

        # ======= Series de 30 días: ventas e ingresos =======
        # Agrupamos por fecha (usando func.date, funciona en SQLite/Postgres)
        rows = s.execute(
            select(
                func.date(Purchase.created_at).label("d"),
                func.count(Purchase.id).label("cnt"),
                func.coalesce(func.sum(Purchase.amount_cents), 0).label("sum_cents"),
            )
            .where(
                and_(
                    Purchase.status == "approved",
                    Purchase.created_at >= datetime.combine(start_date, datetime.min.time())
                )
            )
            .group_by(func.date(Purchase.created_at))
            .order_by(func.date(Purchase.created_at))
        ).all()

    # Normalizamos a una serie continua de 30 días
    by_date = {str(r.d): (int(r.cnt or 0), int(r.sum_cents or 0)) for r in rows}
    labels = []
    sales_series = []
    revenue_series_ars = []
    for i in range(30):
        d = start_date + timedelta(days=i)
        key = d.strftime("%Y-%m-%d")
        labels.append(key)
        cnt, cents = by_date.get(key, (0, 0))
        sales_series.append(cnt)
        revenue_series_ars.append(round(cents / 100.0, 2))

    return render_template(
        "admin_summary.html",
        total_users=total_users,
        total_notes=total_notes,
        total_purchases=total_purchases,
        total_approved=total_approved,
        total_pending=total_pending,
        total_rejected=total_rejected,
        gross_cents=gross_cents,
        mp_commission_cents=mp_commission_cents,
        apy_commission_cents=apy_commission_cents,
        net_cents=net_cents,
        top_notes=top_notes,
        reported_notes=reported_notes,
        last_purchases=last_purchases,
        # series para Chart.js
        labels=labels,
        sales_series=sales_series,
        revenue_series_ars=revenue_series_ars,
    )
