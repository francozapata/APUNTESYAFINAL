from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from ..app import db
from ..models import Faq

admin_faq_bp = Blueprint("admin_faq", __name__, template_folder="../templates/admin")

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        if not getattr(current_user, "is_admin", False) and not getattr(current_user, "es_admin", False):
            flash("No tenés permiso para acceder.","warning")
            return redirect(url_for("helpcenter.faq_index"))
        return f(*args, **kwargs)
    return wrapper

@admin_faq_bp.route("/admin/faq")
@login_required
@admin_required
def list_faq():
    q = request.args.get("q", "").strip()
    query = Faq.query
    if q:
        like = f"%{q}%"
        query = query.filter(Faq.question.ilike(like))
    faqs = query.order_by(Faq.category.asc(), Faq.position.asc(), Faq.created_at.desc()).all()
    return render_template("admin/faq_list.html", faqs=faqs, q=q)

@admin_faq_bp.route("/admin/faq/new", methods=["GET","POST"])
@login_required
@admin_required
def new_faq():
    if request.method == "POST":
        question = request.form.get("question","").strip()
        answer   = request.form.get("answer","").strip()
        category = request.form.get("category","General").strip() or "General"
        position = int(request.form.get("position","0") or 0)
        is_active = request.form.get("is_active") == "on"

        if not question or not answer:
            flash("Pregunta y respuesta son obligatorias","warning")
            return redirect(url_for("admin_faq.new_faq"))

        faq = Faq(question=question, answer=answer, category=category, position=position, is_active=is_active)
        db.session.add(faq)
        db.session.commit()
        flash("FAQ creada","success")
        return redirect(url_for("admin_faq.list_faq"))
    return render_template("admin/faq_form.html", faq=None)

@admin_faq_bp.route("/admin/faq/<int:faq_id>/edit", methods=["GET","POST"])
@login_required
@admin_required
def edit_faq(faq_id):
    faq = Faq.query.get_or_404(faq_id)
    if request.method == "POST":
        faq.question = request.form.get("question","").strip()
        faq.answer   = request.form.get("answer","").strip()
        faq.category = request.form.get("category","General").strip() or "General"
        faq.position = int(request.form.get("position","0") or 0)
        faq.is_active = request.form.get("is_active") == "on"
        db.session.commit()
        flash("FAQ actualizada","success")
        return redirect(url_for("admin_faq.list_faq"))
    return render_template("admin/faq_form.html", faq=faq)

@admin_faq_bp.route("/admin/faq/<int:faq_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_faq(faq_id):
    faq = Faq.query.get_or_404(faq_id)
    db.session.delete(faq)
    db.session.commit()
    flash("FAQ eliminada","success")
    return redirect(url_for("admin_faq.list_faq"))
