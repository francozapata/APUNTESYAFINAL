from flask import Blueprint, render_template, request, abort
from sqlalchemy import or_
from ..app import db
from ..models import Faq

helpcenter_bp = Blueprint("helpcenter", __name__, template_folder="../templates/help")

@helpcenter_bp.route("/ayuda")
@helpcenter_bp.route("/faq")
def faq_index():
    q = request.args.get("q", "", type=str).strip()
    cat = request.args.get("cat", "", type=str).strip()

    query = Faq.query.filter_by(is_active=True)
    if cat:
        query = query.filter(Faq.category == cat)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Faq.question.ilike(like), Faq.answer.ilike(like)))

    faqs = query.order_by(Faq.category.asc(), Faq.position.asc(), Faq.created_at.desc()).all()
    categories = [c[0] for c in db.session.query(Faq.category).filter_by(is_active=True).distinct().all()]

    jsonld = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": f.question,
                "acceptedAnswer": {"@type": "Answer", "text": f.answer}
            } for f in faqs[:30]
        ]
    }
    return render_template("help/faq_index.html", faqs=faqs, categories=categories, q=q, cat=cat, jsonld=jsonld)

@helpcenter_bp.route("/faq/<int:faq_id>-<slug>")
def faq_detail(faq_id, slug):
    faq = Faq.query.filter_by(id=faq_id, is_active=True).first()
    if not faq:
        return abort(404)
    faq.view_count = faq.view_count + 1
    db.session.commit()
    return render_template("help/faq_detail.html", faq=faq)

@helpcenter_bp.route("/api/faq")
def faq_api():
    q = request.args.get("q","").strip()
    cat = request.args.get("cat","").strip()
    query = Faq.query.filter_by(is_active=True)
    if cat: query = query.filter(Faq.category==cat)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Faq.question.ilike(like), Faq.answer.ilike(like)))
    data = [{
        "id": f.id, "question": f.question, "answer": f.answer,
        "category": f.category, "position": f.position,
        "url": f"/faq/{f.id}-{f.slug}"
    } for f in query.order_by(Faq.position.asc()).all()]
    return {"items": data}
