"""SQLAlchemy models.

The app runs on both SQLite (local) and Postgres (Supabase). We therefore use
SQLAlchemy's generic JSON type instead of dialect-specific ones.
"""

from datetime import datetime

from flask_login import UserMixin
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    JSON,
    Text,
    ForeignKey,
    Boolean,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base


Base = declarative_base()


class User(Base, UserMixin):
    __tablename__ = "users"

    phone = Column(String, unique=True, index=True)
    phone_verified = Column(Boolean, default=False)
    phone_verified_at = Column(DateTime)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    imagen_de_perfil: Mapped[str] = mapped_column(String(255), nullable=True)

    university: Mapped[str] = mapped_column(String(120), nullable=False)
    faculty: Mapped[str] = mapped_column(String(120), nullable=False)
    career: Mapped[str] = mapped_column(String(120), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Mercado Pago OAuth
    mp_user_id: Mapped[str] = mapped_column(String(64), nullable=True)
    mp_access_token: Mapped[str] = mapped_column(Text, nullable=True)
    mp_refresh_token: Mapped[str] = mapped_column(Text, nullable=True)
    mp_token_expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Legacy single-field contact (kept for backward compatibility)
    seller_contact: Mapped[str] = mapped_column(String(255), nullable=True)

    # Structured contact fields
    contact_email: Mapped[str] = mapped_column(String(255), nullable=True)
    contact_whatsapp: Mapped[str] = mapped_column(String(64), nullable=True)
    contact_instagram: Mapped[str] = mapped_column(String(80), nullable=True)

    # Visibility controls
    contact_visible_public: Mapped[bool] = mapped_column(Boolean, default=True)
    contact_visible_buyers: Mapped[bool] = mapped_column(Boolean, default=True)

    notes = relationship("Note", back_populates="seller")


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    university: Mapped[str] = mapped_column(String(120), nullable=False)
    faculty: Mapped[str] = mapped_column(String(120), nullable=False)
    career: Mapped[str] = mapped_column(String(120), nullable=False)

    # IMPORTANT: price_cents stores the seller NET price (what the seller wants to receive).
    price_cents: Mapped[int] = mapped_column(Integer, default=0)

    # ✅ En Supabase existe notes.seller_net_cents. Lo agregamos para que el modelo y la BD coincidan.
    # Si hoy no lo usás, no molesta; queda en 0.
    seller_net_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Moderation (AI + manual)
    moderation_status: Mapped[str] = mapped_column(String(32), default="pending_ai")
    moderation_reason: Mapped[str] = mapped_column(Text, nullable=True)

    # AI decision payload
    ai_decision: Mapped[str] = mapped_column(String(16), nullable=True)  # approve|review|deny
    ai_confidence: Mapped[int] = mapped_column(Integer, nullable=True)  # 0..1000
    ai_score_quality: Mapped[int] = mapped_column(Integer, nullable=True)  # 0..1000
    ai_score_copyright: Mapped[int] = mapped_column(Integer, nullable=True)  # 0..1000
    ai_score_mismatch: Mapped[int] = mapped_column(Integer, nullable=True)  # 0..1000
    ai_model: Mapped[str] = mapped_column(String(80), nullable=True)
    ai_summary: Mapped[str] = mapped_column(Text, nullable=True)
    ai_raw: Mapped[dict] = mapped_column(JSON, nullable=True)

    manual_review_due_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    moderated_by_admin_id: Mapped[int] = mapped_column(Integer, nullable=True)
    moderated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Preview (generated on upload)
    preview_pages: Mapped[dict] = mapped_column(JSON, nullable=True)
    preview_images: Mapped[dict] = mapped_column(JSON, nullable=True)

    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_reported: Mapped[bool] = mapped_column(Boolean, default=False)

    seller_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    seller = relationship("User", back_populates="notes")

    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Combo(Base):
    __tablename__ = "combos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seller_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # price_cents = precio final comprador (según tu migración de Supabase)
    price_cents: Mapped[int] = mapped_column(Integer, default=0)

    # ✅ EN SUPABASE ES NOT NULL. Este era el faltante que te rompía.
    seller_net_cents: Mapped[int] = mapped_column(Integer, default=0)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Moderation fields
    moderation_status: Mapped[str] = mapped_column(String(32), default="pending_ai")
    moderation_reason: Mapped[str] = mapped_column(Text, nullable=True)

    ai_decision: Mapped[str] = mapped_column(String(16), nullable=True)
    ai_confidence: Mapped[int] = mapped_column(Integer, nullable=True)  # 0..1000
    ai_model: Mapped[str] = mapped_column(String(80), nullable=True)
    ai_summary: Mapped[str] = mapped_column(Text, nullable=True)
    ai_raw: Mapped[dict] = mapped_column(JSON, nullable=True)

    manual_review_due_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    moderated_by_admin_id: Mapped[int] = mapped_column(Integer, nullable=True)
    moderated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ComboNote(Base):
    __tablename__ = "combo_notes"
    combo_id: Mapped[int] = mapped_column(
        ForeignKey("combos.id", ondelete="CASCADE"), primary_key=True
    )
    note_id: Mapped[int] = mapped_column(
        ForeignKey("notes.id", ondelete="CASCADE"), primary_key=True
    )


class Purchase(Base):
    __tablename__ = "purchases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    buyer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), nullable=False)

    payment_id: Mapped[str] = mapped_column(String(64), nullable=True)
    preference_id: Mapped[str] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending, approved, rejected, cancelled
    amount_cents: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AdminAction(Base):
    __tablename__ = "admin_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    admin_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    ip: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# --- Academic taxonomy (auto-learning dropdowns) ---
class University(Base):
    __tablename__ = "universities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, nullable=False, index=True)


class Faculty(Base):
    __tablename__ = "faculties"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    university_id: Mapped[int] = mapped_column(ForeignKey("universities.id"), nullable=False, index=True)


class Career(Base):
    __tablename__ = "careers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id"), nullable=False, index=True)


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id = Column(Integer, primary_key=True)
    provider = Column(String(32), nullable=False, default="mercadopago")
    provider_id = Column(String(128), nullable=False, unique=True)
    topic = Column(String(64), nullable=True)
    action = Column(String(64), nullable=True)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("note_id", "buyer_id", name="uq_review_note_buyer"),
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_review_rating_1_5"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), nullable=False, index=True)
    buyer_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DownloadLog(Base):
    """Tracks downloads for metrics and anti-abuse."""
    __tablename__ = "download_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # ✅ En Supabase: note_id puede ser NULL y existe combo_id.
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id"), nullable=True, index=True)
    combo_id: Mapped[int] = mapped_column(Integer, nullable=True)

    is_free: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class Notification(Base):
    """Simple in-app notification (also usable for email mirroring)."""
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="info")
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=True)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class Otp(Base):
    __tablename__ = "otps"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    channel = Column(String, nullable=False)
    code_hash = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    attempts = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
