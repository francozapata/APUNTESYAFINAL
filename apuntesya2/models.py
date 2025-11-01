from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from flask_login import UserMixin
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base
from sqlalchemy import Integer, String, DateTime, Text, ForeignKey, Boolean

Base = declarative_base()

class User(Base, UserMixin):
    __tablename__ = "users"
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

    notes = relationship("Note", back_populates="seller")

class Note(Base):
    __tablename__ = "notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    university: Mapped[str] = mapped_column(String(120), nullable=False)
    faculty: Mapped[str] = mapped_column(String(120), nullable=False)
    career: Mapped[str] = mapped_column(String(120), nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_reported: Mapped[bool] = mapped_column(Boolean, default=False)

    seller_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    seller = relationship("User", back_populates="notes")

    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g., deactivate_user, soft_delete_note
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)  # 'user' | 'note'
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
