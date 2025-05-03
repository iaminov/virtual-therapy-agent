"""SQLAlchemy database models for therapeutic sessions."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base model with common fields."""

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SessionStatus(str, Enum):
    """Therapeutic session status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SafetyFlag(str, Enum):
    """Safety concern flag types."""

    CRISIS = "crisis"
    SELF_HARM = "self_harm"
    SUBSTANCE_ABUSE = "substance_abuse"
    INAPPROPRIATE_REQUEST = "inappropriate_request"
    MEDICAL_ADVICE = "medical_advice"


class TherapeuticSession(Base):
    """Core therapeutic session model."""

    __tablename__ = "therapeutic_sessions"

    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[SessionStatus] = mapped_column(
        String(50), default=SessionStatus.ACTIVE, index=True
    )
    title: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)
    session_notes: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    safety_score: Mapped[float] = mapped_column(Float, default=1.0)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )
    safety_events: Mapped[list["SafetyEvent"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ConversationMessage(Base):
    """Individual conversation message within a session."""

    __tablename__ = "conversation_messages"

    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("therapeutic_sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text)
    # Renamed to avoid conflict with SQLAlchemy base class
    message_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, name="metadata"
    )  # Column name in DB stays as 'metadata'
    safety_score: Mapped[float | None] = mapped_column(Float)
    token_count: Mapped[int | None] = mapped_column(Integer)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)

    session: Mapped[TherapeuticSession] = relationship(back_populates="messages")


class SafetyEvent(Base):
    """Safety-related events and interventions."""

    __tablename__ = "safety_events"

    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("therapeutic_sessions.id", ondelete="CASCADE"), index=True
    )
    flag_type: Mapped[SafetyFlag] = mapped_column(String(50), index=True)
    severity_score: Mapped[float] = mapped_column(Float)
    triggered_by_message_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("conversation_messages.id", ondelete="SET NULL")
    )
    description: Mapped[str] = mapped_column(Text)
    intervention_taken: Mapped[str | None] = mapped_column(Text)
    resolved: Mapped[bool] = mapped_column(default=False)

    session: Mapped[TherapeuticSession] = relationship(back_populates="safety_events")
