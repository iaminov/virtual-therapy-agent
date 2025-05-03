"""Repository layer for data access operations."""

from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from therapeutic_agent.storage.models import (
    ConversationMessage,
    MessageRole,
    SafetyEvent,
    SafetyFlag,
    SessionStatus,
    TherapeuticSession,
)


class SessionRepository:
    """Repository for therapeutic session operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_session(
        self, user_id: str, title: str | None = None
    ) -> TherapeuticSession:
        """Create a new therapeutic session."""
        session = TherapeuticSession(
            user_id=user_id, title=title, last_activity=datetime.now(timezone.utc)
        )
        self._session.add(session)
        await self._session.flush()
        return session

    async def get_session(self, session_id: UUID) -> TherapeuticSession | None:
        """Get a session by ID with related messages."""
        stmt = (
            select(TherapeuticSession)
            .options(selectinload(TherapeuticSession.messages))
            .where(TherapeuticSession.id == session_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_sessions(
        self, user_id: str, status: SessionStatus | None = None, limit: int = 50
    ) -> Sequence[TherapeuticSession]:
        """Get sessions for a user, optionally filtered by status."""
        stmt = select(TherapeuticSession).where(TherapeuticSession.user_id == user_id)

        if status:
            stmt = stmt.where(TherapeuticSession.status == status)

        stmt = stmt.order_by(desc(TherapeuticSession.last_activity)).limit(limit)

        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update_session_activity(self, session_id: UUID) -> None:
        """Update the last activity timestamp for a session."""
        stmt = (
            update(TherapeuticSession)
            .where(TherapeuticSession.id == session_id)
            .values(last_activity=datetime.now(timezone.utc))
        )
        await self._session.execute(stmt)

    async def update_session_status(
        self, session_id: UUID, status: SessionStatus
    ) -> None:
        """Update session status."""
        stmt = (
            update(TherapeuticSession)
            .where(TherapeuticSession.id == session_id)
            .values(status=status)
        )
        await self._session.execute(stmt)

    async def get_active_session_count(self, user_id: str) -> int:
        """Get count of active sessions for a user."""
        stmt = select(func.count(TherapeuticSession.id)).where(
            TherapeuticSession.user_id == user_id,
            TherapeuticSession.status == SessionStatus.ACTIVE,
        )
        result = await self._session.execute(stmt)
        return result.scalar() or 0


class MessageRepository:
    """Repository for conversation message operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_message(
        self,
        session_id: UUID,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
        safety_score: float | None = None,
        token_count: int | None = None,
        processing_time_ms: int | None = None,
    ) -> ConversationMessage:
        """Add a new message to a session."""
        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            message_metadata=metadata,
            safety_score=safety_score,
            token_count=token_count,
            processing_time_ms=processing_time_ms,
        )
        self._session.add(message)

        stmt = (
            update(TherapeuticSession)
            .where(TherapeuticSession.id == session_id)
            .values(
                message_count=TherapeuticSession.message_count + 1,
                last_activity=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(stmt)

        await self._session.flush()
        return message

    async def get_session_messages(
        self, session_id: UUID, limit: int | None = None
    ) -> Sequence[ConversationMessage]:
        """Get messages for a session, optionally limited."""
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at)
        )

        if limit:
            stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)
        return result.scalars().all()


class SafetyRepository:
    """Repository for safety event operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_safety_event(
        self,
        session_id: UUID,
        flag_type: SafetyFlag,
        severity_score: float,
        description: str,
        triggered_by_message_id: UUID | None = None,
        intervention_taken: str | None = None,
    ) -> SafetyEvent:
        """Create a new safety event."""
        event = SafetyEvent(
            session_id=session_id,
            flag_type=flag_type,
            severity_score=severity_score,
            triggered_by_message_id=triggered_by_message_id,
            description=description,
            intervention_taken=intervention_taken,
        )
        self._session.add(event)

        stmt = (
            update(TherapeuticSession)
            .where(TherapeuticSession.id == session_id)
            .values(
                safety_score=func.least(TherapeuticSession.safety_score, severity_score)
            )
        )
        await self._session.execute(stmt)

        await self._session.flush()
        return event

    async def get_unresolved_safety_events(
        self, session_id: UUID
    ) -> Sequence[SafetyEvent]:
        """Get unresolved safety events for a session."""
        stmt = (
            select(SafetyEvent)
            .where(
                SafetyEvent.session_id == session_id, SafetyEvent.resolved.is_(False)
            )
            .order_by(desc(SafetyEvent.severity_score))
        )

        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def resolve_safety_event(self, event_id: UUID) -> None:
        """Mark a safety event as resolved."""
        stmt = (
            update(SafetyEvent).where(SafetyEvent.id == event_id).values(resolved=True)
        )
        await self._session.execute(stmt)
