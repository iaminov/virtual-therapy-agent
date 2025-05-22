"""High-level session management orchestrating database, safety, and AI components."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from therapeutic_agent.core.anthropic_client import AnthropicTherapeuticClient
from therapeutic_agent.core.config import get_settings
from therapeutic_agent.core.exceptions import (
    SessionLimitExceededError,
    SessionNotFoundError,
)
from therapeutic_agent.safety.engine import SafetyEngine
from therapeutic_agent.safety.validators import SafetyLevel, SafetyResult
from therapeutic_agent.storage.database import get_database_manager
from therapeutic_agent.storage.models import MessageRole, SafetyFlag, SessionStatus
from therapeutic_agent.storage.repository import (
    MessageRepository,
    SafetyRepository,
    SessionRepository,
)

logger = structlog.get_logger()


class TherapeuticSessionManager:
    """Orchestrates therapeutic sessions with safety validation and AI integration."""

    def __init__(self) -> None:
        self._anthropic_client = AnthropicTherapeuticClient()
        self._safety_engine = SafetyEngine()
        self._settings = get_settings()

    async def create_session(
        self, user_id: str, title: str | None = None
    ) -> dict[str, Any]:
        """Create a new therapeutic session with validation."""
        db_manager = await get_database_manager()

        async with db_manager.get_session() as db_session:
            session_repo = SessionRepository(db_session)

            active_count = await session_repo.get_active_session_count(user_id)
            max_sessions = self._settings.security.max_sessions_per_user
            if active_count >= max_sessions:
                raise SessionLimitExceededError(
                    f"User has reached maximum active sessions ({max_sessions})"
                )

            session = await session_repo.create_session(user_id, title)

            logger.info(
                "Created therapeutic session",
                session_id=str(session.id),
                user_id=user_id,
                title=title,
            )

            return {
                "session_id": str(session.id),
                "user_id": session.user_id,
                "title": session.title,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "message_count": 0,
            }

    async def send_message(
        self, session_id: UUID, user_message: str, user_id: str | None = None
    ) -> dict[str, Any]:
        """Process message with safety validation and generate response."""
        if not user_message.strip():
            raise ValueError("Message content cannot be empty")

        db_manager = await get_database_manager()

        async with db_manager.get_session() as db_session:
            session_repo = SessionRepository(db_session)
            message_repo = MessageRepository(db_session)
            safety_repo = SafetyRepository(db_session)

            session = await session_repo.get_session(session_id)
            if not session:
                raise SessionNotFoundError(f"Session {session_id} not found")

            if user_id and session.user_id != user_id:
                raise SessionNotFoundError("Session not found for this user")

            if session.status != SessionStatus.ACTIVE:
                raise ValueError(f"Session is {session.status}, cannot send messages")

            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in session.messages[-20:]  # Last 20 messages for context
            ]

            safety_result = await self._safety_engine.validate_content(
                user_message,
                context={
                    "session_id": str(session_id),
                    "conversation_history": conversation_history,
                    "session_metadata": {
                        "message_count": session.message_count,
                        "safety_score": session.safety_score,
                    },
                },
            )

            await message_repo.add_message(
                session_id=session_id,
                role=MessageRole.USER,
                content=user_message,
                safety_score=(
                    safety_result.confidence if not safety_result.is_safe else None
                ),
                metadata={"safety_validated": True},
            )

            if not safety_result.is_safe:
                await self._handle_safety_violation(
                    safety_repo, session_id, safety_result, user_message
                )

                response_content = (
                    safety_result.suggested_response
                    or self._get_default_safety_response(safety_result.level)
                )

                assistant_message = await message_repo.add_message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=response_content,
                    metadata={
                        "safety_intervention": True,
                        "original_safety_level": safety_result.level,
                    },
                )

                return {
                    "message_id": str(assistant_message.id),
                    "content": response_content,
                    "role": "assistant",
                    "safety_intervention": True,
                    "safety_level": safety_result.level,
                    "timestamp": assistant_message.created_at.isoformat(),
                }

            try:
                ai_response = (
                    await self._anthropic_client.generate_therapeutic_response(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        session_context={
                            "session_length": session.message_count + 1,
                            "safety_score": session.safety_score,
                            "last_activity": session.last_activity.isoformat(),
                        },
                    )
                )

                assistant_message = await message_repo.add_message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=ai_response["content"],
                    token_count=ai_response["usage"]["output_tokens"],
                    processing_time_ms=ai_response["processing_time_ms"],
                    metadata={
                        "model": ai_response["model"],
                        "input_tokens": ai_response["usage"]["input_tokens"],
                        "stop_reason": ai_response["stop_reason"],
                    },
                )

                await session_repo.update_session_activity(session_id)

                logger.info(
                    "Generated therapeutic response",
                    session_id=str(session_id),
                    response_length=len(ai_response["content"]),
                    processing_time_ms=ai_response["processing_time_ms"],
                    tokens_used=ai_response["usage"]["input_tokens"]
                    + ai_response["usage"]["output_tokens"],
                )

                return {
                    "message_id": str(assistant_message.id),
                    "content": ai_response["content"],
                    "role": "assistant",
                    "safety_intervention": False,
                    "timestamp": assistant_message.created_at.isoformat(),
                    "metadata": {
                        "processing_time_ms": ai_response["processing_time_ms"],
                        "tokens_used": ai_response["usage"]["input_tokens"]
                        + ai_response["usage"]["output_tokens"],
                    },
                }

            except Exception as e:
                logger.error(
                    "Failed to generate AI response",
                    session_id=str(session_id),
                    error=str(e),
                )

                fallback_response = (
                    "I apologize, but I'm having difficulty processing "
                    "your message right now. Could you please try "
                    "rephrasing your question or concern?"
                )

                assistant_message = await message_repo.add_message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=fallback_response,
                    metadata={"fallback_response": True, "error": str(e)},
                )

                return {
                    "message_id": str(assistant_message.id),
                    "content": fallback_response,
                    "role": "assistant",
                    "safety_intervention": False,
                    "error": "AI response generation failed",
                    "timestamp": assistant_message.created_at.isoformat(),
                }

    async def get_session(
        self, session_id: UUID, user_id: str | None = None
    ) -> dict[str, Any]:
        """Retrieve session details with conversation history."""
        db_manager = await get_database_manager()

        async with db_manager.get_session() as db_session:
            session_repo = SessionRepository(db_session)
            session = await session_repo.get_session(session_id)

            if not session:
                raise SessionNotFoundError(f"Session {session_id} not found")

            if user_id and session.user_id != user_id:
                raise SessionNotFoundError("Session not found for this user")

            messages = [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat(),
                    "metadata": msg.message_metadata,
                }
                for msg in session.messages
            ]

            return {
                "session_id": str(session.id),
                "user_id": session.user_id,
                "title": session.title,
                "status": session.status,
                "summary": session.summary,
                "safety_score": session.safety_score,
                "message_count": session.message_count,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "messages": messages,
            }

    async def end_session(
        self, session_id: UUID, user_id: str | None = None
    ) -> dict[str, Any]:
        """End therapeutic session and generate summary."""
        db_manager = await get_database_manager()

        async with db_manager.get_session() as db_session:
            session_repo = SessionRepository(db_session)
            session = await session_repo.get_session(session_id)

            if not session:
                raise SessionNotFoundError(f"Session {session_id} not found")

            if user_id and session.user_id != user_id:
                raise SessionNotFoundError("Session not found for this user")

            if session.status != SessionStatus.ACTIVE:
                raise ValueError("Session is not active")

            if session.message_count > 5:
                try:
                    conversation_history = [
                        {"role": msg.role, "content": msg.content}
                        for msg in session.messages
                    ]

                    analysis = (
                        await self._anthropic_client.analyze_conversation_patterns(
                            conversation_history, "therapeutic_progress"
                        )
                    )

                    session.summary = analysis["analysis"][
                        :1000
                    ]  # Limit summary length

                except Exception as e:
                    logger.error(
                        "Failed to generate session summary",
                        session_id=str(session_id),
                        error=str(e),
                    )

            await session_repo.update_session_status(
                session_id, SessionStatus.COMPLETED
            )

            logger.info(
                "Ended therapeutic session",
                session_id=str(session_id),
                message_count=session.message_count,
            )

            return {
                "session_id": str(session_id),
                "status": SessionStatus.COMPLETED,
                "message_count": session.message_count,
                "summary": session.summary,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_safety_violation(
        self,
        safety_repo: SafetyRepository,
        session_id: UUID,
        safety_result: SafetyResult,
        user_message: str,
    ) -> None:
        """Handle safety violations by creating events and taking appropriate action."""
        safety_flag_mapping = {
            "crisis": SafetyFlag.CRISIS,
            "self_harm": SafetyFlag.SELF_HARM,
            "substance_abuse": SafetyFlag.SUBSTANCE_ABUSE,
            "medical_advice": SafetyFlag.MEDICAL_ADVICE,
            "inappropriate_request": SafetyFlag.INAPPROPRIATE_REQUEST,
        }

        flag_type = safety_flag_mapping.get(
            (
                safety_result.category.value
                if safety_result.category
                else "inappropriate_request"
            ),
            SafetyFlag.INAPPROPRIATE_REQUEST,
        )

        await safety_repo.create_safety_event(
            session_id=session_id,
            flag_type=flag_type,
            severity_score=1.0 - safety_result.confidence,
            description=safety_result.explanation,
            intervention_taken="Automated safety response provided",
        )

        logger.warning(
            "Safety violation detected",
            session_id=str(session_id),
            category=safety_result.category,
            level=safety_result.level,
            confidence=safety_result.confidence,
        )

    def _get_default_safety_response(self, level: SafetyLevel) -> str:
        """Get default response for safety interventions."""
        if level == SafetyLevel.CRITICAL:
            return (
                "I'm very concerned about your safety. "
                "Please reach out for immediate help:\n\n"
                "🚨 National Suicide Prevention Lifeline: 988\n"
                "🚨 Crisis Text Line: Text HOME to 741741\n"
                "🚨 Emergency Services: 911\n\n"
                "You don't have to go through this alone. "
                "Professional help is available right now."
            )
        elif level == SafetyLevel.WARNING:
            return (
                "I'm concerned about what you've shared and want to "
                "make sure you have support. If you're having thoughts "
                "of self-harm, please consider reaching out to:\n\n"
                "• National Suicide Prevention Lifeline: 988\n"
                "• Crisis Text Line: Text HOME to 741741\n"
                "• A trusted friend, family member, or mental health "
                "professional\n\n"
                "Would you like to talk about what's going on and "
                "how you're feeling?"
            )
        else:
            return (
                "I want to make sure we're focusing on what's most "
                "helpful for you. Let's explore your feelings and "
                "experiences in a way that supports your wellbeing. "
                "How are you feeling right now?"
            )
