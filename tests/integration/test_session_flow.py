"""Integration tests for complete therapeutic session flows."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from therapeutic_agent.core.exceptions import (
    SessionLimitExceededError,
    SessionNotFoundError,
)
from therapeutic_agent.core.session_manager import TherapeuticSessionManager
from therapeutic_agent.storage.models import (
    SessionStatus,
    TherapeuticSession,
)


def create_mock_db_manager():
    """Create a mock database manager with async context manager support."""
    mock_db_manager = AsyncMock()
    mock_session = AsyncMock()

    @asynccontextmanager
    async def mock_get_session():
        yield mock_session

    mock_db_manager.get_session = mock_get_session
    return mock_db_manager, mock_session


class TestSessionCreationFlow:
    """Test complete session creation and management flow."""

    async def test_create_session_success(self, sample_user_id: str) -> None:
        """Test successful session creation."""
        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            # Create mock database session and repository
            mock_session_repo = MagicMock()
            mock_session_repo.get_active_session_count = AsyncMock(return_value=0)

            # Create mock session object
            mock_therapeutic_session = MagicMock(spec=TherapeuticSession)
            mock_therapeutic_session.id = UUID("12345678-1234-1234-1234-123456789012")
            mock_therapeutic_session.user_id = sample_user_id
            mock_therapeutic_session.title = "Test Session"
            mock_therapeutic_session.status = SessionStatus.ACTIVE
            mock_therapeutic_session.created_at = datetime.now(timezone.utc)

            mock_session_repo.create_session = AsyncMock(
                return_value=mock_therapeutic_session
            )

            # Patch SessionRepository constructor
            with patch(
                "therapeutic_agent.core.session_manager.SessionRepository",
                return_value=mock_session_repo,
            ):
                manager = TherapeuticSessionManager()

                with patch.object(manager, "_anthropic_client"):
                    session_data = await manager.create_session(
                        sample_user_id, "Test Session"
                    )

                assert "session_id" in session_data
                assert session_data["user_id"] == sample_user_id
                assert session_data["title"] == "Test Session"
                assert session_data["status"] == SessionStatus.ACTIVE
                assert session_data["message_count"] == 0

    async def test_session_limit_enforcement(self, sample_user_id: str) -> None:
        """Test that session limits are enforced."""
        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            # Create mock database session and repository
            mock_session_repo = MagicMock()
            mock_session_repo.get_active_session_count = AsyncMock(
                return_value=10
            )  # Exceed limit

            # Patch SessionRepository constructor
            with patch(
                "therapeutic_agent.core.session_manager.SessionRepository",
                return_value=mock_session_repo,
            ):
                manager = TherapeuticSessionManager()

                with pytest.raises(SessionLimitExceededError):
                    await manager.create_session(sample_user_id)


class TestMessageFlow:
    """Test message sending and processing flow."""

    async def test_safe_message_flow(
        self, sample_user_id: str, safe_message: str
    ) -> None:
        """Test complete flow for safe message processing."""
        session_id = UUID("12345678-1234-1234-1234-123456789012")

        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            # Mock repositories
            mock_session_repo = MagicMock()
            mock_message_repo = MagicMock()
            mock_safety_repo = MagicMock()

            # Mock session
            mock_therapeutic_session = MagicMock(spec=TherapeuticSession)
            mock_therapeutic_session.id = session_id
            mock_therapeutic_session.user_id = sample_user_id
            mock_therapeutic_session.status = SessionStatus.ACTIVE
            mock_therapeutic_session.messages = []
            mock_therapeutic_session.message_count = 0
            mock_therapeutic_session.safety_score = 1.0
            mock_therapeutic_session.last_activity = datetime.now(timezone.utc)

            mock_session_repo.get_session = AsyncMock(
                return_value=mock_therapeutic_session
            )
            mock_session_repo.update_session_activity = AsyncMock()

            # Mock message creation
            mock_message = MagicMock()
            mock_message.id = UUID("22222222-2222-2222-2222-222222222222")
            mock_message.created_at = datetime.now(timezone.utc)
            mock_message_repo.add_message = AsyncMock(return_value=mock_message)

            with (
                patch(
                    "therapeutic_agent.core.session_manager.SessionRepository",
                    return_value=mock_session_repo,
                ),
                patch(
                    "therapeutic_agent.core.session_manager.MessageRepository",
                    return_value=mock_message_repo,
                ),
                patch(
                    "therapeutic_agent.core.session_manager.SafetyRepository",
                    return_value=mock_safety_repo,
                ),
            ):
                manager = TherapeuticSessionManager()

                with (
                    patch.object(manager, "_safety_engine") as mock_safety,
                    patch.object(manager, "_anthropic_client") as mock_ai,
                ):
                    from therapeutic_agent.safety.validators import (
                        SafetyLevel,
                        SafetyResult,
                    )

                    mock_safety.validate_content = AsyncMock(
                        return_value=SafetyResult(
                            is_safe=True,
                            level=SafetyLevel.SAFE,
                            category=None,
                            confidence=0.9,
                            explanation="Content is safe",
                        )
                    )
                    mock_ai.generate_therapeutic_response = AsyncMock(
                        return_value={
                            "content": "Thank you for sharing that with me.",
                            "model": "claude-3-sonnet-20240229",
                            "role": "assistant",
                            "usage": {"input_tokens": 50, "output_tokens": 25},
                            "processing_time_ms": 800,
                            "stop_reason": "end_turn",
                        }
                    )

                    response = await manager.send_message(
                        session_id, safe_message, sample_user_id
                    )

                    assert "message_id" in response
                    assert response["role"] == "assistant"
                    assert response["safety_intervention"] is False
                    assert "Thank you for sharing" in response["content"]

    async def test_crisis_message_intervention(
        self, sample_user_id: str, crisis_message: str
    ) -> None:
        """Test safety intervention for crisis messages."""
        session_id = UUID("12345678-1234-1234-1234-123456789012")

        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            # Mock repositories
            mock_session_repo = MagicMock()
            mock_message_repo = MagicMock()
            mock_safety_repo = MagicMock()
            mock_safety_repo.create_safety_event = AsyncMock()

            # Mock session
            mock_therapeutic_session = MagicMock(spec=TherapeuticSession)
            mock_therapeutic_session.id = session_id
            mock_therapeutic_session.user_id = sample_user_id
            mock_therapeutic_session.status = SessionStatus.ACTIVE
            mock_therapeutic_session.messages = []

            mock_session_repo.get_session = AsyncMock(
                return_value=mock_therapeutic_session
            )

            # Mock message creation
            mock_message = MagicMock()
            mock_message.id = UUID("22222222-2222-2222-2222-222222222222")
            mock_message.created_at = datetime.now(timezone.utc)
            mock_message_repo.add_message = AsyncMock(return_value=mock_message)

            with (
                patch(
                    "therapeutic_agent.core.session_manager.SessionRepository",
                    return_value=mock_session_repo,
                ),
                patch(
                    "therapeutic_agent.core.session_manager.MessageRepository",
                    return_value=mock_message_repo,
                ),
                patch(
                    "therapeutic_agent.core.session_manager.SafetyRepository",
                    return_value=mock_safety_repo,
                ),
            ):
                manager = TherapeuticSessionManager()

                with patch.object(manager, "_safety_engine") as mock_safety:
                    from therapeutic_agent.safety.validators import (
                        SafetyCategory,
                        SafetyLevel,
                        SafetyResult,
                    )

                    mock_safety.validate_content = AsyncMock(
                        return_value=SafetyResult(
                            is_safe=False,
                            level=SafetyLevel.CRITICAL,
                            category=SafetyCategory.CRISIS,
                            confidence=0.9,
                            explanation="Crisis detected",
                            suggested_response="Please call 988 immediately",
                        )
                    )

                    response = await manager.send_message(
                        session_id, crisis_message, sample_user_id
                    )

                    assert response["safety_intervention"] is True
                    assert (
                        "988" in response["content"]
                        or "crisis" in response["content"].lower()
                    )

    async def test_session_not_found_error(self, sample_user_id: str) -> None:
        """Test error handling for non-existent sessions."""
        session_id = UUID("00000000-0000-0000-0000-000000000000")

        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            mock_session_repo = MagicMock()
            mock_session_repo.get_session = AsyncMock(
                return_value=None
            )  # Session not found

            with patch(
                "therapeutic_agent.core.session_manager.SessionRepository",
                return_value=mock_session_repo,
            ):
                manager = TherapeuticSessionManager()

                with pytest.raises(SessionNotFoundError):
                    await manager.send_message(session_id, "Hello", sample_user_id)


class TestSessionCompletion:
    """Test session ending and summarization flow."""

    async def test_end_session_with_summary(self, sample_user_id: str) -> None:
        """Test ending session with automatic summary generation."""
        session_id = UUID("12345678-1234-1234-1234-123456789012")

        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            mock_session_repo = MagicMock()

            # Mock session with messages
            mock_therapeutic_session = MagicMock(spec=TherapeuticSession)
            mock_therapeutic_session.id = session_id
            mock_therapeutic_session.user_id = sample_user_id
            mock_therapeutic_session.status = SessionStatus.ACTIVE
            mock_therapeutic_session.message_count = 10
            mock_therapeutic_session.messages = []
            mock_therapeutic_session.summary = None

            mock_session_repo.get_session = AsyncMock(
                return_value=mock_therapeutic_session
            )
            mock_session_repo.update_session_status = AsyncMock()

            with patch(
                "therapeutic_agent.core.session_manager.SessionRepository",
                return_value=mock_session_repo,
            ):
                manager = TherapeuticSessionManager()

                with patch.object(manager, "_anthropic_client") as mock_ai:
                    mock_ai.analyze_conversation_patterns = AsyncMock(
                        return_value={
                            "analysis": (
                                "User showed good progress in " "emotional regulation"
                            )
                        }
                    )

                    result = await manager.end_session(session_id, sample_user_id)

                    assert result["status"] == SessionStatus.COMPLETED
                    assert "ended_at" in result
                    assert "summary" in result

    async def test_end_session_error_handling(self, sample_user_id: str) -> None:
        """Test error handling during session ending."""
        session_id = UUID("00000000-0000-0000-0000-000000000000")

        with patch(
            "therapeutic_agent.core.session_manager.get_database_manager",
            new_callable=AsyncMock,
        ) as mock_get_db:
            mock_db_manager, mock_session = create_mock_db_manager()
            mock_get_db.return_value = mock_db_manager

            mock_session_repo = MagicMock()
            mock_session_repo.get_session = AsyncMock(
                return_value=None
            )  # Session not found

            with patch(
                "therapeutic_agent.core.session_manager.SessionRepository",
                return_value=mock_session_repo,
            ):
                manager = TherapeuticSessionManager()

                with pytest.raises(SessionNotFoundError):
                    await manager.end_session(session_id, sample_user_id)
