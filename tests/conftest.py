"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from therapeutic_agent.core.anthropic_client import AnthropicTherapeuticClient
from therapeutic_agent.storage.database import DatabaseManager
from therapeutic_agent.storage.models import Base
from therapeutic_agent.storage.repository import (
    MessageRepository,
    SafetyRepository,
    SessionRepository,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db_manager() -> AsyncGenerator[DatabaseManager, None]:
    """Create test database manager with in-memory SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    manager = DatabaseManager()
    manager._engine = engine
    manager._session_factory = AsyncSession.bind(engine)

    yield manager

    await engine.dispose()


@pytest.fixture
async def db_session(
    test_db_manager: DatabaseManager,
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session for testing."""
    async with test_db_manager.get_session() as session:
        yield session


@pytest.fixture
async def session_repo(db_session: AsyncSession) -> SessionRepository:
    """Get session repository for testing."""
    return SessionRepository(db_session)


@pytest.fixture
async def message_repo(db_session: AsyncSession) -> MessageRepository:
    """Get message repository for testing."""
    return MessageRepository(db_session)


@pytest.fixture
async def safety_repo(db_session: AsyncSession) -> SafetyRepository:
    """Get safety repository for testing."""
    return SafetyRepository(db_session)


@pytest.fixture
def mock_anthropic_client() -> AsyncMock:
    """Mock Anthropic client for testing."""
    client = AsyncMock(spec=AnthropicTherapeuticClient)
    client.generate_therapeutic_response.return_value = {
        "content": "Thank you for sharing. How are you feeling about that?",
        "model": "claude-3-sonnet-20240229",
        "role": "assistant",
        "usage": {"input_tokens": 150, "output_tokens": 50},
        "processing_time_ms": 1200,
        "stop_reason": "end_turn",
    }
    client.analyze_conversation_patterns.return_value = {
        "analysis_type": "therapeutic_progress",
        "analysis": "User showed good engagement and emotional awareness",
        "conversation_length": 10,
        "tokens_used": 300,
    }
    return client


@pytest.fixture
def sample_user_id() -> str:
    """Generate sample user ID for testing."""
    return f"test_user_{uuid4().hex[:8]}"


@pytest.fixture
def crisis_message() -> str:
    """Sample crisis message for testing."""
    return "I can't take it anymore. I want to kill myself and end the pain."


@pytest.fixture
def safe_message() -> str:
    """Sample safe message for testing."""
    return "I've been feeling anxious lately about work stress."


@pytest.fixture
def medical_advice_request() -> str:
    """Sample medical advice request for testing."""
    msg = "Can you diagnose what's wrong with me? "
    msg += "I have these symptoms and need medication recommendations."
    return msg
