"""FastAPI application with therapeutic endpoints and middleware."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from uuid import UUID

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from therapeutic_agent.core.config import get_settings
from therapeutic_agent.core.exceptions import (
    AnthropicAPIError,
    SafetyViolationError,
    SessionLimitExceededError,
    SessionNotFoundError,
    TherapeuticAgentException,
)
from therapeutic_agent.core.session_manager import TherapeuticSessionManager
from therapeutic_agent.storage.database import get_database_manager

logger = structlog.get_logger()


class CreateSessionRequest(BaseModel):
    """Request model for creating a therapeutic session."""

    user_id: str = Field(..., min_length=1, max_length=255)
    title: str | None = Field(None, max_length=255)


class SendMessageRequest(BaseModel):
    """Request model for sending a message to a session."""

    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str | None = Field(None, min_length=1, max_length=255)


class SessionResponse(BaseModel):
    """Response model for session data."""

    session_id: str
    user_id: str
    title: str | None
    status: str
    created_at: str
    message_count: int


class MessageResponse(BaseModel):
    """Response model for message responses."""

    message_id: str
    content: str
    role: str
    timestamp: str
    safety_intervention: bool = False
    metadata: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan with database initialization."""
    logger.info("Starting therapeutic agent API")
    await get_database_manager()
    yield
    logger.info("Shutting down therapeutic agent API")


app = FastAPI(
    title="Therapeutic Agent API",
    description="Production-ready ethical virtual therapist powered by Claude",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = TherapeuticSessionManager()


@app.exception_handler(TherapeuticAgentException)
async def therapeutic_exception_handler(
    request: Request, exc: TherapeuticAgentException
):
    """Handle therapeutic agent specific exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    if isinstance(exc, SessionNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, SessionLimitExceededError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exc, SafetyViolationError):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, AnthropicAPIError):
        status_code = status.HTTP_502_BAD_GATEWAY

    logger.error(
        "API exception", error=str(exc), status_code=status_code, details=exc.details
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "type": exc.__class__.__name__,
        },
    )


@app.post(
    "/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(request: CreateSessionRequest) -> dict[str, Any]:
    """Create a new therapeutic session."""
    try:
        session_data = await session_manager.create_session(
            user_id=request.user_id, title=request.title
        )
        return session_data
    except Exception as e:
        logger.error("Failed to create session", error=str(e), user_id=request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        )


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: UUID, request: SendMessageRequest) -> dict[str, Any]:
    """Send a message to a therapeutic session."""
    try:
        response = await session_manager.send_message(
            session_id=session_id, user_message=request.message, user_id=request.user_id
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session(session_id: UUID, user_id: str | None = None) -> dict[str, Any]:
    """Get session details with conversation history."""
    try:
        session_data = await session_manager.get_session(session_id, user_id)
        return session_data
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )


@app.post("/sessions/{session_id}/end")
async def end_session(session_id: UUID, user_id: str | None = None) -> dict[str, Any]:
    """End a therapeutic session."""
    try:
        result = await session_manager.end_session(session_id, user_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "therapeutic-agent"}


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "therapeutic_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
