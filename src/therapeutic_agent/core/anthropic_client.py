"""Anthropic API client with therapeutic-focused prompting and error handling."""

import time
from typing import Any

import structlog
from anthropic import AsyncAnthropic
from anthropic.types import Message

from therapeutic_agent.core.config import get_settings
from therapeutic_agent.core.exceptions import AnthropicAPIError

logger = structlog.get_logger()


class TherapeuticPromptBuilder:
    """Builds therapeutic context-aware prompts for Anthropic Claude."""

    THERAPEUTIC_SYSTEM_PROMPT = (
        "You are a compassionate, ethical virtual therapist assistant. "
        "Your role is to provide supportive, evidence-based therapeutic "
        "guidance while maintaining strict ethical boundaries.\n\n"
        "Core principles:\n"
        "1. Always prioritize user safety and wellbeing\n"
        "2. Provide empathetic, non-judgmental responses\n"
        "3. Use evidence-based therapeutic techniques (CBT, mindfulness, etc.)\n"
        "4. Never provide medical diagnoses or prescriptions\n"
        "5. Maintain professional boundaries at all times\n"
        "6. Encourage professional help when appropriate\n"
        "7. Recognize and respond to crisis situations appropriately\n\n"
        "Response guidelines:\n"
        "- Be warm, authentic, and validating\n"
        "- Ask clarifying questions to better understand\n"
        "- Reflect emotions and summarize key points\n"
        "- Suggest practical coping strategies when appropriate\n"
        "- Stay within your scope of practice as a therapeutic assistant\n"
        "- If safety concerns arise, prioritize immediate safety resources\n\n"
        "Remember: You are a supportive tool, not a replacement for "
        "human professional therapy."
    )

    @classmethod
    def build_conversation_prompt(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_context: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        """Build a complete conversation prompt with therapeutic context."""
        messages = [{"role": "system", "content": self.THERAPEUTIC_SYSTEM_PROMPT}]

        if session_context:
            context_summary = self._build_context_summary(session_context)
            if context_summary:
                messages.append(
                    {"role": "system", "content": f"Session context: {context_summary}"}
                )

        if conversation_history:
            messages.extend(conversation_history[-10:])  # Last 10 messages for context

        messages.append({"role": "user", "content": user_message})
        return messages

    @classmethod
    def _build_context_summary(cls, context: dict[str, Any]) -> str:
        """Build context summary from session metadata."""
        parts = []

        if context.get("session_length"):
            parts.append(f"Session #{context['session_length']}")

        if context.get("primary_concerns"):
            concerns = ", ".join(context["primary_concerns"][:3])
            parts.append(f"Primary concerns: {concerns}")

        if context.get("therapeutic_goals"):
            goals = ", ".join(context["therapeutic_goals"][:2])
            parts.append(f"Goals: {goals}")

        return "; ".join(parts)


class AnthropicTherapeuticClient:
    """Enhanced Anthropic client optimized for therapeutic interactions."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._prompt_builder = TherapeuticPromptBuilder()
        self._model = "claude-3-sonnet-20240229"
        self._max_tokens = 1000

    async def generate_therapeutic_response(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_context: dict[str, Any] | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate therapeutic response with error handling and metrics."""
        start_time = time.time()

        try:
            messages = self._prompt_builder.build_conversation_prompt(
                user_message=user_message,
                conversation_history=conversation_history,
                session_context=session_context,
            )

            logger.info(
                "Generating therapeutic response",
                user_message_length=len(user_message),
                history_length=len(conversation_history) if conversation_history else 0,
                has_context=bool(session_context),
            )

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=temperature,
                messages=messages,  # type: ignore[arg-type]
            )

            processing_time = int((time.time() - start_time) * 1000)

            return self._format_response(response, processing_time)

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(
                "Anthropic API error",
                error=str(e),
                processing_time_ms=processing_time,
                user_message_length=len(user_message),
            )
            raise AnthropicAPIError(
                message="Failed to generate therapeutic response",
                details={
                    "original_error": str(e),
                    "processing_time_ms": processing_time,
                    "model": self._model,
                },
            )

    def _format_response(
        self, response: Message, processing_time_ms: int
    ) -> dict[str, Any]:
        """Format Anthropic response with metadata."""
        content = ""
        if response.content and len(response.content) > 0:
            content = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content[0])
            )

        return {
            "content": content,
            "model": response.model,
            "role": response.role,
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
            "processing_time_ms": processing_time_ms,
            "stop_reason": response.stop_reason,
        }

    async def analyze_conversation_patterns(
        self,
        conversation_history: list[dict[str, str]],
        analysis_type: str = "therapeutic_progress",
    ) -> dict[str, Any]:
        """Analyze conversation patterns for therapeutic insights."""
        if not conversation_history:
            return {"analysis": "No conversation history available", "insights": []}

        analysis_prompts = {
            "therapeutic_progress": """Analyze this therapeutic conversation for:
1. Key themes and concerns expressed
2. Emotional patterns and progress indicators
3. Therapeutic alliance quality
4. Areas needing focus in future sessions
Provide concise, professional insights.""",
            "safety_assessment": """Review this conversation for:
1. Any safety concerns or risk factors
2. Crisis indicators or escalation patterns
3. Protective factors mentioned
4. Recommended safety interventions
Focus on clinical safety assessment.""",
        }

        prompt = analysis_prompts.get(
            analysis_type, analysis_prompts["therapeutic_progress"]
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical supervisor analyzing "
                        "therapeutic conversations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nConversation:\n"
                    + "\n".join(
                        [
                            f"{msg['role']}: {msg['content']}"
                            for msg in conversation_history[-15:]
                        ]
                    ),
                },
            ]

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=500,
                temperature=0.3,
                messages=messages,  # type: ignore[arg-type]
            )

            content = ""
            if response.content and len(response.content) > 0:
                content = (
                    response.content[0].text
                    if hasattr(response.content[0], "text")
                    else str(response.content[0])
                )

            return {
                "analysis_type": analysis_type,
                "analysis": content,
                "conversation_length": len(conversation_history),
                "tokens_used": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if response.usage
                    else 0
                ),
            }

        except Exception as e:
            logger.error(
                "Conversation analysis failed",
                error=str(e),
                analysis_type=analysis_type,
            )
            raise AnthropicAPIError(
                message="Failed to analyze conversation patterns",
                details={"original_error": str(e), "analysis_type": analysis_type},
            )
