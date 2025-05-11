"""Core safety validation engine orchestrating multiple validators."""

import asyncio
from typing import Any

import structlog

from therapeutic_agent.safety.validators import (
    BaseValidator,
    CrisisValidator,
    MedicalAdviceValidator,
    SafetyLevel,
    SafetyResult,
    SelfHarmValidator,
    TherapeuticBoundaryValidator,
)

logger = structlog.get_logger()


class SafetyEngine:
    """Orchestrates multiple safety validators for comprehensive content analysis."""

    def __init__(self) -> None:
        self._validators: list[BaseValidator] = [
            CrisisValidator(),
            SelfHarmValidator(),
            MedicalAdviceValidator(),
            TherapeuticBoundaryValidator(),
        ]

    async def validate_content(
        self, content: str, context: dict[str, Any] | None = None
    ) -> SafetyResult:
        """Run content through all safety validators and return aggregated result."""
        if not content.strip():
            return SafetyResult(
                is_safe=True,
                level=SafetyLevel.SAFE,
                category=None,
                confidence=1.0,
                explanation="Empty content is safe",
            )

        validation_tasks = [
            validator.validate(content, context) for validator in self._validators
        ]

        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Filter out exceptions explicitly for typing
        valid_results: list[SafetyResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(
                    "Validator failed",
                    validator=self._validators[i].__class__.__name__,
                    error=str(result),
                )
                continue
            valid_results.append(result)

        if not valid_results:
            logger.warning("No validators succeeded, defaulting to unsafe")
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.WARNING,
                category=None,
                confidence=0.1,
                explanation="Safety validation failed - unable to assess content",
            )

        return self._aggregate_results(valid_results)

    def _aggregate_results(self, results: list[SafetyResult]) -> SafetyResult:
        """Aggregate multiple validation results into a single safety assessment."""
        unsafe_results = [r for r in results if not r.is_safe]

        if not unsafe_results:
            highest_confidence = max(r.confidence for r in results)
            return SafetyResult(
                is_safe=True,
                level=SafetyLevel.SAFE,
                category=None,
                confidence=highest_confidence,
                explanation="Content passed all safety validations",
            )

        most_critical = max(
            unsafe_results, key=lambda r: (self._level_priority(r.level), r.confidence)
        )

        # Calculate average confidence for the most critical category
        matching_results = [
            r for r in unsafe_results if r.category == most_critical.category
        ]
        if matching_results:
            avg_confidence = sum(r.confidence for r in matching_results) / len(
                matching_results
            )
        else:
            avg_confidence = most_critical.confidence

        # Boost confidence if multiple violations detected
        confidence_boost = min(0.15, 0.05 * (len(unsafe_results) - 1))
        aggregated_confidence = min(0.95, avg_confidence + confidence_boost)

        explanations = [r.explanation for r in unsafe_results if r.explanation]
        combined_explanation = "; ".join(explanations[:3])

        return SafetyResult(
            is_safe=False,
            level=most_critical.level,
            category=most_critical.category,
            confidence=aggregated_confidence,
            explanation=combined_explanation,
            suggested_response=most_critical.suggested_response,
        )

    def _level_priority(self, level: SafetyLevel) -> int:
        """Get numeric priority for safety level (higher = more critical)."""
        priority_map = {
            SafetyLevel.SAFE: 0,
            SafetyLevel.CAUTION: 1,
            SafetyLevel.WARNING: 2,
            SafetyLevel.CRITICAL: 3,
        }
        return priority_map.get(level, 0)

    async def validate_conversation_context(
        self,
        messages: list[dict[str, str]],
        session_metadata: dict[str, Any] | None = None,
    ) -> SafetyResult:
        """Validate entire conversation context for patterns across messages."""
        if not messages:
            return SafetyResult(
                is_safe=True,
                level=SafetyLevel.SAFE,
                category=None,
                confidence=1.0,
                explanation="No conversation history to validate",
            )

        combined_content = " ".join(
            msg.get("content", "")
            for msg in messages[-10:]  # Last 10 messages for context
            if msg.get("role") == "user"
        )

        return await self.validate_content(
            combined_content,
            context={
                "conversation_length": len(messages),
                "session_metadata": session_metadata,
            },
        )
