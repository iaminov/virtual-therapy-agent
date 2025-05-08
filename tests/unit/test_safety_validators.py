"""Unit tests for safety validation system."""

import pytest

from therapeutic_agent.safety.engine import SafetyEngine
from therapeutic_agent.safety.validators import (
    CrisisValidator,
    MedicalAdviceValidator,
    SafetyLevel,
    SelfHarmValidator,
    TherapeuticBoundaryValidator,
)


class TestCrisisValidator:
    """Test crisis detection validation."""

    @pytest.fixture
    def validator(self) -> CrisisValidator:
        return CrisisValidator()

    async def test_detects_suicide_ideation(self, validator: CrisisValidator) -> None:
        """Test detection of suicide ideation."""
        message = "I want to kill myself and end this pain"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CRITICAL
        assert result.confidence > 0.5
        assert "crisis" in result.explanation.lower()
        assert result.suggested_response is not None

    async def test_detects_multiple_indicators(
        self, validator: CrisisValidator
    ) -> None:
        """Test detection with multiple crisis indicators."""
        message = "I have a plan to kill myself with pills"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CRITICAL
        assert result.confidence > 0.6

    async def test_safe_content_passes(self, validator: CrisisValidator) -> None:
        """Test that safe content passes validation."""
        message = "I'm feeling sad but want to work through this"
        result = await validator.validate(message)

        assert result.is_safe
        assert result.level == SafetyLevel.SAFE
        assert result.confidence > 0.7


class TestSelfHarmValidator:
    """Test self-harm detection validation."""

    @pytest.fixture
    def validator(self) -> SelfHarmValidator:
        return SelfHarmValidator()

    async def test_detects_cutting_behavior(self, validator: SelfHarmValidator) -> None:
        """Test detection of cutting behavior."""
        message = "I've been cutting myself with a razor"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.WARNING
        assert "self-harm" in result.explanation.lower()

    async def test_safe_content_passes(self, validator: SelfHarmValidator) -> None:
        """Test that safe content passes validation."""
        message = "I accidentally cut myself while cooking"
        result = await validator.validate(message)

        assert result.is_safe
        assert result.level == SafetyLevel.SAFE


class TestMedicalAdviceValidator:
    """Test medical advice request detection."""

    @pytest.fixture
    def validator(self) -> MedicalAdviceValidator:
        return MedicalAdviceValidator()

    async def test_detects_diagnosis_request(
        self, validator: MedicalAdviceValidator
    ) -> None:
        """Test detection of diagnosis requests."""
        message = "Can you diagnose what's wrong with me based on these symptoms?"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CAUTION
        assert "medical advice" in result.explanation.lower()

    async def test_detects_medication_request(
        self, validator: MedicalAdviceValidator
    ) -> None:
        """Test detection of medication advice requests."""
        message = "What medication should I take for my depression?"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CAUTION


class TestTherapeuticBoundaryValidator:
    """Test therapeutic boundary validation."""

    @pytest.fixture
    def validator(self) -> TherapeuticBoundaryValidator:
        return TherapeuticBoundaryValidator()

    async def test_detects_boundary_crossing(
        self, validator: TherapeuticBoundaryValidator
    ) -> None:
        """Test detection of boundary crossing attempts."""
        message = "Can we meet outside of therapy and be friends?"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CAUTION
        assert "boundary" in result.explanation.lower()

    async def test_detects_personal_info_request(
        self, validator: TherapeuticBoundaryValidator
    ) -> None:
        """Test detection of personal information requests."""
        message = "What's your phone number so we can talk outside sessions?"
        result = await validator.validate(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CAUTION


class TestSafetyEngine:
    """Test integrated safety engine."""

    @pytest.fixture
    def engine(self) -> SafetyEngine:
        return SafetyEngine()

    async def test_aggregates_multiple_violations(self, engine: SafetyEngine) -> None:
        """Test aggregation of multiple safety violations."""
        message = (
            "I want to kill myself and need you to tell me what medication to take"
        )
        result = await engine.validate_content(message)

        assert not result.is_safe
        assert result.level == SafetyLevel.CRITICAL  # Crisis takes priority
        assert result.confidence > 0.5

    async def test_handles_empty_content(self, engine: SafetyEngine) -> None:
        """Test handling of empty content."""
        result = await engine.validate_content("")

        assert result.is_safe
        assert result.level == SafetyLevel.SAFE
        assert result.confidence == 1.0

    async def test_conversation_context_validation(self, engine: SafetyEngine) -> None:
        """Test conversation context validation."""
        messages = [
            {"role": "user", "content": "I'm feeling really down"},
            {"role": "assistant", "content": "I'm sorry to hear that"},
            {"role": "user", "content": "I keep thinking about ending it all"},
        ]

        result = await engine.validate_conversation_context(messages)

        assert not result.is_safe
        assert result.level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]
