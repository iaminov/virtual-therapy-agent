"""Content safety validation rules and processors."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


class SafetyLevel(str, Enum):
    """Safety assessment levels."""

    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


class SafetyCategory(str, Enum):
    """Categories of safety concerns."""

    CRISIS = "crisis"
    SELF_HARM = "self_harm"
    SUBSTANCE_ABUSE = "substance_abuse"
    INAPPROPRIATE_REQUEST = "inappropriate_request"
    MEDICAL_ADVICE = "medical_advice"
    PERSONAL_INFO = "personal_info"
    THERAPEUTIC_BOUNDARY = "therapeutic_boundary"


@dataclass
class SafetyResult:
    """Result of safety validation assessment."""

    is_safe: bool
    level: SafetyLevel
    category: SafetyCategory | None
    confidence: float
    explanation: str
    suggested_response: str | None = None


class SafetyPattern(BaseModel):
    """Pattern-based safety rule configuration."""

    category: SafetyCategory
    patterns: list[str]
    level: SafetyLevel
    confidence_boost: float = 0.0
    context_required: bool = False


class BaseValidator(ABC):
    """Abstract base class for safety validators."""

    @abstractmethod
    async def validate(
        self, content: str, context: dict[str, Any] | None = None
    ) -> SafetyResult:
        """Validate content for safety concerns."""
        pass


class CrisisValidator(BaseValidator):
    """Validates content for crisis indicators."""

    CRISIS_PATTERNS = [
        r"\b(?:kill(?:ing)?|hurt(?:ing)?|harm(?:ing)?)\s+(?:myself|me)\b",
        r"\b(?:suicide|suicidal|end(?:ing)?\s+(?:my\s+)?life)\b",
        r"\b(?:end(?:ing)?\s+it(?:\s+all)?)\b",  # Matches "ending it", "end it all"
        r"\b(?:want\s+to\s+die|wish\s+i\s+was\s+dead|better\s+off\s+dead)\b",
        r"\b(?:overdose|pills?\s+to\s+(?:die|end))\b",
        r"\b(?:plan\s+to\s+(?:kill|hurt|harm)|have\s+a\s+plan)\b",
        r"\b(?:can\'?t\s+(?:go\s+on|take\s+it|live)|no\s+point\s+in\s+living)\b",
    ]

    CRISIS_KEYWORDS = {
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "want to die",
        "overdose",
        "pills to die",
        "hurt myself",
        "harm myself",
        "cut myself",
    }

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CRISIS_PATTERNS
        ]

    async def validate(
        self, content: str, context: dict[str, Any] | None = None
    ) -> SafetyResult:
        """Validate for crisis indicators."""
        content_lower = content.lower()

        pattern_matches = sum(
            1 for pattern in self._compiled_patterns if pattern.search(content)
        )
        keyword_matches = sum(
            1 for keyword in self.CRISIS_KEYWORDS if keyword in content_lower
        )

        total_indicators = pattern_matches + keyword_matches

        if total_indicators >= 2:
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.CRITICAL,
                category=SafetyCategory.CRISIS,
                confidence=min(0.9, 0.4 + (total_indicators * 0.15)),
                explanation=(
                    "Multiple crisis indicators detected requiring "
                    "immediate intervention"
                ),
                suggested_response=(
                    "I'm very concerned about what you've shared. "
                    "Your safety is the most important thing right now. "
                    "Please reach out to a crisis hotline immediately: "
                    "National Suicide Prevention Lifeline at 988. "
                    "If you're in immediate danger, please call "
                    "emergency services at 911."
                ),
            )
        elif total_indicators == 1:
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.WARNING,
                category=SafetyCategory.CRISIS,
                confidence=0.7,
                explanation="Potential crisis indicator detected",
                suggested_response=(
                    "I'm concerned about what you've mentioned. "
                    "It's important to talk to someone who can help. "
                    "Consider contacting the National Suicide Prevention "
                    "Lifeline at 988 for support."
                ),
            )

        return SafetyResult(
            is_safe=True,
            level=SafetyLevel.SAFE,
            category=None,
            confidence=0.8,
            explanation="No crisis indicators detected",
        )


class SelfHarmValidator(BaseValidator):
    """Validates content for self-harm indicators."""

    SELF_HARM_PATTERNS = [
        r"\b(?:cut(?:ting)?|scratch(?:ing)?|burn(?:ing)?)\s+(?:myself|me|my\s+\w+)\b",
        r"\b(?:razor|blade|knife)\s+(?:on|to)\s+(?:myself|my\s+\w+)\b",
        r"\b(?:self[-\s]?harm|self[-\s]?injur)",
        r"\b(?:hurt\s+myself|inflict\s+pain)\b",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SELF_HARM_PATTERNS
        ]

    async def validate(
        self, content: str, context: dict[str, Any] | None = None
    ) -> SafetyResult:
        """Validate for self-harm indicators."""
        # Check for accidental/non-intentional mentions
        content_lower = content.lower()
        accidental_keywords = [
            "accidentally",
            "accident",
            "cooking",
            "shaving",
            "working",
        ]
        is_accidental = any(keyword in content_lower for keyword in accidental_keywords)

        matches = sum(
            1 for pattern in self._compiled_patterns if pattern.search(content)
        )

        if matches >= 1 and not is_accidental:
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.WARNING,
                category=SafetyCategory.SELF_HARM,
                confidence=0.8,
                explanation="Self-harm behavior indicators detected",
                suggested_response=(
                    "I'm concerned about the self-harm behaviors "
                    "you've mentioned. These feelings are valid, "
                    "but there are healthier ways to cope. Please "
                    "consider reaching out to a mental health "
                    "professional or crisis line for support."
                ),
            )

        return SafetyResult(
            is_safe=True,
            level=SafetyLevel.SAFE,
            category=None,
            confidence=0.9,
            explanation="No self-harm indicators detected",
        )


class MedicalAdviceValidator(BaseValidator):
    """Validates requests for medical advice."""

    MEDICAL_PATTERNS = [
        r"\b(?:diagnose|diagnosis|what(?:\s+do\s+i|\'s\s+wrong))\b"
        r".*\b(?:symptoms?|condition|disease|disorder)\b",
        r"\b(?:should\s+i\s+take|recommend(?:ation)?|"
        r"prescribe|medication|medicine|drug)\b",
        r"\b(?:doctor|physician|medical|health)\s+(?:advice|opinion|recommendation)\b",
        r"\b(?:treatment\s+for|cure\s+for|how\s+to\s+treat)\b",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.MEDICAL_PATTERNS
        ]

    async def validate(
        self, content: str, context: dict[str, Any] | None = None
    ) -> SafetyResult:
        """Validate for inappropriate medical advice requests."""
        matches = sum(
            1 for pattern in self._compiled_patterns if pattern.search(content)
        )

        if matches >= 1:
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.CAUTION,
                category=SafetyCategory.MEDICAL_ADVICE,
                confidence=0.7,
                explanation="Request for medical advice detected",
                suggested_response=(
                    "I understand you're looking for guidance, but "
                    "I can't provide medical advice or diagnoses. "
                    "For health-related concerns, it's important to "
                    "consult with qualified healthcare professionals "
                    "who can properly assess your situation."
                ),
            )

        return SafetyResult(
            is_safe=True,
            level=SafetyLevel.SAFE,
            category=None,
            confidence=0.8,
            explanation="No inappropriate medical advice requests detected",
        )


class TherapeuticBoundaryValidator(BaseValidator):
    """Validates therapeutic relationship boundaries."""

    BOUNDARY_PATTERNS = [
        r"\b(?:can\s+we\s+(?:meet|be\s+friends)|personal\s+(?:relationship|contact))\b",
        r"\b(?:outside\s+of\s+(?:therapy|session)|real\s+life|in\s+person)\b",
        r"\b(?:phone\s+number|address|where\s+do\s+you\s+live|personal\s+info)\b",
        r"\b(?:date|romantic|sexual|intimate)\b.*\b(?:relationship|feelings)\b",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.BOUNDARY_PATTERNS
        ]

    async def validate(
        self, content: str, context: dict[str, Any] | None = None
    ) -> SafetyResult:
        """Validate therapeutic boundary maintenance."""
        matches = sum(
            1 for pattern in self._compiled_patterns if pattern.search(content)
        )

        if matches >= 1:
            return SafetyResult(
                is_safe=False,
                level=SafetyLevel.CAUTION,
                category=SafetyCategory.THERAPEUTIC_BOUNDARY,
                confidence=0.6,
                explanation="Therapeutic boundary concerns detected",
                suggested_response=(
                    "I appreciate your trust in our therapeutic "
                    "relationship. To maintain professional boundaries "
                    "that best support your wellbeing, I need to keep "
                    "our interactions focused on your therapeutic goals "
                    "and within this context."
                ),
            )

        return SafetyResult(
            is_safe=True,
            level=SafetyLevel.SAFE,
            category=None,
            confidence=0.9,
            explanation="No boundary concerns detected",
        )
