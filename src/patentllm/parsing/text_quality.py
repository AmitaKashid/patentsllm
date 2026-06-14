"""Text quality utilities used by parsing, chunking, and audit stages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patentllm.parsing.text_cleaning import normalize_inline_spacing

_ALLOWED_SYMBOLS = set(".,;:()[]{}%-/+°=<>_&'\"*@#€$™®µ–—")
_OCR_ARTIFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bOo\b"),
    re.compile(r"=\s*blend", re.IGNORECASE),
    re.compile(r"\bAMIE\b", re.IGNORECASE),
    re.compile(r"\bDiisseldorf\b", re.IGNORECASE),
    re.compile(r"\bWIPOIPCT\b", re.IGNORECASE),
    re.compile(r"\bWorld\s+Intellectual\w*\s+Property\w*\s+Organization\w*", re.IGNORECASE),
    re.compile(r"\(10\)\s*\(10\)|\(19\)\s*\(19\)|\(22\)\s*32\)"),
    re.compile(r"(?:[=|_~^]{2,}|[A-Z]?\|[A-Z]?\|[A-Z]?)"),
)
_TABLE_SIGNAL_RE = re.compile(r"\b(?:Table|TABLE)\s*\d+[A-Z]?\b")
_FIGURE_SIGNAL_RE = re.compile(r"\b(?:Figure|FIG\.)\s*\d+[A-Z]?\b", re.IGNORECASE)
_DESIGNATED_STATES_RE = re.compile(r"\bDesignated\s+States\b|\bARIPO\b|\bEurasian\b|\bEuropean\b|\bOAPI\b", re.IGNORECASE)
_COUNTRY_CODE_RUN_RE = re.compile(r"\b(?:[A-Z]{2},\s*){8,}[A-Z]{2}\b")


@dataclass(frozen=True, slots=True)
class TextQuality:
    """Compact quality assessment for one text block."""

    weird_character_ratio: float
    artifact_hits: list[str] = field(default_factory=list)
    table_detected: bool = False
    figure_detected: bool = False
    designated_states_detected: bool = False
    country_code_run_detected: bool = False

    @property
    def is_suspicious_ocr(self) -> bool:
        return self.weird_character_ratio > 0.035 or bool(self.artifact_hits)

    @property
    def is_cover_page_noise(self) -> bool:
        return self.designated_states_detected or self.country_code_run_detected

    def flags(self) -> list[str]:
        values: list[str] = []
        if self.is_suspicious_ocr:
            values.append("suspicious_ocr")
        if self.table_detected:
            values.append("table_text_detected")
        if self.figure_detected:
            values.append("figure_text_detected")
        if self.is_cover_page_noise:
            values.append("cover_page_layout_noise")
        return values


def assess_text_quality(text: str) -> TextQuality:
    """Assess OCR/layout noise without depending on external NLP packages."""

    normalized = normalize_inline_spacing(text or "")
    weird = 0
    for char in normalized:
        if char.isalnum() or char.isspace() or char in _ALLOWED_SYMBOLS:
            continue
        weird += 1

    ratio = weird / max(1, len(normalized))
    hits = [pattern.pattern for pattern in _OCR_ARTIFACT_PATTERNS if pattern.search(normalized)]

    return TextQuality(
        weird_character_ratio=ratio,
        artifact_hits=hits,
        table_detected=bool(_TABLE_SIGNAL_RE.search(normalized)),
        figure_detected=bool(_FIGURE_SIGNAL_RE.search(normalized)),
        designated_states_detected=bool(_DESIGNATED_STATES_RE.search(normalized)),
        country_code_run_detected=bool(_COUNTRY_CODE_RUN_RE.search(normalized)),
    )


def sanitize_cover_field(value: str | None, *, max_chars: int = 500) -> str | None:
    """Keep a cover-page field only if it is short and not polluted by layout noise."""

    if not value:
        return None
    cleaned = normalize_inline_spacing(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,.-")
    if not cleaned:
        return None

    quality = assess_text_quality(cleaned)
    if len(cleaned) > max_chars:
        return None
    if quality.is_cover_page_noise:
        return None
    if len(quality.artifact_hits) >= 2:
        return None
    return cleaned


def clean_abstract_text(value: str | None, *, max_chars: int = 1800) -> str | None:
    """Remove obvious cover-page artifacts from abstract text while preserving technical content."""

    if not value:
        return None
    text = normalize_inline_spacing(value)

    # Stop at clear transition points from the cover page/body repeat.
    stop_patterns = [
        r"\bWO\s+\d{4}/\d{4,}\b",
        r"\bPCT/[A-Z]{2}\d{4}/\d+\b",
        r"\bFIELD\s+OF\s+INVENTION\b",
        r"\bTECHNICAL\s+FIELD\b",
        r"\bBACKGROUND\b",
        r"\bFigure\s+\d+\b",
    ]
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[: match.start()]

    # Remove common OCR margin debris without rewriting chemistry.
    text = re.sub(r"\bOo\b|\bGQ\b|\bSN\b|^[=&~_\-—]+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ;,.-")
    if len(text) < 60:
        return None
    return text[:max_chars]
