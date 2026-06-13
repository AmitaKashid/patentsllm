"""Text normalization helpers for patent parsing."""

from __future__ import annotations

import re
import unicodedata

_PAGE_HEADER_RE = re.compile(
    r"^(?:WO\s*\d{4}/?\d{5,}|US\s*\d{4}/?\d+|EP\s*\d+|PCT/[A-Z0-9/]+|\d{1,3})$",
    flags=re.IGNORECASE,
)
_SOFT_HYPHEN_RE = re.compile(r"\u00ad")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTIBLANK_RE = re.compile(r"\n{3,}")


def estimate_tokens(text: str) -> int:
    """Return a lightweight token estimate without adding a tokenizer dependency."""

    words = re.findall(r"\S+", text or "")
    return max(1, int(len(words) * 1.3)) if words else 0


def normalize_unicode(text: str) -> str:
    """Normalize Unicode while keeping patent symbols readable."""

    value = unicodedata.normalize("NFKC", text or "")
    value = _SOFT_HYPHEN_RE.sub("", value)
    return value


def normalize_inline_spacing(text: str) -> str:
    """Collapse excessive horizontal whitespace inside a line."""

    value = normalize_unicode(text).replace("\r", "\n")
    value = _MULTISPACE_RE.sub(" ", value)
    return value.strip()


def normalize_page_text(text: str) -> str:
    """Normalize extracted page text while preserving useful line boundaries."""

    value = normalize_unicode(text).replace("\r", "\n")
    cleaned_lines: list[str] = []

    for raw_line in value.split("\n"):
        line = normalize_inline_spacing(raw_line)
        if not line:
            cleaned_lines.append("")
            continue
        if is_probable_header_or_footer(line):
            continue
        cleaned_lines.append(line)

    normalized = "\n".join(cleaned_lines)
    normalized = _MULTIBLANK_RE.sub("\n\n", normalized)
    return normalized.strip()


def is_probable_header_or_footer(line: str) -> bool:
    """Detect repeated patent page headers/footers."""

    stripped = line.strip()
    if len(stripped) > 120:
        return False
    return bool(_PAGE_HEADER_RE.match(stripped))


def join_wrapped_lines(lines: list[str]) -> str:
    """Join OCR/PDF line wraps into a readable paragraph."""

    paragraph = " ".join(line.strip() for line in lines if line.strip())
    # Repair only simple OCR/PDF line hyphenation such as poly- propylene.
    paragraph = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", paragraph)
    return normalize_inline_spacing(paragraph)


def canonical_section_name(raw: str) -> str:
    """Convert a heading into a stable section identifier."""

    value = normalize_inline_spacing(raw).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"
