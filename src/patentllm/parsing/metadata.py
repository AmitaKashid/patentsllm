"""Defensive metadata extraction for OCR-heavy PCT-style patent PDFs."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from patentllm.parsing.models import PageRecord, PatentMetadata
from patentllm.parsing.text_cleaning import join_wrapped_lines, normalize_inline_spacing
from patentllm.parsing.text_quality import clean_abstract_text, sanitize_cover_field

_PUBLICATION_FROM_FILENAME_RE = re.compile(
    r"(?P<country>[A-Z]{2})[_-]?(?P<number>\d{4,})[_-]?(?P<kind>[A-Z]\d?)?",
    re.IGNORECASE,
)
_PUBLICATION_IN_TEXT_RE = re.compile(
    r"\b(?P<country>WO|US|EP)\s*(?P<number>\d{4}/?\d{4,}|\d{7,})\s*(?P<kind>A\d?|B\d?)?\b",
    re.IGNORECASE,
)
_FIELD_PATTERNS = {
    "publication_date": re.compile(r"International\s+Publication\s+Date\s+([^\n]+)", re.IGNORECASE),
    "application_number": re.compile(r"International\s+Application\s+Number:?\s*([^\n]+)", re.IGNORECASE),
    "filing_date": re.compile(r"International\s+Filing\s+Date:?\s*([^\n]+)", re.IGNORECASE),
    "priority_date": re.compile(r"Priority\s+Data:?\s*([^\n]+(?:\n(?!\(\d{2}\)).+)*)", re.IGNORECASE),
    "applicant": re.compile(r"\(71\)\s*Applicant[s]?:?\s*([^\n]+(?:\n(?!\(7[24]\)|\(8[14]\)|\(10\)|\(43\)).+)*)", re.IGNORECASE),
    "inventors": re.compile(r"\(72\)\s*Inventor[s]?:?\s*([^\n]+(?:\n(?!\(7[14]\)|\(74\)|\(8[14]\)|\(10\)|\(43\)).+)*)", re.IGNORECASE),
    "classification": re.compile(r"\(51\)\s*International\s+Patent\s+Classification:?\s*([^\n]+(?:\n(?!\(\d{2}\)).+)*)", re.IGNORECASE),
}
_PATENT_CITATION_RE = re.compile(
    r"\b(?:WO\s*\d{4}/\d{4,}|US\s*(?:Patent\s*)?(?:Application\s*)?(?:No\.\s*)?\d[\d,./]+|EP\s*\d{6,}|JP\s*\d{4}-\d{4,})\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(r"\bTable\s+\d+[A-Z]?\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\b(?:Figure|FIG\.)\s+\d+[A-Z]?\b", re.IGNORECASE)
_CLASSIFICATION_CODE_RE = re.compile(r"\b[A-H][O0]?\d{2}[A-ZJT]\s*\d{1,3}/\d{1,6}\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b")
_APPLICATION_RE = re.compile(r"\bPCT/[A-Z]{2}\d{4}/\d{5,}\b", re.IGNORECASE)
_COVER_NOISE_RE = re.compile(r"\b(?:Designated States|ARIPO|Eurasian|European|OAPI|Published:|Declarations under Rule)\b", re.IGNORECASE)
_TITLE_SKIP_RE = re.compile(r"\b(?:WIPO|PCT|INTERNATIONAL|PUBLICATION|DESIGNATED STATES|BUREAU|APPLICATION NUMBER)\b", re.IGNORECASE)


def extract_metadata(pdf_path: Path, document_id: str, pages: list[PageRecord]) -> PatentMetadata:
    """Extract metadata defensively from filename, cover page, and early body pages."""

    source_file = pdf_path.name
    page_count = _pdf_page_count(pdf_path)
    first_pages_text = "\n".join(page.text for page in pages[:4])
    body_text_sample = "\n".join(page.text for page in pages[: min(20, len(pages))])

    publication_number, kind_code = _extract_publication_number(source_file, first_pages_text)
    raw_fields = {key: _extract_raw_field(pattern, first_pages_text) for key, pattern in _FIELD_PATTERNS.items()}

    title = _extract_title(first_pages_text, source_file)
    abstract = _extract_abstract(first_pages_text, body_text_sample)
    classifications = _extract_classifications(raw_fields.get("classification"), first_pages_text)

    application_number = _clean_application_number(raw_fields.get("application_number"), first_pages_text)
    publication_date = _clean_date(raw_fields.get("publication_date"))
    filing_date = _clean_date(raw_fields.get("filing_date"))
    priority_date = _clean_date(raw_fields.get("priority_date"))
    applicant = sanitize_cover_field(_truncate_at_cover_noise(raw_fields.get("applicant")), max_chars=260)
    inventors = sanitize_cover_field(_truncate_at_cover_noise(raw_fields.get("inventors")), max_chars=500)

    warnings: list[str] = []
    for name, value in {
        "applicant": raw_fields.get("applicant"),
        "inventors": raw_fields.get("inventors"),
        "publication_date": raw_fields.get("publication_date"),
        "application_number": raw_fields.get("application_number"),
        "filing_date": raw_fields.get("filing_date"),
        "priority_date": raw_fields.get("priority_date"),
    }.items():
        cleaned_value = locals().get(name)
        if value and not cleaned_value:
            warnings.append(f"Discarded noisy cover-page field: {name}.")

    citations = _deduplicate(_PATENT_CITATION_RE.findall(body_text_sample))[:100]
    tables = _deduplicate(_TABLE_RE.findall(body_text_sample))[:100]
    figures = _deduplicate(_FIGURE_RE.findall(body_text_sample))[:100]

    return PatentMetadata(
        document_id=document_id,
        source_file=source_file,
        page_count=page_count,
        publication_number=publication_number,
        kind_code=kind_code,
        title=title,
        abstract=abstract,
        applicant=applicant,
        inventors=inventors,
        publication_date=publication_date,
        application_number=application_number,
        filing_date=filing_date,
        priority_date=priority_date,
        ipc_cpc_classifications=classifications,
        cited_patent_literature=citations,
        detected_tables=tables,
        detected_figures=figures,
        raw_cover_fields={key: value for key, value in raw_fields.items() if value},
        metadata_warnings=warnings,
    )


def _pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as doc:
        return len(doc)


def _extract_publication_number(source_file: str, text: str) -> tuple[str | None, str | None]:
    for raw in (source_file, text):
        match = _PUBLICATION_FROM_FILENAME_RE.search(raw) or _PUBLICATION_IN_TEXT_RE.search(raw)
        if not match:
            continue
        country = match.group("country").upper()
        number = match.group("number").replace("/", "")
        kind = match.groupdict().get("kind")
        kind = kind.upper() if kind else None
        return (f"{country}-{number}-{kind}", kind) if kind else (f"{country}-{number}", None)
    return None, None


def _extract_title(text: str, source_file: str) -> str | None:
    candidates: list[str] = []

    marker = re.search(r"\(54\)\s*Title:?\s*(.+?)(?:\n\s*\(\d{2}\)|\n\s*Abstract|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if marker:
        candidates.append(marker.group(1))

    # Body title often appears as an all-uppercase line after the repeated WO/PCT header.
    lines = [normalize_inline_spacing(line) for line in text.splitlines()]
    for index, line in enumerate(lines[:160]):
        candidate = line.strip(" -–—")
        if not candidate:
            continue
        if _TITLE_SKIP_RE.search(candidate):
            continue
        if 8 <= len(candidate) <= 220 and _looks_like_title(candidate):
            # Prefer lines followed by technical/body headings or title continuation lines.
            next_lines = " ".join(lines[index + 1 : index + 4])
            if re.search(r"\b(?:FIELD|TECHNICAL|BACKGROUND|CROSS REFERENCE|DESCRIPTION)\b", next_lines, re.IGNORECASE) or candidate.isupper():
                candidates.append(candidate)

    for candidate in candidates:
        cleaned = _clean_title(candidate)
        if cleaned:
            return cleaned

    fallback = Path(source_file).stem.replace("_", " ")
    return fallback if fallback else None


def _looks_like_title(value: str) -> bool:
    if value.count(",") >= 4:
        return False
    if len(value.split()) > 22:
        return False
    alpha = sum(1 for char in value if char.isalpha())
    if alpha < 6:
        return False
    uppercase_ratio = sum(1 for char in value if char.isupper()) / max(1, alpha)
    return uppercase_ratio > 0.65


def _clean_title(value: str) -> str | None:
    cleaned = normalize_inline_spacing(value)
    cleaned = re.sub(r"\s+US[-–—]?\s*$", " USING", cleaned)
    cleaned = cleaned.replace("US- ING", "USING")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—;,.:")
    if not cleaned or _TITLE_SKIP_RE.search(cleaned):
        return None
    if len(cleaned) < 8 or len(cleaned) > 220:
        return None
    return cleaned


def _extract_abstract(first_pages_text: str, body_text_sample: str) -> str | None:
    patterns = [
        r"(?:\(57\)\s*)?Abstract:?\s*(.+?)(?:\n\s*(?:Description|Technical\s+Field|Field\s+of\s+the\s+Invention|Background|\[?0001\]?|Claims?\b))",
        r"\bABSTRACT\b\s*(.+?)(?:\n\s*(?:FIELD|TECHNICAL FIELD|BACKGROUND|DESCRIPTION|\[?0001\]?))",
    ]
    for corpus in (first_pages_text, body_text_sample):
        for pattern in patterns:
            match = re.search(pattern, corpus, flags=re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = clean_abstract_text(match.group(1))
                if cleaned:
                    return cleaned
    return None


def _extract_raw_field(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    value = join_wrapped_lines(match.group(1).splitlines())
    value = re.sub(r"\s{2,}", " ", value).strip(" ;,")
    return value[:2_500] if value else None


def _truncate_at_cover_noise(value: str | None) -> str | None:
    if not value:
        return None
    match = _COVER_NOISE_RE.search(value)
    if match:
        value = value[: match.start()]
    return value


def _clean_date(value: str | None) -> str | None:
    if not value:
        return None
    match = _DATE_RE.search(value)
    return match.group(0) if match else None


def _clean_application_number(raw_value: str | None, text: str) -> str | None:
    corpus = " ".join(value for value in (raw_value, text[:3_000]) if value)
    match = _APPLICATION_RE.search(corpus)
    return match.group(0).upper() if match else None


def _extract_classifications(raw_classification: str | None, text: str) -> list[str]:
    candidates: list[str] = []
    if raw_classification:
        candidates.extend(_CLASSIFICATION_CODE_RE.findall(raw_classification))
    candidates.extend(_CLASSIFICATION_CODE_RE.findall(text[:5_000]))
    return _deduplicate([_normalize_classification(value) for value in candidates])


def _normalize_classification(value: str) -> str:
    cleaned = normalize_inline_spacing(value).upper()
    cleaned = cleaned.replace("CO8", "C08").replace("C09T", "C09J")
    cleaned = cleaned.replace("C09F", "C09J") if cleaned.startswith("C09F") else cleaned
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_inline_spacing(value)
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
