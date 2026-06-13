"""Heuristic metadata extraction for PCT-style patent PDFs."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from patentllm.parsing.models import PageRecord, PatentMetadata
from patentllm.parsing.text_cleaning import join_wrapped_lines, normalize_inline_spacing

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
    "application_number": re.compile(r"International\s+Application\s+Number\s+([^\n]+)", re.IGNORECASE),
    "filing_date": re.compile(r"International\s+Filing\s+Date\s+([^\n]+)", re.IGNORECASE),
    "priority_date": re.compile(r"Priority\s+Data\s+([^\n]+)", re.IGNORECASE),
    "applicant": re.compile(r"\(71\)\s*Applicant[s]?:?\s*([^\n]+(?:\n(?!\(\d{2}\)).+)*)", re.IGNORECASE),
    "inventors": re.compile(r"\(72\)\s*Inventor[s]?:?\s*([^\n]+(?:\n(?!\(\d{2}\)).+)*)", re.IGNORECASE),
    "classification": re.compile(r"\(51\)\s*International\s+Patent\s+Classification:?\s*([^\n]+(?:\n(?!\(\d{2}\)).+)*)", re.IGNORECASE),
}
_PATENT_CITATION_RE = re.compile(
    r"\b(?:WO\s*\d{4}/\d{4,}|US\s*(?:Patent\s*)?(?:Application\s*)?(?:No\.\s*)?\d[\d,./]+|EP\s*\d{6,}|JP\s*\d{4}-\d{4,})\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(r"\bTable\s+\d+\b", re.IGNORECASE)
_FIGURE_RE = re.compile(r"\b(?:Figure|FIG\.)\s+\d+[A-Z]?\b", re.IGNORECASE)
_CLASSIFICATION_CODE_RE = re.compile(r"\b[A-H][0-9]{2}[A-Z]\s*\d{1,3}/\d{1,6}\b")


def extract_metadata(pdf_path: Path, document_id: str, pages: list[PageRecord]) -> PatentMetadata:
    """Extract lightweight metadata from filename, cover page, and early body pages."""

    source_file = pdf_path.name
    page_count = _pdf_page_count(pdf_path)
    first_pages_text = "\n".join(page.text for page in pages[:4])
    all_text_sample = "\n".join(page.text for page in pages[: min(20, len(pages))])

    publication_number, kind_code = _extract_publication_number(source_file, first_pages_text)
    title = _extract_title(first_pages_text, source_file)
    abstract = _extract_abstract(first_pages_text)
    field_values = {key: _extract_field(pattern, first_pages_text) for key, pattern in _FIELD_PATTERNS.items()}

    classifications = _extract_classifications(field_values.get("classification"), first_pages_text)
    citations = _deduplicate(_PATENT_CITATION_RE.findall(all_text_sample))[:100]
    tables = _deduplicate(_TABLE_RE.findall(all_text_sample))[:100]
    figures = _deduplicate(_FIGURE_RE.findall(all_text_sample))[:100]

    return PatentMetadata(
        document_id=document_id,
        source_file=source_file,
        page_count=page_count,
        publication_number=publication_number,
        kind_code=kind_code,
        title=title,
        abstract=abstract,
        applicant=field_values.get("applicant"),
        inventors=field_values.get("inventors"),
        publication_date=field_values.get("publication_date"),
        application_number=field_values.get("application_number"),
        filing_date=field_values.get("filing_date"),
        priority_date=field_values.get("priority_date"),
        ipc_cpc_classifications=classifications,
        cited_patent_literature=citations,
        detected_tables=tables,
        detected_figures=figures,
        raw_cover_fields={key: value for key, value in field_values.items() if value},
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
    title_marker = re.search(r"\(54\)\s*Title:?[\s\n]*(.+)", text, flags=re.IGNORECASE)
    if title_marker:
        return normalize_inline_spacing(title_marker.group(1))[:300]

    title_of_invention = re.search(
        r"Title\s+of\s+Invention\s+(.+?)(?:\n\s*(?:Technical\s+Field|Background|\[0001\]))",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if title_of_invention:
        return normalize_inline_spacing(title_of_invention.group(1))[:300]

    # Early body pages usually repeat title in uppercase before TECHNICAL FIELD.
    for line in text.splitlines()[:80]:
        candidate = normalize_inline_spacing(line)
        if 8 <= len(candidate) <= 180 and candidate.isupper():
            if not any(skip in candidate for skip in ("WIPO", "PCT", "INTERNATIONAL", "PUBLICATION")):
                return candidate

    return Path(source_file).stem.replace("_", " ")


def _extract_abstract(text: str) -> str | None:
    match = re.search(
        r"(?:\(57\)\s*)?Abstract:?\s*(.+?)(?:\n\s*(?:Description|Technical\s+Field|Field\s+of\s+the\s+Invention|Background|\[0001\]))",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        abstract = normalize_inline_spacing(match.group(1))
        return abstract[:2_000] if abstract else None
    return None


def _extract_field(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    value = join_wrapped_lines(match.group(1).splitlines())
    value = re.sub(r"\s{2,}", " ", value).strip(" ;,")
    return value[:2_000] if value else None


def _extract_classifications(raw_classification: str | None, text: str) -> list[str]:
    candidates = []
    if raw_classification:
        candidates.extend(_CLASSIFICATION_CODE_RE.findall(raw_classification))
    candidates.extend(_CLASSIFICATION_CODE_RE.findall(text[:4_000]))
    return _deduplicate(candidates)


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
