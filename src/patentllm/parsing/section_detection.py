"""Section detection for patent documents.

The detector is intentionally conservative. OCR patent PDFs often produce
short fragments, search-report labels, and sentence-like lines that can look
like headings. For RAG, a wrong section label is worse than a missed subsection,
because section labels are later used as retrieval metadata.
"""

from __future__ import annotations

import re

from patentllm.parsing.text_cleaning import canonical_section_name, normalize_inline_spacing


KNOWN_SECTION_ALIASES: dict[str, str] = {
    # Front matter / abstract
    "abstract": "abstract",
    "57_abstract": "abstract",

    # Main patent body
    "description": "description",
    "title_of_invention": "title",
    "field_of_invention": "technical_field",
    "technical_field": "technical_field",
    "technical_background": "background",
    "background": "background",
    "background_art": "background",
    "background_of_the_invention": "background",

    # Summary / disclosure
    "summary": "summary",
    "summary_of_the_invention": "summary",
    "brief_summary_of_the_invention": "summary",
    "disclosure_of_the_invention": "summary",
    "problems_to_be_solved_by_the_invention": "summary",
    "means_for_solving_the_problems": "summary",
    "solution_to_problem": "summary",
    "technical_problem": "summary",
    "advantageous_effects_of_invention": "summary",
    "effects_of_the_invention": "summary",

    # Drawings
    "brief_description_of_the_drawings": "brief_description_of_drawings",
    "brief_description_of_drawings": "brief_description_of_drawings",
    "brief_description_of_the_drawing": "brief_description_of_drawings",
    "brief_description_of_figures": "brief_description_of_drawings",
    "brief_description_of_the_figures": "brief_description_of_drawings",
    "brief_description_of_the_figure": "brief_description_of_drawings",

    # Detailed description
    "detailed_description": "detailed_description",
    "detailed_description_of_the_invention": "detailed_description",
    "description_of_embodiments": "detailed_description",
    "description_of_the_invention": "detailed_description",
    "mode_for_carrying_out_the_invention": "detailed_description",
    "best_mode_for_carrying_out_the_invention": "detailed_description",
    "embodiments": "detailed_description",

    # Examples / experiments
    "examples": "examples",
    "example": "examples",
    "experimental_part": "experimental_part",
    "experiments": "experimental_part",
    "test_methods": "experimental_part",
    "evaluation": "experimental_part",
    "preparation_of_samples": "experimental_part",
    "preparation_of_the_hot_melt_adhesive_compositions": "experimental_part",

    # Claims
    "claims": "claims",
    "claim": "claims",
    "we_claim": "claims",
    "i_we_claim": "claims",
    "what_is_claimed": "claims",

    # Citations / search report
    "citation_list": "citation_list",
    "patent_literature": "citation_list",
    "non_patent_literature": "citation_list",
    "references_cited": "citation_list",
    "international_search_report": "search_report",
    "international_preliminary_report_on_patentability": "search_report",
    "written_opinion_of_the_international_searching_authority": "search_report",
}


CLAIM_LIKE_RE = re.compile(r"^\s*\d{1,3}\s*[\.)]\s+")
PARAGRAPH_MARKER_RE = re.compile(r"^\s*\[\d{4}\]")

CLAIM_HEADING_RE = re.compile(
    r"\b(?:CLAIMS?|WE CLAIM|I/WE CLAIM|WHAT IS CLAIMED)\b\s*:?",
    flags=re.IGNORECASE,
)

SEARCH_REPORT_RE = re.compile(
    r"^[A-D]\.\s*(CLASSIFICATION OF SUBJECT MATTER|FIELDS SEARCHED|DOCUMENTS CONSIDERED TO BE RELEVANT|FURTHER DOCUMENTS)$",
    flags=re.IGNORECASE,
)


def detect_section_heading(line: str) -> str | None:
    """Return a canonical section name if a line is a reliable patent heading."""

    candidate = normalize_inline_spacing(line).strip(" :.-")
    if not candidate:
        return None

    if PARAGRAPH_MARKER_RE.match(candidate):
        return None

    if CLAIM_LIKE_RE.match(candidate):
        return None

    # Claim headings can appear as "CLAIMS", "We claim:", "I/We claim:",
    # or "What is Claimed:" depending on jurisdiction and OCR layout.
    if CLAIM_HEADING_RE.search(candidate):
        return "claims"

    if len(candidate) > 90:
        return None

    canonical = canonical_section_name(candidate)

    if canonical in KNOWN_SECTION_ALIASES:
        return KNOWN_SECTION_ALIASES[canonical]

    if SEARCH_REPORT_RE.match(candidate):
        return "search_report"

    return None


def is_evidence_section(section: str) -> bool:
    """Return True for sections that usually contain concrete evidence."""

    normalized = canonical_section_name(section)
    return normalized in {
        "examples",
        "experimental_part",
        "brief_description_of_drawings",
        "citation_list",
        "search_report",
    }