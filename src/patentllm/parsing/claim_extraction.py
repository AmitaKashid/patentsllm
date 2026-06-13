"""Robust patent claim extraction for OCR-heavy patent PDFs.

This module is deliberately defensive. It does not try to interpret legal scope; it
tries to preserve claim boundaries without letting cover-page text, description
paragraphs, table rows, or international-search-report references become claim chunks.

Design principles:
- Prefer explicit numbered claims, e.g. ``1. A composition ...``.
- Remove the International Search Report only after the real claim block has started.
- If OCR dropped a short ``1.`` line, infer a partial claim 1 from the paragraphs
  immediately before the first numbered dependent claim and mark it as low confidence.
- If OCR dropped all claim numbers, create a low-confidence claim block only from the
  late-document claim pages, not from the whole description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patentllm.parsing.models import ParagraphRecord
from patentllm.parsing.text_cleaning import normalize_inline_spacing

_CLAIM_START_RE = re.compile(r"(?m)^\s*(?P<number>\d{1,3})\s*[\.)]\s+(?P<body>.+?)\s*$")
_SEARCH_REPORT_RE = re.compile(
    r"\bINTERNATIONAL\s+SEARCH\s+REPORT\b|\bInternational\s+Search\s+Report\b|\bForm\s+PCT/ISA/210\b",
    re.IGNORECASE,
)
_CLAIM_LANGUAGE_RE = re.compile(
    r"\b(?:claim|claims|comprising|comprises|wherein|according\s+to|method|use|composition|adhesive|laminate|polymer|system|article|substrate|process|homopolymer|copolymer|propylene|ethylene|disposable|product|products|consists?\s+of)\b",
    re.IGNORECASE,
)
_DEPENDENT_CLAIM_RE = re.compile(
    r"\b(?:according\s+to|of|as\s+defined\s+in|wherein|where)\s+(?:any\s+one\s+of\s+)?claims?\s*\d+|\bclaims?\s*\d+\b",
    re.IGNORECASE,
)
_INDEPENDENT_START_RE = re.compile(
    r"^\s*(?:A|An|The|Use\s+of|A\s+method|A\s+composition|A\s+paper\s+product|A\s+hot\s+melt|A\s+laminate)\b",
    re.IGNORECASE,
)
_TABLE_OR_SEARCH_NOISE_RE = re.compile(
    r"\b(?:DOCUMENTS\s+CONSIDERED|CLASSIFICATION\s+OF\s+SUBJECT\s+MATTER|Special\s+categories|patent\s+family\s+annex|Form\s+PCT/ISA/210)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """One extracted claim candidate."""

    claim_number: int | None
    text: str
    paragraph_ids: list[str]
    page_numbers: list[int]
    confidence: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ClaimExtractionResult:
    """Claim extraction result and sequence quality information."""

    claims: list[ExtractedClaim]
    detected_numbers: list[int]
    missing_numbers: list[int]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    paragraph: ParagraphRecord


@dataclass(frozen=True, slots=True)
class _ClaimCandidate:
    number: int
    start: int
    end: int
    body: str


def extract_claims_from_paragraphs(
    claim_paragraphs: list[ParagraphRecord],
    *,
    max_claim_number: int = 100,
) -> ClaimExtractionResult:
    """Extract legal-claim chunks from paragraphs assigned to the claims section.

    The input may still contain OCR/layout contamination. This function first locates
    the most plausible numbered claim block, then prunes cover-page/search-report noise.
    """

    if not claim_paragraphs:
        return ClaimExtractionResult(claims=[], detected_numbers=[], missing_numbers=[], warnings=[])

    ordered = sorted(claim_paragraphs, key=lambda p: (p.page_number, p.sequence_index))
    combined, spans = _combine_paragraphs(ordered)
    if not combined.strip():
        return ClaimExtractionResult(claims=[], detected_numbers=[], missing_numbers=[], warnings=[])

    candidates = _candidate_claim_starts(combined, max_claim_number=max_claim_number)
    candidates = [c for c in candidates if _candidate_has_claim_language(combined, c)]

    if not candidates:
        return _fallback_unnumbered_claim_block(ordered)

    first_index, warnings = _select_first_real_claim_candidate(candidates)
    first_candidate = candidates[first_index]

    # If OCR dropped a short independent claim 1 line, synthesize a low-confidence
    # partial claim 1 from the immediately preceding claim-like paragraph fragments.
    inferred_claims: list[ExtractedClaim] = []
    if first_candidate.number > 1:
        inferred = _infer_missing_claim_one(first_candidate, spans)
        if inferred:
            inferred_claims.append(inferred)
            warnings.append(
                "Claim 1 was inferred from text immediately before the first numbered dependent claim."
            )
        else:
            warnings.append("Claim 1 was not found and could not be inferred safely.")

    claim_start = inferred_claims[0].paragraph_ids[0] if inferred_claims else None
    working_text, working_spans, offset_shift = _trim_before_candidate_or_inferred(
        combined=combined,
        spans=spans,
        first_candidate=first_candidate,
        inferred_claim=inferred_claims[0] if inferred_claims else None,
    )
    candidates = _candidate_claim_starts(working_text, max_claim_number=max_claim_number)
    candidates = [c for c in candidates if _candidate_has_claim_language(working_text, c)]

    if inferred_claims and candidates and candidates[0].number == 1:
        # Do not double-count if trimming exposed an explicit claim 1.
        inferred_claims = []

    working_text = _remove_search_report_tail_after_claim_start(working_text)
    candidates = _candidate_claim_starts(working_text, max_claim_number=max_claim_number)
    candidates = [c for c in candidates if _candidate_has_claim_language(working_text, c)]

    claims = [*inferred_claims]
    last_number = claims[-1].claim_number if claims and claims[-1].claim_number is not None else None

    for index, candidate in enumerate(candidates):
        next_start = candidates[index + 1].start if index + 1 < len(candidates) else len(working_text)
        text = normalize_inline_spacing(working_text[candidate.start:next_start])

        if last_number is not None and candidate.number <= last_number:
            # OCR sometimes duplicates the preceding claim number inside the same claim.
            # Preserve the text, but do not create a second claim with a repeated number.
            if claims and text:
                previous = claims[-1]
                claims[-1] = ExtractedClaim(
                    claim_number=previous.claim_number,
                    text=normalize_inline_spacing(previous.text + " " + text),
                    paragraph_ids=previous.paragraph_ids,
                    page_numbers=previous.page_numbers,
                    confidence="medium" if previous.confidence == "high" else previous.confidence,
                    warnings=[*previous.warnings, f"Merged repeated/non-increasing claim candidate {candidate.number} into previous claim."],
                )
            continue

        if not _is_claim_text(text):
            continue

        confidence = "high"
        item_warnings: list[str] = []
        if last_number is not None and candidate.number > last_number + 1:
            confidence = "medium"
            item_warnings.append(f"Gap before claim {candidate.number}; expected {last_number + 1}.")

        paragraph_ids, page_numbers = _paragraphs_for_span(working_spans, candidate.start, next_start)
        claims.append(
            ExtractedClaim(
                claim_number=candidate.number,
                text=text,
                paragraph_ids=paragraph_ids,
                page_numbers=page_numbers,
                confidence=confidence,
                warnings=item_warnings,
            )
        )
        last_number = candidate.number

    detected_numbers = [c.claim_number for c in claims if c.claim_number is not None]
    missing_numbers = _missing_numbers(detected_numbers)
    if missing_numbers:
        warnings.append(f"Missing claim numbers after extraction: {missing_numbers}")

    return ClaimExtractionResult(
        claims=claims,
        detected_numbers=detected_numbers,
        missing_numbers=missing_numbers,
        warnings=warnings,
    )


def _combine_paragraphs(paragraphs: list[ParagraphRecord]) -> tuple[str, list[_Span]]:
    parts: list[str] = []
    spans: list[_Span] = []
    offset = 0
    for paragraph in paragraphs:
        text = normalize_inline_spacing(paragraph.text)
        if not text:
            continue
        if parts:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(text)
        offset += len(text)
        spans.append(_Span(start=start, end=offset, paragraph=paragraph))
    return "".join(parts), spans


def _candidate_claim_starts(text: str, *, max_claim_number: int) -> list[_ClaimCandidate]:
    matches = list(_CLAIM_START_RE.finditer(text))
    candidates: list[_ClaimCandidate] = []
    for index, match in enumerate(matches):
        number = int(match.group("number"))
        if not 1 <= number <= max_claim_number:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        candidates.append(
            _ClaimCandidate(
                number=number,
                start=match.start(),
                end=end,
                body=match.group("body"),
            )
        )
    return candidates


def _candidate_has_claim_language(text: str, candidate: _ClaimCandidate) -> bool:
    sample = normalize_inline_spacing(text[candidate.start : min(candidate.start + 800, len(text))])
    if _TABLE_OR_SEARCH_NOISE_RE.search(sample[:300]):
        return False
    if candidate.number == 1:
        return bool(_INDEPENDENT_START_RE.search(candidate.body) or _CLAIM_LANGUAGE_RE.search(sample))
    return bool(_DEPENDENT_CLAIM_RE.search(sample) or _CLAIM_LANGUAGE_RE.search(sample))


def _select_first_real_claim_candidate(candidates: list[_ClaimCandidate]) -> tuple[int, list[str]]:
    warnings: list[str] = []
    for index, candidate in enumerate(candidates):
        if candidate.number == 1:
            if index:
                warnings.append(f"Discarded {index} numbered candidates before claim 1 as pre-claim noise.")
            return index, warnings
    # No explicit claim 1. Use the earliest small claim number. Later logic may infer claim 1.
    best_index = min(range(len(candidates)), key=lambda i: (candidates[i].number, candidates[i].start))
    if best_index:
        warnings.append(f"Discarded {best_index} candidates before the first plausible claim candidate.")
    return best_index, warnings


def _infer_missing_claim_one(first_candidate: _ClaimCandidate, spans: list[_Span]) -> ExtractedClaim | None:
    candidate_span_index = next((i for i, s in enumerate(spans) if s.start <= first_candidate.start <= s.end), None)
    if candidate_span_index is None:
        return None

    candidate_page = spans[candidate_span_index].paragraph.page_number
    selected: list[_Span] = []
    for span in reversed(spans[:candidate_span_index]):
        paragraph = span.paragraph
        if paragraph.page_number < candidate_page - 1:
            break
        text = normalize_inline_spacing(paragraph.text)
        if _SEARCH_REPORT_RE.search(text) or _TABLE_OR_SEARCH_NOISE_RE.search(text):
            break
        if len(text) < 2:
            continue
        selected.append(span)
        # Two to six preceding fragments are usually enough for OCR-dropped claim 1 starts.
        if len(selected) >= 6:
            break

    selected = list(reversed(selected))
    text = normalize_inline_spacing(" ".join(s.paragraph.text for s in selected))
    if not text or not _CLAIM_LANGUAGE_RE.search(text):
        return None

    return ExtractedClaim(
        claim_number=1,
        text=text,
        paragraph_ids=[s.paragraph.paragraph_id for s in selected],
        page_numbers=sorted({s.paragraph.page_number for s in selected}),
        confidence="low",
        warnings=["Claim number was missing in OCR; inferred as partial claim 1 from preceding fragments."],
    )


def _trim_before_candidate_or_inferred(
    *,
    combined: str,
    spans: list[_Span],
    first_candidate: _ClaimCandidate,
    inferred_claim: ExtractedClaim | None,
) -> tuple[str, list[_Span], int]:
    if inferred_claim and inferred_claim.paragraph_ids:
        first_span = next((s for s in spans if s.paragraph.paragraph_id == inferred_claim.paragraph_ids[0]), None)
        start = first_span.start if first_span else first_candidate.start
    else:
        start = first_candidate.start

    trimmed = combined[start:]
    shifted_spans = [
        _Span(start=max(0, s.start - start), end=max(0, s.end - start), paragraph=s.paragraph)
        for s in spans
        if s.end >= start
    ]
    return trimmed, shifted_spans, start


def _remove_search_report_tail_after_claim_start(text: str) -> str:
    match = _SEARCH_REPORT_RE.search(text)
    if match and match.start() > 250:
        return text[: match.start()]
    return text


def _is_claim_text(text: str) -> bool:
    if not text or len(text) < 20:
        return False
    if _TABLE_OR_SEARCH_NOISE_RE.search(text[:500]):
        return False
    return bool(_CLAIM_LANGUAGE_RE.search(text[:1200]))


def _paragraphs_for_span(spans: list[_Span], start: int, end: int) -> tuple[list[str], list[int]]:
    paragraph_ids: list[str] = []
    page_numbers: list[int] = []
    for span in spans:
        if span.end < start or span.start > end:
            continue
        paragraph_ids.append(span.paragraph.paragraph_id)
        page_numbers.append(span.paragraph.page_number)
    return paragraph_ids, sorted(set(page_numbers))


def _fallback_unnumbered_claim_block(paragraphs: list[ParagraphRecord]) -> ClaimExtractionResult:
    """Create low-confidence claims when OCR removed all claim numbers.

    This is intentionally conservative: only late-document, claim-like text is used.
    """

    pages = [p.page_number for p in paragraphs]
    if not pages:
        return ClaimExtractionResult(claims=[], detected_numbers=[], missing_numbers=[], warnings=[])

    # Search-report pages are excluded. Front/description pages are excluded by taking
    # only the late portion of the section, because claims in PCT publications normally
    # appear near the end before the search report.
    max_page = max(pages)
    min_late_page = max(1, int(max_page * 0.70))
    candidate_paragraphs = [
        p
        for p in paragraphs
        if p.page_number >= min_late_page
        and not _SEARCH_REPORT_RE.search(p.text)
        and not _TABLE_OR_SEARCH_NOISE_RE.search(p.text)
    ]

    if not candidate_paragraphs:
        return ClaimExtractionResult(
            claims=[],
            detected_numbers=[],
            missing_numbers=[],
            warnings=["No numbered claim starts were detected and no safe late claim block was found."],
        )

    # Start at the first independent claim-like phrase in the late block.
    start_index = next(
        (
            i
            for i, p in enumerate(candidate_paragraphs)
            if _INDEPENDENT_START_RE.search(p.text) and _CLAIM_LANGUAGE_RE.search(p.text)
        ),
        None,
    )
    if start_index is None:
        start_index = next((i for i, p in enumerate(candidate_paragraphs) if _CLAIM_LANGUAGE_RE.search(p.text)), None)
    if start_index is None:
        return ClaimExtractionResult(
            claims=[],
            detected_numbers=[],
            missing_numbers=[],
            warnings=["No numbered claim starts were detected and late text did not look like claims."],
        )

    selected = candidate_paragraphs[start_index:]
    text = normalize_inline_spacing(" ".join(p.text for p in selected))
    if not text:
        return ClaimExtractionResult(claims=[], detected_numbers=[], missing_numbers=[], warnings=[])

    claim = ExtractedClaim(
        claim_number=1,
        text=text,
        paragraph_ids=[p.paragraph_id for p in selected],
        page_numbers=sorted({p.page_number for p in selected}),
        confidence="low",
        warnings=["All claim numbers appear to be missing from OCR; emitted one low-confidence claim block."],
    )
    return ClaimExtractionResult(
        claims=[claim],
        detected_numbers=[1],
        missing_numbers=[],
        warnings=["No numbered claim starts were detected; emitted low-confidence unnumbered claim block as claim 1."],
    )


def _missing_numbers(numbers: list[int]) -> list[int]:
    if not numbers:
        return []
    unique = sorted(set(numbers))
    return [value for value in range(1, unique[-1] + 1) if value not in unique]


def claim_sequence_score(result: ClaimExtractionResult) -> float:
    """Return a simple 0-1 sequence-completeness score."""

    if not result.detected_numbers:
        return 0.0
    max_number = max(result.detected_numbers)
    if max_number <= 0:
        return 0.0
    return max(0.0, 1.0 - (len(result.missing_numbers) / max_number))
