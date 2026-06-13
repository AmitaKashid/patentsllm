"""Patent-aware chunking.

This chunker is deliberately conservative:
- claims are extracted through a dedicated defensive extractor;
- metadata is kept as a non-embeddable filter chunk unless explicitly used later;
- table/figure-heavy OCR text is flagged so it does not silently pollute RAG evidence;
- legal/semantic boundaries use zero overlap; overlap is used only for forced splits.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from patentllm.parsing.claim_extraction import ClaimExtractionResult, ExtractedClaim, extract_claims_from_paragraphs
from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import ParagraphRecord, PatentChunk, PatentMetadata
from patentllm.parsing.section_detection import is_evidence_section
from patentllm.parsing.text_cleaning import estimate_tokens, normalize_inline_spacing
from patentllm.parsing.text_quality import assess_text_quality

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9\(])")
_EMBEDDABLE_CHILD_TYPES = {
    "claim",
    "claim_clause_window",
    "description_paragraph_window",
    "description_paragraph_window_forced_split",
    "evidence_paragraph_window",
    "evidence_paragraph_window_forced_split",
    "title_abstract_claims",
}
_NON_EMBEDDABLE_SECTIONS = {"front_page", "search_report", "metadata"}


class PatentChunker:
    """Build embedding-ready chunks from parsed patent paragraphs."""

    def __init__(self, config: PatentParserConfig) -> None:
        self.config = config
        self.last_claim_extraction: ClaimExtractionResult | None = None

    def build_chunks(
        self,
        metadata: PatentMetadata,
        paragraphs: list[ParagraphRecord],
    ) -> list[PatentChunk]:
        chunks: list[PatentChunk] = [self._metadata_chunk(metadata)]

        claim_chunks = self._claim_chunks(metadata, paragraphs)
        tac_chunk = self._title_abstract_claims_chunk(metadata, claim_chunks)
        if tac_chunk:
            chunks.append(tac_chunk)

        chunks.extend(claim_chunks)

        description_chunks = self._paragraph_window_chunks(metadata, paragraphs)
        chunks.extend(description_chunks)

        chunks.extend(self._parent_chunks(metadata, claim_chunks + description_chunks))
        return chunks

    def _metadata_chunk(self, metadata: PatentMetadata) -> PatentChunk:
        lines = [
            "Patent metadata (not primary evidence)",
            f"Publication number: {metadata.publication_number or 'unknown'}",
            f"Kind code: {metadata.kind_code or 'unknown'}",
            f"Title: {metadata.title or 'unknown'}",
            f"Application number: {metadata.application_number or 'unknown'}",
            f"Publication date: {metadata.publication_date or 'unknown'}",
            f"Filing date: {metadata.filing_date or 'unknown'}",
            f"Priority date: {metadata.priority_date or 'unknown'}",
            f"Classifications: {', '.join(metadata.ipc_cpc_classifications) if metadata.ipc_cpc_classifications else 'unknown'}",
        ]
        if metadata.applicant:
            lines.append(f"Applicant: {metadata.applicant}")
        if metadata.inventors:
            lines.append(f"Inventors: {metadata.inventors}")
        if metadata.abstract:
            lines.append(f"Abstract: {metadata.abstract}")
        if metadata.cited_patent_literature:
            lines.append("Cited patent literature: " + "; ".join(metadata.cited_patent_literature[:20]))
        if metadata.detected_tables:
            lines.append("Detected tables: " + "; ".join(metadata.detected_tables[:20]))
        if metadata.detected_figures:
            lines.append("Detected figures: " + "; ".join(metadata.detected_figures[:20]))

        return self._make_chunk(
            metadata=metadata,
            section="metadata",
            chunk_type="metadata",
            retrieval_tier="metadata",
            text="\n".join(lines),
            page_start=1,
            page_end=1,
            paragraph_ids=[],
            source_page_numbers=[1],
            extra_metadata={
                "overlap_tokens": 0,
                "embeddable_reason": "Metadata is a filter/display record, not primary technical evidence.",
                "metadata_warnings": metadata.metadata_warnings,
            },
            force_embeddable=False,
        )

    def _title_abstract_claims_chunk(
        self,
        metadata: PatentMetadata,
        claim_chunks: list[PatentChunk],
    ) -> PatentChunk | None:
        parts = [f"Title: {metadata.title}" if metadata.title else ""]
        if metadata.abstract:
            parts.append(f"Abstract: {metadata.abstract}")

        # Prefer independent claim 1 and then early claims. Avoid noisy low-confidence claim chunks.
        selected_claims = [
            chunk
            for chunk in claim_chunks
            if chunk.metadata.get("claim_confidence") in {"high", "medium"}
            and chunk.metadata.get("claim_window_index", 0) == 0
        ]
        selected_claims.sort(key=lambda chunk: int(chunk.metadata.get("claim_number") or 999))

        included = 0
        for claim_chunk in selected_claims:
            claim_number = claim_chunk.metadata.get("claim_number")
            label = f"Claim {claim_number}" if claim_number is not None else "Claim evidence"
            candidate = "\n\n".join(part for part in parts + [f"{label}: {claim_chunk.text}"] if part.strip())
            if estimate_tokens(candidate) > self.config.tac_max_tokens:
                break
            parts.append(f"{label}: {claim_chunk.text}")
            included += 1

        text = "\n\n".join(part for part in parts if part.strip())
        if not text.strip():
            return None

        pages = sorted({1, *[p for c in selected_claims[:included] for p in c.source_page_numbers]})
        return self._make_chunk(
            metadata=metadata,
            section="title_abstract_claims",
            chunk_type="title_abstract_claims",
            retrieval_tier="document_view",
            text=text,
            page_start=min(pages) if pages else 1,
            page_end=max(pages) if pages else 1,
            paragraph_ids=[pid for c in selected_claims[:included] for pid in c.paragraph_ids],
            source_page_numbers=pages or [1],
            extra_metadata={"overlap_tokens": 0, "claim_count_included": included},
        )

    def _claim_chunks(
        self,
        metadata: PatentMetadata,
        paragraphs: list[ParagraphRecord],
    ) -> list[PatentChunk]:
        claim_paragraphs = [p for p in paragraphs if p.section == "claims"]
        extraction = extract_claims_from_paragraphs(claim_paragraphs)
        self.last_claim_extraction = extraction
        if not extraction.claims:
            return []

        chunks: list[PatentChunk] = []
        for claim in extraction.claims:
            chunks.extend(self._chunks_for_claim(metadata, claim))
        return chunks

    def _chunks_for_claim(self, metadata: PatentMetadata, claim: ExtractedClaim) -> list[PatentChunk]:
        pages = sorted(set(claim.page_numbers)) or [None]
        extra = {
            "claim_number": claim.claim_number,
            "claim_confidence": claim.confidence,
            "claim_warnings": claim.warnings,
            "overlap_tokens": 0,
        }

        if estimate_tokens(claim.text) <= self.config.claim_max_tokens:
            return [
                self._make_chunk(
                    metadata=metadata,
                    section="claims",
                    chunk_type="claim",
                    retrieval_tier="child",
                    text=claim.text,
                    page_start=min(p for p in pages if p is not None),
                    page_end=max(p for p in pages if p is not None),
                    paragraph_ids=claim.paragraph_ids,
                    source_page_numbers=[p for p in pages if p is not None],
                    extra_metadata=extra,
                )
            ]

        windows = self._forced_text_windows(
            metadata=metadata,
            section="claims",
            chunk_type="claim_clause_window",
            retrieval_tier="child",
            text=claim.text,
            max_tokens=self.config.claim_max_tokens,
            overlap_tokens=self.config.forced_split_overlap_tokens,
            paragraph_ids=claim.paragraph_ids,
            source_page_numbers=[p for p in pages if p is not None],
            extra_metadata=extra,
        )
        return [
            self._replace_chunk_metadata(
                chunk,
                {
                    **chunk.metadata,
                    "claim_window_index": chunk.metadata.get("window_index", 0),
                    "claim_number": claim.claim_number,
                    "claim_confidence": claim.confidence,
                    "claim_warnings": claim.warnings,
                },
            )
            for chunk in windows
        ]

    def _paragraph_window_chunks(
        self,
        metadata: PatentMetadata,
        paragraphs: list[ParagraphRecord],
    ) -> list[PatentChunk]:
        eligible = [p for p in paragraphs if p.section != "claims"]
        grouped: dict[str, list[ParagraphRecord]] = defaultdict(list)
        for paragraph in eligible:
            grouped[paragraph.section].append(paragraph)

        chunks: list[PatentChunk] = []
        for section, section_paragraphs in grouped.items():
            evidence = is_evidence_section(section)
            target_tokens = self.config.evidence_target_tokens if evidence else self.config.description_target_tokens
            max_tokens = self.config.evidence_max_tokens if evidence else self.config.description_max_tokens
            chunk_type = "evidence_paragraph_window" if evidence else "description_paragraph_window"
            chunks.extend(
                self._merge_paragraph_windows(
                    metadata=metadata,
                    section=section,
                    paragraphs=section_paragraphs,
                    chunk_type=chunk_type,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                )
            )
        return chunks

    def _merge_paragraph_windows(
        self,
        metadata: PatentMetadata,
        section: str,
        paragraphs: list[ParagraphRecord],
        chunk_type: str,
        target_tokens: int,
        max_tokens: int,
    ) -> list[PatentChunk]:
        chunks: list[PatentChunk] = []
        buffer: list[ParagraphRecord] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            chunks.append(self._chunk_from_paragraphs(metadata, section, chunk_type, buffer, "child"))
            buffer = []
            buffer_tokens = 0

        for paragraph in paragraphs:
            if paragraph.token_estimate > max_tokens:
                flush()
                chunks.extend(
                    self._forced_text_windows(
                        metadata=metadata,
                        section=section,
                        chunk_type=f"{chunk_type}_forced_split",
                        retrieval_tier="child",
                        text=paragraph.text,
                        max_tokens=max_tokens,
                        overlap_tokens=self.config.forced_split_overlap_tokens,
                        paragraph_ids=[paragraph.paragraph_id],
                        source_page_numbers=[paragraph.page_number],
                        extra_metadata={"original_paragraph_id": paragraph.paragraph_id},
                    )
                )
                continue

            if buffer and (buffer_tokens + paragraph.token_estimate > max_tokens or buffer_tokens >= target_tokens):
                flush()

            buffer.append(paragraph)
            buffer_tokens += paragraph.token_estimate

        flush()
        return chunks

    def _chunk_from_paragraphs(
        self,
        metadata: PatentMetadata,
        section: str,
        chunk_type: str,
        paragraphs: list[ParagraphRecord],
        retrieval_tier: str,
    ) -> PatentChunk:
        text = "\n\n".join(p.text for p in paragraphs)
        pages = [p.page_number for p in paragraphs]
        return self._make_chunk(
            metadata=metadata,
            section=section,
            chunk_type=chunk_type,
            retrieval_tier=retrieval_tier,
            text=text,
            page_start=min(pages),
            page_end=max(pages),
            paragraph_ids=[p.paragraph_id for p in paragraphs],
            source_page_numbers=sorted(set(pages)),
            extra_metadata={"overlap_tokens": 0},
        )

    def _parent_chunks(
        self,
        metadata: PatentMetadata,
        child_chunks: list[PatentChunk],
    ) -> list[PatentChunk]:
        grouped: dict[str, list[PatentChunk]] = defaultdict(list)
        for chunk in child_chunks:
            if chunk.retrieval_tier == "child" and chunk.embeddable:
                grouped[chunk.section].append(chunk)

        parents: list[PatentChunk] = []
        for section, chunks in grouped.items():
            buffer: list[PatentChunk] = []
            buffer_tokens = 0

            def flush() -> None:
                nonlocal buffer, buffer_tokens
                if not buffer:
                    return
                text = "\n\n".join(c.text for c in buffer)
                pages = [p for c in buffer for p in c.source_page_numbers]
                parents.append(
                    self._make_chunk(
                        metadata=metadata,
                        section=section,
                        chunk_type="parent_section_window",
                        retrieval_tier="parent",
                        text=text,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        paragraph_ids=[pid for c in buffer for pid in c.paragraph_ids],
                        source_page_numbers=sorted(set(pages)),
                        extra_metadata={
                            "child_chunk_ids": [c.chunk_id for c in buffer],
                            "overlap_tokens": 0,
                        },
                    )
                )
                buffer = []
                buffer_tokens = 0

            for chunk in chunks:
                if buffer and buffer_tokens + chunk.token_estimate > self.config.parent_max_tokens:
                    flush()
                buffer.append(chunk)
                buffer_tokens += chunk.token_estimate
                if buffer_tokens >= self.config.parent_target_tokens:
                    flush()
            flush()

        return parents

    def _forced_text_windows(
        self,
        metadata: PatentMetadata,
        section: str,
        chunk_type: str,
        retrieval_tier: str,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
        paragraph_ids: list[str],
        source_page_numbers: list[int],
        extra_metadata: dict[str, object] | None = None,
    ) -> list[PatentChunk]:
        units = _split_into_sentence_like_units(text)
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = estimate_tokens(unit)
            if current and current_tokens + unit_tokens > max_tokens:
                windows.append(" ".join(current).strip())
                current = _tail_overlap(current, overlap_tokens)
                current_tokens = estimate_tokens(" ".join(current))
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            windows.append(" ".join(current).strip())

        chunks: list[PatentChunk] = []
        for index, window in enumerate(windows):
            metadata_extra = dict(extra_metadata or {})
            metadata_extra.update({"window_index": index, "overlap_tokens": overlap_tokens})
            chunks.append(
                self._make_chunk(
                    metadata=metadata,
                    section=section,
                    chunk_type=chunk_type,
                    retrieval_tier=retrieval_tier,
                    text=window,
                    page_start=min(source_page_numbers) if source_page_numbers else None,
                    page_end=max(source_page_numbers) if source_page_numbers else None,
                    paragraph_ids=paragraph_ids,
                    source_page_numbers=sorted(set(source_page_numbers)),
                    extra_metadata=metadata_extra,
                )
            )
        return chunks

    def _make_chunk(
        self,
        metadata: PatentMetadata,
        section: str,
        chunk_type: str,
        retrieval_tier: str,
        text: str,
        page_start: int | None,
        page_end: int | None,
        paragraph_ids: list[str],
        source_page_numbers: list[int],
        extra_metadata: dict[str, object] | None = None,
        force_embeddable: bool | None = None,
    ) -> PatentChunk:
        clean_text = normalize_inline_spacing(text.replace("\n", " \n ")).replace(" \n ", "\n")
        quality = assess_text_quality(clean_text)
        quality_flags = quality.flags()

        embeddable = chunk_type in _EMBEDDABLE_CHILD_TYPES and section not in _NON_EMBEDDABLE_SECTIONS
        if quality.is_suspicious_ocr and quality.table_detected:
            embeddable = False
            quality_flags.append("excluded_from_embedding_due_to_table_ocr_noise")
        if chunk_type == "metadata":
            embeddable = False
        if force_embeddable is not None:
            embeddable = force_embeddable

        extra = dict(extra_metadata or {})
        extra.update(
            {
                "weird_character_ratio": round(quality.weird_character_ratio, 4),
                "table_detected": quality.table_detected,
                "figure_detected": quality.figure_detected,
                "quality_flags": quality_flags,
            }
        )

        chunk_id = _stable_chunk_id(metadata.document_id, section, chunk_type, clean_text, page_start, page_end)
        return PatentChunk(
            chunk_id=chunk_id,
            document_id=metadata.document_id,
            source_file=metadata.source_file,
            publication_number=metadata.publication_number,
            title=metadata.title,
            section=section,
            chunk_type=chunk_type,
            retrieval_tier=retrieval_tier,
            page_start=page_start,
            page_end=page_end,
            text=clean_text,
            token_estimate=estimate_tokens(clean_text),
            paragraph_ids=paragraph_ids,
            source_page_numbers=source_page_numbers,
            metadata=extra,
            embeddable=embeddable,
            quality_flags=quality_flags,
        )

    @staticmethod
    def _replace_chunk_metadata(chunk: PatentChunk, metadata: dict[str, object]) -> PatentChunk:
        return PatentChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            publication_number=chunk.publication_number,
            title=chunk.title,
            section=chunk.section,
            chunk_type=chunk.chunk_type,
            retrieval_tier=chunk.retrieval_tier,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            token_estimate=chunk.token_estimate,
            paragraph_ids=chunk.paragraph_ids,
            source_page_numbers=chunk.source_page_numbers,
            metadata=metadata,
            embeddable=chunk.embeddable,
            quality_flags=chunk.quality_flags,
        )


def _split_into_sentence_like_units(text: str) -> list[str]:
    units = [unit.strip() for unit in _SENTENCE_SPLIT_RE.split(text) if unit.strip()]
    if len(units) <= 1:
        words = text.split()
        return [" ".join(words[index : index + 120]) for index in range(0, len(words), 120)]
    return units


def _tail_overlap(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    reversed_tail: list[str] = []
    total = 0
    for unit in reversed(units):
        reversed_tail.append(unit)
        total += estimate_tokens(unit)
        if total >= overlap_tokens:
            break
    return list(reversed(reversed_tail))


def _stable_chunk_id(
    document_id: str,
    section: str,
    chunk_type: str,
    text: str,
    page_start: int | None,
    page_end: int | None,
) -> str:
    payload = f"{document_id}|{section}|{chunk_type}|{page_start}|{page_end}|{text}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{document_id}:{chunk_type}:{section}:p{page_start or 'x'}-{page_end or 'x'}:{digest}"
