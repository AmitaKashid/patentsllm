"""Patent-aware chunking."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from patentllm.parsing.config import PatentParserConfig
from patentllm.parsing.models import ParagraphRecord, PatentChunk, PatentMetadata
from patentllm.parsing.section_detection import is_evidence_section
from patentllm.parsing.text_cleaning import estimate_tokens, normalize_inline_spacing

_CLAIM_START_RE = re.compile(r"^\s*(?P<number>\d{1,3})\s*[\.)]\s+(?P<body>.+)", flags=re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9\(])")


class PatentChunker:
    """Build embedding-ready chunks from parsed patent paragraphs."""

    def __init__(self, config: PatentParserConfig) -> None:
        self.config = config

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
            "Patent metadata",
            f"Publication number: {metadata.publication_number or 'unknown'}",
            f"Kind code: {metadata.kind_code or 'unknown'}",
            f"Title: {metadata.title or 'unknown'}",
            f"Applicant: {metadata.applicant or 'unknown'}",
            f"Inventors: {metadata.inventors or 'unknown'}",
            f"Publication date: {metadata.publication_date or 'unknown'}",
            f"Application number: {metadata.application_number or 'unknown'}",
            f"Filing date: {metadata.filing_date or 'unknown'}",
            f"Priority date: {metadata.priority_date or 'unknown'}",
            f"Classifications: {', '.join(metadata.ipc_cpc_classifications) if metadata.ipc_cpc_classifications else 'unknown'}",
        ]
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
            extra_metadata={"overlap_tokens": 0},
        )

    def _title_abstract_claims_chunk(
        self,
        metadata: PatentMetadata,
        claim_chunks: list[PatentChunk],
    ) -> PatentChunk | None:
        parts = [f"Title: {metadata.title or ''}".strip()]
        if metadata.abstract:
            parts.append(f"Abstract: {metadata.abstract}")

        for claim_chunk in claim_chunks:
            candidate = "\n\n".join(parts + [f"Claim evidence: {claim_chunk.text}"])
            if estimate_tokens(candidate) > self.config.tac_max_tokens:
                break
            parts.append(f"Claim evidence: {claim_chunk.text}")

        text = "\n\n".join(part for part in parts if part.strip())
        if not text.strip():
            return None

        pages = sorted({1, *[p for c in claim_chunks for p in c.source_page_numbers]})
        return self._make_chunk(
            metadata=metadata,
            section="title_abstract_claims",
            chunk_type="title_abstract_claims",
            retrieval_tier="document_view",
            text=text,
            page_start=min(pages) if pages else 1,
            page_end=max(pages) if pages else 1,
            paragraph_ids=[pid for c in claim_chunks for pid in c.paragraph_ids],
            source_page_numbers=pages or [1],
            extra_metadata={"overlap_tokens": 0, "claim_count_included": len(claim_chunks)},
        )

    def _claim_chunks(
        self,
        metadata: PatentMetadata,
        paragraphs: list[ParagraphRecord],
    ) -> list[PatentChunk]:
        claim_paragraphs = [p for p in paragraphs if p.section == "claims"]
        if not claim_paragraphs:
            return []

        claims = self._split_claims(claim_paragraphs)
        chunks: list[PatentChunk] = []

        for claim_number, claim_text, claim_paragraph_ids, pages in claims:
            if estimate_tokens(claim_text) <= self.config.claim_max_tokens:
                chunks.append(
                    self._make_chunk(
                        metadata=metadata,
                        section="claims",
                        chunk_type="claim",
                        retrieval_tier="child",
                        text=claim_text,
                        page_start=min(pages),
                        page_end=max(pages),
                        paragraph_ids=claim_paragraph_ids,
                        source_page_numbers=sorted(set(pages)),
                        extra_metadata={"claim_number": claim_number, "overlap_tokens": 0},
                    )
                )
            else:
                chunks.extend(
                    self._forced_text_windows(
                        metadata=metadata,
                        section="claims",
                        chunk_type="claim_clause_window",
                        retrieval_tier="child",
                        text=claim_text,
                        max_tokens=self.config.claim_max_tokens,
                        overlap_tokens=self.config.forced_split_overlap_tokens,
                        paragraph_ids=claim_paragraph_ids,
                        source_page_numbers=pages,
                        extra_metadata={"claim_number": claim_number},
                    )
                )
        return chunks

    def _split_claims(
        self,
        claim_paragraphs: list[ParagraphRecord],
    ) -> list[tuple[int | None, str, list[str], list[int]]]:
        combined = "\n".join(p.text for p in claim_paragraphs)
        starts = list(re.finditer(r"(?m)^\s*(\d{1,3})\s*[\.)]\s+", combined))

        if not starts:
            return [
                (
                    None,
                    normalize_inline_spacing(combined),
                    [p.paragraph_id for p in claim_paragraphs],
                    [p.page_number for p in claim_paragraphs],
                )
            ]

        claims: list[tuple[int | None, str, list[str], list[int]]] = []
        for idx, start in enumerate(starts):
            end = starts[idx + 1].start() if idx + 1 < len(starts) else len(combined)
            raw = combined[start.start() : end]
            match = _CLAIM_START_RE.match(raw.strip())
            claim_number = int(match.group("number")) if match else None
            claims.append(
                (
                    claim_number,
                    normalize_inline_spacing(raw),
                    [p.paragraph_id for p in claim_paragraphs],
                    [p.page_number for p in claim_paragraphs],
                )
            )
        return claims

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
            if chunk.retrieval_tier == "child":
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
    ) -> PatentChunk:
        clean_text = normalize_inline_spacing(text.replace("\n", " \n ")).replace(" \n ", "\n")
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
            metadata=dict(extra_metadata or {}),
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
