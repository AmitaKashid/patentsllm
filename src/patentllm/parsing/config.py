"""Configuration for patent PDF parsing and chunking.

This module is intentionally configuration-only. Keep parsing heuristics here
instead of scattering magic numbers across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatentParserConfig:
    """Runtime configuration for patent parsing.

    The defaults target scanned PCT-style patent PDFs:
    - Use native PDF text when available.
    - Fall back to OCR when native text is too short.
    - Preserve patent structure before chunking.
    - Use no overlap for legal/semantic boundaries; overlap only for forced splits.
    """

    # Text extraction
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 240
    min_native_chars_per_page: int = 250
    compute_ocr_confidence: bool = True
    low_ocr_confidence_threshold: float = 65.0

    # Paragraph parsing
    min_paragraph_chars: int = 35
    max_heading_chars: int = 120

    # Child chunks: description / evidence
    description_target_tokens: int = 650
    description_max_tokens: int = 850
    evidence_target_tokens: int = 700
    evidence_max_tokens: int = 900

    # Claims
    claim_max_tokens: int = 1_000

    # Used only when a paragraph or claim is too long and must be split artificially.
    forced_split_overlap_tokens: int = 96

    # Parent chunks for later report synthesis.
    parent_target_tokens: int = 1_400
    parent_max_tokens: int = 1_800

    # Combined patent representation: title + abstract + claims.
    tac_max_tokens: int = 1_900

    # Optional quick smoke-test limit.
    max_pages: int | None = None
    tessdata_dir: str | None = None