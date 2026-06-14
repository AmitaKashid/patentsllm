"""Deterministic patent-key generation for PatentLLM reports.

The patent key should not depend on LLM generation. It is a structured table
built from:
- report task target publications
- patent_metadata.csv
- evidence_pack.json

This avoids missing headings, missing patents, inconsistent owners, and
LLM-generated table drift.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PATENT_KEY_HEADING = "## Patent key and extracted R&D signal"


@dataclass(frozen=True, slots=True)
class PatentMetadataRow:
    """Structured metadata for one patent publication."""

    document_id: str
    patent_publication: str
    owner: str
    priority_year: str
    core_technology: str
    main_rd_relevance: str


def build_patent_key_markdown(
    evidence_pack: dict[str, Any],
    metadata_csv_path: Path | None = None,
) -> str:
    """Build deterministic patent-key markdown."""

    metadata_by_publication = (
        load_metadata_by_publication(metadata_csv_path)
        if metadata_csv_path and metadata_csv_path.exists()
        else {}
    )

    evidence_by_publication = group_evidence_by_publication(evidence_pack)
    coverage_by_publication = group_coverage_by_publication(evidence_pack)

    target_publications = evidence_pack.get("task", {}).get("target_publications", [])
    if not target_publications:
        target_publications = sorted(
            set(evidence_by_publication) | set(coverage_by_publication)
        )

    lines = [
        PATENT_KEY_HEADING,
        "",
        "| # | Patent | Owner | Priority | Core technology | Main R&D relevance | Evidence |",
        "|---:|---|---|---:|---|---|---|",
    ]

    for index, publication in enumerate(target_publications, start=1):
        publication_norm = normalize_publication(publication)

        metadata = metadata_by_publication.get(publication_norm)
        coverage = coverage_by_publication.get(publication_norm, {})
        evidence_items = evidence_by_publication.get(publication_norm, [])

        owner = first_non_empty(
            metadata.owner if metadata else "",
            str(coverage.get("owner", "")),
            first_value_from_evidence(evidence_items, "owner"),
            "not available",
        )

        priority_year = first_non_empty(
            metadata.priority_year if metadata else "",
            str(coverage.get("priority_year", "")),
            first_value_from_evidence(evidence_items, "priority_year"),
            "not available",
        )

        core_technology = first_non_empty(
            metadata.core_technology if metadata else "",
            infer_core_technology(evidence_items),
            "not available in structured metadata",
        )

        main_rd_relevance = first_non_empty(
            metadata.main_rd_relevance if metadata else "",
            infer_rd_relevance(evidence_items),
            "not available in structured metadata",
        )

        evidence_refs = select_evidence_refs(evidence_items)

        lines.append(
            "| "
            f"{index} | "
            f"{escape_md(publication)} | "
            f"{escape_md(owner)} | "
            f"{escape_md(priority_year)} | "
            f"{escape_md(core_technology)} | "
            f"{escape_md(main_rd_relevance)} | "
            f"{escape_md(evidence_refs)} |"
        )

    return "\n".join(lines).strip() + "\n"


def append_or_replace_patent_key(
    report_text: str,
    patent_key_markdown: str,
) -> str:
    """Append deterministic patent key after removing LLM-generated key artifacts.

    This removes:
    1. any proper old patent-key section starting with:
       ## Patent key and extracted R&D signal

    2. any malformed LLM-generated patent-key table starting with:
       # | Patent | Owner | Priority | Core technology | Main R&D relevance

    The deterministic patent key is then appended once.
    """

    cleaned_report = remove_patent_key_artifacts(report_text).rstrip()

    return cleaned_report + "\n\n" + patent_key_markdown.strip() + "\n"


def remove_patent_key_artifacts(report_text: str) -> str:
    """Remove old or malformed patent-key sections from a generated report."""

    cleaned = report_text

    # Remove a proper existing patent-key section and everything after it.
    proper_patent_key_pattern = re.compile(
        r"\n##\s+Patent\s+key\s+and\s+extracted\s+R&D\s+signal\b.*\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(proper_patent_key_pattern, "", cleaned)

    # Remove malformed LLM key table blocks like:
    # | Patent | Owner | Priority | Core technology | Main R&D relevance
    # | EP2756049A1 | ...
    malformed_key_pattern = re.compile(
        r"\n#\s*\|\s*Patent\s*\|\s*Owner\s*\|\s*Priority\s*\|\s*Core\s+technology\s*\|\s*Main\s+R&D\s+relevance\b.*?\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(malformed_key_pattern, "", cleaned)

    return cleaned.rstrip()


def load_metadata_by_publication(path: Path) -> dict[str, PatentMetadataRow]:
    """Load patent metadata CSV indexed by normalized publication number."""

    rows: dict[str, PatentMetadataRow] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            publication = str(row.get("patent_publication", "")).strip()
            if not publication:
                continue

            rows[normalize_publication(publication)] = PatentMetadataRow(
                document_id=str(row.get("document_id", "")).strip(),
                patent_publication=publication,
                owner=str(row.get("owner", "")).strip(),
                priority_year=str(row.get("priority_year", "")).strip(),
                core_technology=str(row.get("core_technology", "")).strip(),
                main_rd_relevance=str(row.get("main_rd_relevance", "")).strip(),
            )

    return rows


def group_evidence_by_publication(
    evidence_pack: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Group evidence items by patent publication."""

    grouped: dict[str, list[dict[str, Any]]] = {}

    for item in evidence_pack.get("evidence_items", []):
        publication = normalize_publication(str(item.get("patent_publication", "")))
        if not publication:
            continue
        grouped.setdefault(publication, []).append(item)

    return grouped


def group_coverage_by_publication(
    evidence_pack: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Group coverage rows by patent publication."""

    grouped: dict[str, dict[str, Any]] = {}

    for row in evidence_pack.get("coverage", []):
        publication = normalize_publication(str(row.get("patent_publication", "")))
        if not publication:
            continue
        grouped[publication] = row

    return grouped


def normalize_publication(value: str) -> str:
    """Normalize publication identifiers for matching."""

    return (
        value.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .strip()
    )


def first_non_empty(*values: str) -> str:
    """Return first non-empty string."""

    for value in values:
        cleaned = str(value).strip()
        if cleaned:
            return cleaned

    return ""


def first_value_from_evidence(
    evidence_items: list[dict[str, Any]],
    key: str,
) -> str:
    """Return first non-empty field value from evidence items."""

    for item in evidence_items:
        value = str(item.get(key, "")).strip()
        if value:
            return value

    return ""


def select_evidence_refs(evidence_items: list[dict[str, Any]]) -> str:
    """Select compact evidence references for the patent-key row."""

    preferred_categories = [
        "claim",
        "polymer",
        "formulation",
        "application",
        "performance",
    ]

    refs: list[str] = []
    seen: set[str] = set()

    for category in preferred_categories:
        for item in evidence_items:
            if str(item.get("selection_category", "")) != category:
                continue

            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue

            refs.append(f"[{evidence_id}]")
            seen.add(evidence_id)
            break

    return ", ".join(refs) if refs else "not available"


def infer_core_technology(evidence_items: list[dict[str, Any]]) -> str:
    """Fallback core-technology inference from evidence categories.

    This is intentionally conservative. Prefer patent_metadata.csv when present.
    """

    claim_text = first_text_for_category(evidence_items, "claim").lower()
    polymer_text = first_text_for_category(evidence_items, "polymer").lower()
    text = f"{claim_text} {polymer_text}"

    if "olefin block copolymer" in text or "obc" in text:
        return "Olefin block copolymer hot-melt adhesive system"

    if "metallocene" in text and "propylene" in text:
        return "Metallocene-catalyzed propylene-based copolymer system"

    if "single site" in text or "ssc-pp" in text:
        return "Single-site catalyst polypropylene blend system"

    if "polar functional" in text or "aliphatic polyester" in text:
        return "Polar-modified polymer and polyester-compatible adhesive system"

    if "ethylene-based copolymer" in text and "wax" in text:
        return "Ethylene-based copolymer and modified wax adhesive system"

    if "polypropylene" in text or "propylene" in text:
        return "Propylene/polypropylene-based hot-melt adhesive system"

    return ""


def infer_rd_relevance(evidence_items: list[dict[str, Any]]) -> str:
    """Fallback R&D relevance inference from evidence categories."""

    application_text = first_text_for_category(evidence_items, "application").lower()
    performance_text = first_text_for_category(evidence_items, "performance").lower()
    formulation_text = first_text_for_category(evidence_items, "formulation").lower()
    text = f"{application_text} {performance_text} {formulation_text}"

    if "stretch laminate" in text:
        return "Checkpoint for stretch-laminate bonding and elastic performance"

    if "elastic" in text or "diaper" in text or "nonwoven" in text:
        return "Checkpoint for hygiene-article bonding and elastic attachment"

    if "viscosity" in text or "coating" in text:
        return "Checkpoint for processability, coating quality, and application temperature"

    if "tackifier" in text or "wax" in text:
        return "Checkpoint for tackifier/wax package optimization"

    return ""


def first_text_for_category(
    evidence_items: list[dict[str, Any]],
    category: str,
) -> str:
    """Return first evidence text for a category."""

    for item in evidence_items:
        if str(item.get("selection_category", "")) == category:
            return str(item.get("text", ""))

    return ""


def escape_md(value: str) -> str:
    """Escape markdown table separators."""

    return str(value).replace("|", "\\|").replace("\n", " ").strip()