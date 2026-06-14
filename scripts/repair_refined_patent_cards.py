"""Repair and validate refined patent cards.

This script fixes common LLM card-format issues:
- supporting_evidence_ids missing even though field_evidence_map contains IDs
- core_technology has no field-level evidence ID
- too few top-level supporting evidence IDs

It does not invent content. It only repairs evidence references using the
original evidence_pack.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TEXT_FIELDS = [
    "core_technology",
    "polymer_route",
    "formulation_strategy",
    "application_focus",
    "performance_evidence",
    "rd_relevance",
]


FIELD_FALLBACK_CATEGORIES = {
    "core_technology": ["claim", "polymer"],
    "polymer_route": ["polymer", "claim"],
    "formulation_strategy": ["formulation", "claim"],
    "application_focus": ["application", "claim"],
    "performance_evidence": ["performance", "application"],
    "rd_relevance": ["application", "performance", "polymer", "formulation", "claim"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair refined patent cards.")

    parser.add_argument("--cards-json", type=Path, required=True)
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/patent_cards_refined_repaired"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cards = json.loads(args.cards_json.read_text(encoding="utf-8"))
    evidence_pack = json.loads(args.evidence_pack.read_text(encoding="utf-8"))

    evidence_by_publication = group_evidence_by_publication(evidence_pack)

    repaired_cards: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for card in cards:
        publication = str(card.get("patent_publication", "")).strip()
        evidence_items = evidence_by_publication.get(publication, [])

        repaired = repair_card(card, evidence_items)
        validation = validate_card(repaired, evidence_items)

        repaired.setdefault("_repair", {})
        repaired["_repair"] = {
            "valid_after_repair": validation["valid"],
            "validation_errors": validation["errors"],
        }

        repaired_cards.append(repaired)
        quality_rows.append(build_quality_row(repaired, validation))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "patent_cards_refined_repaired.json"
    md_path = output_dir / "patent_cards_refined_repaired.md"
    quality_path = output_dir / "patent_card_quality.csv"

    json_path.write_text(
        json.dumps(repaired_cards, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_cards_markdown(repaired_cards, md_path)
    write_quality_csv(quality_rows, quality_path)

    print(f"Wrote repaired refined cards to: {output_dir}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Quality: {quality_path}")


def repair_card(
    card: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    repaired = dict(card)

    valid_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in evidence_items
        if str(item.get("evidence_id", "")).strip()
    }

    field_map = repaired.get("field_evidence_map", {})
    if not isinstance(field_map, dict):
        field_map = {}

    cleaned_field_map: dict[str, list[str]] = {}

    for field in TEXT_FIELDS:
        raw_ids = field_map.get(field, [])

        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]

        if not isinstance(raw_ids, list):
            raw_ids = []

        cleaned_ids = [
            str(eid).strip()
            for eid in raw_ids
            if str(eid).strip() in valid_ids
        ]

        value = str(repaired.get(field, "")).strip().lower()
        field_is_claiming_content = bool(value) and value != "not available in supplied evidence"

        if field_is_claiming_content and not cleaned_ids:
            cleaned_ids = fallback_ids_for_field(
                field=field,
                evidence_items=evidence_items,
                max_ids=2,
            )

        cleaned_field_map[field] = cleaned_ids

    repaired["field_evidence_map"] = cleaned_field_map

    top_level_ids = repaired.get("supporting_evidence_ids", [])
    if isinstance(top_level_ids, str):
        top_level_ids = [top_level_ids]

    if not isinstance(top_level_ids, list):
        top_level_ids = []

    top_level_ids = [
        str(eid).strip()
        for eid in top_level_ids
        if str(eid).strip() in valid_ids
    ]

    union_ids = union_field_map_ids(cleaned_field_map)

    if len(top_level_ids) < 3:
        top_level_ids = merge_unique(
            top_level_ids,
            union_ids,
            representative_ids_by_category(evidence_items),
        )

    repaired["supporting_evidence_ids"] = [
        evidence_id for evidence_id in top_level_ids if evidence_id in valid_ids
    ]

    return repaired


def fallback_ids_for_field(
    field: str,
    evidence_items: list[dict[str, Any]],
    max_ids: int,
) -> list[str]:
    categories = FIELD_FALLBACK_CATEGORIES.get(field, ["claim", "polymer"])

    selected: list[str] = []
    seen: set[str] = set()

    for category in categories:
        for item in evidence_items:
            if str(item.get("selection_category", "")) != category:
                continue

            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue

            selected.append(evidence_id)
            seen.add(evidence_id)
            break

        if len(selected) >= max_ids:
            break

    return selected


def representative_ids_by_category(
    evidence_items: list[dict[str, Any]],
) -> list[str]:
    categories = [
        "claim",
        "polymer",
        "formulation",
        "application",
        "performance",
    ]

    selected: list[str] = []
    seen: set[str] = set()

    for category in categories:
        for item in evidence_items:
            if str(item.get("selection_category", "")) != category:
                continue

            evidence_id = str(item.get("evidence_id", "")).strip()
            if not evidence_id or evidence_id in seen:
                continue

            selected.append(evidence_id)
            seen.add(evidence_id)
            break

    return selected


def union_field_map_ids(field_map: dict[str, list[str]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    for field in TEXT_FIELDS:
        for evidence_id in field_map.get(field, []):
            if evidence_id in seen:
                continue
            ids.append(evidence_id)
            seen.add(evidence_id)

    return ids


def merge_unique(*lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for values in lists:
        for value in values:
            if value in seen:
                continue
            merged.append(value)
            seen.add(value)

    return merged


def validate_card(
    card: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []

    valid_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in evidence_items
        if str(item.get("evidence_id", "")).strip()
    }

    for field in [
        "patent_publication",
        "owner",
        "priority_year",
        "core_technology",
        "polymer_route",
        "formulation_strategy",
        "application_focus",
        "rd_relevance",
        "supporting_evidence_ids",
        "field_evidence_map",
    ]:
        if field not in card:
            errors.append(f"missing field: {field}")

    supporting_ids = card.get("supporting_evidence_ids", [])
    if not isinstance(supporting_ids, list) or not supporting_ids:
        errors.append("no supporting_evidence_ids")
    else:
        for evidence_id in supporting_ids:
            if evidence_id not in valid_ids:
                errors.append(f"invalid supporting evidence id: {evidence_id}")

    field_map = card.get("field_evidence_map", {})
    if not isinstance(field_map, dict):
        errors.append("field_evidence_map is not object")
    else:
        for field in TEXT_FIELDS:
            value = str(card.get(field, "")).strip().lower()
            has_content = bool(value) and value != "not available in supplied evidence"

            ids = field_map.get(field, [])

            if has_content and not ids:
                errors.append(f"field has no evidence ids: {field}")

            for evidence_id in ids:
                if evidence_id not in valid_ids:
                    errors.append(f"invalid field evidence id: {field} -> {evidence_id}")

    return {
        "valid": not errors,
        "errors": errors,
    }


def build_quality_row(
    card: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "patent_publication": card.get("patent_publication", ""),
        "owner": card.get("owner", ""),
        "priority_year": card.get("priority_year", ""),
        "valid": validation["valid"],
        "errors": "; ".join(validation["errors"]),
        "supporting_evidence_count": len(card.get("supporting_evidence_ids", [])),
        "core_technology_present": is_supported_text(card.get("core_technology")),
        "polymer_route_present": is_supported_text(card.get("polymer_route")),
        "formulation_strategy_present": is_supported_text(card.get("formulation_strategy")),
        "application_focus_present": is_supported_text(card.get("application_focus")),
        "performance_evidence_present": is_supported_text(card.get("performance_evidence")),
        "rd_relevance_present": is_supported_text(card.get("rd_relevance")),
    }


def is_supported_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and text != "not available in supplied evidence"


def group_evidence_by_publication(
    evidence_pack: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in evidence_pack.get("evidence_items", []):
        publication = str(item.get("patent_publication", "")).strip()
        if publication:
            grouped[publication].append(item)

    return grouped


def write_cards_markdown(cards: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Repaired Refined Patent Cards", ""]

    for card in cards:
        lines.append(f"## {card['patent_publication']}")
        lines.append("")
        lines.append(f"- Owner: {card['owner']}")
        lines.append(f"- Priority year: {card['priority_year']}")
        lines.append(f"- Core technology: {card['core_technology']}")
        lines.append(f"- Polymer route: {card['polymer_route']}")
        lines.append(f"- Formulation strategy: {card['formulation_strategy']}")
        lines.append(f"- Application focus: {card['application_focus']}")
        lines.append(f"- Performance evidence: {card['performance_evidence']}")
        lines.append(f"- R&D relevance: {card['rd_relevance']}")
        lines.append(
            "- Missing or uncertain evidence: "
            + "; ".join(card.get("missing_or_uncertain_evidence", []))
        )
        lines.append(
            "- Supporting evidence IDs: "
            + ", ".join(f"[{eid}]" for eid in card.get("supporting_evidence_ids", []))
        )
        lines.append("")
        lines.append("Field evidence map:")
        for field, ids in card.get("field_evidence_map", {}).items():
            lines.append(f"- {field}: " + ", ".join(f"[{eid}]" for eid in ids))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "patent_publication",
        "owner",
        "priority_year",
        "valid",
        "errors",
        "supporting_evidence_count",
        "core_technology_present",
        "polymer_route_present",
        "formulation_strategy_present",
        "application_focus_present",
        "performance_evidence_present",
        "rd_relevance_present",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()