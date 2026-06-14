"""Refine raw evidence-pack patent cards with a local LLM.

This script creates analyst-ready grounded patent cards.

Input:
    evidence_pack.json

Output:
    data/processed/patent_cards_refined/<run_id>/
    ├── patent_cards_refined.json
    ├── patent_cards_refined.md
    ├── patent_card_quality.csv
    └── raw_responses/

The refined cards are not free-form summaries. Each field must be grounded in
specific evidence IDs from the supplied evidence pack.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


REQUIRED_CARD_FIELDS = [
    "patent_publication",
    "owner",
    "priority_year",
    "core_technology",
    "polymer_route",
    "formulation_strategy",
    "application_focus",
    "performance_evidence",
    "rd_relevance",
    "missing_or_uncertain_evidence",
    "supporting_evidence_ids",
    "field_evidence_map",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LLM-refined grounded patent cards from evidence pack."
    )

    parser.add_argument(
        "--evidence-pack",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:8b",
        help="Ollama model name, e.g. qwen3:8b.",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/patent_cards_refined"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=12000,
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=900,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evidence_pack = load_json(args.evidence_pack)
    pack_id = str(evidence_pack.get("pack_id", "unknown_pack"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = args.output_dir / f"{run_id}_{pack_id}_{safe_filename(args.model)}"
    raw_dir = run_dir / "raw_responses"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    grouped_evidence = group_evidence_by_publication(evidence_pack)
    coverage_by_publication = group_coverage_by_publication(evidence_pack)

    target_publications = evidence_pack.get("task", {}).get("target_publications", [])
    if not target_publications:
        target_publications = sorted(grouped_evidence)

    refined_cards: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for publication in target_publications:
        print(f"Refining card: {publication}")

        evidence_items = grouped_evidence.get(publication, [])
        coverage = coverage_by_publication.get(publication, {})

        prompt = build_card_prompt(
            publication=publication,
            evidence_items=evidence_items,
            coverage=coverage,
            max_evidence_chars=args.max_evidence_chars,
        )

        raw_response_path = raw_dir / f"{safe_filename(publication)}.json"

        started = time.perf_counter()

        try:
            card = call_ollama_json(
                base_url=args.ollama_base_url,
                model=args.model,
                prompt=prompt,
                raw_response_path=raw_response_path,
                temperature=args.temperature,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
            )

            card = normalize_card(
                card=card,
                publication=publication,
                coverage=coverage,
                evidence_items=evidence_items,
            )

            latency_seconds = round(time.perf_counter() - started, 2)
            validation = validate_card(card, evidence_items)

            card["_generation"] = {
                "model": args.model,
                "latency_seconds": latency_seconds,
                "raw_response_path": str(raw_response_path),
                "valid": validation["valid"],
                "validation_errors": validation["errors"],
            }

        except Exception as exc:
            card = build_failed_card(
                publication=publication,
                coverage=coverage,
                error=f"{type(exc).__name__}: {exc}",
            )
            validation = {
                "valid": False,
                "errors": [card["_generation"]["error"]],
            }

        refined_cards.append(card)

        quality_rows.append(
            build_quality_row(
                card=card,
                validation=validation,
            )
        )

    json_path = run_dir / "patent_cards_refined.json"
    md_path = run_dir / "patent_cards_refined.md"
    quality_path = run_dir / "patent_card_quality.csv"

    json_path.write_text(
        json.dumps(refined_cards, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_cards_markdown(refined_cards, md_path)
    write_quality_csv(quality_rows, quality_path)

    metadata = {
        "run_id": run_id,
        "pack_id": pack_id,
        "model": args.model,
        "evidence_pack": str(args.evidence_pack),
        "cards": len(refined_cards),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "quality_path": str(quality_path),
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote refined cards to: {run_dir}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Quality: {quality_path}")


def build_card_prompt(
    publication: str,
    evidence_items: list[dict[str, Any]],
    coverage: dict[str, Any],
    max_evidence_chars: int,
) -> str:
    evidence_block = "\n\n".join(
        format_evidence_item(item, max_chars=max_evidence_chars)
        for item in evidence_items
    )

    available_ids = [
        str(item.get("evidence_id", "")).strip()
        for item in evidence_items
        if str(item.get("evidence_id", "")).strip()
    ]

    return f"""You are creating a grounded patent intelligence card for an R&D team.

Patent publication:
{publication}

Structured metadata:
owner: {coverage.get("owner", "")}
priority_year: {coverage.get("priority_year", "")}
available_evidence_ids: {", ".join(available_ids)}

Task:
Create one concise, analyst-ready patent card using only the supplied evidence.

Rules:
- Output valid JSON only.
- Do not copy long raw excerpts.
- Convert evidence into clean technical summaries.
- Every field must be directly supported by the supplied evidence.
- Use cautious wording when evidence is incomplete.
- Do not invent performance data, application areas, owners, dates, or technical effects.
- Use only evidence IDs from available_evidence_ids.
- If a field is unsupported, write "not available in supplied evidence".
- Do not provide legal, infringement, FTO, validity, or patentability conclusions.

Required JSON schema:
{{
  "patent_publication": "{publication}",
  "owner": "{coverage.get("owner", "")}",
  "priority_year": "{coverage.get("priority_year", "")}",
  "core_technology": "one sentence",
  "polymer_route": "one sentence",
  "formulation_strategy": "one sentence",
  "application_focus": "one sentence",
  "performance_evidence": "one sentence or not available in supplied evidence",
  "rd_relevance": "one concrete R&D relevance sentence",
  "missing_or_uncertain_evidence": ["short item"],
  "supporting_evidence_ids": ["E1"],
  "field_evidence_map": {{
    "core_technology": ["E1"],
    "polymer_route": ["E1"],
    "formulation_strategy": ["E2"],
    "application_focus": ["E3"],
    "performance_evidence": ["E4"],
    "rd_relevance": ["E1", "E3"]
  }}
}}

Evidence:
{evidence_block}
"""


def format_evidence_item(item: dict[str, Any], max_chars: int) -> str:
    text = clean_text(str(item.get("text", "")))
    text = clip(text, max_chars)

    return (
        f"[{item.get('evidence_id', '')}]\n"
        f"category: {item.get('selection_category', '')}\n"
        f"section: {item.get('section', '')}\n"
        f"chunk_type: {item.get('chunk_type', '')}\n"
        f"text: {text}"
    )


def call_ollama_json(
    base_url: str,
    model: str,
    prompt: str,
    raw_response_path: Path,
    temperature: float,
    num_ctx: int,
    num_predict: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": "You produce strict JSON for patent R&D analysis. Return JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    response = requests.post(url, json=payload, timeout=1800)
    response.raise_for_status()

    data = response.json()
    raw_response_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    content = str(data.get("message", {}).get("content", "")).strip()

    if not content:
        raise RuntimeError(f"Ollama returned empty card response. Raw: {raw_response_path}")

    return parse_json_object(content)


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise TypeError("Expected JSON object from card model.")

    return parsed


def normalize_card(
    card: dict[str, Any],
    publication: str,
    coverage: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    for field in REQUIRED_CARD_FIELDS:
        normalized[field] = card.get(field)

    normalized["patent_publication"] = publication
    normalized["owner"] = first_non_empty(
        str(card.get("owner", "")),
        str(coverage.get("owner", "")),
        "not available",
    )
    normalized["priority_year"] = first_non_empty(
        str(card.get("priority_year", "")),
        str(coverage.get("priority_year", "")),
        "not available",
    )

    text_fields = [
        "core_technology",
        "polymer_route",
        "formulation_strategy",
        "application_focus",
        "performance_evidence",
        "rd_relevance",
    ]

    for field in text_fields:
        value = str(card.get(field, "")).strip()
        normalized[field] = value if value else "not available in supplied evidence"

    missing = card.get("missing_or_uncertain_evidence", [])
    if isinstance(missing, str):
        missing = [missing]
    if not isinstance(missing, list):
        missing = ["not available"]
    normalized["missing_or_uncertain_evidence"] = [str(item) for item in missing]

    supporting_ids = card.get("supporting_evidence_ids", [])
    if isinstance(supporting_ids, str):
        supporting_ids = re.findall(r"E\d+", supporting_ids)
    if not isinstance(supporting_ids, list):
        supporting_ids = []

    valid_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in evidence_items
    }

    normalized["supporting_evidence_ids"] = [
        str(eid).strip()
        for eid in supporting_ids
        if str(eid).strip() in valid_ids
    ]

    field_map = card.get("field_evidence_map", {})
    if not isinstance(field_map, dict):
        field_map = {}

    cleaned_map: dict[str, list[str]] = {}
    for field, ids in field_map.items():
        if isinstance(ids, str):
            ids = re.findall(r"E\d+", ids)
        if not isinstance(ids, list):
            ids = []
        cleaned_map[str(field)] = [
            str(eid).strip()
            for eid in ids
            if str(eid).strip() in valid_ids
        ]

    normalized["field_evidence_map"] = cleaned_map

    return normalized


def validate_card(
    card: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []

    valid_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in evidence_items
    }

    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            errors.append(f"missing field: {field}")

    text_fields = [
        "core_technology",
        "polymer_route",
        "formulation_strategy",
        "application_focus",
        "rd_relevance",
    ]

    for field in text_fields:
        value = str(card.get(field, "")).strip()
        if not value or value == "not available in supplied evidence":
            errors.append(f"unsupported required field: {field}")

    supporting_ids = card.get("supporting_evidence_ids", [])
    if not supporting_ids:
        errors.append("no supporting_evidence_ids")

    for evidence_id in supporting_ids:
        if evidence_id not in valid_ids:
            errors.append(f"invalid supporting evidence id: {evidence_id}")

    field_map = card.get("field_evidence_map", {})
    if not isinstance(field_map, dict):
        errors.append("field_evidence_map is not an object")
    else:
        for field, ids in field_map.items():
            if not ids:
                errors.append(f"field has no evidence ids: {field}")
            for evidence_id in ids:
                if evidence_id not in valid_ids:
                    errors.append(f"invalid field evidence id: {field} -> {evidence_id}")

    return {
        "valid": not errors,
        "errors": errors,
    }


def build_failed_card(
    publication: str,
    coverage: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    return {
        "patent_publication": publication,
        "owner": str(coverage.get("owner", "not available")),
        "priority_year": str(coverage.get("priority_year", "not available")),
        "core_technology": "not available in supplied evidence",
        "polymer_route": "not available in supplied evidence",
        "formulation_strategy": "not available in supplied evidence",
        "application_focus": "not available in supplied evidence",
        "performance_evidence": "not available in supplied evidence",
        "rd_relevance": "not available in supplied evidence",
        "missing_or_uncertain_evidence": ["card generation failed"],
        "supporting_evidence_ids": [],
        "field_evidence_map": {},
        "_generation": {
            "valid": False,
            "error": error,
        },
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


def group_coverage_by_publication(
    evidence_pack: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("patent_publication", "")).strip(): row
        for row in evidence_pack.get("coverage", [])
    }


def write_cards_markdown(cards: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Refined Patent Cards", ""]

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    return " ".join(text.split())


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [TRUNCATED]"


def first_non_empty(*values: str) -> str:
    for value in values:
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return ""


def safe_filename(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


if __name__ == "__main__":
    main()