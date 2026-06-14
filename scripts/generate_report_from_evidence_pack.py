"""Generate a detailed R&D patent report from a structured evidence pack.

Example:
    python scripts/generate_report_from_evidence_pack.py ^
      --evidence-pack data/processed/evidence_packs/20260613_155340/evidence_pack.json ^
      --llms ollama:qwen3:8b ^
      --num-ctx 16000 ^
      --max-new-tokens 4500

This script intentionally does not benchmark model quality.
It only generates report artifacts from a stable evidence pack.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from patentllm.generation.config import GenerationConfig
from patentllm.generation.llm_clients import build_llm_client, parse_llm_spec
from patentllm.generation.report_prompts import (
    SYSTEM_PROMPT,
    PromptBuildConfig,
    build_detailed_rd_report_prompt,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationRunRecord:
    """Metadata for one generated report."""

    run_id: str
    evidence_pack_id: str
    llm_spec: str
    backend: str
    model_name: str
    success: bool
    latency_seconds: float
    report_path: str
    prompt_path: str
    metadata_path: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate detailed patent report from evidence pack."
    )

    parser.add_argument(
        "--evidence-pack",
        type=Path,
        required=True,
        help="Path to evidence_pack.json.",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        default=["ollama:qwen3:8b"],
        help=(
            "LLM specs in backend:model format. Examples: "
            "ollama:qwen3:8b ollama:llama3.1:8b"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/generated_reports"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4500,
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=16000,
    )
    parser.add_argument(
        "--max-chars-per-evidence",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--max-total-evidence-items",
        type=int,
        default=63,
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default="http://localhost:8000",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    evidence_pack = _load_json(args.evidence_pack)
    evidence_pack_id = str(evidence_pack.get("pack_id", "unknown_pack"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{run_id}_{evidence_pack_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_config = PromptBuildConfig(
        max_chars_per_evidence=args.max_chars_per_evidence,
        max_total_evidence_items=args.max_total_evidence_items,
        include_coverage_table=True,
    )

    user_prompt = build_detailed_rd_report_prompt(
        evidence_pack=evidence_pack,
        config=prompt_config,
    )

    prompt_path = run_dir / "prompt_used.md"
    prompt_path.write_text(user_prompt, encoding="utf-8")

    source_path = run_dir / "evidence_sources.csv"
    _write_evidence_sources(evidence_pack=evidence_pack, path=source_path)

    config = GenerationConfig(
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_ctx=args.num_ctx,
        ollama_base_url=args.ollama_base_url,
        vllm_base_url=args.vllm_base_url,
    )

    records: list[GenerationRunRecord] = []

    for raw_spec in args.llms:
        spec = parse_llm_spec(raw_spec)
        safe_name = _safe_filename(f"{spec.backend}_{spec.model_name}")

        LOGGER.info("Generating report with %s:%s", spec.backend, spec.model_name)

        report_path = run_dir / f"report_{safe_name}.md"
        metadata_path = run_dir / f"metadata_{safe_name}.json"

        start = time.perf_counter()

        try:
            client = build_llm_client(spec, config)
            report = client.generate(
                model_name=spec.model_name,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            latency_seconds = time.perf_counter() - start

            report_path.write_text(report, encoding="utf-8")

            metadata = {
                "run_id": run_id,
                "evidence_pack_path": str(args.evidence_pack),
                "evidence_pack_id": evidence_pack_id,
                "llm_spec": raw_spec,
                "backend": spec.backend,
                "model_name": spec.model_name,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "num_ctx": args.num_ctx,
                "max_chars_per_evidence": args.max_chars_per_evidence,
                "max_total_evidence_items": args.max_total_evidence_items,
                "evidence_items_available": len(evidence_pack.get("evidence_items", [])),
                "evidence_items_prompted": min(
                    args.max_total_evidence_items,
                    len(evidence_pack.get("evidence_items", [])),
                ),
                "latency_seconds": round(latency_seconds, 2),
                "report_path": str(report_path),
                "prompt_path": str(prompt_path),
                "source_path": str(source_path),
            }

            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            records.append(
                GenerationRunRecord(
                    run_id=run_id,
                    evidence_pack_id=evidence_pack_id,
                    llm_spec=raw_spec,
                    backend=spec.backend,
                    model_name=spec.model_name,
                    success=True,
                    latency_seconds=round(latency_seconds, 2),
                    report_path=str(report_path),
                    prompt_path=str(prompt_path),
                    metadata_path=str(metadata_path),
                    error="",
                )
            )

            LOGGER.info("Wrote report to %s", report_path)

        except Exception as exc:
            latency_seconds = time.perf_counter() - start
            error_message = f"{type(exc).__name__}: {exc}"

            LOGGER.exception("Generation failed for %s", raw_spec)

            metadata_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "evidence_pack_path": str(args.evidence_pack),
                        "evidence_pack_id": evidence_pack_id,
                        "llm_spec": raw_spec,
                        "backend": spec.backend,
                        "model_name": spec.model_name,
                        "success": False,
                        "latency_seconds": round(latency_seconds, 2),
                        "error": error_message,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            records.append(
                GenerationRunRecord(
                    run_id=run_id,
                    evidence_pack_id=evidence_pack_id,
                    llm_spec=raw_spec,
                    backend=spec.backend,
                    model_name=spec.model_name,
                    success=False,
                    latency_seconds=round(latency_seconds, 2),
                    report_path="",
                    prompt_path=str(prompt_path),
                    metadata_path=str(metadata_path),
                    error=error_message,
                )
            )

    _write_run_summary(records=records, path=run_dir / "generation_summary.csv")
    _write_run_summary_jsonl(records=records, path=run_dir / "generation_summary.jsonl")

    print(f"\nWrote generation outputs to: {run_dir}")
    print(f"Prompt used: {prompt_path}")
    print(f"Evidence sources: {source_path}")
    print(f"Summary: {run_dir / 'generation_summary.csv'}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _write_evidence_sources(evidence_pack: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "evidence_id",
        "document_id",
        "patent_publication",
        "owner",
        "priority_year",
        "chunk_id",
        "chunk_type",
        "section",
        "page_start",
        "page_end",
        "selection_category",
        "selection_reason",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in evidence_pack.get("evidence_items", []):
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def _write_run_summary(records: list[GenerationRunRecord], path: Path) -> None:
    fieldnames = [
        "run_id",
        "evidence_pack_id",
        "llm_spec",
        "backend",
        "model_name",
        "success",
        "latency_seconds",
        "report_path",
        "prompt_path",
        "metadata_path",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def _write_run_summary_jsonl(records: list[GenerationRunRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _safe_filename(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


if __name__ == "__main__":
    main()