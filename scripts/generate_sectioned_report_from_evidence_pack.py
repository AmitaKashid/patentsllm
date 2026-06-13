"""Generate a detailed R&D patent report section-by-section.

This script avoids the failure mode of single-pass report generation where the
model hits max_new_tokens before the report is complete.

Each section receives:
- focused instructions
- section-relevant evidence
- its own output budget

The final report is assembled from generated section files.
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
from patentllm.generation.section_prompts import (
    SECTION_SPECS,
    SYSTEM_PROMPT,
    SectionSpec,
    build_section_prompt,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SectionRunRecord:
    """Metadata for one generated report section."""

    run_id: str
    evidence_pack_id: str
    llm_spec: str
    section_id: str
    section_title: str
    success: bool
    latency_seconds: float
    prompt_path: str
    raw_response_path: str
    section_path: str
    prompt_eval_count: int | None
    eval_count: int | None
    done_reason: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate detailed patent report section-by-section."
    )

    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--llm", type=str, default="ollama:qwen3:8b")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/generated_reports_sectioned"),
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-ctx", type=int, default=12000)
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--vllm-base-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue generating later sections even if one section fails.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    evidence_pack = load_json(args.evidence_pack)
    evidence_pack_id = str(evidence_pack.get("pack_id", "unknown_pack"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = args.output_dir / f"{run_id}_{evidence_pack_id}"
    prompts_dir = run_dir / "prompts"
    raw_dir = run_dir / "raw_responses"
    sections_dir = run_dir / "sections"

    prompts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    sections_dir.mkdir(parents=True, exist_ok=True)

    spec = parse_llm_spec(args.llm)
    client_config = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=1200,
        num_ctx=args.num_ctx,
        think=False,
        fail_on_length_done_reason=True,
        ollama_base_url=args.ollama_base_url,
        vllm_base_url=args.vllm_base_url,
    )
    client = build_llm_client(spec, client_config)

    records: list[SectionRunRecord] = []
    generated_sections: list[tuple[str, Path]] = []

    for section_spec in SECTION_SPECS:
        LOGGER.info("Generating section: %s", section_spec.section_id)

        prompt = build_section_prompt(
            evidence_pack=evidence_pack,
            section_spec=section_spec,
        )

        prompt_path = prompts_dir / f"{section_spec.section_id}_prompt.md"
        raw_response_path = raw_dir / f"{section_spec.section_id}_raw.json"
        section_path = sections_dir / f"{section_spec.section_id}.md"

        prompt_path.write_text(prompt, encoding="utf-8")

        start = time.perf_counter()

        section_client_config = GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=section_spec.max_new_tokens,
            num_ctx=args.num_ctx,
            think=False,
            fail_on_length_done_reason=True,
            ollama_base_url=args.ollama_base_url,
            vllm_base_url=args.vllm_base_url,
        )
        section_client = build_llm_client(spec, section_client_config)

        try:
            section_text = section_client.generate(
                model_name=spec.model_name,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                raw_response_path=raw_response_path,
            )

            latency_seconds = time.perf_counter() - start
            section_path.write_text(section_text.strip() + "\n", encoding="utf-8")

            raw_data = load_json(raw_response_path)
            prompt_eval_count = safe_int(raw_data.get("prompt_eval_count"))
            eval_count = safe_int(raw_data.get("eval_count"))
            done_reason = str(raw_data.get("done_reason", ""))

            records.append(
                SectionRunRecord(
                    run_id=run_id,
                    evidence_pack_id=evidence_pack_id,
                    llm_spec=args.llm,
                    section_id=section_spec.section_id,
                    section_title=section_spec.title,
                    success=True,
                    latency_seconds=round(latency_seconds, 2),
                    prompt_path=str(prompt_path),
                    raw_response_path=str(raw_response_path),
                    section_path=str(section_path),
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                    done_reason=done_reason,
                    error="",
                )
            )

            generated_sections.append((section_spec.section_id, section_path))
            LOGGER.info("Wrote section to %s", section_path)

        except Exception as exc:
            latency_seconds = time.perf_counter() - start
            error = f"{type(exc).__name__}: {exc}"

            LOGGER.exception("Section generation failed: %s", section_spec.section_id)

            raw_data = load_json_if_exists(raw_response_path)
            records.append(
                SectionRunRecord(
                    run_id=run_id,
                    evidence_pack_id=evidence_pack_id,
                    llm_spec=args.llm,
                    section_id=section_spec.section_id,
                    section_title=section_spec.title,
                    success=False,
                    latency_seconds=round(latency_seconds, 2),
                    prompt_path=str(prompt_path),
                    raw_response_path=str(raw_response_path),
                    section_path=str(section_path),
                    prompt_eval_count=safe_int(raw_data.get("prompt_eval_count")),
                    eval_count=safe_int(raw_data.get("eval_count")),
                    done_reason=str(raw_data.get("done_reason", "")),
                    error=error,
                )
            )

            if not args.continue_on_error:
                break

    final_report_path = run_dir / f"final_report_{spec.safe_name}.md"
    assemble_report(generated_sections, final_report_path)

    write_summary_csv(records, run_dir / "section_generation_summary.csv")
    write_summary_jsonl(records, run_dir / "section_generation_summary.jsonl")

    metadata = {
        "run_id": run_id,
        "evidence_pack_path": str(args.evidence_pack),
        "evidence_pack_id": evidence_pack_id,
        "llm": args.llm,
        "temperature": args.temperature,
        "num_ctx": args.num_ctx,
        "final_report_path": str(final_report_path),
        "sections_generated": len(generated_sections),
        "sections_expected": len(SECTION_SPECS),
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote sectioned report outputs to: {run_dir}")
    print(f"Final report: {final_report_path}")
    print(f"Summary: {run_dir / 'section_generation_summary.csv'}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def assemble_report(section_paths: list[tuple[str, Path]], output_path: Path) -> None:
    """Assemble section files into one final report."""

    parts: list[str] = []

    for _, path in section_paths:
        if not path.exists():
            continue
        parts.append(path.read_text(encoding="utf-8").strip())

    output_path.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")


def write_summary_csv(records: list[SectionRunRecord], path: Path) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else [
        "run_id",
        "evidence_pack_id",
        "llm_spec",
        "section_id",
        "section_title",
        "success",
        "latency_seconds",
        "prompt_path",
        "raw_response_path",
        "section_path",
        "prompt_eval_count",
        "eval_count",
        "done_reason",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def write_summary_jsonl(records: list[SectionRunRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()