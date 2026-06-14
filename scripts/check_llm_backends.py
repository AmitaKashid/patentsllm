"""Smoke-test configured LLM backends without running report generation.

This verifies model integration only:
- registry loading
- backend selection
- model call
- latency capture
- error capture

It does not evaluate report quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from patentllm.generation.config import GenerationConfig
from patentllm.generation.llm_clients import build_llm_client, parse_llm_spec
from patentllm.generation.model_registry import load_enabled_models


@dataclass(frozen=True, slots=True)
class BackendCheckResult:
    run_id: str
    model_id: str
    backend: str
    model_name: str
    success: bool
    latency_ms: float
    output_preview: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check configured LLM backends.")
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("configs/llm_models.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/llm_backend_checks"),
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = GenerationConfig(
        ollama_base_url=args.ollama_base_url,
        vllm_base_url=args.vllm_base_url,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    models = load_enabled_models(args.registry_path)
    results: list[BackendCheckResult] = []

    system_prompt = "You are a backend health-check assistant."
    user_prompt = "Return exactly this text and nothing else: BACKEND_OK"

    for model in models:
        print(f"Checking {model.id} | {model.backend}:{model.model_name}")

        start = time.perf_counter()
        try:
            spec = parse_llm_spec(model.cli_spec)
            client = build_llm_client(spec, config)

            output = client.generate(
                model_name=spec.model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            latency_ms = (time.perf_counter() - start) * 1000.0

            results.append(
                BackendCheckResult(
                    run_id=run_id,
                    model_id=model.id,
                    backend=model.backend,
                    model_name=model.model_name,
                    success=True,
                    latency_ms=round(latency_ms, 2),
                    output_preview=output[:300],
                    error="",
                )
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0

            results.append(
                BackendCheckResult(
                    run_id=run_id,
                    model_id=model.id,
                    backend=model.backend,
                    model_name=model.model_name,
                    success=False,
                    latency_ms=round(latency_ms, 2),
                    output_preview="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    write_jsonl(results, run_dir / "backend_check_results.jsonl")
    write_csv(results, run_dir / "backend_check_results.csv")
    write_markdown(results, run_dir / "backend_check_report.md")

    print(f"\nWrote backend check outputs to: {run_dir}")


def write_jsonl(results: list[BackendCheckResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def write_csv(results: list[BackendCheckResult], path: Path) -> None:
    fieldnames = [
        "run_id",
        "model_id",
        "backend",
        "model_name",
        "success",
        "latency_ms",
        "output_preview",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def write_markdown(results: list[BackendCheckResult], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# LLM Backend Integration Check\n\n")
        file.write("This report checks whether configured LLM backends are callable.\n\n")
        file.write("| Model ID | Backend | Model | Success | Latency ms | Error |\n")
        file.write("|---|---|---|---:|---:|---|\n")

        for result in results:
            error = result.error.replace("|", "\\|")
            file.write(
                f"| `{result.model_id}` "
                f"| `{result.backend}` "
                f"| `{result.model_name}` "
                f"| {result.success} "
                f"| {result.latency_ms} "
                f"| {error} |\n"
            )


if __name__ == "__main__":
    main()