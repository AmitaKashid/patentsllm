# scripts/generate_report.py

"""Generate patent reports using retrieved chunks and pluggable LLM backends."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from patentllm.generation.config import GenerationConfig
from patentllm.generation.llm_clients import parse_llm_spec
from patentllm.generation.report_generator import PatentReportGenerator

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PatentLLM reports.")

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Business or technical question to answer using patent evidence.",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        default=["ollama:qwen3:8b"],
        help=(
            "LLM specs in backend:model format. Examples: "
            "ollama:qwen3:8b transformers:Qwen/Qwen3-8B vllm:Qwen/Qwen3-8B"
        ),
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/processed/vector_store/chroma"),
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="patent_chunks_bge_m3",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="BAAI/bge-m3",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
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
        default=2500,
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

    model_specs = [parse_llm_spec(raw_spec) for raw_spec in args.llms]

    config = GenerationConfig(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model_name,
        output_dir=args.output_dir,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        ollama_base_url=args.ollama_base_url,
        vllm_base_url=args.vllm_base_url,
    )

    LOGGER.info("Generating report using collection=%s", config.collection_name)
    LOGGER.info("LLMs: %s", ", ".join(args.llms))

    generator = PatentReportGenerator(config)
    results = generator.generate(query=args.query, model_specs=model_specs)
    run_dir = generator.write_results(results)

    LOGGER.info("Wrote generated reports to %s", run_dir)


if __name__ == "__main__":
    main()