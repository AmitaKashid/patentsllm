# src/patentllm/generation/config.py

"""Configuration for LLM-based report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Runtime configuration for report generation."""

    persist_dir: Path = Path("data/processed/vector_store/chroma")
    collection_name: str = "patent_chunks_bge_m3"

    embedding_backend: str = "sentence_transformer"
    embedding_model_name: str = "BAAI/bge-m3"

    output_dir: Path = Path("data/processed/generated_reports")

    top_k: int = 8
    temperature: float = 0.1
    max_new_tokens: int = 2500

    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000"

    transformers_device_map: str = "auto"
    transformers_torch_dtype: str = "auto"