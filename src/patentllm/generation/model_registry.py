"""Model registry for PatentLLM generation backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


LLMBackend = Literal["ollama", "transformers", "vllm"]


@dataclass(frozen=True, slots=True)
class RegisteredLLM:
    """One configured LLM entry."""

    id: str
    backend: LLMBackend
    model_name: str
    enabled: bool

    @property
    def cli_spec(self) -> str:
        return f"{self.backend}:{self.model_name}"


def load_model_registry(path: Path) -> list[RegisteredLLM]:
    """Load configured LLMs from JSON registry."""

    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_models = data.get("models", [])

    if not isinstance(raw_models, list):
        raise ValueError("Model registry must contain a list field named 'models'.")

    models: list[RegisteredLLM] = []

    for raw in raw_models:
        model_id = str(raw.get("id", "")).strip()
        backend = str(raw.get("backend", "")).strip().lower()
        model_name = str(raw.get("model_name", "")).strip()
        enabled = bool(raw.get("enabled", False))

        if not model_id:
            raise ValueError(f"Model entry is missing id: {raw}")

        if backend not in {"ollama", "transformers", "vllm"}:
            raise ValueError(f"Unsupported backend for {model_id}: {backend}")

        if not model_name:
            raise ValueError(f"Model entry is missing model_name: {model_id}")

        models.append(
            RegisteredLLM(
                id=model_id,
                backend=backend,  # type: ignore[arg-type]
                model_name=model_name,
                enabled=enabled,
            )
        )

    return models


def load_enabled_models(path: Path) -> list[RegisteredLLM]:
    """Load only enabled models."""

    return [model for model in load_model_registry(path) if model.enabled]