# src/patentllm/generation/llm_clients.py

"""LLM clients for local and open-source report generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import requests

from patentllm.generation.config import GenerationConfig


LLMBackend = Literal["ollama", "transformers", "vllm"]


@dataclass(frozen=True, slots=True)
class LLMModelSpec:
    """One model/backend combination.

    Format used in CLI:
        ollama:qwen3:8b
        transformers:Qwen/Qwen3-8B
        vllm:Qwen/Qwen3-8B
    """

    backend: LLMBackend
    model_name: str

    @property
    def safe_name(self) -> str:
        return (
            f"{self.backend}_{self.model_name}"
            .replace("/", "_")
            .replace(":", "_")
            .replace("\\", "_")
        )


class BaseLLMClient(ABC):
    """Common interface for all generation backends."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate report text."""


class OllamaClient(BaseLLMClient):
    """Client for Ollama local chat models."""

    def generate(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"

        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_new_tokens,
            },
        }

        response = requests.post(url, json=payload, timeout=900)
        response.raise_for_status()

        data = response.json()
        content = data.get("message", {}).get("content")

        if not content:
            raise RuntimeError(f"Ollama returned empty response for model={model_name}")

        return str(content)


class TransformersClient(BaseLLMClient):
    """Direct Hugging Face Transformers generation backend.

    This backend loads the model inside Python. It is useful for controlled
    experiments, but it can be memory-heavy on laptops.
    """

    def generate(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformers backend requires: pip install -e \".[generation-transformers]\""
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.config.transformers_device_map,
            torch_dtype=self.config.transformers_torch_dtype,
            trust_remote_code=True,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = _build_transformers_prompt(tokenizer, messages)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        do_sample = self.config.temperature > 0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not text:
            raise RuntimeError(f"Transformers returned empty response for model={model_name}")

        return text


class VLLMClient(BaseLLMClient):
    """Client for a vLLM OpenAI-compatible chat server."""

    def generate(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        url = f"{self.config.vllm_base_url.rstrip('/')}/v1/chat/completions"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
        }

        response = requests.post(url, json=payload, timeout=900)
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError(f"vLLM returned no choices for model={model_name}")

        content = choices[0].get("message", {}).get("content")

        if not content:
            raise RuntimeError(f"vLLM returned empty response for model={model_name}")

        return str(content)


def parse_llm_spec(raw_spec: str) -> LLMModelSpec:
    """Parse backend:model_name from the CLI."""

    if ":" not in raw_spec:
        raise ValueError(
            f"Invalid model spec: {raw_spec}. Expected format like 'ollama:qwen3:8b'."
        )

    backend_raw, model_name = raw_spec.split(":", 1)
    backend = backend_raw.strip().lower()

    if backend not in {"ollama", "transformers", "vllm"}:
        raise ValueError(
            f"Unsupported backend: {backend}. Use one of: ollama, transformers, vllm."
        )

    if not model_name.strip():
        raise ValueError(f"Missing model name in spec: {raw_spec}")

    return LLMModelSpec(backend=backend, model_name=model_name.strip())  # type: ignore[arg-type]


def build_llm_client(spec: LLMModelSpec, config: GenerationConfig) -> BaseLLMClient:
    """Create the right client for a model spec."""

    if spec.backend == "ollama":
        return OllamaClient(config)

    if spec.backend == "transformers":
        return TransformersClient(config)

    if spec.backend == "vllm":
        return VLLMClient(config)

    raise ValueError(f"Unsupported backend: {spec.backend}")


def _build_transformers_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """Use model chat template when available; otherwise use a safe fallback."""

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        system = messages[0]["content"]
        user = messages[1]["content"]
        return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"