from dataclasses import dataclass
from typing import Dict, List, Tuple, Any 
import json
import time 
import threading
import os


# Coarse price bins (share of regular price) for small test sets.
DEFAULT_PRICE_BINS = (
    ("0-40%", 0.0, 0.4),
    ("60-100%", 0.6, 1.0),
    ("120-160%", 1.2, 1.6),
    ("180-200%", 1.8, 2.0),
)
STRATEGIES = (
    "direct",  # control: one Yes/No answer per call
    "verbalized",  # 1. verbalized distribution
    "population",  # 3. population framing
    "knowledge_restriction",  # 5. persona-consistent knowledge restriction
    "paraphrase",  # 6. LLM-generated diverse persona paraphrases
    "fewshot_self",  # 4a. twin's own wave 3 pricing answers as few-shot
)
DEFAULT_RESTRICTION = (
    "This person does not know typical retail prices or what other stores "
    "charge beyond their own everyday shopping experience. They have no "
    "training in economics or marketing and decide quickly, by gut feeling, "
    "as they would in a real store. Do not use knowledge, calculations, or "
    "price comparisons this person would not make."
)


@dataclass
class LLMOut:
    response: str
    input_token: int
    output_token: int
    model_name: str
    provider: str


@dataclass
class ExperimentConfig:
    model: str = "gpt-4.1-mini"
    temperature: float | None = 1.0  # None = omit (needed for reasoning models)
    n_samples: int = 5  # calls per twin x item, all strategies
    strategies: tuple = STRATEGIES
    persona_mode: str = "blocks"  # "demographics" | "blocks" | "full_text"
    persona_blocks: tuple = ("Demographics", "Economic preferences")
    max_persona_chars: int = 20000
    condition_key: str = (
        "price_bin"  # "price_bin" | "price_level" | "product" | "product_price"
    )
    price_bins: tuple = DEFAULT_PRICE_BINS
    prob_samples: int = 1  # calls per item for probability strategies
    fewshot_k: int = 10  # wave 3 examples shown in fewshot_self
    json_mode: bool = True  # force JSON output for JSON-returning prompts
    population_size: int = 100
    knowledge_restriction_text: str = DEFAULT_RESTRICTION
    n_bootstrap: int = 1000
    n_baseline_reps: int = 200  # simulation reps for baseline heterogeneity
    seed: int = 42
    max_workers: int = 8
    max_retries: int = 2
    max_output_tokens: int = 200
    cache_path: str | None = "cache/llm_cache.jsonl"
    output_dir: str = "results"


class ResponseCache:
    def __init__(self, path: str | None):
        self.path = path
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, text: str) -> None:
        with self._lock:
            self._data[key] = text
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "text": text}) + "\n")


class OpenAIClient:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(self, model: str, max_retries: int = 2):
        from openai import OpenAI  # imported lazily so mock runs need no SDK

        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model
        self.max_retries = max_retries

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None,
        max_output_tokens: int,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            instructions=system,
            input=user,
            max_output_tokens=max(16, max_output_tokens),
        )
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        if temperature is not None:
            kwargs["temperature"] = temperature
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.responses.create(**kwargs).output_text
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise
                wait = 2**attempt
                # log.warning("API error (%s); retrying in %ss", exc, wait)
                time.sleep(wait)
        return ""
