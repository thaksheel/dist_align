from dataclasses import dataclass, field
from enum import Enum


@dataclass
class LLMOut:
    response: str
    input_token: int
    output_token: int
    model_name: str
    provider: str


class OpenModelSelection(Enum):
    tllama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qwen38_27b = "Qwen/Qwen3.8-27B"
    llama31_8b = "meta-llama/Llama-3.1-8B-Instruct"
    qwen25_7b = "Qwen/Qwen2.5-7B-Instruct"
    qwen3_8b = "Qwen/Qwen3-8B"


class CloseModelSelection(Enum):
    gpt6_luna = "gpt-6-luna"
    gpt6_sol = "gpt-6-sol"
    gpt6_astra = "gpt-6-astra"
    gpt56_sol = "gpt-5.6-sol"
    gpt56_luna = "gpt-5.6-luna"
    gpt56_terra = "gpt-5.6-terra"
    opus55 = "claude-opus-5-5"
    sonnet55 = "claude-sonnet-5-5"
    haiku55 = "claude-haiku-5-5"


def _category_blocks() -> dict:
    return {
        "marketing": ["product preferences pricing"],
        "public_health": ["disease"],
        "political_science": ["myside"],
        "economics": [
            "allais",
            "wta wtp",
            "sunk cost",
            "absolute vs relative",
            "less is more",
            "probability matching",
        ],
        "psychology": [
            "base rate",
            "linda",
            "outcome bias",
            "anchoring",
            "proportion dominance",
            "false consensus",
            "non experimental heuristics",
        ],
    }


@dataclass
class ExperimentConfig:
    """Every setting for one experiment run, shared by all classes.

    Parameters
    ----------
    hf_name : str
        Hugging Face dataset id.
    category : str
        Question category to load; a key of `category_blocks`.
    category_blocks : dict of str to list of str
        Block-name keywords for each category, matched ignoring case, spaces
        and punctuation.
    truth_column : str
        Column with ground-truth answers.
    reference_column : str
        Column with the same questions from another wave (retest and few-shot).
    persona_blocks : tuple of str or None
        Persona blocks to include. None uses the full persona text.
    max_persona_chars : int
        Persona length cap.
    condition_key : {"price_bin", "price_level", "product"}
        How pricing items are grouped into conditions.
    price_bins : tuple of (str, float, float)
        Label, lower and upper share of the regular price for each bin.
    n_twins : int
        Twins in the test set.
    n_units : int or None
        Products (or questions) in the test set. None uses all.
    strategies : tuple of str
        Strategies to run, in order. Choose from ("direct", "verbalized", "population", "fewshot_self", "knowledge_restriction", "paraphrase")
    prob_strategies : tuple of str
        Strategies that return a probability rather than Yes/No.
    n_samples : int
        Calls per item for Yes/No strategies (also the number of paraphrases).
    prob_samples : int
        Calls per item for probability strategies.
    fewshot_k : int
        Reference-wave examples shown in `fewshot_self`.
    population_size : int
        Group size in `population`.
    knowledge_restriction_text : str
        Knowledge limits added in `knowledge_restriction`.
    json_mode : bool
        Ask the provider for JSON output on probability prompts.
    n_bootstrap : int
        Bootstrap draws for the JSD confidence interval.
    n_sim_reps : int
        Simulated Yes/No draws for heterogeneity of probability predictors.
    pilot_items : int
        Items per strategy in `pilot()`.
    max_workers : int
        Parallel LLM calls (1 is used for "hf").
    max_retries : int
        Retries per failed LLM call.
    cache_path : str or None
        JSONL response cache. None keeps the cache in memory only.
    output_dir : str
        Folder for saved runs.
    save_all_prompts : bool
        Also save every unique prompt, not only one example per strategy.
    seed : int
        Seed for test-set selection, few-shot examples and simulation.
    """

    hf_name: str = "LLM-Digital-Twin/Twin-2K-500"
    category: str = "marketing"
    category_blocks: dict = field(default_factory=_category_blocks)
    truth_column: str = "wave4_Q_wave4_A"
    reference_column: str = "wave4_Q_wave1_3_A"
    persona_blocks: tuple | None = ("Demographics", "Economic preferences")
    max_persona_chars: int = 20000
    condition_key: str = "price_bin"
    price_bins: tuple = (
        ("0-40%", 0.0, 0.4),
        ("60-100%", 0.6, 1.0),
        ("120-160%", 1.2, 1.6),
        ("180-200%", 1.8, 2.0),
    )
    n_twins: int = 20
    n_units: int | None = 10

    strategies: tuple = ("direct", "verbalized", "population", "fewshot_self")
    prob_strategies: tuple = ("verbalized", "population", "fewshot_self")
    n_samples: int = 3
    prob_samples: int = 1
    fewshot_k: int = 10
    population_size: int = 100
    knowledge_restriction_text: str = (
        "This person does not know typical retail prices or what other stores "
        "charge beyond their own everyday shopping experience. They have no "
        "training in economics or marketing and decide quickly, by gut feeling, "
        "as they would in a real store. Do not use knowledge, calculations, or "
        "price comparisons this person would not make."
    )
    json_mode: bool = True

    n_bootstrap: int = 1000
    n_sim_reps: int = 200
    pilot_items: int = 10

    max_workers: int = 6
    max_retries: int = 2
    cache_path: str | None = "cache/llm_cache.jsonl"
    output_dir: str = "exports"
    save_all_prompts: bool = False
    seed: int = 42
