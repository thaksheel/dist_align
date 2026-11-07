import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import extract_json, strip_reasoning


@dataclass
class Prompt:
    """One LLM prompt.

    Parameters
    ----------
    strategy : str
        Strategy that produced the prompt.
    system, user : str
        Message contents.
    kind : {"binary", "prob", "paraphrases"}
        Expected answer format.
    """

    strategy: str
    system: str
    user: str
    kind: str

    @property
    def messages(self) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


class PromptStrategies:
    """Prompt templates and answer parsers for every strategy.

    Each strategy is a method named after it that returns a `Prompt`. To add one,
    write the method and add its name to `config.strategies` (and to
    `config.prob_strategies` if it returns a probability).

    Parameters
    ----------
    config : ExperimentConfig
        Uses `prob_strategies`, `n_samples`, `prob_samples`, `population_size`
        and `knowledge_restriction_text`.

    Methods
    -------
    build(strategy, persona, item, examples=None)
        Prompt for one strategy, persona and item.
    paraphrase_generation(persona, k)
        Prompt that rewrites a persona k different ways.
    parse(prompt, text)
        0/1 for binary prompts, a probability for probability prompts.
    n_calls(strategy)
        Calls per item for a strategy.
    """

    YES_NO = "Reply with exactly one word: Yes or No."
    PROB_JSON = (
        "Return JSON only, with probabilities summing to 1: "
        '{"yes": <0-1>, "no": <0-1>}'
    )

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def n_calls(self, strategy: str) -> int:
        cfg = self.config
        return cfg.prob_samples if strategy in cfg.prob_strategies else cfg.n_samples

    def build(
        self,
        strategy: str,
        persona: str,
        item: pd.Series,
        examples: pd.DataFrame | None = None,
    ) -> Prompt:
        if strategy == "fewshot_self":
            return self.fewshot_self(persona, item, examples)
        return getattr(self, strategy)(persona, item)

    @staticmethod
    def question(item: pd.Series) -> str:
        return (
            f"Please consider the following product category: {item['category']}.\n"
            "Suppose you are in a grocery store, and you see the following product "
            f"in that category: {item['product']}.\n"
            f"The product is priced at: ${item['price']:.2f}.\n"
            "Would you or would you not purchase the product?"
        )

    @staticmethod
    def respondent(persona: str, extra: str = "") -> str:
        extra = f"\n{extra}\n" if extra else ""
        return (
            "You are taking part in a survey as the person described below. "
            f"Answer exactly as this person would, not as an AI assistant.\n{extra}"
            f"\nPERSON:\n{persona}"
        )

    def direct(self, persona, item) -> Prompt:
        return Prompt(
            "direct",
            self.respondent(persona),
            f"{self.question(item)}\n\n{self.YES_NO}",
            "binary",
        )

    def paraphrase(self, persona, item) -> Prompt:
        return Prompt(
            "paraphrase",
            self.respondent(persona),
            f"{self.question(item)}\n\n{self.YES_NO}",
            "binary",
        )

    def knowledge_restriction(self, persona, item) -> Prompt:
        extra = f"KNOWLEDGE LIMITS: {self.config.knowledge_restriction_text}"
        return Prompt(
            "knowledge_restriction",
            self.respondent(persona, extra),
            f"{self.question(item)}\n\n{self.YES_NO}",
            "binary",
        )

    def verbalized(self, persona, item) -> Prompt:
        return Prompt(
            "verbalized",
            self.respondent(persona),
            f"{self.question(item)}\n\nGive the probability that this person "
            f"would purchase the product. {self.PROB_JSON}",
            "prob",
        )

    def population(self, persona, item) -> Prompt:
        n = self.config.population_size
        return Prompt(
            "population",
            "You are an expert in consumer behavior estimating how groups of real "
            "shoppers respond to survey questions.",
            f"Imagine {n} people who all match this profile:\n{persona}\n\n"
            f"Each of them is asked:\n{self.question(item)}\n\n"
            f"How many of the {n} would purchase the product? Return JSON only, with "
            f'counts summing to {n}: {{"purchase": <count>, "not_purchase": <count>}}',
            "prob",
        )

    def fewshot_self(self, persona, item, examples: pd.DataFrame) -> Prompt:
        shots = "\n".join(
            f"- {r.product} ({r.category}), priced at ${r.price:.2f}: "
            f"{'Would purchase' if r.answer == 1 else 'Would NOT purchase'}"
            for r in examples.itertuples()
        )
        return Prompt(
            "fewshot_self",
            self.respondent(persona),
            "Earlier, this person answered the same kind of question about other "
            f"products:\n{shots}\n\nNow they are asked:\n{self.question(item)}\n\n"
            "Based on this person's profile and earlier answers, give the probability "
            f"that they would purchase the product. {self.PROB_JSON}",
            "prob",
        )

    def paraphrase_generation(self, persona: str, k: int) -> Prompt:
        return Prompt(
            "paraphrase_generation",
            "You rewrite personal profiles for survey research.",
            f"Below is a profile of one survey respondent. Write {k} different "
            "descriptions of this same person.\n\nRules:\n"
            "- Keep every fact exactly as given. Do not add or change facts.\n"
            "- Make each description as different as possible from all the others you "
            "have written: vary the voice (first person, third person, bullet list, "
            "interview, a friend describing them, a diary entry), which facts come "
            "first or are emphasized, tone, and length.\n"
            "- Each description should read like a distinct, realistic way this person "
            f"could be portrayed.\n\nPROFILE:\n{persona}\n\n"
            'Return JSON only: {"personas": ["...", "..."]}',
            "paraphrases",
        )

    def parse(self, prompt: Prompt, text: str) -> float | None:
        if prompt.kind == "binary":
            m = re.search(r"\b(yes|no)\b", strip_reasoning(text).lower())
            return float(m.group(1) == "yes") if m else None
        d = extract_json(text) or {}
        pos = next(
            (float(v) for k, v in d.items() if k.lower() in ("yes", "purchase")), None
        )
        neg = next(
            (float(v) for k, v in d.items() if k.lower() in ("no", "not_purchase")),
            None,
        )
        if pos is None:
            return None
        if neg is None or pos + neg == 0:
            return float(np.clip(pos if pos <= 1 else pos / 100, 0, 1))
        return float(np.clip(pos / (pos + neg), 0, 1))

    @staticmethod
    def parse_paraphrases(text: str) -> list[str]:
        return [
            p for p in (extract_json(text) or {}).get("personas", []) if str(p).strip()
        ]
