import threading
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .config import ExperimentConfig
from .dataset import Twin2KDataset
from .metrics import DistributionMetrics
from .strategies import Prompt, PromptStrategies
from .utils import ResponseCache, stable_hash, task_rng
from .llm_provider import LLMProvider


class RecoverDistributionAlignment:
    """Runs prompting strategies on the test set, evaluates them and saves the results.

    Parameters
    ----------
    config : ExperimentConfig
        Run settings.
    dataset : Twin2KDataset
        Loaded dataset with a selected test set.
    strategies : PromptStrategies
        Prompt templates and parsers.
    provider : LLMProvider
        Any object with `invoke(messages, json_mode) -> LLMOut` and
        `provider`, `model_name`, `temperature` attributes.

    Attributes
    ----------
    results : pandas.DataFrame
        One row per LLM call.
    metrics : pandas.DataFrame
        One row per strategy, baseline and human retest.
    ranking : pandas.DataFrame
        Strategies ranked by distributional fit.
    prompt_examples : dict
        First prompt and response seen for each strategy.
    paraphrases : dict
        Generated personas per twin (`paraphrase` strategy).

    Methods
    -------
    pilot(n_items=None)
        Run a few items per strategy; returns parse-failure and mean-prediction summary.
    run(strategies=None)
        Run strategies (default: `config.strategies`) on the test set.
    evaluate()
        Compute `metrics` and `ranking`.
    save(tag=None)
        Write per-call results, summaries, prompt examples and plots.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        dataset: Twin2KDataset,
        strategies: PromptStrategies,
        provider: LLMProvider,
    ):
        self.config = config
        self.dataset = dataset
        self.strategies = strategies
        self.provider = provider
        self.metrics_engine = DistributionMetrics(config, dataset.unit)
        self.cache = ResponseCache(config.cache_path)
        self.results = self.metrics = self.ranking = None
        self.prompt_examples, self.paraphrases, self.all_prompts = {}, {}, {}
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._lock = threading.Lock()

    @property
    def test_items(self) -> pd.DataFrame:
        return self.dataset.test_items

    @property
    def retest_items(self) -> pd.DataFrame:
        ref, unit = self.dataset.reference_items, self.dataset.unit
        return ref[
            ref["pid"].isin(self.test_items["pid"])
            & ref[unit].isin(self.test_items[unit])
        ]

    @property
    def workers(self) -> int:
        return 1 if self.provider.provider == "hf" else self.config.max_workers

    def call_llm(self, prompt: Prompt, sample_idx: int) -> dict:
        json_mode = self.config.json_mode and prompt.kind != "binary"
        key = stable_hash(
            [
                self.provider.provider,
                self.provider.model_name,
                self.provider.temperature,
                prompt.messages,
                sample_idx,
                json_mode,
            ]
        )
        hit = self.cache.get(key)
        if hit:
            return dict(
                response=hit["response"],
                input_tokens=hit["input_tokens"],
                output_tokens=hit["output_tokens"],
                cached=True,
                latency_s=0.0,
                error=None,
            )
        for attempt in range(self.config.max_retries + 1):
            start = time.time()
            try:
                out = self.provider.invoke(
                    messages=prompt.messages, json_mode=json_mode
                )
                rec = dict(
                    response=str(out.response),
                    input_tokens=out.input_token,
                    output_tokens=out.output_token,
                )
                self.cache.set(key, rec)
                return rec | dict(
                    cached=False, latency_s=round(time.time() - start, 3), error=None
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2**attempt)
        return dict(
            response="",
            input_tokens=None,
            output_tokens=None,
            cached=False,
            latency_s=None,
            error=error,
        )

    def n_calls_planned(self) -> int:
        n = sum(
            len(self.test_items) * self.strategies.n_calls(s)
            for s in self.config.strategies
        )
        return n + self.test_items["pid"].nunique() * (
            "paraphrase" in self.config.strategies
        )

    def _remember(self, prompt: Prompt, response: str, **context) -> str:
        prompt_id = stable_hash(prompt.messages, 12)
        with self._lock:
            self.prompt_examples.setdefault(
                prompt.strategy,
                dict(
                    strategy=prompt.strategy,
                    kind=prompt.kind,
                    prompt_id=prompt_id,
                    system=prompt.system,
                    user=prompt.user,
                    response=response,
                    **context,
                ),
            )
            if self.config.save_all_prompts:
                self.all_prompts.setdefault(
                    prompt_id,
                    dict(
                        prompt_id=prompt_id,
                        strategy=prompt.strategy,
                        system=prompt.system,
                        user=prompt.user,
                    ),
                )
        return prompt_id

    def get_paraphrases(self, pid: str) -> list[str]:
        if pid not in self.paraphrases:
            k = self.config.n_samples
            prompt = self.strategies.paraphrase_generation(
                self.dataset.build_persona(pid), k
            )
            out = self.call_llm(prompt, sample_idx=-1)
            self._remember(prompt, out["response"], pid=pid)
            variants = self.strategies.parse_paraphrases(out["response"]) or [
                self.dataset.build_persona(pid)
            ]
            self.paraphrases[pid] = [variants[i % len(variants)] for i in range(k)]
        return self.paraphrases[pid]

    def _tasks(self, strategy: str, items: pd.DataFrame) -> list:
        return [
            (strategy, row, s)
            for _, row in items.iterrows()
            for s in range(self.strategies.n_calls(strategy))
        ]

    def _run_one(self, strategy: str, item: pd.Series, s: int) -> dict:
        pid, unit = item["pid"], item[self.dataset.unit]
        persona = (
            self.get_paraphrases(pid)[s]
            if strategy == "paraphrase"
            else self.dataset.build_persona(pid)
        )
        examples = (
            self.dataset.fewshot_pool(pid, unit) if strategy == "fewshot_self" else None
        )
        prompt = self.strategies.build(strategy, persona, item, examples)
        out = self.call_llm(prompt, s)
        prompt_id = self._remember(
            prompt, out["response"], pid=pid, unit=unit, sample_idx=s
        )
        value = self.strategies.parse(prompt, out["response"])
        sampled = (
            value
            if prompt.kind == "binary" or value is None
            else float(
                task_rng(self.config.seed, strategy, pid, unit, s).random() < value
            )
        )
        return dict(
            strategy=strategy,
            pid=pid,
            **{self.dataset.unit: unit},
            product=item["product"],
            price=item["price"],
            price_ratio=item["price_ratio"],
            condition=item["condition"],
            human_answer=item["answer"],
            sample_idx=s,
            prompt_kind=prompt.kind,
            prompt_id=prompt_id,
            value=value,
            sampled_answer=sampled,
            parse_ok=value is not None,
            **out,
            provider=self.provider.provider,
            model=self.provider.model_name,
        )

    def _run_sequential(self, strategy: str, items: pd.DataFrame) -> list:
        if strategy == "paraphrase":
            for pid in tqdm(items["pid"].unique(), desc="paraphrase personas"):
                self.get_paraphrases(pid)
        return [
            self._run_one(*t) for t in tqdm(self._tasks(strategy, items), desc=strategy)
        ]

    def _run_parallel(self, strategy: str, items: pd.DataFrame) -> list:
        with ThreadPoolExecutor(self.config.max_workers) as ex:
            if strategy == "paraphrase":
                pids = items["pid"].unique()
                list(
                    tqdm(
                        ex.map(self.get_paraphrases, pids),
                        total=len(pids),
                        desc="paraphrase personas",
                    )
                )
            tasks = self._tasks(strategy, items)
            return list(
                tqdm(
                    ex.map(lambda t: self._run_one(*t), tasks),
                    total=len(tasks),
                    desc=strategy,
                )
            )

    def _run(
        self, strategies: tuple, items: pd.DataFrame, parallel: bool
    ) -> pd.DataFrame:
        run_strategy = self._run_parallel if parallel else self._run_sequential
        frames = [
            pd.DataFrame(run_strategy(strategy, items)) for strategy in strategies
        ]
        results = pd.concat(frames, ignore_index=True)
        results.insert(0, "call_id", range(len(results)))
        return results

    def run(
        self, strategies: tuple | None = None, parallel: bool = False
    ) -> pd.DataFrame:
        self.results = self._run(
            strategies or self.config.strategies, self.test_items, parallel
        )
        return self.results

    def evaluate(self) -> pd.DataFrame:
        self.metrics = self.metrics_engine.evaluate(
            self.results, self.test_items, self.retest_items
        )
        self.ranking = self.metrics_engine.rank(self.metrics)
        return self.metrics

    def condition_table(self) -> pd.DataFrame:
        return self.metrics_engine.condition_table(
            self.results, self.test_items, self.retest_items
        )

    def token_usage(self) -> pd.DataFrame:
        return self.results.groupby("strategy", sort=False).agg(
            calls=("call_id", "count"),
            cached=("cached", "sum"),
            errors=("error", "count"),
            input_tokens=("input_tokens", "sum"),
            output_tokens=("output_tokens", "sum"),
            mean_latency_s=("latency_s", "mean"),
        )
