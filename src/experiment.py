import hashlib, logging, re, os, json, time 
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, wasserstein_distance

from . import ExperimentConfig, ResponseCache 


log = logging.getLogger("dist_align")

# Strategies that return a probability. They are evaluated on the probability
# itself (no Bernoulli sampling noise) and need fewer calls per item.
PROB_STRATEGIES = {"verbalized", "population", "fewshot_self"}


def _clean(text: Any) -> str:
    TAG_RE = re.compile(r"<[^>]+>")
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", str(text or ""))).strip()


def _iter_questions(obj: Any, block: str | None = None):
    """Yield (block_name, question_dict) for every question in a persona JSON."""
    if isinstance(obj, dict):
        block = obj.get("BlockName", block)
        if "QuestionText" in obj:
            yield block, obj
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _iter_questions(v, block)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_questions(v, block)


class DistributionAlignmentExperiment:
    """Load Twin-2K-500 pricing data, run prompting strategies, and evaluate."""

    def __init__(self, config: ExperimentConfig, client: Any = None):
        self.config = config
        self.client = client
        self.cache = ResponseCache(self.config.cache_path)
        self.raw: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None  # wave 4 ground truth
        self.retest_items: pd.DataFrame | None = None  # wave 3 answers
        self.test_items: pd.DataFrame | None = None
        self.results: pd.DataFrame | None = None
        self._personas: dict[str, str] = {}
        self._paraphrases: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ data
    def load_data(
        self,
        source: str = "hf",
        path: str | None = None,
        hf_name: str = "LLM-Digital-Twin/Twin-2K-500",
        dataframe: pd.DataFrame | None = None,
        regular_prices: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Load the `wave_split` config and extract pricing items.

        source: "hf" (Hugging Face), "parquet", "jsonl", or "dataframe".
        regular_prices: optional {product_text: regular_price}. If omitted,
            the regular price is inferred as max observed price / 2 across all
            participants (prices span 0-200% of regular).
        """
        if source == "hf":
            from datasets import load_dataset

            df = load_dataset(hf_name, "wave_split")["data"].to_pandas()
        elif source == "parquet":
            df = pd.read_parquet(path)
        elif source == "jsonl":
            df = pd.read_json(path, lines=True)
        elif source == "dataframe":
            df = dataframe.copy()
        else:
            raise ValueError(f"Unknown source: {source}")

        df["pid"] = df["pid"].astype(str)
        self.raw = df.set_index("pid", drop=False)
        self.items = self._extract_pricing(df, "wave4_Q_wave4_A")
        if "wave4_Q_wave1_3_A" in df.columns:
            self.retest_items = self._extract_pricing(df, "wave4_Q_wave1_3_A")
        self._assign_price_levels(regular_prices)
        log.info(
            "Loaded %d twins, %d pricing answers, %d products",
            df.shape[0],
            len(self.items),
            self.items.product_id.nunique(),
        )
        return self.items 

    @staticmethod
    def _to_json(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return value

    def _extract_pricing(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        rows = []
        PRICE_RE = re.compile(r"priced at:?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
        CATEGORY_RE = re.compile(r"product category:\s*(.*?)\.?\s*Suppose", re.I | re.S)
        PRODUCT_RE = re.compile(
            r"in that category:\s*(.*?)\.?\s*The product is priced", re.I | re.S
        )
        for pid, blob in zip(df["pid"], df[column]):
            for ll, q in _iter_questions(self._to_json(blob)):
                text = _clean(q.get("QuestionText"))
                pm = PRICE_RE.search(text)
                if not pm:
                    continue
                cat = CATEGORY_RE.search(text)
                prod = PRODUCT_RE.search(text)
                rows.append(
                    dict(
                        pid=str(pid),
                        qid=q.get("QuestionID"),
                        category=cat.group(1).strip() if cat else "",
                        product=(
                            prod.group(1).strip() if prod else q.get("QuestionID", "")
                        ),
                        price=float(pm.group(1).replace(",", "")),
                        answer=self._answer_to_binary(q),
                        question_text=text,
                    )
                )
        out = pd.DataFrame(rows)
        if out.empty:
            raise ValueError(
                f"No pricing questions found in '{column}'. Run "
                "inspect_pricing_items() and adjust PRICE_RE if needed."
            )
        return out.dropna(subset=["answer"]).astype({"answer": int})

    @staticmethod
    def _answer_to_binary(q: dict) -> int | None:
        ans = q.get("Answers") or {}
        text = ans.get("SelectedText")
        if text is None:
            pos, opts = ans.get("SelectedByPosition"), q.get("Options") or []
            if isinstance(pos, int) and 1 <= pos <= len(opts):
                text = opts[pos - 1]
        if isinstance(text, str):
            t = text.strip().lower()
            if t.startswith("yes"):
                return 1
            if t.startswith("no"):
                return 0
        return None

    def _assign_price_levels(self, regular_prices: dict[str, float] | None) -> None:
        frames = [f for f in (self.items, self.retest_items) if f is not None]
        allp = pd.concat(frames)
        products = sorted(allp["product"].unique())
        pid_map = {p: f"P{i + 1:02d}" for i, p in enumerate(products)}
        regular = (
            regular_prices or (allp.groupby("product")["price"].max() / 2).to_dict()
        )
        for f in frames:
            f["product_id"] = f["product"].map(pid_map)
            f["regular_price"] = f["product"].map(regular)
            ratio = f["price"] / f["regular_price"].replace(0, np.nan)
            f["price_ratio"] = (np.round(ratio / 0.2) * 0.2).clip(0, 2).round(1)
            f["condition"] = self._condition_of(f)

    def _condition_of(self, f: pd.DataFrame) -> pd.Series:
        key = self.config.condition_key
        if key == "price_bin":

            def to_bin(r):  # numbered labels keep bins in price order when sorted
                for i, (label, lo, hi) in enumerate(self.config.price_bins, 1):
                    if lo - 1e-9 <= r <= hi + 1e-9:
                        return f"{i}: {label}"
                return "other"

            return f["price_ratio"].map(to_bin)
        if key == "price_level":
            return f["price_ratio"]
        if key == "product":
            return f["product_id"]
        if key == "product_price":
            return f["product_id"] + "@" + f["price_ratio"].map(lambda r: f"{r:.1f}")
        raise ValueError(f"Unknown condition_key: {key}")

    def inspect_pricing_items(self, n: int = 2) -> None:
        """Print raw pricing questions so you can verify the parser."""
        blob = self._to_json(self.raw.iloc[0]["wave4_Q_wave4_A"])
        shown = 0
        PRICE_RE = re.compile(r"priced at:?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
        for block, q in _iter_questions(blob):
            if PRICE_RE.search(_clean(q.get("QuestionText"))):
                print(f"[block: {block}]")
                print(json.dumps(q, indent=2)[:1500])
                shown += 1
                if shown >= n:
                    break
        print("\nParsed items (head):")
        print(self.items.drop(columns="question_text").head(10).to_string())
        print("\nAnswers per price level:")
        print(self.items.groupby("price_ratio")["answer"].agg(["count", "mean"]))

    def select_test_set(
        self,
        n_twins: int = 10,
        n_products: int | None = 10,
        pids: list[str] | None = None,
        products: list[str] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Pick a small random subset of twins and products for testing."""
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        all_pids = sorted(self.items["pid"].unique())
        all_prods = sorted(self.items["product_id"].unique())
        pids = pids or list(
            rng.choice(all_pids, size=min(n_twins, len(all_pids)), replace=False)
        )
        if products is None:
            products = (
                all_prods
                if n_products is None
                else list(
                    rng.choice(
                        all_prods, size=min(n_products, len(all_prods)), replace=False
                    )
                )
            )
        mask = self.items["pid"].isin(pids) & self.items["product_id"].isin(products)
        self.test_items = self.items[mask].reset_index(drop=True)
        log.info(
            "Test set: %d twins x %d products = %d items",
            len(pids),
            len(products),
            len(self.test_items),
        )
        return self.test_items

    # --------------------------------------------------------------- persona
    def build_persona(self, pid: str) -> str:
        if pid in self._personas:
            return self._personas[pid]
        row = self.raw.loc[pid]
        if self.config.persona_mode == "full_text":
            text = str(row["wave1_3_persona_text"])
        else:
            blocks = (
                ("Demographics",)
                if self.config.persona_mode == "demographics"
                else self.config.persona_blocks
            )
            lines, seen = [], set()
            PRICE_RE = re.compile(
                r"priced at:?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.I
            )
            for block, q in _iter_questions(self._to_json(row["wave1_3_persona_json"])):
                seen.add(block)
                qtext = _clean(q.get("QuestionText"))
                if block not in blocks or not qtext or PRICE_RE.search(qtext):
                    continue  # skip other blocks; leakage guard for pricing items
                if q.get("Answers"):
                    lines.append(f"Q: {qtext}\nA: {self._format_answer(q['Answers'])}")
            missing = [b for b in blocks if b not in seen]
            if missing and not getattr(self, "_warned_blocks", False):
                log.warning(
                    "Persona blocks not found: %s. Available: %s",
                    missing,
                    sorted(b for b in seen if b),
                )
                self._warned_blocks = True
            text = "\n".join(lines)
        text = text[: self.config.max_persona_chars]
        self._personas[pid] = text
        return text

    @staticmethod
    def _format_answer(ans: Any) -> str:
        if not isinstance(ans, dict):
            return _clean(ans)
        if "SelectedText" in ans:
            v = ans["SelectedText"]
            return "; ".join(map(_clean, v)) if isinstance(v, list) else _clean(v)
        if "Text" in ans:
            return _clean(ans["Text"])
        return json.dumps(ans, ensure_ascii=False)[:500]

    # --------------------------------------------------------------- prompts
    @staticmethod
    def item_text(item: pd.Series) -> str:
        """Original Twin-2K-500 question wording."""
        return (
            f"Please consider the following product category: {item['category']}.\n"
            "Suppose you are in a grocery store, and you see the following "
            f"product in that category: {item['product']}.\n"
            f"The product is priced at: ${item['price']:.2f}.\n"
            "Would you or would you not purchase the product?"
        )

    @staticmethod
    def _respondent_system(persona: str, extra: str = "") -> str:
        return (
            "You are taking part in a survey as the person described below. "
            "Answer exactly as this person would, not as an AI assistant.\n"
            f"{extra}\n\nPERSON:\n{persona}"
        ).replace("\n\n\n", "\n\n")

    def build_prompt(
        self, strategy: str, persona: str, item: pd.Series
    ) -> tuple[str, str, str]:
        """Return (system, user, output_kind) for one call."""
        q = self.item_text(item)
        if strategy in ("direct", "paraphrase"):
            return (
                self._respondent_system(persona),
                f"{q}\n\nReply with exactly one word: Yes or No.",
                "binary",
            )
        if strategy == "knowledge_restriction":
            extra = f"\nKNOWLEDGE LIMITS: {self.config.knowledge_restriction_text}"
            return (
                self._respondent_system(persona, extra),
                f"{q}\n\nReply with exactly one word: Yes or No.",
                "binary",
            )
        if strategy == "verbalized":
            return (
                self._respondent_system(persona),
                f"{q}\n\nGive the probability that this person would purchase "
                "the product. Return JSON only, with probabilities summing to 1: "
                '{"yes": <0-1>, "no": <0-1>}',
                "prob",
            )
        if strategy == "fewshot_self":
            shots = self.fewshot_examples(item["pid"], item["product_id"])
            return (
                self._respondent_system(persona),
                "Earlier, this person answered the same kind of question about "
                f"other products:\n{shots}\n\nNow they are asked:\n{q}\n\n"
                "Based on this person's profile and earlier answers, give the "
                "probability that they would purchase the product. Return JSON "
                'only, with probabilities summing to 1: {"yes": <0-1>, "no": <0-1>}',
                "prob",
            )
        if strategy == "population":
            n = self.config.population_size
            system = (
                "You are an expert in consumer behavior estimating how "
                "groups of real shoppers respond to survey questions."
            )
            user = (
                f"Imagine {n} people who all match this profile:\n{persona}\n\n"
                f"Each of them is asked:\n{q}\n\n"
                f"How many of the {n} would purchase the product? Return JSON "
                f"only, with counts summing to {n}: "
                '{"purchase": <count>, "not_purchase": <count>}'
            )
            return system, user, "prob"
        raise ValueError(f"Unknown strategy: {strategy}")

    def fewshot_examples(self, pid: str, product_id: str) -> str:
        """The twin's own wave 3 pricing answers, excluding the target product."""
        if self.retest_items is None:
            raise RuntimeError("fewshot_self needs wave 3 answers (wave4_Q_wave1_3_A).")
        pool = self.retest_items[
            (self.retest_items["pid"] == pid)
            & (self.retest_items["product_id"] != product_id)
        ]
        k = min(self.config.fewshot_k, len(pool))
        idx = self._task_rng("fewshot", pid, product_id).choice(
            len(pool), k, replace=False
        )
        lines = []
        for _, r in pool.iloc[sorted(idx)].iterrows():
            verdict = "Would purchase" if r["answer"] == 1 else "Would NOT purchase"
            lines.append(
                f"- {r['product']} ({r['category']}), priced at "
                f"${r['price']:.2f}: {verdict}"
            )
        return "\n".join(lines)

    def paraphrase_prompt(self, persona: str, k: int) -> tuple[str, str]:
        system = "You rewrite personal profiles for survey research."
        user = (
            f"Below is a profile of one survey respondent. Write {k} different "
            "descriptions of this same person.\n\nRules:\n"
            "- Keep every fact exactly as given. Do not add or change facts.\n"
            "- Make each description as different as possible from all the "
            "others you have written: vary the voice (first person, third person, "
            "bullet list, interview, a friend describing them, a diary entry), "
            "which facts come first or are emphasized, tone, and length.\n"
            "- Each description should read like a distinct, realistic way this "
            "person could be portrayed.\n\n"
            f"PROFILE:\n{persona}\n\n"
            'Return JSON only: {"personas": ["...", "..."]}'
        )
        return system, user

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _extract_json(text: str) -> Any:
        t = re.sub(r"```(?:json)?", "", text or "").strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", t, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def parse_binary(text: str) -> int | None:
        m = re.search(r"\b(yes|no)\b", (text or "").lower())
        return None if not m else int(m.group(1) == "yes")

    @classmethod
    def parse_probability(cls, text: str) -> float | None:
        """Parse {"yes": p, "no": q} or {"purchase": n, "not_purchase": m}."""
        d = cls._extract_json(text)
        if not isinstance(d, dict):
            return None
        pos = neg = None
        for k, v in d.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            kl = k.lower().strip()
            if kl.startswith("no") or "not" in kl:
                neg = v
            elif kl in ("yes", "purchase", "buy", "would_purchase"):
                pos = v
        if pos is None:
            return None
        if neg is None:
            return float(np.clip(pos if pos <= 1 else pos / 100, 0, 1))
        total = pos + neg
        return None if total <= 0 else float(np.clip(pos / total, 0, 1))

    # --------------------------------------------------------------- calling
    def _call(
        self,
        system: str,
        user: str,
        sample_idx: int,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        max_tokens = max_tokens or self.config.max_output_tokens
        json_mode = json_mode and self.config.json_mode
        key = hashlib.sha256(
            json.dumps(
                [
                    self.client.model,
                    self.config.temperature,
                    system,
                    user,
                    sample_idx,
                    max_tokens,
                    json_mode,
                ]
            ).encode()
        ).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        text = self.client.complete(
            system, user, self.config.temperature, max_tokens, json_mode=json_mode
        )
        self.cache.set(key, text)
        return text

    def _task_rng(self, *parts) -> np.random.Generator:
        h = hashlib.md5("|".join(map(str, (self.config.seed,) + parts)).encode())
        return np.random.default_rng(int(h.hexdigest()[:8], 16))

    def get_paraphrases(self, pid: str) -> list[str]:
        """Strategy 6: one call per twin returns k mutually diverse paraphrases."""
        if pid in self._paraphrases:
            return self._paraphrases[pid]
        k = self.config.n_samples
        system, user = self.paraphrase_prompt(self.build_persona(pid), k)
        text = self._call(
            system, user, sample_idx=-1, max_tokens=700 * k, json_mode=True
        )
        d = self._extract_json(text)
        variants = [
            v for v in (d or {}).get("personas", []) if isinstance(v, str) and v.strip()
        ]
        if not variants:
            log.warning(
                "Paraphrase parse failed for pid %s; using original persona", pid
            )
            variants = [self.build_persona(pid)]
        variants = [variants[i % len(variants)] for i in range(k)]
        self._paraphrases[pid] = variants
        return variants

    def _run_one(self, strategy: str, item: pd.Series, s: int) -> dict:
        pid = item["pid"]
        persona = (
            self.get_paraphrases(pid)[s]
            if strategy == "paraphrase"
            else self.build_persona(pid)
        )
        system, user, kind = self.build_prompt(strategy, persona, item)
        text = self._call(system, user, sample_idx=s, json_mode=(kind == "prob"))
        prob = None
        if kind == "binary":
            answer = self.parse_binary(text)
        else:
            prob = self.parse_probability(text)
            answer = (
                None
                if prob is None
                else int(
                    self._task_rng(strategy, pid, item["product_id"], s).random() < prob
                )
            )
        return dict(
            strategy=strategy,
            pid=pid,
            product_id=item["product_id"],
            condition=item["condition"],
            price_ratio=item["price_ratio"],
            sample_idx=s,
            prob=prob,
            answer=answer,
            value=prob if kind == "prob" else answer,
            raw=(text or "")[:300],
        )

    def run_strategy(self, strategy: str) -> pd.DataFrame:
        if self.test_items is None:
            raise RuntimeError("Call select_test_set() first.")
        if strategy == "paraphrase":
            with ThreadPoolExecutor(self.config.max_workers) as ex:
                list(ex.map(self.get_paraphrases, self.test_items["pid"].unique()))
        n = self.n_calls_per_item(strategy)
        tasks = [(item, s) for _, item in self.test_items.iterrows() for s in range(n)]
        out = []
        with ThreadPoolExecutor(self.config.max_workers) as ex:
            futs = [ex.submit(self._run_one, strategy, it, s) for it, s in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                out.append(f.result())
                if i % 200 == 0 or i == len(futs):
                    log.info("[%s] %d/%d calls", strategy, i, len(futs))
        return pd.DataFrame(out)

    def n_calls_per_item(self, strategy: str) -> int:
        return (
            self.config.prob_samples if strategy in PROB_STRATEGIES else self.config.n_samples
        )

    def run(self, strategies: tuple | None = None) -> pd.DataFrame:
        frames = [self.run_strategy(s) for s in (strategies or self.config.strategies)]
        self.results = pd.concat(frames, ignore_index=True)
        return self.results

    # --------------------------------------------------------------- metrics
    @staticmethod
    def bernoulli_jsd(p: float, q: float) -> float:
        """Jensen-Shannon divergence (base 2, range 0-1) of Bernoulli(p), Bernoulli(q)."""
        return float(jensenshannon([p, 1 - p], [q, 1 - q], base=2) ** 2)

    def _bootstrap_ci(self, values: np.ndarray) -> tuple[float, float]:
        rng = np.random.default_rng(self.config.seed)
        idx = rng.integers(0, len(values), size=(self.config.n_bootstrap, len(values)))
        means = values[idx].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def condition_metrics(self, pred: pd.Series, human: pd.Series) -> dict:
        """Moon et al. binary metrics over conditions."""
        common = human.index.intersection(pred.index)
        p, h = pred.loc[common].astype(float), human.loc[common].astype(float)
        jsd = np.array([self.bernoulli_jsd(a, b) for a, b in zip(p, h)])
        if len(common) >= 3 and np.ptp(p) > 1e-12 and np.ptp(h) > 1e-12:
            r = float(pearsonr(p, h)[0])
        else:
            r = np.nan  # undefined: constant predictions, as in Moon et al.
        lo, hi = self._bootstrap_ci(jsd) if len(jsd) else (np.nan, np.nan)
        return dict(
            mean_MAE=float(np.abs(p - h).mean()),
            pattern_r=r,
            mean_JSD=float(jsd.mean()),
            JSD_CI_low=lo,
            JSD_CI_high=hi,
            n_conditions=len(common),
        )

    @staticmethod
    def _share_per_twin(df: pd.DataFrame, by: list[str]) -> np.ndarray:
        return df.groupby(by)["answer"].mean().to_numpy()

    def heterogeneity_metrics(
        self, pred_shares: np.ndarray, human_shares: np.ndarray
    ) -> dict:
        """Compare distributions of per-twin purchase share (person-level spread)."""
        hs = human_shares.std()
        return dict(
            het_wasserstein=float(wasserstein_distance(pred_shares, human_shares)),
            het_sd_ratio=float(pred_shares.std() / hs) if hs > 0 else np.nan,
        )

    @staticmethod
    def individual_metrics(p_hat: np.ndarray, y: np.ndarray) -> dict:
        """Expected agreement of a sampled response with the twin, and Brier score."""
        return dict(
            indiv_agreement=float(np.mean(p_hat * y + (1 - p_hat) * (1 - y))),
            indiv_brier=float(np.mean((p_hat - y) ** 2)),
        )

    def _simulate_shares(self, cell_probs: pd.DataFrame) -> np.ndarray:
        """Per-twin purchase shares when each cell is Bernoulli(prob)."""
        rng = np.random.default_rng(self.config.seed)
        draws = (
            rng.random((self.config.n_baseline_reps, len(cell_probs)))
            < cell_probs["prob"].to_numpy()
        )
        shares = []
        pids = cell_probs["pid"].to_numpy()
        for rep in draws:
            shares.append(pd.Series(rep.astype(float)).groupby(pids).mean().to_numpy())
        return np.concatenate(shares)

    def strategy_shares(self, strategy: str, valid: pd.DataFrame) -> np.ndarray:
        """Per-twin purchase shares for a strategy.

        Binary strategies: one share per (twin, sample run).
        Probability strategies: simulated by drawing Bernoulli(p) per item, so
        the result includes the same binomial noise as the human shares.
        """
        if strategy in PROB_STRATEGIES:
            cells = valid.groupby(["pid", "product_id"])["value"].mean()
            cells = cells.rename("prob").reset_index()
            return self._simulate_shares(cells)
        return valid.groupby(["pid", "sample_idx"])["value"].mean().to_numpy()

    def evaluate(self, results: pd.DataFrame | None = None) -> pd.DataFrame:
        results = self.results if results is None else results
        human = self.test_items
        h_rate = human.groupby("condition")["answer"].mean()
        h_shares = self._share_per_twin(human, ["pid"])
        rows = []

        # LLM strategies
        for strat, g in results.groupby("strategy"):
            valid = g.dropna(subset=["value"])
            row = dict(
                predictor=strat,
                n_calls=len(g),
                parse_fail_rate=float(g["value"].isna().mean()),
            )
            row |= self.condition_metrics(
                valid.groupby("condition")["value"].mean(), h_rate
            )
            row |= self.heterogeneity_metrics(
                self.strategy_shares(strat, valid), h_shares
            )
            cell = valid.groupby(["pid", "product_id"])["value"].mean().rename("p_hat")
            m = human.merge(cell.reset_index(), on=["pid", "product_id"])
            row |= self.individual_metrics(
                m["p_hat"].to_numpy(), m["answer"].to_numpy()
            )
            rows.append(row)

        # Baselines from human data (Moon et al.): interpretive anchors
        pooled = human["answer"].mean()
        baselines = {
            "baseline_marginal": pd.Series(pooled, index=h_rate.index),
            "baseline_uniform": pd.Series(0.5, index=h_rate.index),
            "baseline_mean_delta": h_rate,
        }
        for name, rate in baselines.items():
            row = dict(predictor=name, n_calls=0, parse_fail_rate=0.0)
            row |= self.condition_metrics(rate, h_rate)
            cells = human[["pid", "product_id", "condition", "answer"]].copy()
            cells["prob"] = cells["condition"].map(rate).astype(float)
            row |= self.heterogeneity_metrics(self._simulate_shares(cells), h_shares)
            row |= self.individual_metrics(
                cells["prob"].to_numpy(), cells["answer"].to_numpy()
            )
            rows.append(row)

        # Human test-retest reference (wave 3 answers, independently drawn prices)
        if self.retest_items is not None:
            rt = self.retest_items[
                self.retest_items["pid"].isin(human["pid"].unique())
                & self.retest_items["product_id"].isin(human["product_id"].unique())
            ]
            if len(rt):
                row = dict(
                    predictor="human_retest_wave3", n_calls=0, parse_fail_rate=0.0
                )
                row |= self.condition_metrics(
                    rt.groupby("condition")["answer"].mean(), h_rate
                )
                row |= self.heterogeneity_metrics(
                    self._share_per_twin(rt, ["pid"]), h_shares
                )
                row |= dict(indiv_agreement=np.nan, indiv_brier=np.nan)
                rows.append(row)

        self.metrics = pd.DataFrame(rows).set_index("predictor")
        return self.metrics

    def rank_strategies(self, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
        """Rank LLM strategies by how well they recover the human distribution.

        Three distributional criteria, each ranked (1 = best), then averaged:
            - mean_JSD:        condition-level Bernoulli JSD (Moon et al.)
            - het_wasserstein: distance of the per-twin purchase-share distribution
            - |het_sd_ratio-1|: how close the person-level spread is to humans
        Also flags whether each strategy beats the marginal baseline, the
        reference no model beat for distributional fit in Moon et al.
        """
        metrics = self.metrics if metrics is None else metrics
        is_llm = ~metrics.index.str.startswith(("baseline", "human"))
        m = metrics[is_llm].copy()
        ranks = pd.DataFrame(
            {
                "rank_JSD": m["mean_JSD"].rank(),
                "rank_het_wasserstein": m["het_wasserstein"].rank(),
                "rank_sd_ratio": (m["het_sd_ratio"] - 1).abs().rank(),
            }
        )
        ranks["avg_rank"] = ranks.mean(axis=1)
        cols = [
            "mean_JSD",
            "het_wasserstein",
            "het_sd_ratio",
            "mean_MAE",
            "pattern_r",
            "indiv_agreement",
        ]
        out = m[cols].join(ranks).sort_values("avg_rank")
        if "baseline_marginal" in metrics.index:
            marg = metrics.loc["baseline_marginal"]
            out["beats_marginal_JSD"] = out["mean_JSD"] < marg["mean_JSD"]
            out["beats_marginal_het"] = out["het_wasserstein"] < marg["het_wasserstein"]
        self.ranking = out
        return out

    def condition_table(self, results: pd.DataFrame | None = None) -> pd.DataFrame:
        """Purchase rate per condition for humans and every strategy."""
        results = self.results if results is None else results
        table = {"human_wave4": self.test_items.groupby("condition")["answer"].mean()}
        for strat, g in results.groupby("strategy"):
            table[strat] = (
                g.dropna(subset=["value"]).groupby("condition")["value"].mean()
            )
        n = self.test_items.groupby("condition").size().rename("n_items")
        return pd.DataFrame(table).join(n)

    # --------------------------------------------------------------- output
    def save(self, tag: str | None = None) -> str:
        out = os.path.join(self.config.output_dir, tag or time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out, exist_ok=True)
        self.test_items.to_csv(os.path.join(out, "test_items.csv"), index=False)
        self.results.to_csv(os.path.join(out, "llm_responses.csv"), index=False)
        self.metrics.to_csv(os.path.join(out, "metrics.csv"))
        self.condition_table().to_csv(os.path.join(out, "condition_rates.csv"))
        if getattr(self, "ranking", None) is not None:
            self.ranking.to_csv(os.path.join(out, "strategy_ranking.csv"))
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(
                asdict(self.config) | {"client_model": self.client.model}, f, indent=2
            )
        if self._paraphrases:
            with open(os.path.join(out, "paraphrases.json"), "w") as f:
                json.dump(self._paraphrases, f, indent=2)
        self.plot(out)
        return out

    def plot(self, out_dir: str) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        ct = self.condition_table().drop(columns="n_items")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for col in ct.columns:
            ax[0].plot(
                ct.index.astype(str),
                ct[col],
                marker="o",
                lw=2.5 if col == "human_wave4" else 1.2,
                color="black" if col == "human_wave4" else None,
                label=col,
            )
        ax[0].set(
            title=f"Purchase rate by {self.config.condition_key}",
            xlabel=self.config.condition_key,
            ylabel="purchase rate",
        )
        ax[0].tick_params(axis="x", rotation=45)
        ax[0].legend(fontsize=8)
        bins = np.linspace(0, 1, 11)
        ax[1].hist(
            self._share_per_twin(self.test_items, ["pid"]),
            bins=bins,
            density=True,
            histtype="step",
            lw=2.5,
            color="black",
            label="human_wave4",
        )
        for strat, g in self.results.groupby("strategy"):
            v = g.dropna(subset=["value"])
            ax[1].hist(
                self.strategy_shares(strat, v),
                bins=bins,
                density=True,
                histtype="step",
                lw=1.2,
                label=strat,
            )
        ax[1].set(
            title="Per-twin purchase share (heterogeneity)",
            xlabel="share of products purchased",
            ylabel="density",
        )
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "distribution_plots.png"), dpi=150)
        plt.close(fig)
