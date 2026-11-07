import json
import re

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .utils import clean_text, iter_questions, norm_name, task_rng, to_json

PRICE_RE = re.compile(r"priced at:?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)
CATEGORY_RE = re.compile(r"product category:\s*(.*?)\.?\s*Suppose", re.I | re.S)
PRODUCT_RE = re.compile(
    r"in that category:\s*(.*?)\.?\s*The product is priced", re.I | re.S
)


class Twin2KDataset:
    """Twin-2K-500 data: questions by category, test set, personas and few-shot pools.

    Parameters
    ----------
    config : ExperimentConfig
        Uses the data settings (`hf_name`, `category`, `category_blocks`,
        `truth_column`, `reference_column`, persona, condition and test-set fields).

    Attributes
    ----------
    raw : pandas.DataFrame
        One row per participant, indexed by `pid`.
    items : pandas.DataFrame
        Ground-truth answers for the category, one row per person x question.
        For marketing, adds product, price, price_ratio, condition and a 0/1 answer.
    reference_items : pandas.DataFrame
        The same questions answered in the reference wave.
    test_items : pandas.DataFrame
        Subset chosen by `select_test_set`.
    report : dict
        Rows found and kept during extraction.
    unit : str
        Column identifying a question within a person ("product_id" or "qid").

    Methods
    -------
    load(dataframe=None)
        Load the data (Hugging Face by default) and build items for `config.category`.
    inventory()
        Every block in the data with its category and counts.
    get_items(category, column)
        Generic rows for any category.
    select_test_set(n_twins=None, n_units=None)
        Random twins x units; defaults come from the config.
    build_persona(pid)
        Persona text for one twin.
    fewshot_pool(pid, exclude_unit)
        The twin's reference-wave answers, excluding one unit.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.raw = self.items = self.reference_items = self.test_items = None
        self.report = {}
        self._personas = {}

    @property
    def is_pricing(self) -> bool:
        return self.config.category == "marketing"

    @property
    def unit(self) -> str:
        return "product_id" if self.is_pricing else "qid"

    def load(self, dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
        if dataframe is None:
            from datasets import load_dataset

            dataframe = load_dataset(self.config.hf_name, "wave_split")[
                "data"
            ].to_pandas()
        df = dataframe.copy()
        df["pid"] = df["pid"].astype(str)
        self.raw = df.set_index("pid", drop=False)
        cfg = self.config
        self.items = self.get_items(cfg.category, cfg.truth_column)
        self.reference_items = self.get_items(cfg.category, cfg.reference_column)
        self.report = {"category": cfg.category, "people": len(df)}
        if self.is_pricing:
            self.items = self._prepare_pricing(self.items, "truth")
            self.reference_items = self._prepare_pricing(
                self.reference_items, "reference"
            )
            self._assign_conditions()
        return self.items

    def category_of(self, block) -> str | None:
        b = norm_name(block)
        for category, keywords in self.config.category_blocks.items():
            if any(norm_name(k) in b for k in keywords):
                return category
        return None

    def inventory(self) -> pd.DataFrame:
        rows = []
        for col in (
            "wave1_3_persona_json",
            self.config.reference_column,
            self.config.truth_column,
        ):
            counts = {}
            for blob in self.raw[col]:
                per_person = {}
                for block, _ in iter_questions(to_json(blob)):
                    per_person[block] = per_person.get(block, 0) + 1
                for block, n in per_person.items():
                    counts.setdefault(block, []).append(n)
            rows += [
                dict(
                    column=col,
                    block=b,
                    category=self.category_of(b) or "",
                    people=len(ns),
                    questions_per_person=round(np.mean(ns), 1),
                )
                for b, ns in counts.items()
            ]
        return pd.DataFrame(rows).sort_values(
            ["column", "category", "block"], ignore_index=True
        )

    def get_items(self, category: str, column: str) -> pd.DataFrame:
        rows = []
        for pid, blob in zip(self.raw["pid"], self.raw[column]):
            for block, q in iter_questions(to_json(blob)):
                if self.category_of(block) != category or not q.get("Answers"):
                    continue
                ans, options = q["Answers"], q.get("Options") or []
                pos = ans.get("SelectedByPosition")
                text = ans.get("SelectedText")
                if text is None and isinstance(pos, int) and 0 < pos <= len(options):
                    text = options[pos - 1]
                rows.append(
                    dict(
                        pid=str(pid),
                        block=str(block).strip(),
                        qid=q.get("QuestionID"),
                        question_type=q.get("QuestionType"),
                        question_text=clean_text(q.get("QuestionText")),
                        options=json.dumps(options) if options else None,
                        answer_text=self._format_answer(ans if text is None else text),
                        answer_position=pos if isinstance(pos, int) else None,
                        answer_raw=json.dumps(ans, ensure_ascii=False),
                    )
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _format_answer(ans) -> str:
        if isinstance(ans, list):
            return "; ".join(map(clean_text, ans))
        if isinstance(ans, dict):
            if "SelectedText" in ans:
                return Twin2KDataset._format_answer(ans["SelectedText"])
            return clean_text(
                ans.get("Text", json.dumps(ans, ensure_ascii=False)[:500])
            )
        return clean_text(ans)

    def _prepare_pricing(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        price = df["question_text"].str.extract(PRICE_RE, expand=False)
        out = df[price.notna()].copy()
        out["price"] = price.dropna().str.replace(",", "").astype(float)
        out["category"] = (
            out["question_text"].str.extract(CATEGORY_RE, expand=False).str.strip()
        )
        out["product"] = (
            out["question_text"].str.extract(PRODUCT_RE, expand=False).str.strip()
        )
        t = out["answer_text"].str.strip().str.lower()
        out["answer"] = np.where(
            t.str.startswith("yes"), 1, np.where(t.str.startswith("no"), 0, np.nan)
        )
        kept = out.dropna(subset=["answer"]).astype({"answer": int})
        self.report[label] = dict(
            rows_in_block=len(df),
            with_price=len(out),
            readable_answers=len(kept),
            per_person=round(len(kept) / kept["pid"].nunique(), 1),
        )
        cols = [
            "pid",
            "qid",
            "block",
            "category",
            "product",
            "price",
            "answer",
            "answer_text",
            "question_text",
        ]
        return kept[cols].reset_index(drop=True)

    def _assign_conditions(self) -> None:
        both = pd.concat([self.items, self.reference_items])
        ids = {
            p: f"P{i + 1:02d}" for i, p in enumerate(sorted(both["product"].unique()))
        }
        regular = both.groupby("product")["price"].max() / 2
        bins = self.config.price_bins
        to_bin = lambda r: next(
            f"{i}: {lab}"
            for i, (lab, lo, hi) in enumerate(bins, 1)
            if lo - 1e-9 <= r <= hi + 1e-9
        )
        for f in (self.items, self.reference_items):
            f["product_id"] = f["product"].map(ids)
            f["regular_price"] = f["product"].map(regular)
            f["price_ratio"] = (
                np.round(f["price"] / f["regular_price"] / 0.2) * 0.2
            ).round(1)
            f["condition"] = {
                "price_bin": f["price_ratio"].map(to_bin),
                "price_level": f["price_ratio"],
                "product": f["product_id"],
            }[self.config.condition_key]

    def select_test_set(
        self, n_twins: int | None = None, n_units: int | None = None
    ) -> pd.DataFrame:
        cfg = self.config
        n_twins = n_twins or cfg.n_twins
        n_units = n_units or cfg.n_units
        rng = np.random.default_rng(cfg.seed)
        pids = rng.choice(sorted(self.items["pid"].unique()), n_twins, replace=False)
        units = sorted(self.items[self.unit].unique())
        if n_units:
            units = rng.choice(units, n_units, replace=False)
        mask = self.items["pid"].isin(pids) & self.items[self.unit].isin(units)
        self.test_items = self.items[mask].reset_index(drop=True)
        return self.test_items

    def build_persona(self, pid: str) -> str:
        if pid not in self._personas:
            cfg = self.config
            row = self.raw.loc[pid]
            if cfg.persona_blocks is None:
                text = str(row["wave1_3_persona_text"])
            else:
                wanted = {norm_name(b) for b in cfg.persona_blocks}
                text = "\n".join(
                    f"Q: {clean_text(q['QuestionText'])}\nA: {self._format_answer(q['Answers'])}"
                    for block, q in iter_questions(to_json(row["wave1_3_persona_json"]))
                    if norm_name(block) in wanted
                    and q.get("Answers")
                    and not PRICE_RE.search(clean_text(q["QuestionText"]))
                )
            self._personas[pid] = text[: cfg.max_persona_chars]
        return self._personas[pid]

    def fewshot_pool(self, pid: str, exclude_unit: str) -> pd.DataFrame:
        ref = self.reference_items
        pool = ref[(ref["pid"] == pid) & (ref[self.unit] != exclude_unit)]
        k = min(self.config.fewshot_k, len(pool))
        idx = task_rng(self.config.seed, "fewshot", pid, exclude_unit).choice(
            len(pool), k, replace=False
        )
        return pool.iloc[sorted(idx)]
