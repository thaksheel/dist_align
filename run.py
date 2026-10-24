import logging
import sys

import pandas as pd

from src import (
    DistributionAlignmentExperiment,
    ExperimentConfig,
    MockClient,
    OpenAIClient,
)

# ----------------------------------------------------------------- settings
N_TWINS = 20
N_PRODUCTS = 10  # 20 x 10 = 200 test samples
STRATEGIES = ("direct", "verbalized", "population", "fewshot_self")
N_SAMPLES = 3  # calls per item for Yes/No strategies (direct)
PROB_SAMPLES = 1  # calls per item for probability strategies
PERSONA_BLOCKS = ("Demographics", "Economic preferences")
FEWSHOT_K = 10  # wave 3 examples per fewshot_self prompt
MODEL = "gpt-6-luna"
TEMPERATURE = 1.0  # None for reasoning models
SEED = 42
PILOT_ITEMS = 10  # items per strategy in the pilot check
MAX_PARSE_FAIL = 0.05  # stop if a strategy's pilot parse failures exceed this
USE_MOCK = False  # True = synthetic data + mock LLM, no API calls
# ---------------------------------------------------------------------------


def pilot_check(exp: DistributionAlignmentExperiment) -> bool:
    """Run a few items per strategy and check parsing and persona content."""
    full = exp.test_items
    pilot_items = full.sample(n=min(PILOT_ITEMS, len(full)), random_state=SEED)
    exp.test_items = pilot_items
    pilot = exp.run()
    exp.test_items = full

    pid = full["pid"].iloc[0]
    persona = exp.build_persona(pid)
    print(f"\n--- Pilot check ({PILOT_ITEMS} items) ---")
    print(f"Persona for twin {pid}: {len(persona)} chars. Start:\n{persona[:400]}...\n")

    summary = pilot.groupby("strategy").agg(
        parse_fail_rate=("value", lambda v: v.isna().mean()),
        mean_predicted=("value", "mean"),
    )
    summary["human_rate"] = pilot_items["answer"].mean()
    print(summary.round(3).to_string())

    bad = summary.index[summary["parse_fail_rate"] > MAX_PARSE_FAIL].tolist()
    if bad:
        print(
            f"\nStopping: parse failures above {MAX_PARSE_FAIL:.0%} for {bad}. "
            "Check the 'raw' column below before the full run."
        )
        print(pilot[pilot["value"].isna()][["strategy", "raw"]].head(10).to_string())
        return False
    print("Pilot passed.\n")
    return True


def main() -> pd.DataFrame | None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    cfg = ExperimentConfig(
        model=MODEL,
        temperature=TEMPERATURE,
        strategies=STRATEGIES,
        n_samples=N_SAMPLES,
        prob_samples=PROB_SAMPLES,
        persona_mode="blocks",
        persona_blocks=PERSONA_BLOCKS,
        condition_key="price_bin",
        fewshot_k=FEWSHOT_K,
        seed=SEED,
        cache_path=None if USE_MOCK else "cache/llm_cache.jsonl",
    )
    client = (
        MockClient(seed=SEED) if USE_MOCK else OpenAIClient(cfg.model, cfg.max_retries)
    )
    exp = DistributionAlignmentExperiment(cfg, client)

    # 1. Load the dataset
    if USE_MOCK:
        exp.load_data(
            source="dataframe",
            dataframe=DistributionAlignmentExperiment.make_mock_dataset(seed=SEED),
        )
    else:
        exp.load_data(source="hf")

    # 2. Select 200 test samples
    exp.select_test_set(n_twins=N_TWINS, n_products=N_PRODUCTS)
    n_items = len(exp.test_items)
    n_calls = sum(n_items * exp.n_calls_per_item(s) for s in STRATEGIES)
    print(
        f"\nTest set: {n_items} samples ({exp.test_items.pid.nunique()} twins x "
        f"{exp.test_items.product_id.nunique()} products), ~{n_calls} LLM calls"
    )
    print(
        "Items per price bin:",
        exp.test_items["condition"].value_counts().sort_index().to_dict(),
    )

    # 3. Pilot, then full run
    if not pilot_check(exp):
        return None
    exp.run()
    metrics = exp.evaluate()
    ranking = exp.rank_strategies()

    # 4. Report
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("=== All metrics (strategies, baselines, human retest) ===")
        print(metrics.drop(columns=["n_conditions"]).to_string())
        print("\n=== Strategy ranking for distribution recovery (1 = best) ===")
        print(ranking.to_string())
        print("\n=== Purchase rate by price bin ===")
        print(exp.condition_table().round(3).to_string())

    print(f"\nBest at recovering the human distribution: {ranking.index[0]}")
    if not ranking["beats_marginal_JSD"].any():
        print("Note: no strategy beats the marginal baseline on JSD.")
    print(f"Saved to: {exp.save()}")
    return ranking


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
