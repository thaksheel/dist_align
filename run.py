import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv 

from src import (
    DistributionAlignmentExperiment,
    ExperimentConfig,
    OpenAIClient,
    STRATEGIES,
)

load_dotenv()
config = ExperimentConfig(
    model="gpt-6-luna",
    temperature=1,
    strategies=("direct", "verbalized", "population", "fewshot_self"),
    n_samples=1,
    prob_samples=1,
    persona_mode="blocks",
    persona_blocks=("Demographics", "Economic preferences"),
    condition_key="price_bin",
    fewshot_k=10,
    seed=42,
)
client = OpenAIClient(config.model, config.max_retries)
exp = DistributionAlignmentExperiment(config, client)

# Select 200 test samples
items = exp.load_data(source="hf")
test_items = exp.select_test_set(
    n_twins=3, # 2058 people 
    n_products=3, # 40 products 
)  # n_twins x n_products = test samples
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
results = exp.run()
metrics = exp.evaluate()
ranking = exp.rank_strategies()

# Report
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

print("END")
