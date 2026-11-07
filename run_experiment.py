import os
import pandas as pd
from dotenv import load_dotenv

from src import (
    RecoverDistributionAlignment,
    ExperimentConfig,
    PromptStrategies,
    Twin2KDataset,
    LLMProvider,
    CloseModelSelection,
    ResultsReporter
)

load_dotenv()

config = ExperimentConfig(
    n_twins=10,
    n_units=5,
    strategies=(
        "direct",
        "verbalized",
        "population",
        "fewshot_self",
        "knowledge_restriction",
        "paraphrase",
    ),
    cache_path="./exports/llm_cache.jsonl", 
    output_dir="./exports"
)
provider = LLMProvider(
    provider="openai",
    model_name=CloseModelSelection.gpt6_luna.value,
    token=os.getenv("OPENAI_KEY"),
    max_new_tokens=4096,
    temperature=1,
    load_in_4bit=False,
)

dataset = Twin2KDataset(config)
dataset.load()
dataset.select_test_set()
strategies = PromptStrategies(config)
rda = RecoverDistributionAlignment(config, dataset, strategies, provider)

print(dataset.report)
print(f"{len(dataset.test_items)} items, ~{rda.n_calls_planned()} calls")

df_results = rda.run()
df_eval = rda.evaluate()

rr = ResultsReporter(rda)
with pd.option_context("display.width", 200, "display.precision", 3):
    print(rda.metrics)
    print(rda.ranking)
print("Saved to", rr.save())
print("END")
