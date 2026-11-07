import pandas as pd
import os 
from dotenv import load_dotenv

from src import (
    DistributionAlignmentExperiment,
    ExperimentConfig,
    PromptStrategies,
    Twin2KDataset,
    LLMProvider, 
    CloseModelSelection
)

load_dotenv()

config = ExperimentConfig(
    n_twins=20,
    n_units=10,
    strategies=(
        "direct",
        "verbalized",
        "population",
        "fewshot_self",
        "knowledge_restriction",
        "paraphrase",
    ),
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
experiment = DistributionAlignmentExperiment(config, dataset, strategies, provider)

print(dataset.report)
print(f"{len(dataset.test_items)} items, ~{experiment.n_calls_planned()} calls")
print(experiment.pilot())

experiment.run()
experiment.evaluate()
with pd.option_context("display.width", 200, "display.precision", 3):
    print(experiment.metrics)
    print(experiment.ranking)
print("Saved to", experiment.save())
