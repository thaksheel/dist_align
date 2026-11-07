# Distributional Alignment of LLM Survey Simulations

LLMs used as stand-ins for human survey respondents tend to get the average answer roughly right but lose the spread: simulated people look more alike than real people do. This codebase evaluates prompting strategies that aim to recover the **distribution** of human responses, not just the mean.

Experiments use the [Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) dataset, which pairs about 2,000 detailed participant profiles with their survey answers. The default task is the marketing category: a pricing study in which each person decides whether to buy 40 grocery products shown at random prices (0–200% of the regular price). Evaluation follows Moon et al., *Beyond Averages*.

## Codebase

```
run_experiment.py        example run with OpenAI or a Hugging Face model
dist_align/
  config.py              ExperimentConfig: every setting in one dataclass
  dataset.py             Twin2KDataset: loads data, questions by category, personas, test set
  strategies.py          PromptStrategies: prompt templates and answer parsers
  llm_provider.py        LLMProvider: OpenAI (default) or local Hugging Face models
  metrics.py             DistributionMetrics: metrics and baselines
  experiment.py          DistributionAlignmentExperiment: runs, evaluates, saves
  reporting.py           ResultsReporter: result files and plots
```

## Strategies

| Strategy | Idea | Answer |
|---|---|---|
| `direct` | The model plays the person and answers Yes or No (control). | Yes/No |
| `verbalized` | The model gives the probability the person would buy. | Probability |
| `population` | The model estimates how many of 100 similar people would buy. | Probability |
| `fewshot_self` | Shows the person's own earlier pricing answers before asking for a probability. | Probability |
| `knowledge_restriction` | Tells the model what the person would not know. | Yes/No |
| `paraphrase` | Rewrites each persona several different ways; each version answers once. | Yes/No |

Yes/No strategies are called `n_samples` times per item and averaged. Probability strategies are called `prob_samples` times and scored on the probability itself.

## Evaluation

Each strategy's answers are compared with the real answers, grouped into price bins:

- **Mean:** mean absolute error of purchase rates.
- **Pattern:** correlation of purchase rates across price bins.
- **Distribution:** Jensen-Shannon divergence between predicted and human Yes/No distributions, with a bootstrap confidence interval.
- **Heterogeneity:** how closely the spread of each person's purchase share matches humans (Wasserstein distance and spread ratio, where 1 = human spread).
- **Individual:** agreement with each person's own answers.

Results are compared against three baselines (marginal, uniform and mean-delta) and against the same people's answers from another wave (human retest), which sets a realistic ceiling.

## Usage

Install the requirements and add your API key to a `.env` file (`OPENAI_API_KEY`, or `HF_TOKEN` for Hugging Face models):

```bash
pip install -r requirements.txt
```

All settings live in `ExperimentConfig`. A run creates the config, builds each class and calls its methods:

```python
from dist_align import (DistributionAlignmentExperiment, ExperimentConfig,
                        PromptStrategies, Twin2KDataset)
from dist_align.llm_provider import LLMProvider

config = ExperimentConfig(
    model_name="gpt-4.1-mini",
    n_twins=20,
    n_units=10,
    strategies=("direct", "verbalized", "population", "fewshot_self"),
)

dataset = Twin2KDataset(config)
dataset.load()
dataset.select_test_set()

experiment = DistributionAlignmentExperiment(
    config, dataset, PromptStrategies(config), LLMProvider.from_config(config))
experiment.pilot()       # quick check on a few items
experiment.run()
experiment.evaluate()
experiment.save()        # exports/<timestamp>/
```

Or run the scripts:

```bash
python run_experiment.py    # real run
python run_mock.py          # offline test of the whole pipeline
python inspect_dataset.py   # see the data's question blocks and categories
```

To use an open-source model, set `ExperimentConfig(provider="hf", model_name="Qwen/Qwen2.5-7B-Instruct")`.

Common settings: `model_name`, `temperature`, `strategies`, `n_twins`, `n_units`, `n_samples`, `persona_blocks`, `condition_key` and `output_dir`. The `ExperimentConfig` docstring lists every field.

## Outputs

Each run is saved to `exports/<timestamp>/`:

| Path | Contents |
|---|---|
| `per_call_results.csv` | One row per LLM call: prompt id, full response, parsed answer, tokens, latency |
| `summary/` | Metrics, strategy ranking, purchase rates by price bin, token usage, run summary |
| `prompts/` | One full example prompt and response per strategy |
| `data/` | The test set and the data extraction report |
| `plots/` | Demand curves, per-person spread, metric summary |

## Notes

- Strategies currently support the marketing (pricing) category. Other categories (public health, political science, economics, psychology) load as data for future work; their mapping to question blocks is set in `category_blocks` and can be checked with `inspect_dataset.py`.
- LLM responses are cached in `cache/llm_cache.jsonl`, so re-running does not repeat paid calls.
- Results from `run_mock.py` only test the code; the mock model ignores personas.
