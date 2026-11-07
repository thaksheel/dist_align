import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, wasserstein_distance

from .config import ExperimentConfig


class DistributionMetrics:
    """Mean, pattern, distribution, heterogeneity and individual metrics, plus baselines.

    Follows Moon et al. ("Beyond Averages") for binary outcomes and adds
    person-level heterogeneity.

    Parameters
    ----------
    config : ExperimentConfig
        Uses `prob_strategies`, `n_bootstrap`, `n_sim_reps` and `seed`.
    unit : str
        Column identifying a question within a person.

    Methods
    -------
    evaluate(results, human, retest=None)
        One row per strategy, baseline and human retest with all metrics.
    rank(metrics)
        Strategies ranked by mean_JSD, het_wasserstein and |het_sd_ratio - 1|.
    condition_table(results, human, retest=None)
        Purchase rate per condition for humans and every strategy.

    Notes
    -----
    mean_MAE        mean |predicted rate - human rate| over conditions
    pattern_r       Pearson r of condition rates (NaN if constant)
    mean_JSD        mean Bernoulli Jensen-Shannon divergence (base 2) over conditions,
                    with a bootstrap 95% CI
    het_wasserstein distance between per-twin purchase-share distributions
    het_sd_ratio    predicted / human spread of per-twin shares (1 = match)
    indiv_agreement mean of p*y + (1-p)*(1-y) over twin x item
    indiv_brier     mean of (p - y)^2 over twin x item
    """

    def __init__(self, config: ExperimentConfig, unit: str):
        self.config = config
        self.unit = unit

    @staticmethod
    def jsd(p: float, q: float) -> float:
        return float(jensenshannon([p, 1 - p], [q, 1 - q], base=2) ** 2)

    def condition_metrics(self, pred: pd.Series, human: pd.Series) -> dict:
        p, h = pred.reindex(human.index).astype(float), human.astype(float)
        jsd = np.array([self.jsd(a, b) for a, b in zip(p, h)])
        rng = np.random.default_rng(self.config.seed)
        boot = jsd[rng.integers(0, len(jsd), (self.config.n_bootstrap, len(jsd)))].mean(
            1
        )
        r = pearsonr(p, h)[0] if len(p) > 2 and np.ptp(p) > 0 else np.nan
        return dict(
            mean_MAE=float(np.abs(p - h).mean()),
            pattern_r=float(r),
            mean_JSD=float(jsd.mean()),
            JSD_CI_low=float(np.percentile(boot, 2.5)),
            JSD_CI_high=float(np.percentile(boot, 97.5)),
        )

    @staticmethod
    def heterogeneity_metrics(pred: np.ndarray, human: np.ndarray) -> dict:
        return dict(
            het_wasserstein=float(wasserstein_distance(pred, human)),
            het_sd_ratio=float(pred.std() / human.std()),
        )

    @staticmethod
    def individual_metrics(p: np.ndarray, y: np.ndarray) -> dict:
        return dict(
            indiv_agreement=float(np.mean(p * y + (1 - p) * (1 - y))),
            indiv_brier=float(np.mean((p - y) ** 2)),
        )

    def simulate_shares(self, cells: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.config.seed)
        draws = (
            rng.random((self.config.n_sim_reps, len(cells))) < cells["prob"].to_numpy()
        )
        pids = cells["pid"].to_numpy()
        return np.concatenate(
            [pd.Series(d.astype(float)).groupby(pids).mean().to_numpy() for d in draws]
        )

    def strategy_shares(self, strategy: str, valid: pd.DataFrame) -> np.ndarray:
        if strategy in self.config.prob_strategies:
            cells = (
                valid.groupby(["pid", self.unit])["value"]
                .mean()
                .rename("prob")
                .reset_index()
            )
            return self.simulate_shares(cells)
        return valid.groupby(["pid", "sample_idx"])["value"].mean().to_numpy()

    @staticmethod
    def human_shares(df: pd.DataFrame) -> np.ndarray:
        return df.groupby("pid")["answer"].mean().to_numpy()

    def evaluate(
        self,
        results: pd.DataFrame,
        human: pd.DataFrame,
        retest: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        h_rate = human.groupby("condition")["answer"].mean()
        h_shares = self.human_shares(human)
        rows = []
        for strategy, g in results.groupby("strategy", sort=False):
            valid = g.dropna(subset=["value"])
            cell = (
                valid.groupby(["pid", self.unit])["value"]
                .mean()
                .rename("p")
                .reset_index()
            )
            m = human.merge(cell, on=["pid", self.unit])
            rows.append(
                dict(
                    predictor=strategy,
                    type="strategy",
                    n_calls=len(g),
                    parse_fail_rate=float(g["value"].isna().mean()),
                )
                | self.condition_metrics(
                    valid.groupby("condition")["value"].mean(), h_rate
                )
                | self.heterogeneity_metrics(
                    self.strategy_shares(strategy, valid), h_shares
                )
                | self.individual_metrics(m["p"].to_numpy(), m["answer"].to_numpy())
            )

        baselines = {
            "baseline_marginal": pd.Series(human["answer"].mean(), h_rate.index),
            "baseline_uniform": pd.Series(0.5, h_rate.index),
            "baseline_mean_delta": h_rate,
        }
        for name, rate in baselines.items():
            cells = human[["pid", "condition", "answer"]].assign(
                prob=human["condition"].map(rate).astype(float)
            )
            rows.append(
                dict(predictor=name, type="baseline", n_calls=0, parse_fail_rate=0.0)
                | self.condition_metrics(rate, h_rate)
                | self.heterogeneity_metrics(self.simulate_shares(cells), h_shares)
                | self.individual_metrics(
                    cells["prob"].to_numpy(), cells["answer"].to_numpy()
                )
            )

        if retest is not None:
            rows.append(
                dict(
                    predictor="human_retest",
                    type="reference",
                    n_calls=0,
                    parse_fail_rate=0.0,
                )
                | self.condition_metrics(
                    retest.groupby("condition")["answer"].mean(), h_rate
                )
                | self.heterogeneity_metrics(self.human_shares(retest), h_shares)
            )
        return pd.DataFrame(rows).set_index("predictor")

    @staticmethod
    def rank(metrics: pd.DataFrame) -> pd.DataFrame:
        m = metrics[metrics["type"] == "strategy"]
        ranks = pd.DataFrame(
            {
                "rank_JSD": m["mean_JSD"].rank(),
                "rank_het_wasserstein": m["het_wasserstein"].rank(),
                "rank_sd_ratio": (m["het_sd_ratio"] - 1).abs().rank(),
            }
        )
        ranks["avg_rank"] = ranks.mean(axis=1)
        marg = metrics.loc["baseline_marginal"]
        out = m[
            [
                "mean_JSD",
                "het_wasserstein",
                "het_sd_ratio",
                "mean_MAE",
                "pattern_r",
                "indiv_agreement",
                "parse_fail_rate",
            ]
        ].join(ranks)
        out["beats_marginal_JSD"] = out["mean_JSD"] < marg["mean_JSD"]
        out["beats_marginal_het"] = out["het_wasserstein"] < marg["het_wasserstein"]
        return out.sort_values("avg_rank")

    @staticmethod
    def condition_table(
        results: pd.DataFrame, human: pd.DataFrame, retest: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        table = {"human": human.groupby("condition")["answer"].mean()}
        if retest is not None:
            table["human_retest"] = retest.groupby("condition")["answer"].mean()
        for strategy, g in results.groupby("strategy", sort=False):
            table[strategy] = g.groupby("condition")["value"].mean()
        n = human.groupby("condition").size().rename("n_items")
        return pd.DataFrame(table).join(n).sort_index()
