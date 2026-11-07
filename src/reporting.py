import json
import os
import time
from dataclasses import asdict
import numpy as np

from .experiment import RecoverDistributionAlignment


class ResultsReporter:
    """Saves one finished experiment to `<config.output_dir>/<tag>/`.

    Parameters
    ----------
    experiment : DistributionAlignmentExperiment
        An experiment after `run()`; `evaluate()` is called if needed.

    Notes
    -----
    Folder layout::

        per_call_results.csv      one row per LLM call
        summary/                  metrics, strategy_ranking, condition_rates,
                                    token_usage (.csv) and run_summary.json
        data/                     test_items.csv, dataset_report.json
        prompts/                  examples.md, <strategy>.txt, examples.json,
                                    paraphrases.json, all_prompts.jsonl
        plots/                    demand_curves, heterogeneity, metric_summary (.png)
    """

    STYLE = {
        "direct": ("#2a78d6", "o"),
        "verbalized": ("#eb6834", "s"),
        "population": ("#1baf7a", "^"),
        "knowledge_restriction": ("#eda100", "D"),
        "paraphrase": ("#e87ba4", "v"),
        "fewshot_self": ("#008300", "P"),
    }
    HUMAN, RETEST, BASELINE = "#1f1f1e", "#8a8984", "#c3c2b7"
    INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"

    def __init__(self, runner: RecoverDistributionAlignment):
        self.runner = runner
        self.cfg = runner.config

    def style(self, name: str) -> tuple:
        return self.STYLE.get(name, ("#6250d6", "X"))

    def save(self, tag: str | None = None) -> str:
        if self.runner.metrics is None:
            self.runner.evaluate()
        out = os.path.join(self.cfg.output_dir, tag or time.strftime("%Y%m%d_%H%M%S"))
        for sub in ("summary", "data", "prompts", "plots"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        self.runner.results.to_csv(os.path.join(out, "per_call_results.csv"), index=False)
        self.runner.test_items.to_csv(os.path.join(out, "data", "test_items.csv"), index=False)
        self.write_json(
            os.path.join(out, "data", "dataset_report.json"), self.runner.dataset.report
        )
        self.save_summary(os.path.join(out, "summary"))
        self.save_prompts(os.path.join(out, "prompts"))
        self.plot(os.path.join(out, "plots"))
        return out

    @staticmethod
    def write_json(path: str, obj) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    def save_summary(self, folder: str) -> None:
        self.runner.metrics.to_csv(os.path.join(folder, "metrics.csv"))
        self.runner.ranking.to_csv(os.path.join(folder, "strategy_ranking.csv"))
        self.runner.condition_table().to_csv(os.path.join(folder, "condition_rates.csv"))
        self.runner.token_usage().to_csv(os.path.join(folder, "token_usage.csv"))
        self.write_json(
            os.path.join(folder, "run_summary.json"),
            dict(
                started_at=self.runner.started_at,
                saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=asdict(self.cfg) | {"token": None},
                test_set=dict(
                    items=len(self.runner.test_items),
                    twins=self.runner.test_items["pid"].nunique(),
                    units=self.runner.test_items[self.runner.dataset.unit].nunique(),
                    items_per_condition=self.runner.test_items["condition"]
                    .value_counts()
                    .sort_index()
                    .to_dict(),
                ),
                calls=dict(
                    total=len(self.runner.results),
                    cached=int(self.runner.results["cached"].sum()),
                    errors=int(self.runner.results["error"].count()),
                ),
                best_strategy=self.runner.ranking.index[0],
                any_beats_marginal_JSD=bool(self.runner.ranking["beats_marginal_JSD"].any()),
            ),
        )

    def save_prompts(self, folder: str) -> None:
        md = ["# Prompt examples\n\nOne real call per strategy from this run.\n"]
        for name, ex in self.runner.prompt_examples.items():
            ctx = ", ".join(
                f"{k}={ex[k]}" for k in ("pid", "unit", "sample_idx") if k in ex
            )
            with open(os.path.join(folder, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(
                    f"=== {name} ({ex['kind']}) | {ctx} ===\n\n--- SYSTEM ---\n"
                    f"{ex['system']}\n\n--- USER ---\n{ex['user']}\n\n"
                    f"--- RESPONSE ---\n{ex['response']}\n"
                )
            md.append(
                f"## {name}\n\n*{ex['kind']} | {ctx}*\n\n**System**\n\n```\n"
                f"{ex['system']}\n```\n\n**User**\n\n```\n{ex['user']}\n```\n\n"
                f"**Response**\n\n```\n{ex['response']}\n```\n"
            )
        with open(os.path.join(folder, "examples.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        self.write_json(os.path.join(folder, "examples.json"), self.runner.prompt_examples)
        if self.runner.paraphrases:
            self.write_json(os.path.join(folder, "paraphrases.json"), self.runner.paraphrases)
        if self.runner.all_prompts:
            with open(
                os.path.join(folder, "all_prompts.jsonl"), "w", encoding="utf-8"
            ) as f:
                f.writelines(json.dumps(r) + "\n" for r in self.runner.all_prompts.values())

    def plot(self, folder: str) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.edgecolor": self.GRID,
                "axes.labelcolor": self.MUTED,
                "xtick.color": self.MUTED,
                "ytick.color": self.MUTED,
                "axes.titlecolor": self.INK,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.color": self.GRID,
                "grid.linewidth": 0.6,
                "axes.axisbelow": True,
                "legend.frameon": False,
                "figure.dpi": 150,
            }
        )
        for name, fn in (
            ("demand_curves", self.plot_demand_curves),
            ("heterogeneity", self.plot_heterogeneity),
            ("metric_summary", self.plot_metric_summary),
        ):
            fig = fn(plt)
            fig.savefig(
                os.path.join(folder, f"{name}.png"),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

    def plot_demand_curves(self, plt):
        ct = self.runner.condition_table()
        x = np.arange(len(ct))
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(
            x,
            ct["human"],
            color=self.HUMAN,
            lw=2.5,
            marker="o",
            ms=8,
            label="Human",
            zorder=5,
        )
        ax.plot(
            x,
            ct["human_retest"],
            color=self.RETEST,
            lw=2,
            ls="--",
            marker="o",
            ms=7,
            label="Human retest",
        )
        for s in self.runner.results["strategy"].unique():
            c, m = self.style(s)
            ax.plot(
                x,
                ct[s],
                color=c,
                lw=2,
                marker=m,
                ms=8,
                label=s,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
        ax.set_xticks(x, [str(i) for i in ct.index], rotation=0 if len(ct) <= 6 else 45)
        ax.set(
            ylim=(0, 1),
            ylabel="Purchase rate",
            xlabel=f"Condition ({self.cfg.condition_key})",
        )
        ax.set_title("Purchase rate by condition", loc="left", fontsize=12)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
        return fig

    def plot_heterogeneity(self, plt):
        me = self.runner.metrics_engine
        human = me.human_shares(self.runner.test_items)
        strategies = list(self.runner.results["strategy"].unique())
        cols = min(3, len(strategies))
        rows = int(np.ceil(len(strategies) / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(3.6 * cols, 2.9 * rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        bins = np.linspace(0, 1, 11)
        for ax, s in zip(axes.flat, strategies):
            valid = self.runner.results[self.runner.results["strategy"] == s].dropna(subset=["value"])
            ax.hist(
                human,
                bins=bins,
                density=True,
                color=self.BASELINE,
                edgecolor="white",
                linewidth=2,
                label="Human",
            )
            ax.hist(
                me.strategy_shares(s, valid),
                bins=bins,
                density=True,
                histtype="step",
                color=self.style(s)[0],
                lw=2,
                label=s,
            )
            ax.set_title(
                f"{s}  (SD ratio {self.runner.metrics.loc[s, 'het_sd_ratio']:.2f})",
                loc="left",
                fontsize=10,
            )
            ax.legend(fontsize=8, loc="upper right")
        for ax in axes.flat[len(strategies) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Share of products purchased")
        for ax in axes[:, 0]:
            ax.set_ylabel("Density")
        fig.suptitle(
            "Same Person purchase share: strategy vs human",
            x=0.01,
            ha="left",
            fontsize=12,
            color=self.INK,
        )
        fig.tight_layout()
        return fig

    def plot_metric_summary(self, plt):
        m = self.runner.metrics.sort_values(
            "type",
            key=lambda t: t.map({"strategy": 0, "baseline": 1, "reference": 2}),
            kind="stable",
        )
        colors = [
            (
                self.style(p)[0]
                if t == "strategy"
                else self.RETEST if t == "reference" else self.BASELINE
            )
            for p, t in zip(m.index, m["type"])
        ]
        panels = [
            ("mean_JSD", "Mean JSD (lower is better)", None),
            ("het_wasserstein", "Heterogeneity distance (lower is better)", None),
            ("het_sd_ratio", "Spread ratio (1 = human spread)", 1.0),
            ("mean_MAE", "Mean absolute error (lower is better)", None),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(m) + 1.6), sharey=True)
        y = np.arange(len(m))
        for ax, (col, title, ref) in zip(axes, panels):
            vals = m[col].to_numpy(float)
            ax.barh(y, vals, color=colors, height=0.7, edgecolor="white", linewidth=2)
            label_x = vals
            if col == "mean_JSD":
                hi = m["JSD_CI_high"].to_numpy(float)
                ax.errorbar(
                    vals,
                    y,
                    xerr=[vals - m["JSD_CI_low"].to_numpy(float), hi - vals],
                    fmt="none",
                    ecolor=self.MUTED,
                    elinewidth=1,
                    capsize=3,
                )
                label_x = np.fmax(vals, hi)
            if ref:
                ax.axvline(ref, color=self.INK, lw=1, ls="--")
            for yi, v, lx in zip(y, vals, label_x):
                ax.text(lx, yi, f"  {v:.3f}", va="center", fontsize=8, color=self.INK)
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.25)
        axes[0].set_yticks(y, list(m.index))
        axes[0].invert_yaxis()
        fig.suptitle(
            "Strategies vs baselines (grey) and human retest (dark grey)",
            x=0.01,
            ha="left",
            fontsize=12,
            color=self.INK,
        )
        fig.tight_layout()
        return fig
