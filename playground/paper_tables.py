"""Tables 1-3: searched hyperparameters, runs per budget, fit quality and extrapolation of the scaling law.

Table 3 uses the recipe of paper_scaling_trends.ipynb: frontier over the grid points in (0.95e11, 1.05e16) FLOPs,
Huber loss in log space (ROC AUC as log(1 - AUC)); held out: the best run per grid point from 1.78e16 to 1e17 FLOPs.

Usage: python playground/paper_tables.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from scaling.utils import fit_parametric_form, functional_form_L0

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "nanotabpfn_model_tabiclv1_prior.csv"
OUT = HERE / "tables"

FLOPS = "total_flops_rounded"
BUDGET = "config/num_flops_training"
FIT_FLOPS_MIN, FIT_FLOPS_MAX = 1e11 * 0.95, 1e16 * 1.05
HELD_OUT_MAX = 1e17 * 1.05
TABLE_1_MAX_BUDGET = 1e16  # Table 1 lists the values of the runs with a nominal budget up to 1e16 FLOPs
INITIAL_GRID = [[0, 0.1, 0.5], [0.1, 1, 2, 5, 10, 50, 100], [-1, -0.5, 0, 0.5, 1]]
DELTA = 1e-3
METRICS = {
    "val/val_loss": ("Synthetic validation loss", False),
    "real_data/nll": ("Real-world NLL", False),
    "real_data/roc_auc": ("Real-world ROC AUC", True),
}
HYPERPARAMETERS = {
    "config/lr": "Learning rate",
    "config/weight_decay": "Weight decay",
    "config/effective_batch_size": "Batch size",
    "config/model_config.embedding_size": "Embedding size",
    "config/model_config.num_layers": "Layers",
    "config/model_config.num_attention_heads": "Attention heads",
    "config/max_features": "Max. features per table",
    "config/num_datapoints_max": "Max. datapoints per table",
}


def fmt(v: float) -> str:
    return f"{v:.3g}" if 0 < v < 1 else f"{v:g}"


def pareto_frontier(df: pd.DataFrame, metric: str, maximize: bool) -> pd.DataFrame:
    """Best run per FLOP grid point, kept only where it improves on every smaller budget."""
    best = df.sort_values([FLOPS, metric], ascending=[True, not maximize]).drop_duplicates(FLOPS)
    running = best[metric].cummax() if maximize else best[metric].cummin()
    if maximize:
        improves = best[metric] > running.shift(fill_value=-np.inf)
    else:
        improves = best[metric] < running.shift(fill_value=np.inf)
    return best[improves]


def best_per_level(df: pd.DataFrame, metric: str, maximize: bool) -> pd.DataFrame:
    return df.sort_values([FLOPS, metric], ascending=[True, not maximize]).drop_duplicates(FLOPS)


def fit_law(flops: np.ndarray, values: np.ndarray, maximize: bool) -> np.ndarray:
    y = np.log(1 - values) if maximize else np.log(values)
    params, _ = fit_parametric_form(functional_form_L0, flops, y, list(product(*INITIAL_GRID)), None, DELTA)
    return params


def predict(params: np.ndarray, flops, maximize: bool) -> np.ndarray:
    loss = functional_form_L0(np.asarray(flops, dtype=float), params, return_log_loss=False)
    return 1 - loss if maximize else loss


def table_1(runs: pd.DataFrame) -> pd.DataFrame:
    rows = [(label, ", ".join(fmt(v) for v in sorted(runs[col].unique()))) for col, label in HYPERPARAMETERS.items()]
    n_params = runs["parameters"]
    rows.append(("Parameters N", f"{n_params.min():,.0f} to {n_params.max():,.0f} ({n_params.nunique()} distinct)"))
    cells = sorted((runs["config/max_features"] * runs["config/num_datapoints_max"]).unique())
    rows.append(("Max. cells per table", ", ".join(fmt(v) for v in cells)))
    return pd.DataFrame(rows, columns=["Hyperparameter", "Values"])


def table_2(runs: pd.DataFrame) -> pd.DataFrame:
    counts = runs.groupby(BUDGET).size()
    return pd.DataFrame({"Budget (FLOPs)": [f"{b:.0e}" for b in counts.index], "Runs": counts.to_numpy()})


def table_3(df: pd.DataFrame) -> pd.DataFrame:
    fit_df = df[(df[FLOPS] > FIT_FLOPS_MIN) & (df[FLOPS] < FIT_FLOPS_MAX)]
    held_df = df[(df[FLOPS] > FIT_FLOPS_MAX) & (df[FLOPS] < HELD_OUT_MAX)]
    rows = []
    for metric, (label, maximize) in METRICS.items():
        frontier = pareto_frontier(fit_df, metric, maximize)
        held = best_per_level(held_df, metric, maximize)
        params = fit_law(frontier[FLOPS].to_numpy(), frontier[metric].to_numpy(), maximize)
        observed, predicted = frontier[metric].to_numpy(), predict(params, frontier[FLOPS], maximize)
        r2 = 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2)
        held_pred = predict(params, held[FLOPS], maximize)
        at_1e17 = np.isclose(held[FLOPS].to_numpy(), 1e17)
        rows.append({
            "Metric": label,
            "Observed at 1e17": float(held.loc[at_1e17, metric].iloc[0]),
            "Predicted at 1e17": float(held_pred[at_1e17][0]),
            "Fit R2": r2,
            "Extrapolation MAE": float(np.mean(np.abs(held_pred - held[metric].to_numpy()))),
            "frontier points": len(frontier),
            "held-out budgets": len(held),
            "L0": float(1 - np.exp(params[0]) if maximize else np.exp(params[0])),
            "alpha": float(-params[2]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, index_col=[0, 1])
    runs = df.groupby(level=0).first()
    tables = {
        "table1_search_space": table_1(runs[runs[BUDGET] <= TABLE_1_MAX_BUDGET]),
        "table2_runs_per_budget": table_2(runs),
        "table3_extrapolation": table_3(df),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n{name}\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
