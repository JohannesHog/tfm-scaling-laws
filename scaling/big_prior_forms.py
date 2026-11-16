"""Functional forms for the wide-prior ROC AUC frontier (Appendix C: Table 4, Figure 7).

Frontier of the wide-prior runs (max_features 16, num_datapoints_max 400) over 0.95e11 < FLOPs < 1e16; the best
wide-prior run above that is held out. Each form is fitted in ROC AUC space with the Huber loss from a fixed grid of
starts, backtested leave-future-out (fit up to a cut, score the points above it) and compared with the all-runs law.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from scaling.utils import (
    double_sigmoid_functional_form,
    fit_parametric_form,
    functional_form_L0,
    gap_powerlaw_deficit,
    logistic,
    sigmoid_functional_form,
)

DATA_PATH = Path("data") / "nanotabpfn_model_tabiclv1_prior.csv"

FLOPS, AUC = "total_flops_rounded", "real_data/roc_auc"
FLOPS_MIN, FLOPS_MAX = 1e11 * 0.95, 1e16 * 1.05
BIG_PRIOR = {"config/max_features": 16, "config/num_datapoints_max": 400}
CUTS = (14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5)  # log10 FLOPs; each leaves >= 3 test points
DELTA = 1e-3
PLOT_GRID = np.logspace(11, 19, 321)
ALL_RUNS_GRID = [[0, 0.1, 0.5], [0.1, 1, 2, 5, 10, 50, 100], [-1, -0.5, 0, 0.5, 1]]  # notebook's initial_grid_min


@dataclass
class Form:
    name: str
    key: str
    family: str
    func: Callable
    grid: list[list[float]]
    bounds: list[tuple[float, float]] | None


FORMS = [
    Form(
        "Logistic, notebook start grid",
        "logistic_nb",
        "single step",
        sigmoid_functional_form,
        [[0.85, 0.9, 0.95, 0.99, 0.999], [2, 4, 6, 8, 10, 12], [0.1, 0.3, 1.0, 3.0, 10.0], [0.45, 0.5]],
        None,
    ),
    Form(
        "Logistic",
        "logistic",
        "single step",
        logistic,
        [[0.87, 0.9, 0.95], [12.5, 13, 13.5, 14], [0.5, 1, 2, 4, 8], [0.5, 0.54]],
        [(0.5, 1), (10, 17), (0.01, 50), (0.3, 0.6)],
    ),
    Form(
        "Power law + deficit",
        "gap_pl_deficit",
        "power-law tail",
        gap_powerlaw_deficit,
        [
            [np.log(0.08), np.log(0.11), np.log(0.13)],
            [np.log(0.05), np.log(0.1)],
            [0.1, 0.3],
            [0.15, 0.25],
            [13, 13.5],
            [2, 6],
        ],
        [(np.log(1e-4), np.log(0.5)), (-10, 2), (0, 3), (0, 0.5), (10, 17), (0.01, 50)],
    ),
    Form(
        "Two logistics",
        "two_logistics",
        "two steps",
        double_sigmoid_functional_form,
        [[0.45, 0.5], [0.15, 0.2], [13.3], [5, 10], [0.15, 0.25, 0.35], [12.5, 13.5, 14.5], [0.3, 0.6, 1.0]],
        [(0.3, 0.6), (0, 0.5), (12, 15), (0.5, 50), (0, 0.5), (10, 18), (0.01, 5)],
    ),
]


# --- data -----------------------------------------------------------------------
def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Running maximum of ROC AUC over FLOPs, best run per FLOP level (the notebook's get_pareto_frontier)."""
    best = df.sort_values([FLOPS, AUC], ascending=[True, False]).drop_duplicates(FLOPS)
    return best[best[AUC] > best[AUC].cummax().shift(fill_value=-np.inf)]


def best_per_level(df: pd.DataFrame) -> pd.DataFrame:
    """Best run per FLOP level (the notebook's get_optimal_values)."""
    return df.sort_values([FLOPS, AUC], ascending=[True, False]).drop_duplicates(FLOPS)


def load(data_path: str | Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(data_path, index_col=[0, 1])
    big = raw[(raw[list(BIG_PRIOR)] == pd.Series(BIG_PRIOR)).all(axis=1)]
    in_range = (raw[FLOPS] > FLOPS_MIN) & (raw[FLOPS] < FLOPS_MAX)
    return {
        "big": pareto_frontier(big[in_range.loc[big.index]]),
        "big_held_out": best_per_level(big[big[FLOPS] > FLOPS_MAX]),
        "all": pareto_frontier(raw[in_range]),
        "all_held_out": best_per_level(raw[raw[FLOPS] > FLOPS_MAX]),
    }


# --- fitting and scoring ----------------------------------------------------------
def fit(form: Form, flops: np.ndarray, auc: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        params, _ = fit_parametric_form(form.func, flops, auc, list(product(*form.grid)), form.bounds, DELTA)
    return params


def predict(form: Form, params: np.ndarray, flops) -> np.ndarray:
    return np.asarray(form.func(np.atleast_1d(np.asarray(flops, dtype=float)), params), dtype=float)


def all_runs_curve(frontier: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    """The notebook's 'best prior / flops' fit: functional_form_L0 on log(1 - AUC)."""
    flops = frontier[FLOPS].to_numpy()
    params, _ = fit_parametric_form(
        functional_form_L0, flops, np.log(1 - frontier[AUC].to_numpy()), list(product(*ALL_RUNS_GRID)), None, DELTA
    )
    return lambda c: 1 - functional_form_L0(np.atleast_1d(np.asarray(c, dtype=float)), params, return_log_loss=False)


def backtest(form: Form, flops: np.ndarray, auc: np.ndarray, held_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cut in CUTS:
        train = np.log10(flops) <= cut + 1e-6
        params = fit(form, flops[train], auc[train])
        test_flops = np.r_[flops[~train], held_out[FLOPS].to_numpy()]
        err = np.r_[auc[~train], held_out[AUC].to_numpy()] - predict(form, params, test_flops)
        rows.append(
            {
                "form": form.key,
                "cut": cut,
                "n_train": int(train.sum()),
                "mae": np.abs(err).mean(),
                "bias": err.mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate(form: Form, data: dict[str, pd.DataFrame], total: Callable) -> tuple[dict, np.ndarray, pd.DataFrame]:
    flops, auc = data["big"][FLOPS].to_numpy(), data["big"][AUC].to_numpy()
    params = fit(form, flops, auc)
    res = auc - predict(form, params, flops)
    over = predict(form, params, PLOT_GRID) - total(PLOT_GRID)
    held = data["big_held_out"]
    bt = backtest(form, flops, auc, held)
    summary = {
        "form": form.name,
        "key": form.key,
        "rmse": np.sqrt(float((res**2).sum()) / len(auc)),
        "backtest_mae": bt["mae"].mean(),
        "max_excess": over.max(),
        "params": " ".join(f"{v:.4g}" for v in params),
    }
    return summary, params, bt


def run(data_path: str | Path = DATA_PATH, verbose: bool = True) -> dict:
    """Fit and backtest every form. Returns the frontiers, the all-runs curve, the parameters per form and the
    Table 4 summary."""
    data = load(data_path)
    total = all_runs_curve(data["all"])
    summaries, params = [], {}
    for form in FORMS:
        summary, params[form.key], _ = evaluate(form, data, total)
        summaries.append(summary)
        if verbose:
            print(f"{form.name:32s} RMSE {summary['rmse']:.4f}  backtest MAE {summary['backtest_mae']:.3f}", flush=True)
    return {"data": data, "total": total, "params": params, "summary": pd.DataFrame(summaries)}
