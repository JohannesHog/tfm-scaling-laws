"""Evaluation-set selection bias of the compute frontier (Section 4.3, Appendix A.4, Figure 4).

For n selection datasets drawn from the other 20, the best run per FLOP level is picked on their mean and scored on
the held-out dataset; pooled over the 21 targets this gives an in-sample and an honest frontier. Their gap is the
selection bias as a function of n, compared with the residual of the scaling law (``functional_form_L0``).

Modes: ``transfer`` (the paper) and ``loo`` (leave-one-dataset-out, secondary). Aggregates are unweighted means over
datasets; a row is used only when every dataset of the set is non-NaN. Results are cached per ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scaling.utils import fit_parametric_form, functional_form_L0

# Fitting recipe, identical to the "Fit scaling laws for the individual parts" cell of paper_scaling_trends.ipynb.
INITIAL_GRID_MIN = [
    [0, 0.1, 0.5],
    [0.1, 1, 2, 5, 10, 50, 100],
    [-1, -0.5, 0, 0.5, 1],
]
HUBER_DELTA = 1e-3
MAXIMIZE_METRICS = ("roc_auc", "accuracy", "balanced_accuracy")
FLOPS_COLUMN = "total_flops_rounded"
RUN_LABEL_COLUMN = "_run_label"  # added by run_experiment so that the selected run can be reported after reset_index
DEFAULT_OUT_DIR_LOO = Path("playground") / "cache" / "selection_bias_loo"
MODES = ("transfer", "loo")
DEFAULT_OUT_DIR_TRANSFER = Path("playground") / "cache" / "selection_bias"
LOO_DEFAULT_N_TRAIN = (1, 2, 3, 5, 8, 12, 16, 20)
LOO_DEFAULT_N_REPS = 20
LOO_SEED_STRIDE = 1000
LOO_QUANTITIES = (
    "mae", "rmse", "bias", "mae_raw_frontier", "bias_raw_frontier", "mae_transfer", "selection_gap", "law_beats_raw",
)
QUANTITIES = (
    "mae",
    "rmse",
    "bias",
    "mae_raw_frontier",
    "rmse_raw_frontier",
    "bias_raw_frontier",
    "mae_transfer",
    "mae_minus_full_pool",
    "selection_gap",
    "law_beats_raw",
)
PARAM_QUANTITIES = ("L0", "a", "alpha")
N_SLASHES = 2  # real_data/<dataset>/<metric>


def is_maximize(metric: str) -> bool:
    """ROC AUC and accuracies are maximised, NLL is minimised."""
    return metric in MAXIMIZE_METRICS


@dataclass
class ExperimentConfig:
    """Settings of one experiment run. ``n_jobs``, ``force`` and ``out_dir`` do not affect results."""

    data_path: str = "data/nanotabpfn_model_tabiclv1_prior.csv"
    out_dir: str = str(DEFAULT_OUT_DIR_TRANSFER)
    n_reps: int = LOO_DEFAULT_N_REPS
    n_train: tuple = LOO_DEFAULT_N_TRAIN
    base_seed: int = 0
    metrics: tuple = ("roc_auc", "nll")
    flops_min: float = 0.95e11
    flops_max: float = 1.05e16
    error_threshold: dict = field(default_factory=lambda: {"roc_auc": 0.01, "nll": 0.02})
    n_jobs: int = -1
    force: bool = False
    mode: str = "transfer"
    # How the honest (held-out) curve is built; part of the cache key so that older results are recomputed.
    # "carried": running best on the in-sample scores, held-out score carried from the level that achieves it.
    honest_curve: str = "carried"

    @classmethod
    def loo(cls, **kwargs) -> ExperimentConfig:
        """Config of the leave-one-dataset-out mode (its own output directory)."""
        return cls(**{"mode": "loo", "out_dir": str(DEFAULT_OUT_DIR_LOO), **kwargs})

    @classmethod
    def transfer(cls, **kwargs) -> ExperimentConfig:
        """Config of the recipe-transfer mode (the default)."""
        return cls(**{"mode": "transfer", "out_dir": str(DEFAULT_OUT_DIR_TRANSFER), **kwargs})

    def result_key(self) -> dict:
        """The part of the config that determines the results (compared against ``config.json``)."""
        key = asdict(self)
        for ignored in ("n_jobs", "force", "out_dir", "data_path"):
            key.pop(ignored)
        return json.loads(json.dumps(key))


# ----------------------------------------------------------------------------------------------------------------------
# Data and aggregation
# ----------------------------------------------------------------------------------------------------------------------
def load_data(path, flops_min: float = 0.95e11, flops_max: float = 1.05e16) -> pd.DataFrame:
    """Load the CSV and keep ``flops_min < total_flops_rounded < flops_max`` (as the analysis notebook)."""
    df = pd.read_csv(path, index_col=[0, 1])
    return df[(df[FLOPS_COLUMN] > flops_min) & (df[FLOPS_COLUMN] < flops_max)]


def real_datasets(df: pd.DataFrame) -> list[str]:
    """Sorted names of the real-data benchmark datasets (columns ``real_data/<dataset>/<metric>``)."""
    return sorted({c.split("/")[1] for c in df.columns if c.startswith("real_data/") and c.count("/") == N_SLASHES})


def aggregate(df: pd.DataFrame, datasets, metric: str) -> pd.Series:
    """Unweighted mean over ``real_data/<d>/<metric>``; NaN if any member dataset is NaN (complete-case rule)."""
    cols = [f"real_data/{d}/{metric}" for d in datasets]
    return df[cols].mean(axis=1, skipna=False)


def _best_per_level(values: pd.Series, flops: pd.Series, maximize: bool) -> pd.Series:
    """Best finite value at each FLOP level, indexed by (sorted) level."""
    frame = pd.DataFrame({"f": np.asarray(flops, dtype=float), "v": np.asarray(values, dtype=float)}).dropna()
    grouped = frame.groupby("f")["v"]
    return grouped.max() if maximize else grouped.min()


def frontier_points(values: pd.Series, flops: pd.Series, maximize: bool) -> pd.DataFrame:
    """Pareto frontier over FLOPs (best run per level, strictly improving levels only), as the notebook's function.

    Returns a frame with columns ``flops`` and ``value``; empty if there is no finite value.
    """
    best = _best_per_level(values, flops, maximize)
    keep_f, keep_v = [], []
    record = -np.inf if maximize else np.inf
    for level, value in best.items():
        if (value > record) if maximize else (value < record):
            keep_f.append(level)
            keep_v.append(value)
            record = value
    return pd.DataFrame({"flops": keep_f, "value": keep_v})


def _running(best: np.ndarray, maximize: bool) -> np.ndarray:
    """Running best along the grid, keeping NaN where a level has no value."""
    series = pd.Series(best)
    return (series.cummax() if maximize else series.cummin()).to_numpy()


def running_best(values: pd.Series, flops: pd.Series, maximize: bool, grid) -> np.ndarray:
    """Best per level, then running best (cummax / cummin) over ``grid``; NaN where a level has no finite row."""
    best = _best_per_level(values, flops, maximize).reindex(np.asarray(grid, dtype=float)).to_numpy()
    return _running(best, maximize)


def carried_running_best(select, score, maximize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Running best of ``select`` over the grid and, at each level, the ``score`` of the level that achieves it.

    ``score`` is never compared, so a curve built from held-out scores involves no selection on them: at level ``k``
    it reports the held-out score of the (level, run) that the in-sample running best points to. Levels whose
    ``select`` is NaN are NaN in both outputs (as ``_running``); later levels carry on from the last finite record.
    Ties keep the earlier level.
    """
    select, score = np.asarray(select, dtype=float), np.asarray(score, dtype=float)
    best, carried = np.full(len(select), np.nan), np.full(len(select), np.nan)
    record, record_score = np.nan, np.nan
    for k, (s, t) in enumerate(zip(select, score)):
        if np.isnan(s):
            continue
        if np.isnan(record) or ((s > record) if maximize else (s < record)):
            record, record_score = s, t
        best[k], carried[k] = record, record_score
    return best, carried


def transfer_running_best(
    select_values: pd.Series, score_values: pd.Series, flops: pd.Series, maximize: bool, grid
) -> np.ndarray:
    """Per level pick the run that is best on ``select_values``; running best over levels on those runs' selection
    scores; return the ``score_values`` of the run at the level that achieves it (no running best on the scores)."""
    chosen = selected_values(select_values, select_values, flops, maximize, grid)
    scored = selected_values(select_values, score_values, flops, maximize, grid)
    return carried_running_best(chosen, scored, maximize)[1]


def selected_positions(select_values: pd.Series, flops: pd.Series, maximize: bool, grid) -> np.ndarray:
    """Positional row index of the run that is best on ``select_values`` at each grid level; -1 where none."""
    frame = pd.DataFrame(
        {"f": np.asarray(flops, dtype=float), "s": np.asarray(select_values, dtype=float), "pos": np.arange(len(flops))}
    ).dropna(subset=["s"])
    frame = frame.sort_values(["f", "s"], ascending=[True, not maximize], kind="stable")
    chosen = frame.drop_duplicates(subset="f", keep="first").set_index("f")["pos"]
    return chosen.reindex(np.asarray(grid, dtype=float)).fillna(-1).astype(int).to_numpy()


def selected_values(
    select_values: pd.Series, score_values: pd.Series, flops: pd.Series, maximize: bool, grid
) -> np.ndarray:
    """Per level the ``score_values`` of the run that is best on ``select_values`` (no running best); NaN if none."""
    pos = selected_positions(select_values, flops, maximize, grid)
    score = np.asarray(score_values, dtype=float)
    return np.where(pos >= 0, score[np.clip(pos, 0, None)], np.nan)


def run_labels(data: pd.DataFrame) -> np.ndarray:
    """String label of every row's run: ``RUN_LABEL_COLUMN`` if present, else the first index level (the run id of the
    CSV) or the plain index."""
    if RUN_LABEL_COLUMN in data.columns:
        return np.asarray(data[RUN_LABEL_COLUMN].astype(str), dtype=object)
    index = data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index
    return np.asarray(index.astype(str), dtype=object)


# ----------------------------------------------------------------------------------------------------------------------
# Law fitting
# ----------------------------------------------------------------------------------------------------------------------
def fit_law_full(flops, values, maximize: bool) -> tuple[np.ndarray, float, int]:
    """Fit ``functional_form_L0`` to the frontier of (flops, values). Returns (params, huber_loss, n_frontier_points).

    NLL is fitted on ``log(y)``, maximised metrics (ROC AUC) on ``log(1 - y)``. Params are ``[l0, a, b]``.
    """
    pareto = frontier_points(pd.Series(np.asarray(values)), pd.Series(np.asarray(flops)), maximize)
    n_points = len(pareto)
    if n_points == 0:
        return np.full(3, np.nan), np.nan, 0
    y = pareto["value"].to_numpy()
    y_data = np.log(1 - y) if maximize else np.log(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, loss = fit_parametric_form(
            func_form=functional_form_L0,
            x_data=pareto["flops"].to_numpy(),
            y_data=y_data,
            initial_grid=list(product(*INITIAL_GRID_MIN)),
            bounds=None,
            delta=HUBER_DELTA,
        )
    if params is None:
        return np.full(3, np.nan), np.nan, n_points
    return np.asarray(params, dtype=float), float(loss), n_points


def fit_law(flops, values, maximize: bool) -> np.ndarray:
    """Parameters ``[l0, a, b]`` of the law fitted on the frontier of (flops, values)."""
    return fit_law_full(flops, values, maximize)[0]


def predict_law(params, grid, maximize: bool) -> np.ndarray:
    """Predicted metric on ``grid``; ``1 - L`` for maximised metrics (the law is on ``1 - y``)."""
    grid = np.asarray(grid, dtype=float)
    if not np.all(np.isfinite(params)):
        return np.full(grid.shape, np.nan)
    with np.errstate(all="ignore"):
        y = np.asarray(functional_form_L0(grid, params, return_log_loss=False), dtype=float)
    return 1 - y if maximize else y


# ----------------------------------------------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """One (target, repetition): held-out dataset and the pool order whose prefixes are the nested fitting sets."""

    seed: int
    held_out: tuple
    pool: tuple

    def fit_set(self, n: int) -> tuple:
        return self.pool[:n]


def make_splits_loo(datasets, n_reps: int, n_train_sizes, base_seed: int = 0) -> list[Split]:
    """Leave-one-dataset-out splits: every dataset is the held-out target ``n_reps`` times (ordered by dataset, rep).

    For the sorted dataset ``i`` and repetition ``r``: ``rng = default_rng(base_seed + 1000 * i + r)``,
    ``pool = rng.permutation(others)``; fitting sets are nested prefixes of ``pool``.
    """
    names = sorted(datasets)
    if n_reps > LOO_SEED_STRIDE:
        raise ValueError("n_reps must not exceed the seed stride of 1000")
    valid_sizes(n_train_sizes, len(names) - 1)
    splits = []
    for i, target in enumerate(names):
        others = [d for d in names if d != target]
        for rep in range(n_reps):
            seed = base_seed + LOO_SEED_STRIDE * i + rep
            pool = tuple(str(x) for x in np.random.default_rng(seed).permutation(others))
            splits.append(Split(seed=seed, held_out=(target,), pool=pool))
    return splits


def valid_sizes(n_train_sizes, pool_size: int) -> list[int]:
    """Drop fitting-set sizes that exceed the pool (with a warning)."""
    keep = [int(n) for n in n_train_sizes if 1 <= int(n) <= pool_size]
    dropped = sorted(set(int(n) for n in n_train_sizes) - set(keep))
    if dropped:
        warnings.warn(f"dropping fitting-set sizes {dropped} larger than the pool ({pool_size})", stacklevel=2)
    return keep


# ----------------------------------------------------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------------------------------------------------
def _joined(names) -> str:
    return ";".join(names)


def _fit_row(ident: tuple, fit_datasets, held_out, fit_result) -> dict:
    """``ident`` = (rep, seed, n_train, metric)."""
    rep, seed, n_train, metric = ident
    params, loss, n_points = fit_result
    return {
        "rep": rep, "seed": seed, "n_train": n_train, "metric": metric,
        "fit_datasets": _joined(fit_datasets), "held_out": _joined(held_out),
        "l0": params[0], "a": params[1], "b": params[2],
        "L0": np.exp(params[0]), "alpha": -params[2],
        "huber_loss": loss, "n_frontier_points": n_points,
    }


def _curve_frame(ident: tuple, grid, columns: dict) -> pd.DataFrame:
    """``ident`` = (rep, n_train, metric, held_out)."""
    rep, n_train, metric, held_out = ident
    frame = pd.DataFrame(
        {"rep": rep, "n_train": n_train, "metric": metric, "held_out": held_out, "flops": grid, **columns}
    )
    frame["error"] = frame["predicted"] - frame["observed_heldout"]
    return frame


def _run_rep(df: pd.DataFrame, job: tuple, sizes: list[int], metrics, grid: np.ndarray):
    """All fits and curves of one repetition (a pure function of the data columns of fitting and held-out sets)."""
    rep, split = job
    fit_rows, curves = [], []
    flops = df[FLOPS_COLUMN]
    for metric in metrics:
        maximize = is_maximize(metric)
        held = aggregate(df, split.held_out, metric)
        for n in sizes:
            fit_set = split.fit_set(n)
            fit_agg = aggregate(df, fit_set, metric)
            # compute levels that have complete rows on both sides (identical for pred and obs); irrelevant w/o NaN
            ok = fit_agg.notna() & held.notna()
            levels = np.isin(grid, np.unique(flops[ok].to_numpy()))
            g = grid[levels]
            result = fit_law_full(flops[fit_agg.notna()], fit_agg[fit_agg.notna()], maximize)
            fit_rows.append(_fit_row((rep, split.seed, n, metric), fit_set, split.held_out, result))
            curves.append(
                _curve_frame(
                    (rep, n, metric, _joined(split.held_out)), g,
                    {
                        "predicted": predict_law(result[0], g, maximize),
                        "observed_heldout": running_best(held, flops, maximize, g),
                        "observed_transfer": transfer_running_best(fit_agg, held, flops, maximize, g),
                        "raw_frontier": running_best(fit_agg, flops, maximize, g),
                    },
                )
            )
    return fit_rows, curves


def run_experiment(df: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    """Run all repetitions (parallel over repetitions) and return ``splits``, ``fits``, ``curves``, ``errors``."""
    datasets = real_datasets(df)
    if config.mode not in MODES:
        raise ValueError(f"unknown mode {config.mode!r}, expected one of {MODES}")
    sizes = valid_sizes(config.n_train, len(datasets) - 1)
    splits = make_splits_loo(datasets, config.n_reps, sizes, config.base_seed)
    keep = [FLOPS_COLUMN] + [f"real_data/{d}/{m}" for d in datasets for m in config.metrics]
    data = df[keep].copy()
    data[RUN_LABEL_COLUMN] = run_labels(df)
    data = data.reset_index(drop=True)
    grid = np.array(sorted(data[FLOPS_COLUMN].unique()), dtype=float)

    if config.mode == "transfer":
        return run_transfer(data, splits, (datasets, sizes), config, grid)
    outputs = Parallel(n_jobs=config.n_jobs)(
        delayed(_run_rep)(data, job, sizes, config.metrics, grid) for job in enumerate(splits)
    )
    fit_rows = [row for rows, _ in outputs for row in rows]
    curves = pd.concat([c for _, cs in outputs for c in cs], ignore_index=True)

    # Full-benchmark reference law (all datasets, not a held-out quantity; used only for plots and the parameter table).
    for metric in config.metrics:
        agg = aggregate(data, datasets, metric)
        ok = agg.notna()
        result = fit_law_full(data[FLOPS_COLUMN][ok], agg[ok], is_maximize(metric))
        fit_rows.append(_fit_row((-1, -1, "all", metric), datasets, (), result))

    splits_df = pd.DataFrame(
        [{"rep": r, "seed": s.seed, "held_out": _joined(s.held_out), "pool_order": _joined(s.pool)}
         for r, s in enumerate(splits)]
    )
    fits = pd.DataFrame(fit_rows)
    return {"splits": splits_df, "fits": fits, "curves": curves, "errors": compute_errors(curves)}


# ----------------------------------------------------------------------------------------------------------------------
# Errors and summaries
# ----------------------------------------------------------------------------------------------------------------------
def _err_stats(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float, float]:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if not ok.any():
        return np.nan, np.nan, np.nan
    e = pred[ok] - obs[ok]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), float(np.mean(e))


def compute_errors(curves: pd.DataFrame) -> pd.DataFrame:
    """Per (rep, n_train, metric) MAE / RMSE / bias of the law and of the raw frontier against the held-out frontier.

    ``mae_transfer``: law vs transferred-selection target. ``selection_gap``: mean loss in performance caused by
    selecting runs on the fitting datasets (>= 0, in metric units). ``mae_minus_full_pool``: MAE minus the MAE of the
    largest n of the same repetition and metric.
    """
    rows = []
    for (rep, n, metric), g in curves.groupby(["rep", "n_train", "metric"], sort=False):
        pred, obs = g["predicted"].to_numpy(), g["observed_heldout"].to_numpy()
        raw, transfer = g["raw_frontier"].to_numpy(), g["observed_transfer"].to_numpy()
        mae, rmse, bias = _err_stats(pred, obs)
        mae_raw, rmse_raw, bias_raw = _err_stats(raw, obs)
        sign = -1.0 if is_maximize(metric) else 1.0
        ok = np.isfinite(obs) & np.isfinite(transfer)
        gap = float(np.mean(sign * (transfer[ok] - obs[ok]))) if ok.any() else np.nan
        rows.append({
            "rep": rep, "n_train": n, "metric": metric,
            "held_out": g["held_out"].iloc[0] if "held_out" in g else "", "mae": mae, "rmse": rmse, "bias": bias,
            "mae_raw_frontier": mae_raw, "rmse_raw_frontier": rmse_raw, "bias_raw_frontier": bias_raw,
            "mae_transfer": _err_stats(pred, transfer)[0], "selection_gap": gap,
            "law_beats_raw": float(mae < mae_raw) if np.isfinite(mae) and np.isfinite(mae_raw) else np.nan,
        })
    errors = pd.DataFrame(rows)
    n_max = errors.groupby(["rep", "metric"])["n_train"].transform("max")
    ref = errors[errors["n_train"] == n_max].set_index(["rep", "metric"])["mae"]
    index = pd.MultiIndex.from_frame(errors[["rep", "metric"]])
    errors["mae_minus_full_pool"] = errors["mae"] - ref.reindex(index).to_numpy()
    return errors


def _describe(x: pd.Series) -> dict:
    x = x.dropna()
    q = x.quantile([0.1, 0.25, 0.75, 0.9]) if len(x) else pd.Series([np.nan] * 4, index=[0.1, 0.25, 0.75, 0.9])
    return {
        "n_reps": len(x), "mean": x.mean(), "std": x.std(), "median": x.median(),
        "q10": q[0.1], "q25": q[0.25], "q75": q[0.75], "q90": q[0.9],
    }


def _long_summary(frame: pd.DataFrame, keys: list[str], quantities) -> pd.DataFrame:
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        for quantity in quantities:
            rows.append({**dict(zip(keys, key_tuple)), "quantity": quantity, **_describe(g[quantity])})
    return pd.DataFrame(rows)


def dedupe_full_pool(frame: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    """Keep one row per (held_out, metric, n_train) where ``n_train >= pool_size`` (repetitions are identical sets)."""
    n = pd.to_numeric(frame["n_train"], errors="coerce")
    keys = [c for c in ("held_out", "metric", "n_train") if c in frame.columns]
    duplicate = frame.duplicated(keys) & (n >= pool_size)
    return frame[~duplicate]


def _threshold_table(median_mae: pd.Series, config: ExperimentConfig, note: str) -> pd.DataFrame:
    """Smallest n whose (median) MAE, indexed by (metric, n_train), is below the metric's threshold."""
    rows = []
    for metric in config.metrics:
        thr = config.error_threshold.get(metric)
        has = metric in median_mae.index.get_level_values(0)
        series = median_mae.loc[metric].sort_index() if has else pd.Series(dtype=float)
        hit = series[series < thr] if thr is not None else series.iloc[0:0]
        rows.append({
            "metric": metric, "threshold": thr,
            "smallest_n_below_threshold": hit.index[0] if len(hit) else np.nan,
            "median_mae_at_that_n": hit.iloc[0] if len(hit) else np.nan,
            "note": note,
        })
    return pd.DataFrame(rows)


def per_dataset_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """LOO: one row per (metric, n_train, held-out dataset); median and q10/q90 over repetitions of each quantity."""
    rows = []
    for (metric, n, target), g in errors.groupby(["metric", "n_train", "held_out"], sort=True):
        row = {"metric": metric, "n_train": n, "held_out": target, "n_samples": len(g)}
        for q in LOO_QUANTITIES:
            x = g[q].dropna()
            row[f"{q}_median"] = x.median()
            row[f"{q}_q10"] = x.quantile(0.1) if len(x) else np.nan
            row[f"{q}_q90"] = x.quantile(0.9) if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def overall_summary(errors: pd.DataFrame, per_dataset: pd.DataFrame) -> pd.DataFrame:
    """LOO overall table per (metric, n_train, quantity) separating the two sources of variation.

    * ``mean_of_median``: mean over target datasets of the per-dataset median over repetitions.
    * ``dataset_q10/q90/std``: spread ACROSS DATASETS of those per-dataset medians.
    * ``sample_q10_mean/q90_mean/width``: spread ACROSS FITTING SAMPLES, the mean over datasets of each dataset's
      q10 / q90 / (q90 - q10) over repetitions.
    * ``mean_all_rows``: mean over all (dataset, repetition) rows (for ``bias``: the signed error against the whole
      benchmark; for ``law_beats_raw``: the share of samples where the law beats the raw frontier).
    """
    rows = []
    for (metric, n), g in per_dataset.groupby(["metric", "n_train"], sort=True):
        all_rows = errors[(errors["metric"] == metric) & (errors["n_train"] == n)]
        for q in LOO_QUANTITIES:
            med, lo, hi = g[f"{q}_median"], g[f"{q}_q10"], g[f"{q}_q90"]
            rows.append({
                "metric": metric, "n_train": n, "quantity": q, "n_datasets": len(g),
                "mean_of_median": med.mean(), "dataset_q10": med.quantile(0.1), "dataset_q90": med.quantile(0.9),
                "dataset_std": med.std(), "sample_q10_mean": lo.mean(), "sample_q90_mean": hi.mean(),
                "sample_width": (hi - lo).mean(), "mean_all_rows": all_rows[q].mean(),
            })
    return pd.DataFrame(rows)


def summarize(
    errors: pd.DataFrame, curves: pd.DataFrame, config: ExperimentConfig, fits: pd.DataFrame | None = None,
    pool_size: int | None = None,
):
    """Summary tables of the ``loo`` mode (the ``transfer`` mode dispatches to :func:`summarize_transfer`).

    ``error_summary`` (pooled over targets and repetitions), ``stability``, ``stability_summary``, ``params_summary``,
    ``per_dataset_summary``, ``overall_summary`` and ``threshold`` (on the overall mean-over-datasets median MAE).
    Fitting sets that are necessarily identical across repetitions (n equal to the pool size) are counted once.
    """
    if config.mode == "transfer":
        return summarize_transfer(errors, curves, config, fits, pool_size)
    pool_size = errors["held_out"].nunique() - 1
    errors = dedupe_full_pool(errors, pool_size)
    if fits is not None:
        fits = dedupe_full_pool(fits, pool_size)
    out = {"error_summary": _long_summary(errors, ["metric", "n_train"], QUANTITIES)}

    stability = curves.groupby(["metric", "n_train", "held_out", "flops"])["predicted"].agg(
        mean="mean", std="std", median="median",
        q10=lambda x: x.quantile(0.1), q90=lambda x: x.quantile(0.9),
    ).reset_index()
    out["stability"] = stability
    out["stability_summary"] = stability.groupby(["metric", "n_train"])["std"].mean().rename("mean_std").reset_index()

    if fits is not None:
        pf = fits.copy()
        pf["n_train"] = pf["n_train"].astype(str)
        order = {str(n): i for i, n in enumerate(config.n_train)}
        params = _long_summary(pf, ["metric", "n_train"], PARAM_QUANTITIES)
        params["_o"] = params["n_train"].map(lambda n: order.get(n, 10**6))
        params = params.sort_values(["metric", "_o", "quantity"]).drop(columns="_o")
        out["params_summary"] = params.reset_index(drop=True)

    out["per_dataset_summary"] = per_dataset_summary(errors)
    out["overall_summary"] = overall_summary(errors, out["per_dataset_summary"])
    mae = out["overall_summary"].query("quantity == 'mae'").set_index(["metric", "n_train"])["mean_of_median"]
    note = "smallest n whose mean-over-datasets median MAE is below the threshold, in this setting only"
    out["threshold"] = _threshold_table(mae, config, note)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Recipe transfer pooled over held-out datasets (mode "transfer")
# ----------------------------------------------------------------------------------------------------------------------
TRANSFER_QUANTITIES = (
    "optimism_mean", "optimism_mae",
    "mae_raw", "rmse_raw", "bias_raw", "mae_law", "rmse_law", "bias_law", "law_fit_residual_mae", "law_beats_raw",
)


def _honest_values(data: pd.DataFrame, fit_sets: dict, metric: str, grid: np.ndarray) -> tuple:
    """Per-target evaluations ``(honest, insample, runs)``, each of shape targets x levels.

    Selection: best run per FLOP level on the mean over the (sorted) fitting set ``fit_sets[d]``.
    ``honest[d, level]``: that run evaluated on the target ``d`` (never in its own selection set);
    ``insample[d, level]``: that run scored on the mean over the same fitting set (the selection value itself);
    ``runs[d, level]``: the label of that run ("" where no run has a selection value).
    """
    maximize = is_maximize(metric)
    flops = data[FLOPS_COLUMN]
    labels = run_labels(data)
    honest, insample, runs = [], [], []
    for target, fit_set in fit_sets.items():
        select = aggregate(data, sorted(fit_set), metric)
        pos = selected_positions(select, flops, maximize, grid)
        score = np.asarray(data[f"real_data/{target}/{metric}"], dtype=float)
        sel = np.asarray(select, dtype=float)
        honest.append(np.where(pos >= 0, score[np.clip(pos, 0, None)], np.nan))
        insample.append(np.where(pos >= 0, sel[np.clip(pos, 0, None)], np.nan))
        runs.append(np.where(pos >= 0, labels[np.clip(pos, 0, None)], ""))
    return np.array(honest), np.array(insample), np.array(runs, dtype=object)


def pool_curves(honest_per_dataset: np.ndarray, insample_per_dataset: np.ndarray, maximize: bool) -> tuple:
    """Pooled ``(honest, insample)`` frontiers over the same selected runs.

    Both are the mean over datasets per level (NaN if any is NaN). The running best over levels is taken on the
    in-sample means only; the honest frontier carries the held-out mean of the level that achieves it, so the
    held-out scores are never used to choose a level.
    """
    insample, honest = carried_running_best(
        np.mean(insample_per_dataset, axis=0), np.mean(honest_per_dataset, axis=0), maximize
    )
    return honest, insample


def _transfer_rep(data: pd.DataFrame, job: tuple, sizes, metrics, grid: np.ndarray) -> tuple:
    """One repetition: for every n and metric the pooled honest curve, its law fit and the per-dataset evaluations."""
    rep, rep_splits = job
    fit_rows, curves, per_dataset = [], [], []
    for metric in metrics:
        maximize = is_maximize(metric)
        for n in sizes:
            fit_sets = {sp.held_out[0]: sp.fit_set(n) for sp in rep_splits}
            values, in_values, runs = _honest_values(data, fit_sets, metric, grid)
            honest, insample = pool_curves(values, in_values, maximize)
            ok = np.isfinite(honest)
            result = fit_law_full(grid[ok], honest[ok], maximize)
            fit_rows.append({**_fit_row((rep, rep, n, metric), (), (), result), "fit_datasets": "", "held_out": ""})
            curves.append(pd.DataFrame({
                "n_train": n, "rep": rep, "metric": metric, "flops": grid,
                "honest": honest, "insample": insample, "law": predict_law(result[0], grid, maximize),
            }))
            targets = list(fit_sets)
            per_dataset.append(pd.DataFrame({
                "n_train": n, "rep": rep, "metric": metric, "held_out": np.repeat(targets, len(grid)),
                "flops": np.tile(grid, len(targets)), "value": values.reshape(-1),
                "insample_value": in_values.reshape(-1), "selected_run": runs.reshape(-1),
            }))
    return fit_rows, curves, per_dataset


def transfer_references(data: pd.DataFrame, datasets: list[str], metrics, grid: np.ndarray) -> tuple:
    """References per metric: ``honest_20`` (select on the other 20 for every target), the in-sample full-benchmark
    frontier (select and evaluate on the mean of all datasets) and the law fitted to each of them."""
    frames, fit_rows = [], []
    for metric in metrics:
        maximize = is_maximize(metric)
        fit_sets = {d: tuple(x for x in datasets if x != d) for d in datasets}
        values, in_values, _ = _honest_values(data, fit_sets, metric, grid)
        honest_20 = pool_curves(values, in_values, maximize)[0]
        agg = aggregate(data, datasets, metric)
        insample = running_best(agg, data[FLOPS_COLUMN], maximize, grid)
        laws = []
        for name, curve in (("honest_20", honest_20), ("insample", insample)):
            ok = np.isfinite(curve)
            result = fit_law_full(grid[ok], curve[ok], maximize)
            fit_rows.append({**_fit_row((-1, -1, name, metric), (), (), result), "fit_datasets": "", "held_out": ""})
            laws.append(predict_law(result[0], grid, maximize))
        frames.append(pd.DataFrame({
            "metric": metric, "flops": grid, "honest_20": honest_20, "law_20": laws[0],
            "full_benchmark_front": insample, "full_benchmark_law": laws[1],
        }))
    return pd.concat(frames, ignore_index=True), fit_rows


def compute_errors_transfer(curves: pd.DataFrame) -> pd.DataFrame:
    """Per (rep, n_train, metric) errors of the honest curve and of the law against ``honest_20`` over the grid.

    Sign convention: ``bias_raw`` > 0 means the honest curve at n is worse than ``honest_20`` (``honest_20 - honest``
    for ROC AUC, ``honest - honest_20`` for NLL). ``bias_law`` is ``law - honest_20`` (pred - obs).
    ``law_fit_residual_mae`` is the in-sample residual of the law against its own honest curve.
    Optimism per level is ``insample - honest`` for ROC AUC and ``honest - insample`` for NLL (positive = the
    in-sample front, selected and scored on the same n datasets, looks better than new data delivers);
    ``optimism_mean`` is its signed mean over levels, ``optimism_mae`` the mean absolute value.
    """
    rows = []
    for (rep, n, metric), g in curves.groupby(["rep", "n_train", "metric"], sort=False):
        honest, law, ref = g["honest"].to_numpy(), g["law"].to_numpy(), g["honest_20"].to_numpy()
        opt_mae, _, opt_bias = _err_stats(g["insample"].to_numpy(), honest)
        mae_raw, rmse_raw, bias_raw = _err_stats(honest, ref)
        mae_law, rmse_law, bias_law = _err_stats(law, ref)
        rows.append({
            "rep": rep, "n_train": n, "metric": metric,
            "optimism_mean": opt_bias if is_maximize(metric) else -opt_bias, "optimism_mae": opt_mae,
            "mae_raw": mae_raw, "rmse_raw": rmse_raw, "bias_raw": -bias_raw if is_maximize(metric) else bias_raw,
            "mae_law": mae_law, "rmse_law": rmse_law, "bias_law": bias_law,
            "law_fit_residual_mae": _err_stats(law, honest)[0],
            "law_beats_raw": float(mae_law < mae_raw) if np.isfinite(mae_law) and np.isfinite(mae_raw) else np.nan,
        })
    return pd.DataFrame(rows)


def compute_references_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Law resolution, noise floor and winner's-curse gap per metric.

    ``law_resolution``: in-sample residual (MAE, RMSE) of the law fitted to the full-benchmark in-sample front
    (select and evaluate on all datasets) against that front. ``noise_floor``: in-sample residual of ``law_20``
    against ``honest_20``.

    ``winners_curse_signed`` = mean(insample - honest_20); ``winners_curse_optimism`` is positive when the in-sample
    frontier is better than ``honest_20`` (the usual case).
    """
    rows = []
    for metric, group in curves.groupby("metric", sort=False):
        g = group.drop_duplicates("flops")
        ref, law20, ins = g["honest_20"].to_numpy(), g["law_20"].to_numpy(), g["full_benchmark_front"].to_numpy()
        noise_mae, noise_rmse, _ = _err_stats(law20, ref)
        gap_mae, gap_rmse, gap_signed = _err_stats(ins, ref)
        res_mae, res_rmse, _ = _err_stats(g["full_benchmark_law"].to_numpy(), ins)
        rows.append({
            "metric": metric, "law_resolution_mae": res_mae, "law_resolution_rmse": res_rmse,
            "noise_floor_mae": noise_mae, "noise_floor_rmse": noise_rmse,
            "winners_curse_mae": gap_mae, "winners_curse_rmse": gap_rmse, "winners_curse_signed": gap_signed,
            "winners_curse_optimism": gap_signed if is_maximize(metric) else -gap_signed,
        })
    return pd.DataFrame(rows)


def run_transfer(data, splits, ctx: tuple, config, grid) -> dict[str, pd.DataFrame]:
    """Recipe-transfer experiment (parallel over repetitions); see the module docstring for the design."""
    datasets, sizes = ctx
    n_reps = config.n_reps
    # make_splits_loo is ordered by (target, repetition): regroup by repetition
    by_rep = [[splits[i * n_reps + r] for i in range(len(datasets))] for r in range(n_reps)]
    outputs = Parallel(n_jobs=config.n_jobs)(
        delayed(_transfer_rep)(data, job, sizes, config.metrics, grid) for job in enumerate(by_rep)
    )
    fit_rows = [row for rows, _, _ in outputs for row in rows]
    curves = pd.concat([c for _, cs, _ in outputs for c in cs], ignore_index=True)
    per_dataset = pd.concat([c for _, _, cs in outputs for c in cs], ignore_index=True)
    refs, ref_fit_rows = transfer_references(data, datasets, config.metrics, grid)
    curves = curves.merge(refs, on=["metric", "flops"], how="left")
    splits_df = pd.DataFrame([
        {"rep": r, "seed": sp.seed, "held_out": _joined(sp.held_out), "pool_order": _joined(sp.pool)}
        for r, rep_splits in enumerate(by_rep) for sp in rep_splits
    ])
    return {
        "splits": splits_df, "fits": pd.DataFrame(fit_rows + ref_fit_rows), "curves": curves,
        "per_dataset_values": per_dataset, "errors": compute_errors_transfer(curves),
        "references": compute_references_table(curves),
    }


def summarize_transfer(errors, curves, config, fits, pool_size: int) -> dict[str, pd.DataFrame]:
    """Summaries across repetitions (n equal to the pool size is one distinct sample and counted once), thresholds."""
    errors = dedupe_full_pool(errors, pool_size)
    out = {"summary": _long_summary(errors, ["metric", "n_train"], TRANSFER_QUANTITIES)}
    stability = curves.groupby(["metric", "n_train", "flops"])["law"].agg(
        mean="mean", std="std", median="median", q10=lambda x: x.quantile(0.1), q90=lambda x: x.quantile(0.9),
    ).reset_index()
    out["stability"] = stability
    spreads = curves.groupby(["metric", "n_train", "flops"]).agg(
        mean_std=("law", "std"), honest_spread=("honest", "std"), insample_spread=("insample", "std"),
    )
    out["stability_summary"] = spreads.groupby(["metric", "n_train"]).mean().reset_index()
    as_int = fits["n_train"].map(lambda v: int(v) if str(v).isdigit() else v)
    pf = dedupe_full_pool(fits.assign(n_train=as_int), pool_size)
    pf = pf.assign(n_train=pf["n_train"].astype(str))
    order = {str(n): i for i, n in enumerate(config.n_train)}
    params = _long_summary(pf, ["metric", "n_train"], PARAM_QUANTITIES)
    params["_o"] = params["n_train"].map(lambda n: order.get(n, 10**6))
    out["params_summary"] = params.sort_values(["metric", "_o", "quantity"]).drop(columns="_o").reset_index(drop=True)
    tables = []
    for basis, quantity in (("law", "mae_law"), ("raw", "mae_raw")):
        med = out["summary"].query("quantity == @quantity").set_index(["metric", "n_train"])["median"]
        note = f"smallest n whose median MAE of the {basis} curve vs honest_20 is below the threshold"
        table = _threshold_table(med, config, note)
        tables.append(table.assign(basis=basis))
    refs = compute_references_table(curves)
    resolution = dict(zip(refs["metric"], refs["law_resolution_mae"]))
    med = out["summary"].query("quantity == 'optimism_mean'").set_index(["metric", "n_train"])["median"]
    note = "smallest n whose median optimism of the in-sample front is below the law resolution (MAE, threshold column)"
    optimism_config = replace(config, error_threshold=resolution)
    tables.append(_threshold_table(med, optimism_config, note).assign(basis="optimism_vs_law_resolution"))
    out["threshold"] = pd.concat(tables, ignore_index=True)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Per-study view (mode "transfer")
# ----------------------------------------------------------------------------------------------------------------------
STUDY_CHUNK = 60
STUDY_THRESHOLD_BASIS = "optimism_study_vs_residual_study"
POOLED_THRESHOLD_BASIS = "optimism_pooled_vs_residual_study"


def _study_rows(chunk: list) -> list[dict]:
    """Optimism and law residual of each study ``(metric, n, held_out, rep, flops, honest_values, insample_values)``."""
    rows = []
    for metric, n, target, rep, flops, honest_values, insample_values in chunk:
        maximize = is_maximize(metric)
        insample, honest = carried_running_best(insample_values, honest_values, maximize)
        opt = insample - honest if maximize else honest - insample
        ok = np.isfinite(insample)
        params, _, n_points = fit_law_full(flops[ok], insample[ok], maximize)
        residual = _err_stats(predict_law(params, flops, maximize), insample)[0]
        rows.append({
            "metric": metric, "n_train": n, "held_out": target, "rep": rep,
            "optimism_study": float(np.nanmean(opt)) if np.isfinite(opt).any() else np.nan,
            "residual_study": residual, "n_frontier_points": n_points, "l0": params[0], "a": params[1], "b": params[2],
        })
    return rows


def compute_per_study(per_dataset_values: pd.DataFrame, pool_size: int, n_jobs: int = -1) -> pd.DataFrame:
    """Per study (held-out dataset, draw, n) optimism and law residual, NOT pooled over the targets.

    The study's in-sample front is the running best over budgets of ``insample_value`` for that (d, rep, n) alone,
    its honest curve the ``value`` on d of the selected run at the budget that achieves that running best (no running
    best over the held-out values themselves). ``optimism_study`` is the mean over
    budgets of ``insample - honest`` (ROC AUC) or ``honest - insample`` (NLL). ``residual_study`` is the MAE over
    budgets of the law fitted (same recipe) to the single in-sample front against that front. Studies with n equal
    to the pool size are identical across draws and counted once per target.
    """
    frame = per_dataset_values.copy()
    frame["n_train"] = pd.to_numeric(frame["n_train"])
    full = frame["n_train"] >= pool_size
    first = frame.loc[full, "rep"].min() if full.any() else None
    frame = frame[~full | (frame["rep"] == first)]
    studies = []
    for (metric, n, target, rep), group in frame.groupby(["metric", "n_train", "held_out", "rep"], sort=True):
        g = group.sort_values("flops")
        studies.append((metric, int(n), target, int(rep), g["flops"].to_numpy(float),
                        g["value"].to_numpy(float), g["insample_value"].to_numpy(float)))
    chunks = [studies[i:i + STUDY_CHUNK] for i in range(0, len(studies), STUDY_CHUNK)]
    outputs = Parallel(n_jobs=n_jobs)(delayed(_study_rows)(c) for c in chunks)
    return pd.DataFrame([row for rows in outputs for row in rows])


def study_threshold_table(summary: pd.DataFrame, config: ExperimentConfig, pooled_optimism: pd.Series) -> pd.DataFrame:
    """Smallest n whose median optimism (per-study, and pooled) is below the median per-study residual at that n."""
    rows = []
    for metric in config.metrics:
        opt = summary[(summary["metric"] == metric) & (summary["quantity"] == "optimism_study")]
        res = summary[(summary["metric"] == metric) & (summary["quantity"] == "residual_study")]
        res_med = res.set_index("n_train")["median"].rename("res")
        series = {
            STUDY_THRESHOLD_BASIS: opt.set_index("n_train")["median"],
            POOLED_THRESHOLD_BASIS: pooled_optimism.loc[metric],
        }
        for basis, optimism in series.items():
            both = optimism.rename("opt").to_frame().join(res_med)
            hit = both[both["opt"] < both["res"]].sort_index()
            rows.append({
                "metric": metric, "threshold": hit["res"].iloc[0] if len(hit) else np.nan,
                "smallest_n_below_threshold": hit.index[0] if len(hit) else np.nan,
                "median_mae_at_that_n": hit["opt"].iloc[0] if len(hit) else np.nan,
                "note": "smallest n whose median optimism is below the median per-study residual (threshold)",
                "basis": basis,
            })
    return pd.DataFrame(rows)


def add_per_study(
    results: dict[str, pd.DataFrame], config: ExperimentConfig, per_study: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Add ``per_study``, ``per_study_summary`` and the per-study threshold row to a ``transfer`` result dict."""
    pool_size = results["splits"]["held_out"].nunique() - 1
    if per_study is None:
        per_study = compute_per_study(results["per_dataset_values"], pool_size, config.n_jobs)
    summary = _long_summary(per_study, ["metric", "n_train"], ("optimism_study", "residual_study"))
    threshold = results["threshold"]
    pooled = results["summary"].query("quantity == 'optimism_mean'").set_index(["metric", "n_train"])["median"]
    threshold = threshold[~threshold["basis"].isin([STUDY_THRESHOLD_BASIS, POOLED_THRESHOLD_BASIS])]
    results.update({
        "per_study": per_study, "per_study_summary": summary,
        "threshold": pd.concat([threshold, study_threshold_table(summary, config, pooled)], ignore_index=True),
    })
    return results


# ----------------------------------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------------------------------
def _file_names() -> dict[str, str]:
    return {
        "splits": "splits.csv", "fits": "fits.csv", "curves": "curves.csv", "errors": "errors.csv",
        "error_summary": "error_summary.csv", "stability": "stability.csv",
        "stability_summary": "stability_summary.csv",
        "params_summary": "params_summary.csv", "threshold": "threshold.csv",
        "per_dataset_summary": "per_dataset_summary.csv", "overall_summary": "overall_summary.csv",
        "per_dataset_values": "per_dataset_values.csv", "summary": "summary.csv", "references": "references.csv",
        "per_study": "per_study.csv", "per_study_summary": "per_study_summary.csv",
    }


def save_results(results: dict[str, pd.DataFrame], out_dir, config: ExperimentConfig | None = None) -> None:
    """Write every table as CSV (and ``config.json`` last, so it marks a complete run)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _file_names()
    for key, frame in results.items():
        frame.to_csv(out_dir / names[key], index=False)
    if config is not None:
        (out_dir / "config.json").write_text(json.dumps(config.result_key(), indent=2))


def load_results(out_dir) -> dict[str, pd.DataFrame]:
    """Read back the tables written by :func:`save_results`."""
    out_dir = Path(out_dir)
    results = {}
    for key, name in _file_names().items():
        if (out_dir / name).exists():
            frame = pd.read_csv(out_dir / name)
            if "n_train" in frame.columns:
                frame["n_train"] = frame["n_train"].map(lambda v: int(v) if str(v).isdigit() else v)
            results[key] = frame
    return results


def cache_matches(config: ExperimentConfig) -> bool:
    path = Path(config.out_dir) / "config.json"
    if not path.exists():
        return False
    stored = json.loads(path.read_text())
    for obsolete in ("data_path", "held_out_size"):  # fields of older runs that no longer exist or are cosmetic
        stored.pop(obsolete, None)
    return stored == config.result_key() and (Path(config.out_dir) / "curves.csv").exists()


def run_cached(config: ExperimentConfig, df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Run the experiment and save the tables, or load them if ``out_dir/config.json`` matches and not ``force``."""
    if not config.force and cache_matches(config):
        print(f"Loading cached results from {config.out_dir}")
        results = load_results(config.out_dir)
        if config.mode == "transfer" and "per_study" not in results:  # table added after the cache was written
            start = time.time()
            add_per_study(results, config)
            save_results({k: results[k] for k in ("per_study", "per_study_summary", "threshold")}, config.out_dir)
            print(f"Per-study table computed in {time.time() - start:.1f} s")
        return results
    start = time.time()
    if df is None:
        df = load_data(config.data_path, config.flops_min, config.flops_max)
    results = run_experiment(df, config)
    pool_size = results["splits"]["held_out"].nunique() - 1
    results.update(summarize(results["errors"], results["curves"], config, results["fits"], pool_size))
    if config.mode == "transfer":
        add_per_study(results, config)
    save_results(results, config.out_dir, config)
    print(f"Experiment finished in {time.time() - start:.1f} s, saved to {config.out_dir}")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Dataset sample efficiency of scaling laws (zero-shot).")
    parser.add_argument("--mode", choices=MODES, default=MODES[0], help="transfer (default, headline) or loo")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out_dir", default=None, help="default depends on --mode")
    parser.add_argument("--n_reps", type=int, default=None, help="default 20")
    parser.add_argument("--n_train", type=int, nargs="+", default=None)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", default=list(ExperimentConfig().metrics))
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    given = {
        "data_path": args.data, "out_dir": args.out_dir, "n_reps": args.n_reps,
        "n_train": tuple(args.n_train) if args.n_train else None,
    }
    given = {k: v for k, v in given.items() if v is not None}
    common = {"base_seed": args.base_seed, "metrics": tuple(args.metrics), "n_jobs": args.n_jobs, "force": args.force}
    config = (ExperimentConfig.loo if args.mode == "loo" else ExperimentConfig.transfer)(**given, **common)
    results = run_cached(config)
    if config.mode == "transfer":
        table = results["summary"].query("quantity in ['mae_raw', 'mae_law']")
        table = table[["metric", "n_train", "quantity", "median", "q10", "q90"]]
    else:
        table = results["overall_summary"].query("quantity == 'mae'")
        cols = ["metric", "n_train", "n_datasets", "mean_of_median", "dataset_q10", "dataset_q90", "sample_width"]
        table = table[cols]
    print(table.to_string(index=False))
    print(results["threshold"].drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
