"""Tests for scaling.dataset_sample_efficiency on a small synthetic benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scaling import dataset_sample_efficiency as dse

DATASETS = [f"ds{i}" for i in range(6)]
LEVELS = [1e12, 3e12, 1e13, 3e13, 1e14]
METRICS = ("roc_auc", "nll")
N_REAL_DATASETS = 21
N_TRANSFER_REPS = 3
REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "nanotabpfn_model_tabiclv1_prior.csv"


def make_frame(seed=0, runs=8) -> pd.DataFrame:
    """Runs x FLOP levels; NLL follows a power law in FLOPs plus run and dataset noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for level in LEVELS:
        for run in range(runs):
            row = {dse.FLOPS_COLUMN: level}
            for i, d in enumerate(DATASETS):
                nll = 0.3 + (0.5 + 0.1 * i) * (level / 1e12) ** -0.25 + 0.05 * rng.random()
                row[f"real_data/{d}/nll"] = nll
                row[f"real_data/{d}/roc_auc"] = 1 - 0.1 * nll
            rows.append(row)
    return pd.DataFrame(rows)


def make_config(**kwargs) -> dse.ExperimentConfig:
    """Leave-one-dataset-out config on the 6-dataset fixture (pool size 5)."""
    base = {"n_reps": 2, "n_train": (1, 2, 4), "n_jobs": 1, "metrics": METRICS}
    base.update(kwargs)
    return dse.ExperimentConfig.loo(**base)


def test_oversized_fit_sets_dropped():
    with pytest.warns(UserWarning, match="dropping"):
        assert dse.valid_sizes([1, 4, 5, 9], pool_size=4) == [1, 4]


def test_aggregate_is_mean_of_selected_datasets():
    df = make_frame()
    chosen = ["ds1", "ds4", "ds5"]
    expected = np.mean([df[f"real_data/{d}/nll"].to_numpy() for d in chosen], axis=0)
    np.testing.assert_allclose(dse.aggregate(df, chosen, "nll").to_numpy(), expected)


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real data not available")
def test_aggregate_matches_real_data_columns():
    df = pd.read_csv(REAL_CSV, index_col=[0, 1]).sample(500, random_state=0)
    names = dse.real_datasets(df)
    assert len(names) == N_REAL_DATASETS
    for metric in METRICS:
        np.testing.assert_allclose(dse.aggregate(df, names, metric), df[f"real_data/{metric}"], atol=1e-12)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_no_leakage_from_held_out_dataset():
    """Perturbing the held-out dataset's columns leaves the fits and predictions for that target bit-identical."""
    df = make_frame()
    config = make_config(n_reps=2)
    base = dse.run_experiment(df, config)
    target = DATASETS[2]
    perturbed_df = df.copy()
    for m in METRICS:
        perturbed_df[f"real_data/{target}/{m}"] += 100.0
    perturbed = dse.run_experiment(perturbed_df, config)

    cols = ["rep", "n_train", "metric", "l0", "a", "b", "huber_loss", "n_frontier_points"]

    def for_target(results, table):
        frame = results[table]
        return frame[frame["held_out"] == target]

    pd.testing.assert_frame_equal(
        for_target(base, "fits").query("n_train != 'all'")[cols],
        for_target(perturbed, "fits").query("n_train != 'all'")[cols],
        check_exact=True,
    )
    for column in ("predicted", "raw_frontier"):
        assert np.array_equal(for_target(base, "curves")[column], for_target(perturbed, "curves")[column])
    assert not np.array_equal(
        for_target(base, "curves")["observed_heldout"], for_target(perturbed, "curves")["observed_heldout"]
    )


@pytest.mark.parametrize("maximize", [False, True])
def test_running_best_envelope(maximize):
    flops = pd.Series([1, 1, 2, 2, 3, 3, 4], dtype=float)
    values = pd.Series([5, 3, 4, 6, 2, 7, 1], dtype=float)
    out = dse.running_best(values, flops, maximize, [1, 2, 3, 4])
    expected = [5, 6, 7, 7] if maximize else [3, 3, 2, 1]
    np.testing.assert_array_equal(out, expected)
    assert np.all(np.diff(out) >= 0) if maximize else np.all(np.diff(out) <= 0)


def test_frontier_points_strictly_improving():
    flops = pd.Series([1, 1, 2, 3, 3, 4], dtype=float)
    values = pd.Series([5, 3, 4, 2, 6, 2], dtype=float)
    pareto = dse.frontier_points(values, flops, maximize=False)
    assert pareto["flops"].tolist() == [1, 3]
    assert pareto["value"].tolist() == [3, 2]


def test_error_metrics_hand_computed():
    pred = [1.0, 2.0, 4.0]
    obs = [1.5, 1.0, 4.0]
    curves = pd.DataFrame({
        "rep": 0, "n_train": 1, "metric": "nll", "flops": [1.0, 2.0, 3.0],
        "predicted": pred, "observed_heldout": obs, "observed_transfer": [2.0, 1.5, 4.5],
        "raw_frontier": [1.0, 1.0, 1.0],
    })
    err = dse.compute_errors(curves).iloc[0]
    e = np.array(pred) - np.array(obs)
    assert err["mae"] == pytest.approx(np.mean(np.abs(e)))
    assert err["rmse"] == pytest.approx(np.sqrt(np.mean(e**2)))
    assert err["bias"] == pytest.approx(np.mean(e))
    assert err["mae_raw_frontier"] == pytest.approx(np.mean(np.abs(np.array([1.0, 1.0, 1.0]) - obs)))
    assert err["selection_gap"] == pytest.approx(np.mean(np.array([2.0, 1.5, 4.5]) - obs))
    assert err["mae_minus_full_pool"] == pytest.approx(0.0)


def test_missing_values_complete_case():
    df = make_frame()
    df.loc[3, "real_data/ds0/nll"] = np.nan
    assert np.isnan(dse.aggregate(df, ["ds0", "ds1"], "nll")[3])
    assert not np.isnan(dse.aggregate(df, ["ds1", "ds2"], "nll")[3])
    # a level whose rows are all NaN for a member dataset has no frontier / running-best value
    level = LEVELS[2]
    mask = df[dse.FLOPS_COLUMN] == level
    df.loc[mask, "real_data/ds0/nll"] = np.nan
    agg = dse.aggregate(df, ["ds0", "ds1"], "nll")
    best = dse.running_best(agg, df[dse.FLOPS_COLUMN], False, LEVELS)
    assert np.isnan(best[2]) and np.isfinite(best[[0, 1]]).all()
    assert level not in dse.frontier_points(agg, df[dse.FLOPS_COLUMN], False)["flops"].tolist()


@pytest.mark.parametrize("maximize", [False, True])
def test_fit_predict_recovers_exact_power_law(maximize):
    levels = [1e12 * 10 ** (i / 2) for i in range(12)]
    flops = np.array(levels)
    true = np.array([np.log(0.2), np.log(2.0), -0.3])
    loss = np.exp(true[0]) + np.exp(true[1]) * (flops / 1.0) ** true[2]
    values = 1 - loss if maximize else loss
    params = dse.fit_law(flops, values, maximize)
    pred = dse.predict_law(params, levels, maximize)
    np.testing.assert_allclose(pred, values, rtol=1e-3, atol=1e-4)


def test_loo_splits_every_dataset_held_out_n_reps_times():
    n_reps = 3
    splits = dse.make_splits_loo(DATASETS, n_reps, [1, 2, 5], base_seed=7)
    assert len(splits) == n_reps * len(DATASETS)
    counts = pd.Series([s.held_out[0] for s in splits]).value_counts()
    assert set(counts.index) == set(DATASETS) and (counts == n_reps).all()
    for s in splits:
        assert len(s.held_out) == 1 and s.held_out[0] not in s.pool
        assert sorted(s.pool + s.held_out) == sorted(DATASETS)
        for n in (1, 2, 5):
            assert s.held_out[0] not in s.fit_set(n)
        assert set(s.fit_set(1)) <= set(s.fit_set(2)) <= set(s.fit_set(5))
    assert splits == dse.make_splits_loo(DATASETS, n_reps, [1, 2, 5], base_seed=7)
    assert splits != dse.make_splits_loo(DATASETS, n_reps, [1, 2, 5], base_seed=8)
    # seed rule: base_seed + 1000 * index of the held-out dataset + repetition
    assert splits[n_reps].seed == 7 + 1000 * 1 + 0


def test_loo_experiment_has_held_out_column_for_every_dataset():
    config = dse.ExperimentConfig.loo(n_reps=2, n_train=(1, 2, 5), n_jobs=1, metrics=METRICS)
    results = dse.run_experiment(make_frame(), config)
    for table in ("curves", "errors"):
        assert set(results[table]["held_out"]) == set(DATASETS)
    summary = dse.summarize(results["errors"], results["curves"], config, results["fits"])
    per_dataset = summary["per_dataset_summary"]
    assert set(per_dataset["held_out"]) == set(DATASETS)
    full = per_dataset[per_dataset["n_train"] == len(DATASETS) - 1]
    assert (full["n_samples"] == 1).all()  # identical full-pool fitting sets are counted once


# ----------------------------------------------------------------------------------------------------------------------
# recipe transfer mode
# ----------------------------------------------------------------------------------------------------------------------
def transfer_config(**kwargs) -> dse.ExperimentConfig:
    base = {"n_reps": 1, "n_train": (1, 2, 5), "n_jobs": 1, "metrics": METRICS}
    base.update(kwargs)
    return dse.ExperimentConfig.transfer(**base)


def test_transfer_identical_datasets_honest_equals_insample_frontier():
    df = make_frame()
    for m in METRICS:
        for d in DATASETS:
            df[f"real_data/{d}/{m}"] = df[f"real_data/ds0/{m}"]
    results = dse.run_experiment(df, transfer_config(n_reps=2))
    curves = results["curves"]
    assert len(curves) > 0
    np.testing.assert_allclose(curves["honest"], curves["full_benchmark_front"], rtol=1e-12)
    np.testing.assert_allclose(curves["honest_20"], curves["full_benchmark_front"], rtol=1e-12)
    refs = results["references"]
    assert np.allclose(refs["winners_curse_mae"], 0.0, atol=1e-12)


def test_transfer_held_out_never_in_own_selection_set():
    results = dse.run_experiment(make_frame(), transfer_config(n_reps=N_TRANSFER_REPS))
    splits = results["splits"]
    assert set(splits["held_out"]) == set(DATASETS)
    assert (splits.groupby("held_out")["rep"].nunique() == N_TRANSFER_REPS).all()
    for _, row in splits.iterrows():
        pool = row["pool_order"].split(";")
        assert row["held_out"] not in pool
        for n in (1, 2, 5):
            assert row["held_out"] not in pool[:n]


def test_transfer_pooled_curve_matches_hand_computed_mean():
    names = ["a", "b", "c"]
    values = {  # (run, level) -> per-dataset nll
        (0, 0): [0.9, 0.8, 0.7], (1, 0): [0.6, 0.95, 0.5],
        (0, 1): [0.5, 0.6, 0.65], (1, 1): [0.7, 0.4, 0.55],
    }
    rows = []
    for (run, level), vals in values.items():
        row = {dse.FLOPS_COLUMN: [1e12, 1e13][level]}
        for d, v in zip(names, vals):
            row[f"real_data/{d}/nll"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    config = transfer_config(n_train=(1, 2), metrics=("nll",))
    results = dse.run_experiment(df, config)
    splits = results["splits"]
    for n in (1, 2):
        per_target, per_target_insample = [], []
        for _, row in splits.iterrows():
            fit_set = row["pool_order"].split(";")[:n]
            level_vals, level_ins = [], []
            for level in (0, 1):
                sel = {run: np.mean([values[(run, level)][names.index(f)] for f in fit_set]) for run in (0, 1)}
                best = min(sel, key=sel.get)
                level_vals.append(values[(best, level)][names.index(row["held_out"])])
                level_ins.append(sel[best])
            per_target.append(level_vals)
            per_target_insample.append(level_ins)
        honest_means, insample_means = np.mean(per_target, axis=0), np.mean(per_target_insample, axis=0)
        # the running best is taken on the in-sample means; the honest curve is read at the level that achieves it
        achieving = [int(np.argmin(insample_means[: k + 1])) for k in range(len(insample_means))]
        expected_honest = honest_means[achieving]
        expected_insample = np.minimum.accumulate(insample_means)
        sub = results["curves"].query("n_train == @n and metric == 'nll'").sort_values("flops")
        np.testing.assert_allclose(sub["honest"].to_numpy(), expected_honest, rtol=1e-12)
        np.testing.assert_allclose(sub["insample"].to_numpy(), expected_insample, rtol=1e-12)


def test_transfer_error_sign_convention_positive_when_worse():
    flops = [1e12, 1e13, 1e14]
    rows = []
    for metric, ref, shift in (("roc_auc", [0.7, 0.8, 0.85], -0.05), ("nll", [0.6, 0.5, 0.4], 0.05)):
        rows.append(pd.DataFrame({
            "rep": 0, "n_train": 3, "metric": metric, "flops": flops, "honest_20": ref,
            "honest": np.array(ref) + shift, "insample": np.array(ref) + shift, "law": np.array(ref) + shift,
        }))
    errors = dse.compute_errors_transfer(pd.concat(rows, ignore_index=True))
    assert (errors["bias_raw"] > 0).all()
    assert errors["bias_raw"].to_numpy() == pytest.approx([0.05, 0.05])
    assert errors["mae_raw"].to_numpy() == pytest.approx([0.05, 0.05])
    # the law's bias is pred - obs: negative for ROC AUC (pred lower), positive for NLL
    by_metric = errors.set_index("metric")["bias_law"]
    assert by_metric["roc_auc"] < 0 < by_metric["nll"]


def test_transfer_identical_datasets_insample_equals_honest_and_zero_optimism():
    df = make_frame()
    for m in METRICS:
        for d in DATASETS:
            df[f"real_data/{d}/{m}"] = df[f"real_data/ds0/{m}"]
    results = dse.run_experiment(df, transfer_config(n_reps=2))
    np.testing.assert_allclose(results["curves"]["insample"], results["curves"]["honest"], rtol=1e-12)
    assert set(results["curves"]["n_train"]) == {1, 2, 5}
    np.testing.assert_allclose(results["errors"]["optimism_mean"], 0.0, atol=1e-12)
    np.testing.assert_allclose(results["errors"]["optimism_mae"], 0.0, atol=1e-12)


@pytest.mark.parametrize(("metric", "shift"), [("roc_auc", 0.1), ("nll", -0.1)])
def test_optimism_equals_constant_shift_of_selection_set(metric, shift):
    """Selection datasets are shifted by a constant relative to the target: optimism is that constant (as a gain)."""
    rng = np.random.default_rng(1)
    rows = []
    for level in LEVELS:
        for _ in range(6):
            v = 0.5 + 0.2 * rng.random()
            rows.append({dse.FLOPS_COLUMN: level, f"real_data/t/{metric}": v,
                         f"real_data/a/{metric}": v + shift, f"real_data/b/{metric}": v + shift})
    data = pd.DataFrame(rows)
    grid = np.array(LEVELS)
    maximize = dse.is_maximize(metric)
    honest_vals, insample_vals, runs = dse._honest_values(data, {"t": ("a", "b")}, metric, grid)  # noqa: SLF001
    assert runs.shape == honest_vals.shape and (runs != "").all()
    honest, insample = dse.pool_curves(honest_vals, insample_vals, maximize)
    curves = pd.DataFrame({
        "rep": 0, "n_train": 2, "metric": metric, "flops": grid,
        "honest": honest, "insample": insample, "law": np.nan, "honest_20": np.nan,
    })
    errors = dse.compute_errors_transfer(curves).iloc[0]
    assert errors["optimism_mean"] == pytest.approx(abs(shift))
    assert errors["optimism_mae"] == pytest.approx(abs(shift))


def test_per_study_optimism_zero_for_identical_datasets():
    df = make_frame()
    for m in METRICS:
        for d in DATASETS:
            df[f"real_data/{d}/{m}"] = df[f"real_data/ds0/{m}"]
    results = dse.run_experiment(df, transfer_config(n_reps=2))
    pool_size = len(DATASETS) - 1
    per_study = dse.compute_per_study(results["per_dataset_values"], pool_size, n_jobs=1)
    assert set(per_study["n_train"]) == {1, 2, 5}
    np.testing.assert_allclose(per_study["optimism_study"], 0.0, atol=1e-12)
    full = per_study[per_study["n_train"] == pool_size]
    assert len(full) == len(DATASETS) * len(METRICS)  # n equal to the pool size: one study per target
    config = transfer_config(n_reps=2, n_jobs=1)
    results.update(dse.summarize(results["errors"], results["curves"], config, results["fits"], pool_size))
    dse.add_per_study(results, config)
    assert dse.STUDY_THRESHOLD_BASIS in set(results["threshold"]["basis"])


def test_curves_have_no_pooled_insample_law_and_constant_full_benchmark_law():
    """Only the (constant) full-benchmark reference law is stored; no per-n pooled in-sample law."""
    results = dse.run_experiment(make_frame(), transfer_config(n_reps=2))
    curves = results["curves"]
    assert "insample_law" not in curves.columns
    assert "residual_pooled" not in results["errors"].columns
    for _, group in curves.groupby(["metric", "flops"]):
        assert group["full_benchmark_law"].nunique() == 1
        assert group["full_benchmark_front"].nunique() == 1


@pytest.mark.parametrize("maximize", [True, False])
def test_carried_running_best_never_selects_on_scores(maximize):
    """The held-out curve follows the level chosen on the selection scores, not the best held-out score so far."""
    sign = 1 if maximize else -1
    select = sign * np.array([0.50, 0.60, 0.55, np.nan, 0.70])
    score = sign * np.array([0.40, 0.45, 0.80, 0.90, 0.30])
    best, carried = dse.carried_running_best(select, score, maximize)
    np.testing.assert_allclose(best, sign * np.array([0.50, 0.60, 0.60, np.nan, 0.70]))
    # level 2 has the best score so far (0.80) but was not chosen on the selection scores: level 1's score is carried
    np.testing.assert_allclose(carried, sign * np.array([0.40, 0.45, 0.45, np.nan, 0.30]))
    running_on_scores = dse._running(score, maximize)  # noqa: SLF001
    assert not np.allclose(np.nan_to_num(carried), np.nan_to_num(running_on_scores))


def test_transfer_running_best_carries_held_out_score_of_selected_level():
    """LOO transferred curve: select on the fitting-set mean per level, running best on those means, carry held-out."""
    flops = pd.Series([1e12, 1e12, 1e13, 1e13, 1e14, 1e14])
    fit_agg = pd.Series([0.60, 0.50, 0.45, 0.70, 0.48, 0.47])   # NLL: run 1 best at 1e12, run 2 at 1e13, run 5 at 1e14
    held = pd.Series([0.70, 0.65, 0.80, 0.30, 0.20, 0.90])
    grid = np.array([1e12, 1e13, 1e14])
    got = dse.transfer_running_best(fit_agg, held, flops, maximize=False, grid=grid)
    # in-sample running best: 0.50 (1e12) -> 0.45 (1e13) -> 0.45 (1e13 still), so the held-out of run 2 is carried
    np.testing.assert_allclose(got, [0.65, 0.80, 0.80])


def test_per_study_honest_curve_is_carried():
    """Per-study rows use the carried honest curve, so optimism can be computed from a non-monotone held-out series."""
    flops = np.array([1e12, 1e13, 1e14])
    rows = dse._study_rows([("nll", 2, "t", 0, flops, np.array([0.6, 0.7, 0.5]), np.array([0.5, 0.4, 0.45]))])  # noqa: SLF001
    insample = np.array([0.5, 0.4, 0.4])
    honest = np.array([0.6, 0.7, 0.7])  # level 1 achieves the in-sample running best at levels 1 and 2
    assert rows[0]["optimism_study"] == pytest.approx(np.mean(honest - insample))


def test_selected_run_column_names_the_chosen_run():
    df = make_frame()
    df.index = pd.MultiIndex.from_arrays([[f"run{i // len(LEVELS)}" for i in range(len(df))], range(len(df))])
    results = dse.run_experiment(df, transfer_config(n_reps=1, n_train=(1,)))
    per = results["per_dataset_values"]
    assert "selected_run" in per.columns and per["selected_run"].str.startswith("run").all()
    row = per.iloc[0]
    sub = df.loc[df.index.get_level_values(0) == row["selected_run"]]
    sub = sub[sub[dse.FLOPS_COLUMN] == row["flops"]]
    assert sub[f"real_data/{row['held_out']}/{row['metric']}"].iloc[0] == pytest.approx(row["value"])
