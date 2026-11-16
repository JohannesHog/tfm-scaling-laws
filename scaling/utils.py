"""Functional forms and fits.

Functional forms take ``(data, params)``; ``functional_form_L0`` predicts in log-loss space, the sigmoid forms in
ROC AUC space. ``fit_parametric_form`` minimises a Huber loss with L-BFGS-B from a grid of starts.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logsumexp

# ----------------------------------------------------------------------------------------------------------------------
# Functional forms
# ----------------------------------------------------------------------------------------------------------------------
N_PARAMS_L0 = 3
N_PARAMS_SIGMOID = 4
N_PARAMS_DOUBLE_SIGMOID = 7


def functional_form_L0(  # noqa: N802 - name used throughout the paper code
    data: list | np.ndarray,
    params: list[float],
    return_log_loss: bool = True,
) -> np.ndarray:
    """`L = L0 + A * C**b` (paper Eq. 1, b = -alpha) with params `[log L0, log A, b]`, computed as a logsumexp.

    `data` are raw C values (FLOPs). Returns log(L), or L if `return_log_loss` is False.
    """
    if len(params) != N_PARAMS_L0:
        raise ValueError(f"Expected {N_PARAMS_L0} parameters for functional form, got {len(params)}.")
    if len(data.shape) != 1:
        raise ValueError(f"Expected data with 1 column (C), got shape {data.shape}.")

    l0, a, b = params
    exponents = np.stack([np.full(data.shape[0], l0), a + b * np.log(data)], axis=-1)
    loss = logsumexp(exponents, axis=-1)
    if not return_log_loss:
        loss = np.exp(loss)
    return loss


def sigmoid_functional_form(
    data: list | np.ndarray,
    params: list[float],
    return_log_loss: bool = True,  # unused, kept for a common signature
) -> np.ndarray:
    """One logistic step in log(C): `a_min + (a_max - a_min) / (1 + exp(-k (log C - log_x0)))`.

    params `[a_max, log_x0, k, a_min]`.
    """
    if len(params) != N_PARAMS_SIGMOID:
        raise ValueError(f"Expected {N_PARAMS_SIGMOID} parameters for functional form, got {len(params)}.")
    a_max, log_x0, k, a_min = params

    return a_min + (a_max - a_min) / (1 + np.exp(-k * (np.log(data) - log_x0)))


def double_sigmoid_functional_form(
    data: list | np.ndarray,
    params: list[float],
    return_log_loss: bool = True,  # unused, kept for a common signature
) -> np.ndarray:
    """Two logistic steps in log10(C): `a_min + h1 expit(k1 (log10 C - x1)) + h2 expit(k2 (log10 C - x2))`.

    params `[a_min, h1, x1, k1, h2, x2, k2]`; the ceiling is `a_min + h1 + h2`.
    """
    if len(params) != N_PARAMS_DOUBLE_SIGMOID:
        raise ValueError(f"Expected {N_PARAMS_DOUBLE_SIGMOID} parameters for functional form, got {len(params)}.")
    a_min, h1, x1, k1, h2, x2, k2 = params
    log10_c = np.log10(np.atleast_1d(np.asarray(data, dtype=float)))

    return a_min + h1 * expit(k1 * (log10_c - x1)) + h2 * expit(k2 * (log10_c - x2))


# Candidate forms for the wide-prior ROC AUC frontier (Appendix C); each maps raw FLOPs to ROC AUC. The
# `return_log_loss` argument is unused and kept for the common signature.
LOG10 = np.log(10)


def _x(c) -> np.ndarray:
    return np.log10(np.atleast_1d(np.asarray(c, dtype=float)))


def logistic(c, p, return_log_loss=True):
    a_max, x0, k, a_min = p
    return a_min + (a_max - a_min) * expit(k * (_x(c) - x0))


def gap_powerlaw_deficit(c, p, return_log_loss=True):
    log_l0, log_b, alpha, h, x0, k = p
    x = _x(c)
    return 1 - np.exp(log_l0) - np.exp(log_b - alpha * LOG10 * (x - 13)) - h * expit(-k * (x - x0))


# ----------------------------------------------------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------------------------------------------------
def huber_loss(
    params: list | np.ndarray,
    data: list | np.ndarray,
    y_true: list | np.ndarray,
    func_form: Callable,
    delta: float = 1e-3,
) -> float:
    """Summed Huber loss of `func_form(data, params)` against `y_true`; `params` first so scipy can minimise it."""
    residuals = np.array(y_true) - np.array(func_form(data, params))
    return np.sum(scipy.special.huber(delta, residuals))


def fit_parametric_form(  # noqa: PLR0913, PLR0917
    func_form: Callable,
    x_data: list | np.ndarray,
    y_data: list | np.ndarray,
    initial_grid: list | np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    delta: float = 1e-3,
) -> tuple[np.ndarray | None, float]:
    """Minimise the Huber loss of `func_form` with L-BFGS-B from every start in `initial_grid`; keep the best.

    `y_data` must already be in the space `func_form` predicts in. Returns `(params, loss)`, `(None, inf)` if no
    start gave a finite loss.
    """
    best_params = None
    best_loss = np.inf
    loss = partial(huber_loss, func_form=func_form, delta=delta)

    for init in initial_grid:
        result = scipy.optimize.minimize(fun=loss, x0=init, args=(x_data, y_data), method="L-BFGS-B", bounds=bounds)
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x

    return best_params, best_loss


def fit_linear_model(
    x_data: np.ndarray | list | pd.Series,
    y_data: np.ndarray | list | pd.Series,
) -> tuple[float, float, float, float, float]:
    """`scipy.stats.linregress` on log-transformed data; returns (slope, intercept, r_value, p_value, std_err)."""
    if isinstance(x_data, pd.Series):
        x_data = x_data.values
    if isinstance(y_data, pd.Series):
        y_data = y_data.values

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_data, y_data)

    return slope, intercept, r_value, p_value, std_err
