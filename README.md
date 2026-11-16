# Towards Reliable Scaling Laws for Tabular Foundation Models

Code for the figures and tables of "Towards Reliable Scaling Laws for Tabular Foundation Models".
Built on [EffingScaling](https://github.com/Neeratyoy/EffingScaling).

## Setup

```bash
pip install -e .             # Python >= 3.10
```

## Data

The only input is `data/nanotabpfn_model_tabiclv1_prior.csv`: the 6,120 training runs of the paper, one row per run
and FLOP grid point $C_k = 10^{11 + k/4}$ (Appendix A.3), published on figshare at
https://doi.org/10.6084/m9.figshare.34912776. `download_data.py` fetches it (96 MB) and checks its MD5:

```bash
python download_data.py
```

```python
df = pd.read_csv("data/nanotabpfn_model_tabiclv1_prior.csv", index_col=[0, 1])   # index: (run, grid_idx)
```

| Column(s) | Meaning |
|---|---|
| `total_flops_rounded` | training compute at the grid point |
| `total_cells` | training cells seen |
| `parameters` | model parameters |
| `config/*` | hyperparameters (Table 1) and nominal budget `config/num_flops_training` |
| `val/val_loss`, `val/small/val_loss`, `val/medium/val_loss` | synthetic validation loss |
| `real_data/{nll,roc_auc,accuracy,balanced_accuracy}` | mean over the 21 TabArena datasets |
| `real_data/<dataset>/<metric>` | per dataset |

## Figures and tables

Run the notebooks from `playground/` and the script from the repository root. Figures are written to
`playground/figures/`, tables to `playground/tables/`; both are committed, as is the Figure 4 experiment cache in `playground/cache/`.

| Paper | Code |
|---|---|
| Figures 1-3 | `paper_scaling_trends.ipynb` |
| Figure 4 | `paper_selection_bias.ipynb` (experiment: `scaling/dataset_sample_efficiency.py`, ~10 min without the cache) |
| Figures 5-6 | `paper_hyperparameter_trends.ipynb` |
| Figure 7, Table 4 | `paper_big_prior_forms.ipynb` (experiment: `scaling/big_prior_forms.py`) |
| Tables 1-3 | `python playground/paper_tables.py` |

`scaling/utils.py` holds the Huber-loss fit (`fit_parametric_form`), the scaling law
$L(C) = A C^{-\alpha} + L_0$ (`functional_form_L0`) and the forms of Appendix C.

MIT license, see `LICENSE`.
