# RAAS: Risk-Aware Adaptive Screening Framework

Reproducible code for:
> **A Risk-Aware Adaptive Screening Framework for Psychiatric Assessment: Development and Validation Study**  
> *Submitted to JMIR Medical Informatics*

## Dataset

Download DASS-42 from Kaggle and place as `data.csv`:  
https://www.kaggle.com/datasets/lucasgreenwell/depression-anxiety-stress-scales-responses

## Requirements

```bash
pip install numpy pandas scikit-learn
```

## Usage

```bash
python raas_experiment.py                          # default settings
python raas_experiment.py --data_path data.csv     # explicit path
python raas_experiment.py --n_bootstrap 2000       # full CI (slow)
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | `data.csv` | Path to DASS-42 dataset |
| `--tau` | `1.2` | Rule engine signal-item threshold |
| `--theta` | `0.75` | Adaptive confidence threshold |
| `--n_bootstrap` | `2000` | Bootstrap resamples for Recall@K CI |
| `--seed` | `42` | Random seed |

## Experiments

| # | Paper section | Description |
|---|---|---|
| 1 | Sec 4.2, Table 1 | 6-method comparison |
| 2 | Sec 4.3, Table 2 | Threshold sensitivity sweep |
| 3 | Sec 4.4 | Signal item ablation |
| 4 | Sec 4.5 | Real case study (test index=13) |

## Expected Results (Table 1)

| Method | Avg dims | Reduction | Recall@K |
|---|---|---|---|
| Full-scale baseline | 3.00 | 0.0% | 1.000 |
| Logistic Regression | 1.15 | 61.7% | 1.000* |
| Random Forest | 1.14 | 62.1% | 0.961 |
| GBM only | 1.15 | 61.7% | 0.989 |
| Rule+GBM (no adaptation) | 1.15 | 61.8% | 0.988 |
| **Adaptive Rule+GBM (proposed)** | **1.15** | **61.7%** | **0.989** |

Bootstrap 95% CI: [0.987, 0.991] (n=2,000, seed=42)

*LR perfect score is an artefact of linearly-constructed labels, not clinical generalizability.

## License

MIT License. Contact: [email address]
