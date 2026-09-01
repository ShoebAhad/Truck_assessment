# Freight Rate Prediction

Predicts freight `posted_rate` for the Spotter ML Assessment. Full analysis, feature engineering,
model comparison, and final training live in `freight_rate_modeling.ipynb`; this README covers
setup and how to reproduce every deliverable from scratch.

## Repo contents

| Path | What it is |
|---|---|
| `freight_rate_modeling.ipynb` | The solution — EDA, data cleaning, feature engineering, model comparison, walk-forward validation, final training, and prediction generation |
| `validation_predictions.csv` | Final predictions for the 12,000 loads in `data/validation.csv` |
| `december_predictions.csv` | Final predictions for `data/december_chart_inputs.csv` |
| `Freight_Rate_Report.pdf` | Short report: train/test split approach and the December prediction chart |
| `score.py`, `requirements.txt` | Provided by the assessment — validates predictions and renders the December chart |
| `requirements-notebook.txt` | Dependencies needed to run the notebook itself |
| `data/` | Provided assessment data (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`) |
| `scorer_results/candidate_december.png` | The chart produced by running `score.py` |

## Setup

```bash
python -m pip install -r requirements-notebook.txt
```

On macOS, XGBoost and LightGBM need the OpenMP runtime, which isn't bundled with the pip wheel:

```bash
brew install libomp
```

No GPU is required. The notebook includes a small PyTorch MLP for comparison purposes, but the
production model (a Ridge + CatBoost blend) trains on CPU in a couple of minutes on this dataset.

## Reproduce everything from scratch

```bash
jupyter nbconvert --to notebook --execute --inplace freight_rate_modeling.ipynb
```

This regenerates `validation_predictions.csv` and `december_predictions.csv`, then calls the
provided scorer itself:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

which validates both files and writes `scorer_results/candidate_december.png`.

## Approach summary

- **Split strategy**: time-based throughout, never random K-fold — `train_test.csv` covers
  Jan–Oct 2025, while both prediction targets (`validation.csv`, `december_chart_inputs.csv`) fall
  entirely in Nov–Dec 2025. Model selection used a single fixed holdout plus a 4-cutoff
  walk-forward robustness check; the production model is retrained on 100% of the labeled data.
- **Target**: `rate_per_mile` (`posted_rate / distance`), recovered by multiplying back out —
  removes distance's dominant scale effect and improves every model tested.
- **Key feature engineering**: sign-corrected `weight`, a 2-harmonic seasonal model for
  `market_index` imputation/extrapolation, cyclical (not raw) date encoding so the model can
  extrapolate into unseen months, out-of-fold target-encoded pickup/delivery/lane, and
  `quote_signal_dev` — an engineered feature that captures a U-shaped relationship invisible to
  linear correlation and became the single most important feature in the model.
- **Final model**: a blend of tuned CatBoost (weight 0.75) and tuned Ridge (weight 0.25) on
  `rate_per_mile`, chosen because it's more robust across a walk-forward check than either model
  alone. Full reasoning, all model comparisons, and every data-quality finding are in the notebook.

See `Freight_Rate_Report.pdf` for the train/test split approach and the final December chart.
