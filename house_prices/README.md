# Regression Prediction (House Prices analogue)

Regression task: predict a continuous target from numeric features — the same
model-building workflow as the Kaggle House Prices competition.

## Problem

> **Kaggle reference**: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The script uses the **Diabetes** dataset from `scikit-learn`
(`sklearn.datasets.load_diabetes`) as a fully bundled, no-download substitute.
Both tasks share the same ML workflow: numeric features → continuous target →
regression models → RMSE/R² evaluation.

## Dataset

Loaded automatically via `load_diabetes(as_frame=True)` — no download needed.

| Feature | Description |
|---------|-------------|
| age     | Age          |
| sex     | Sex          |
| bmi     | Body mass index |
| bp      | Average blood pressure |
| s1–s6   | Six serum measurements |

442 samples. **Target**: quantitative measure of disease progression one year
after baseline (continuous value — treated exactly like a house price).

## Models

* **Linear Regression** — baseline
* **Ridge Regression** — regularized linear model
* **Random Forest Regressor** — ensemble tree model

## Usage

```bash
pip install -r ../requirements.txt
python house_prices_regression.py
```

## Results (example)

| Model              | RMSE   | MAE   | R²    |
|--------------------|--------|-------|-------|
| Linear Regression  | ~53    | ~43   | ~0.52 |
| Ridge Regression   | ~53    | ~43   | ~0.52 |
| Random Forest      | ~55    | ~42   | ~0.48 |
