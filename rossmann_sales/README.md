# Rossmann Store Sales Forecasting

Time-series regression task: predict daily sales for Rossmann drug stores.

## Problem

> **Kaggle link**: https://www.kaggle.com/competitions/rossmann-store-sales

Given historical sales data and store metadata, predict the `Sales` for each
store on each future day. The official evaluation metric is **RMSPE** (Root
Mean Squared Percentage Error).

## Dataset

Generated synthetically to mirror the Rossmann schema — **no Kaggle download
required**. The generator reproduces the key real-world patterns:

| Pattern | Detail |
|---|---|
| Weekly seasonality | Saturday peak, Sunday near-zero |
| Monthly seasonality | December uplift, February dip |
| Long-term trend | Gradual upward drift over ~2.5 years |
| Promotions | ~20% sales uplift on promo days |
| Holidays | State / school holiday dampening |
| Store heterogeneity | 4 store types with different base-sales levels |

**Schema mirrors the Kaggle dataset:**

| Column | Description |
|---|---|
| Store | Store ID (1–50) |
| Date | Calendar date |
| DayOfWeek | 1 = Monday … 7 = Sunday |
| Open | 1 = open, 0 = closed |
| Promo | 1 if a promotion was running |
| StateHoliday | 1 if a state holiday |
| SchoolHoliday | 1 if school holiday |
| Sales | Daily turnover (target) |
| Customers | Number of customers |
| StoreType | Store category (a/b/c/d) |
| Assortment | Product range (basic/extra/extended) |
| CompetitionDistance | Distance to nearest competitor (m) |
| Promo2 | Whether store participates in a continuous promo |

## Feature Engineering

* **Calendar**: year, month, day, day-of-week, week-of-year, quarter
* **Lag sales**: 1-day, 7-day, 30-day lags
* **Rolling averages**: 7-day and 30-day windows (shifted to avoid leakage)

## Models

* **Random Forest Regressor** — strong ensemble baseline
* **Gradient Boosting Regressor** — boosted trees; typically best on tabular data

## Usage

```bash
pip install -r ../requirements.txt
python rossmann_sales_forecasting.py
```

## Results (example)

| Model              | RMSPE  | RMSE   | MAE    | R²     |
|--------------------|--------|--------|--------|--------|
| Random Forest      | ~0.059 | ~471   | ~368   | ~0.945 |
| Gradient Boosting  | ~0.058 | ~466   | ~362   | ~0.947 |

Output plots are saved to `rossmann_results.png` (actual-vs-predicted scatter
and top-10 feature importances).
