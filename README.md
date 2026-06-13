# Analytics and Forecasting Using ML

This repo contains classic problems from Kaggle or HuggingFace solved using machine learning.

---

## Projects

### [Rossmann Store Sales](./Rossmann_sales/)

**Source:** [Kaggle — Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/overview)

Forecast daily sales for 1,115 Rossmann drug stores across Germany using ~1M rows of historical data (Jan 2013 – Jul 2015). The test set covers a 6-week future window (~41k rows).

**Target:** `Sales` (regression) | **Metric:** RMSPE

#### Dataset

| File | Description |
|---|---|
| `train.csv` | Historical sales including the target `Sales` column |
| `test.csv` | Historical data excluding `Sales` (to predict) |
| `store.csv` | Store-level metadata (type, assortment, competition, promotions) |

Key features: store type, assortment level, competition distance, state/school holidays, promotional flags (Promo, Promo2).

#### Approach

| Phase | Description |
|---|---|
| EDA | Sales distributions, time patterns, store-level variance, promo effects |
| Feature Engineering | Date features, competition/promo duration, lag & rolling sales, interaction terms |
| Modeling | LightGBM, XGBoost, CatBoost with time-based validation (train Jan 2013–Jun 2015, validate Jul 2015) |
| Tuning | Optuna hyperparameter search, weighted ensemble |
| Submission | Log1p-transformed target, `Sales=0` override for closed stores |
