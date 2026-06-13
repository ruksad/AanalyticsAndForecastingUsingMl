# Analytics and Forecasting Using ML

Classic Machine Learning problems sourced from **Kaggle** and the **UCI ML Repository**,
solved with scikit-learn. Each problem lives in its own folder with a self-contained
Python script and README.

---

## Problems

| Folder | Task | Type | Dataset |
|---|---|---|---|
| [`titanic/`](titanic/) | Survival prediction | Binary classification | seaborn `titanic` |
| [`house_prices/`](house_prices/) | Regression prediction | Regression | sklearn `diabetes` (bundled) |
| [`iris/`](iris/) | Species classification | Multi-class classification | sklearn `iris` |
| [`rossmann_sales/`](rossmann_sales/) | Store sales forecasting | Time-series regression | Synthetic Rossmann-schema data |

---

## Quick start

```bash
# Install dependencies (Python 3.9+)
pip install -r requirements.txt

# Run any problem
python titanic/titanic_classification.py
python house_prices/house_prices_regression.py
python iris/iris_classification.py
python rossmann_sales/rossmann_sales_forecasting.py
```

All datasets are downloaded automatically — no manual Kaggle downloads needed.

---

## Repository structure

```
AnalyticsAndForecastingUsingMl/
├── requirements.txt
├── titanic/
│   ├── README.md
│   └── titanic_classification.py
├── house_prices/
│   ├── README.md
│   └── house_prices_regression.py
├── rossmann_sales/
│   ├── README.md
│   └── rossmann_sales_forecasting.py
└── iris/
    ├── README.md
    └── iris_classification.py
```
