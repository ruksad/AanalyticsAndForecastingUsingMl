"""
Rossmann Store Sales Forecasting
==================================
Classic Kaggle time-series regression / forecasting problem.
Goal: Predict daily sales for Rossmann drug stores.

Kaggle link: https://www.kaggle.com/competitions/rossmann-store-sales

Dataset: Synthetic data generated to mirror the Rossmann dataset schema
         (no manual download required). The generator reproduces the key
         patterns: store-level trends, weekly seasonality, promotions, and
         holiday effects.

Features engineered:
  * Calendar: year, month, day, day-of-week, week-of-year, quarter
  * Lag sales: 1-day, 7-day, 30-day lags
  * Rolling averages: 7-day and 30-day windows
  * Store metadata: store type, assortment, competition distance, promo

Models:
  * Random Forest Regressor  — strong ensemble baseline
  * Gradient Boosting Regressor — boosted trees, typically best on tabular data

Evaluation metric: RMSPE (Root Mean Squared Percentage Error), the official
                   Kaggle metric for this competition.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
TRAIN_TEST_SPLIT_RATIO = 0.90
PLOT_SAMPLE_SIZE = 500

# ---------------------------------------------------------------------------
# 1. Generate synthetic Rossmann-like data
# ---------------------------------------------------------------------------

NUM_STORES = 50
START_DATE = "2013-01-01"
END_DATE = "2015-07-31"

STORE_TYPES = ["a", "b", "c", "d"]
ASSORTMENTS = ["basic", "extra", "extended"]


def _make_store_metadata(n_stores: int) -> pd.DataFrame:
    """Create one row per store with static attributes."""
    return pd.DataFrame(
        {
            "Store": np.arange(1, n_stores + 1),
            "StoreType": rng.choice(STORE_TYPES, size=n_stores),
            "Assortment": rng.choice(ASSORTMENTS, size=n_stores),
            "CompetitionDistance": rng.integers(100, 20_000, size=n_stores),
            "Promo2": rng.integers(0, 2, size=n_stores),
        }
    )


def generate_data() -> pd.DataFrame:
    """
    Generate a synthetic daily sales dataset that mirrors the Rossmann schema.
    Each row is one (store, date) observation.
    """
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    store_meta = _make_store_metadata(NUM_STORES)

    records = []
    for _, row in store_meta.iterrows():
        store_id = int(row["Store"])

        # Base sales level differs by store type
        base = {"a": 6_000, "b": 9_000, "c": 4_500, "d": 7_500}[row["StoreType"]]
        base += rng.integers(-500, 500)  # store-level noise

        for date in dates:
            # Closed on Sundays ~90% of the time
            if date.dayofweek == 6 and rng.random() < 0.90:
                records.append(
                    {
                        "Store": store_id,
                        "Date": date,
                        "DayOfWeek": date.dayofweek + 1,
                        "Open": 0,
                        "Promo": 0,
                        "StateHoliday": 0,
                        "SchoolHoliday": 0,
                        "Sales": 0,
                        "Customers": 0,
                    }
                )
                continue

            # Weekly seasonality (Mon–Sat peaks)
            dow_effect = [1.0, 0.95, 0.90, 0.92, 1.05, 1.15, 0.50][date.dayofweek]

            # Monthly seasonality (slight dip in Feb, peak in Dec)
            month_effect = 1.0 + 0.05 * np.sin(2 * np.pi * (date.month - 3) / 12)

            # Long-term upward trend
            days_elapsed = (date - pd.Timestamp(START_DATE)).days
            trend = 1.0 + 0.0001 * days_elapsed

            # Promotion boosts sales ~20%
            promo = int(rng.random() < 0.40)
            promo_effect = 1.20 if promo else 1.0

            # State / school holidays slightly reduce foot traffic
            state_holiday = int(rng.random() < 0.02)
            school_holiday = int(rng.random() < 0.10)
            holiday_effect = 0.70 if state_holiday else (0.90 if school_holiday else 1.0)

            sales = int(
                base
                * dow_effect
                * month_effect
                * trend
                * promo_effect
                * holiday_effect
                * (1 + rng.normal(0, 0.05))  # white noise
            )
            sales = max(sales, 0)
            customers = int(sales / rng.uniform(6, 10))

            records.append(
                {
                    "Store": store_id,
                    "Date": date,
                    "DayOfWeek": date.dayofweek + 1,
                    "Open": 1,
                    "Promo": promo,
                    "StateHoliday": state_holiday,
                    "SchoolHoliday": school_holiday,
                    "Sales": sales,
                    "Customers": customers,
                }
            )

    df = pd.DataFrame(records)
    df = df.merge(store_meta, on="Store", how="left")
    df["StoreType"] = df["StoreType"].map({"a": 0, "b": 1, "c": 2, "d": 3})
    df["Assortment"] = df["Assortment"].map({"basic": 0, "extra": 1, "extended": 2})
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def explore_data(df: pd.DataFrame) -> None:
    open_df = df[df["Open"] == 1]
    print("=== Dataset shape ===")
    print(f"Total rows  : {len(df):,}  ({df['Store'].nunique()} stores × {df['Date'].nunique()} days)")
    print(f"Open rows   : {len(open_df):,}")
    print(f"Date range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print("\n=== Sales statistics (open days only) ===")
    print(open_df["Sales"].describe().round(0))
    print("\n=== Average sales by day of week ===")
    print(
        open_df.groupby("DayOfWeek")["Sales"]
        .mean()
        .round(0)
        .rename(index={1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"})
    )


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Calendar features
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Quarter"] = data["Date"].dt.quarter

    # Lag and rolling features per store (open days only)
    data = data.sort_values(["Store", "Date"])
    for lag in [1, 7, 30]:
        data[f"Sales_lag_{lag}"] = (
            data.groupby("Store")["Sales"].shift(lag)
        )

    for window in [7, 30]:
        data[f"Sales_roll_mean_{window}"] = (
            data.groupby("Store")["Sales"]
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Drop rows with NaN lags (first 30 days per store)
    data = data.dropna(subset=[c for c in data.columns if c.startswith("Sales_lag_")])
    return data


# ---------------------------------------------------------------------------
# 4. Train & Evaluate
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Store", "DayOfWeek", "Open", "Promo",
    "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance", "Promo2",
    "Year", "Month", "Day", "WeekOfYear", "Quarter",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_30",
    "Sales_roll_mean_7", "Sales_roll_mean_30",
]
TARGET_COL = "Sales"


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error — official Kaggle metric."""
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = np.maximum(model.predict(X_test), 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmspe_val = rmspe(y_test.values, y_pred)
    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"  RMSPE : {rmspe_val:.4f}  (lower is better; Kaggle metric)")
    print(f"  RMSE  : {rmse:.2f}")
    print(f"  MAE   : {mae:.2f}")
    print(f"  R²    : {r2:.4f}")


def train_and_evaluate(data: pd.DataFrame) -> None:
    # Keep only open days for modelling
    model_data = data[data["Open"] == 1].copy()

    X = model_data[FEATURE_COLS]
    y = model_data[TARGET_COL]

    # Chronological split: last 10% of time as test set
    split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTraining rows : {len(X_train):,} | Test rows : {len(X_test):,}")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=RNG_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RNG_SEED
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate(name, model, X_test, y_test)

    # Feature importances (Gradient Boosting)
    gb_model = models["Gradient Boosting"]
    importances = (
        pd.Series(gb_model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )
    print("\n=== Top-10 Feature Importances (Gradient Boosting) ===")
    print(importances.head(10).round(4))

    # Plot: actual vs predicted (sample 500 test points for clarity)
    y_pred_gb = np.maximum(gb_model.predict(X_test), 0)
    sample_idx = np.linspace(0, len(y_test) - 1, min(PLOT_SAMPLE_SIZE, len(y_test)), dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: actual vs predicted
    axes[0].scatter(
        y_test.values[sample_idx],
        y_pred_gb[sample_idx],
        alpha=0.3, s=10
    )
    lim = [0, max(y_test.max(), y_pred_gb.max()) * 1.05]
    axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Sales")
    axes[0].set_ylabel("Predicted Sales")
    axes[0].set_title("Gradient Boosting – Actual vs Predicted")
    axes[0].legend()

    # Bar: top-10 feature importances
    importances.head(10).plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top-10 Feature Importances (Gradient Boosting)")

    plt.tight_layout()
    plt.savefig("rossmann_results.png", dpi=100)
    print("\nPlot saved to rossmann_results.png")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic Rossmann-like dataset …")
    df = generate_data()
    explore_data(df)

    print("\nEngineering features …")
    data = engineer_features(df)

    train_and_evaluate(data)
