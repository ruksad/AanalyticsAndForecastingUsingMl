"""
Regression Prediction (House Prices analogue)
==============================================
Classic regression problem analogous to the Kaggle House Prices competition.
Goal: Predict a continuous target value from numeric features.

Dataset: Diabetes dataset from scikit-learn — a fully bundled, no-download
         regression benchmark (442 samples, 10 physiological features).
         Target is a quantitative measure of disease progression one year
         after baseline (similar continuous-value regression to house prices).
Models  : Linear Regression, Ridge Regression, and Random Forest Regressor.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCATTER_ALPHA = 0.3
SCATTER_SIZE = 10


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target  # Disease progression one year after baseline
    return X, y


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def explore_data(X: pd.DataFrame, y: pd.Series) -> None:
    print("=== Dataset shape ===")
    print(f"Features : {X.shape}")
    print(f"Target   : {y.shape}")
    print("\n=== Feature descriptions ===")
    print(X.describe().round(4))
    print("\n=== Target distribution (disease progression score) ===")
    print(y.describe().round(2))
    print("\n=== Missing values ===")
    print(X.isnull().sum())


# ---------------------------------------------------------------------------
# 3. Train & evaluate
# ---------------------------------------------------------------------------

def evaluate_model(name: str, model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }

    for name, model in linear_models.items():
        model.fit(X_train_scaled, y_train)
        evaluate_model(name, model, X_test_scaled, y_test)
        cv = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="r2"
        )
        print(f"  CV R² : {cv.mean():.4f} ± {cv.std():.4f}")

    # Random Forest (no scaling needed)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test)
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
    print(f"  CV R² : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\n=== Feature Importances (Random Forest) ===")
    print(importances.sort_values(ascending=False).round(4))

    # Actual vs Predicted plot for Random Forest
    y_pred_rf = rf.predict(X_test)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred_rf, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
    lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual progression score")
    plt.ylabel("Predicted progression score")
    plt.title("Random Forest – Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=100)
    print("\nPlot saved to actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = load_data()
    explore_data(X, y)
    train_and_evaluate(X, y)
