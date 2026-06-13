"""
Titanic Survival Prediction
============================
Classic Kaggle binary classification problem.
Goal: Predict which passengers survived the Titanic disaster.

Dataset: Loaded via seaborn (mirrors the Kaggle Titanic dataset).
Models  : Logistic Regression and Random Forest Classifier.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def explore_data(df: pd.DataFrame) -> None:
    print("=== Dataset Info ===")
    print(df.info())
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Missing values ===")
    print(df.isnull().sum())
    print("\n=== Survival rate ===")
    print(df["survived"].value_counts(normalize=True).round(3))


# ---------------------------------------------------------------------------
# 3. Feature engineering & preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()

    # Fill missing age with median age per sex/class group
    data["age"] = data.groupby(["sex", "pclass"])["age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill missing embark_town with mode
    data["embark_town"] = data["embark_town"].fillna(data["embark_town"].mode()[0])

    # Drop columns with high missingness or low predictive value
    data = data.drop(
        columns=[
            "deck", "embarked", "alive", "who",
            "adult_male", "class", "alone",
        ],
        errors="ignore",
    )

    # Encode categorical columns
    le = LabelEncoder()
    for col in ["sex", "embark_town"]:
        data[col] = le.fit_transform(data[col].astype(str))

    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embark_town"]
    X = data[features]
    y = data["survived"]

    return X, y


# ---------------------------------------------------------------------------
# 4. Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"Test Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))

    # Feature importance from Random Forest
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    print("\n=== Feature Importances (Random Forest) ===")
    print(importances.sort_values(ascending=False).round(4))

    # Confusion matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, model) in zip(axes, models.items()):
        if name == "Logistic Regression":
            ConfusionMatrixDisplay.from_estimator(
                model, X_test_scaled, y_test,
                display_labels=["Not Survived", "Survived"],
                ax=ax, colorbar=False
            )
        else:
            ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test,
                display_labels=["Not Survived", "Survived"],
                ax=ax, colorbar=False
            )
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=100)
    print("\nConfusion matrix saved to confusion_matrices.png")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    explore_data(df)
    X, y = preprocess(df)
    train_and_evaluate(X, y)
