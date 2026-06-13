"""
Iris Species Classification
============================
Classic multi-class classification problem.
Goal: Classify iris flowers into three species based on petal/sepal measurements.

Dataset: Iris dataset from scikit-learn (also on Kaggle and UCI ML Repository).
Models  : K-Nearest Neighbors, Decision Tree, and Support Vector Machine.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = list(iris.target_names)
    return X, y, target_names


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def explore_data(X: pd.DataFrame, y: pd.Series, target_names: list[str]) -> None:
    print("=== Dataset shape ===")
    print(f"Features : {X.shape}")
    print(f"Classes  : {target_names}")
    print("\n=== Feature statistics ===")
    print(X.describe().round(2))
    print("\n=== Class distribution ===")
    print(y.value_counts().rename(index=dict(enumerate(target_names))))


# ---------------------------------------------------------------------------
# 3. Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, target_names: list[str]
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, random_state=42),
    }

    for name, model in models.items():
        use_scaled = name != "Decision Tree"
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5)

        print(f"\n{'=' * 50}")
        print(f"Model: {name}")
        print(f"Test Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
        print(f"CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, models.items()):
        use_scaled = name != "Decision Tree"
        X_te = X_test_scaled if use_scaled else X_test
        ConfusionMatrixDisplay.from_estimator(
            model, X_te, y_test,
            display_labels=target_names, ax=ax, colorbar=False
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=100)
    print("\nConfusion matrices saved to confusion_matrices.png")

    # Decision tree visualisation
    dt_model = models["Decision Tree"]
    plt.figure(figsize=(14, 6))
    plot_tree(
        dt_model,
        feature_names=list(X.columns),
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=100)
    print("Decision tree visualisation saved to decision_tree.png")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y, target_names = load_data()
    explore_data(X, y, target_names)
    train_and_evaluate(X, y, target_names)
