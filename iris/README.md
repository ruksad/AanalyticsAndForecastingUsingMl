# Iris Species Classification

Multi-class classification task: identify the species of an iris flower from its measurements.

## Problem

> **Kaggle link**: https://www.kaggle.com/datasets/uciml/iris  
> **UCI ML Repo**: https://archive.ics.uci.edu/ml/datasets/iris

Given four measurements (sepal length, sepal width, petal length, petal width),
classify each sample into one of three iris species:
`setosa`, `versicolor`, or `virginica`.

## Dataset

Loaded automatically via `sklearn.datasets.load_iris()`.

| Feature           | Unit |
|-------------------|------|
| sepal length (cm) | cm   |
| sepal width (cm)  | cm   |
| petal length (cm) | cm   |
| petal width (cm)  | cm   |

150 samples — 50 per class.

## Models

* **K-Nearest Neighbors** — instance-based classifier
* **Decision Tree** — interpretable rule-based model
* **Support Vector Machine (RBF kernel)** — maximum-margin classifier

## Usage

```bash
pip install -r ../requirements.txt
python iris_classification.py
```

## Results (example)

| Model                   | Test Accuracy | CV Accuracy (5-fold) |
|-------------------------|---------------|----------------------|
| K-Nearest Neighbors     | ~0.97         | ~0.97                |
| Decision Tree           | ~0.97         | ~0.96                |
| Support Vector Machine  | ~0.97         | ~0.97                |
