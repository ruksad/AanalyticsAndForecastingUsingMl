# Titanic Survival Prediction

Binary classification task: predict which passengers survived the Titanic disaster.

## Problem

> **Kaggle link**: https://www.kaggle.com/competitions/titanic

Given passenger information (age, sex, ticket class, etc.), predict whether
they survived (`1`) or not (`0`).

## Dataset

Loaded automatically via `seaborn.load_dataset("titanic")` — no manual download needed.

| Feature      | Description                         |
|--------------|-------------------------------------|
| pclass       | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| sex          | Passenger sex                       |
| age          | Age in years                        |
| sibsp        | # of siblings / spouses aboard      |
| parch        | # of parents / children aboard      |
| fare         | Passenger fare                      |
| embark_town  | Port of embarkation                 |

## Models

* **Logistic Regression** — linear baseline
* **Random Forest Classifier** — ensemble tree model

## Usage

```bash
pip install -r ../requirements.txt
python titanic_classification.py
```

## Results (example)

| Model               | Test Accuracy | CV Accuracy (5-fold) |
|---------------------|---------------|----------------------|
| Logistic Regression | ~0.80         | ~0.80                |
| Random Forest       | ~0.81         | ~0.82                |
