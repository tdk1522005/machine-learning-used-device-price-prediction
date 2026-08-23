# Used Device Price Prediction

A machine learning project for predicting the price of used mobile devices from their specifications. The notebook covers data preparation, preprocessing pipelines, model training, hyperparameter tuning, evaluation, and sample inference.

## Overview

This project compares three regression approaches:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Random Forest and XGBoost are tuned with `GridSearchCV`, and the final models are evaluated using MAE, MSE, RMSE, and R².

## Machine Learning Workflow

```text
Raw device data
      ↓
Data inspection & preprocessing
      ↓
Numeric / categorical feature handling
      ↓
Scikit-learn Pipeline + ColumnTransformer
      ↓
Train / test split
      ↓
Linear Regression / Random Forest / XGBoost
      ↓
GridSearchCV for tree-based models
      ↓
MAE · MSE · RMSE · R²
      ↓
Model comparison & price prediction
```

## Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Selection:** GridSearchCV
- **Environment:** Jupyter Notebook

## Results

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| **XGBoost** | **0.1794** | **0.0511** | **0.2260** | **0.8495** |
| Random Forest | 0.1833 | 0.0540 | 0.2323 | 0.8409 |
| Linear Regression | 0.1886 | 0.0574 | 0.2396 | 0.8309 |

XGBoost achieved the best test-set performance among the three evaluated regressors. During hyperparameter tuning, the best cross-validation R² scores were **0.8528 for XGBoost** and **0.8486 for Random Forest**.

### Best tuned configurations

**Random Forest**

```text
max_depth = 10
max_features = sqrt
min_samples_split = 2
n_estimators = 200
```

**XGBoost**

```text
colsample_bytree = 0.8
learning_rate = 0.03
max_depth = 4
min_child_weight = 1
n_estimators = 300
subsample = 0.8
```

## Repository Structure

```text
machine-learning-used-device-price-prediction/
├── Đồ_án_HMUD.ipynb   # End-to-end analysis and modeling notebook
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tdk1522005/machine-learning-used-device-price-prediction.git
cd machine-learning-used-device-price-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open `Đồ_án_HMUD.ipynb` and run the cells in order.

## What This Project Demonstrates

- Exploratory data analysis and visualization
- Handling numeric and categorical features
- Reproducible preprocessing with Scikit-learn pipelines
- Regression model comparison
- Hyperparameter tuning with GridSearchCV
- Model evaluation with multiple regression metrics
- Inference on new device specifications

## Current Limitations

- The project is currently organized primarily as a single notebook.
- Evaluation is based on the current dataset split and cross-validation setup.
- A production API or web interface is not included yet.

## Planned Improvements

- Refactor reusable preprocessing and inference logic into a `src/` package.
- Add automated tests for preprocessing and prediction.
- Add feature-importance / model-interpretability analysis.
- Add a lightweight prediction API or demo interface.

## Author

**Ta Duy Khanh**  
AI Engineering student — Machine Learning, Deep Learning & Natural Language Processing
