# Invoice Intelligence System

Invoice Intelligence System is a machine learning project for procurement and finance workflows. It contains two main ML modules and a Streamlit app for interactive inference:

- Freight Cost Prediction for regression
- Invoice Flagging for classification

The repository includes training scripts, saved model artifacts, notebooks, and a lightweight UI for manual and batch predictions.

## Overview

### Freight Cost Prediction
- Predicts freight cost from invoice features.
- The current training pipeline uses the `Dollars` feature as the input.
- Compares three regressors:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Saves the best model based on lowest MAE.

### Invoice Flagging
- Flags invoices for manual review, where `0` means low risk and `1` means review needed.
- Builds engineered features from `vendor_invoice` and `purchases` data.
- Creates target labels with business-rule heuristics during preprocessing.
- Compares three classifiers:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Saves the best model based on highest weighted F1.

### Streamlit App
- One interface with two prediction flows:
  - Freight Cost Prediction for single predictions
  - Invoice Flagging for single prediction and batch CSV scoring
- Loads the saved models from each module.
- Supports downloadable CSV output for batch invoice flagging.

## Data Source

The project uses a SQLite database stored at `data/inventory.db`.

Main tables used:

- `vendor_invoice`
- `purchases`

## Workflow

### Freight Cost Prediction
1. Load the `vendor_invoice` table.
2. Prepare the features and target, where `X = Dollars` and `y = Freight`.
3. Train three models and evaluate MAE, RMSE, and R2.
4. Save the best model to `Freight Cost Prediction/models/predict_freight_model.pkl`.

### Invoice Flagging
1. Load the `vendor_invoice` and `purchases` tables.
2. Build purchase-order-level aggregates with pandas.
3. Merge invoice and purchase aggregates and engineer date-delay features.
4. Create the `flag_invoice` target from business rules.
5. Train three classifiers and evaluate Accuracy, Precision, Recall, and F1.
6. Save the best model bundle to `Invoice Flagging/models/invoice_flagging_model.pkl`.

### Inference Flow
1. Open the Streamlit app.
2. The app loads model artifacts with `joblib`.
3. Enter features manually or upload a CSV file for invoice flagging.
4. View the prediction and risk probability when available.

## Project Structure

```text
Invoice Intelligence System/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── inventory.db
├── notebooks/
│   ├── Predicting Freight Cost.ipynb
│   └── Invoice Flagging.ipynb
├── Freight Cost Prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── train.py
│   └── models/
│       └── predict_freight_model.pkl
└── Invoice Flagging/
    ├── data_preprocessing.py
    ├── model_evaluation.py
    ├── train.py
    └── models/
        └── invoice_flagging_model.pkl
```

## Setup

### 1. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python "Freight Cost Prediction/train.py"
python "Invoice Flagging/train.py"
```

### 4. Launch the app

```bash
streamlit run app.py
```

## Notes

- The repository is organized as two independent ML workflows under a single UI.
- Future improvements could include model version metadata, stricter feature validation, artifact tracking, and automated tests for preprocessing and prediction paths.
