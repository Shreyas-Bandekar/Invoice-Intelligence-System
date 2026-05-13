# Invoice Intelligence System

Invoice Intelligence System is an end-to-end machine learning project for procurement and finance workflows. It combines two model pipelines with a Streamlit app for interactive inference.

## Modules

- Freight Cost Prediction (regression)
- Invoice Flagging (classification)

The repository includes data preprocessing, model training, model evaluation, saved artifacts, notebooks, and a UI for single and batch prediction.

## Features

### Freight Cost Prediction
- Predicts freight cost from invoice features.
- Uses the `Dollars` feature as the current model input.
- Trains and compares:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Saves the best model by lowest MAE.

### Invoice Flagging
- Flags invoices for manual review.
- Target meaning:
  - `0` = low risk
  - `1` = review needed
- Uses engineered features from `vendor_invoice` and `purchases`.
- Builds labels with business-rule heuristics during preprocessing.
- Trains and compares:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Saves the best model by highest weighted F1.

### Streamlit App
- Unified interface for both modules.
- Supports:
  - Single prediction for freight cost
  - Single and batch CSV scoring for invoice flagging
- Loads saved model artifacts with `joblib`.
- Provides downloadable CSV output for batch invoice flagging.

## Data Source

SQLite database: `data/inventory.db`

Main tables:
- `vendor_invoice`
- `purchases`

## Workflow

### Freight Cost Prediction
1. Load the `vendor_invoice` table.
2. Prepare features and target (`X = Dollars`, `y = Freight`).
3. Train three regressors and evaluate MAE, RMSE, and R2.
4. Save the best model to `Freight Cost Prediction/models/predict_freight_model.pkl`.

### Invoice Flagging
1. Load `vendor_invoice` and `purchases`.
2. Build purchase-order-level aggregates.
3. Merge aggregates and engineer delay/date features.
4. Create `flag_invoice` labels from business rules.
5. Train three classifiers and evaluate Accuracy, Precision, Recall, and F1.
6. Save the best model bundle to `Invoice Flagging/models/invoice_flagging_model.pkl`.

### Inference
1. Launch the Streamlit app.
2. Load model artifacts.
3. Enter features manually or upload CSV (invoice flagging).
4. View predictions and, where available, risk probability.

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

## Quick Start

### 1. Activate virtual environment

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models

```bash
python "Freight Cost Prediction/train.py"
python "Invoice Flagging/train.py"
```

### 4. Run app

```bash
streamlit run app.py
```

## Notes

- The repository is organized as two independent ML workflows with a shared UI.
- Potential next improvements:
  - Model version metadata
  - Stronger feature validation
  - Artifact tracking
  - Automated tests for preprocessing and prediction paths
  