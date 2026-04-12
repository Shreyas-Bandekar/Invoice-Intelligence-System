# Invoice Intelligence System

A machine learning project for procurement/finance workflows with two modules:

1. Freight Cost Prediction (regression)
2. Invoice Flagging (classification)

The project includes training pipelines, saved models, notebooks, and a Streamlit UI for interactive inference.

## Current System Capabilities

### 1) Freight Cost Prediction
- Predicts invoice freight cost from invoice features.
- Current training script uses the `Dollars` feature as the input.
- Trains and compares:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Automatically selects and saves the best model by lowest MAE.

### 2) Invoice Flagging
- Flags invoices for manual review (`0` = low risk, `1` = review needed).
- Builds engineered features from `vendor_invoice` + `purchases` data.
- Creates target labels with heuristic business rules in preprocessing.
- Trains and compares:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Automatically selects and saves the best model by highest weighted F1.

### 3) Streamlit UI (`app.py`)
- Single app with two modules:
  - Freight Cost Prediction (single prediction)
  - Invoice Flagging (single prediction + batch CSV scoring)
- Loads saved models from each module folder.
- Supports downloadable CSV outputs for batch invoice flagging.

## How the System Works

## Data Source
- SQLite database: `data/inventory.db`
- Main tables used:
  - `vendor_invoice`
  - `purchases`

## Training Flow

### Freight module
1. Load `vendor_invoice` table.
2. Prepare X/y (`X = Dollars`, `y = Freight`).
3. Train 3 models and evaluate MAE/RMSE/R2.
4. Save best model to:
   - `Freight Cost Prediction/models/predict_freight_model.pkl`

### Invoice flagging module
1. Load `vendor_invoice` and `purchases` tables.
2. Build PO-level aggregates with pandas.
3. Merge invoice + purchase aggregates and engineer date-delay features.
4. Create `flag_invoice` target from business rules.
5. Train 3 classifiers and evaluate Accuracy/Precision/Recall/F1.
6. Save best model bundle to:
   - `Invoice Flagging/models/invoice_flagging_model.pkl`

## Inference Flow (UI)
1. User opens Streamlit app.
2. App loads model(s) with `joblib`.
3. User enters features manually or uploads CSV (for invoice flagging).
4. App returns prediction and risk probability (when available).

## Folder Structure

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

## Run Instructions

## 1) Activate environment
```bash
source .venv/bin/activate
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

## 3) Train models
```bash
python "Freight Cost Prediction/train.py"
python "Invoice Flagging/train.py"
```

## 4) Launch UI
```bash
streamlit run app.py
```

## Notes
- Duplicate typo file `reuqirements.txt` and transient cache/checkpoint folders were removed during cleanup.
- If you want stricter production behavior, next improvements are:
  - model/version metadata logging,
  - feature validation contracts,
  - train/test artifact tracking,
  - automated tests for preprocessing and prediction endpoints.
