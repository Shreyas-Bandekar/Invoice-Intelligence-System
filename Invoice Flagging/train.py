from pathlib import Path

import joblib

from data_preprocessing import (
    FEATURE_COLUMNS,
    build_feature_frame,
    load_source_tables,
    prepare_features,
    split_data,
)
from model_evaluation import (
    evaluate_model,
    train_decision_tree,
    train_logistic_regression,
    train_random_forest,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    db_path = project_root / "data" / "inventory.db"
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    vendor_df, purchases_df = load_source_tables(str(db_path))
    df = build_feature_frame(vendor_df, purchases_df)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_model = train_logistic_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    results = [
        evaluate_model(lr_model, X_test, y_test, "Logistic Regression"),
        evaluate_model(dt_model, X_test, y_test, "Decision Tree Classifier"),
        evaluate_model(rf_model, X_test, y_test, "Random Forest Classifier"),
    ]

    best_model_info = max(results, key=lambda x: x["f1"])
    best_model_name = best_model_info["model_name"]

    models_dict = {
        "Logistic Regression": lr_model,
        "Decision Tree Classifier": dt_model,
        "Random Forest Classifier": rf_model,
    }

    model_bundle = {
        "model": models_dict[best_model_name],
        "feature_columns": FEATURE_COLUMNS,
        "best_model_name": best_model_name,
        "metrics": best_model_info,
    }

    model_path = model_dir / "invoice_flagging_model.pkl"
    joblib.dump(model_bundle, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
