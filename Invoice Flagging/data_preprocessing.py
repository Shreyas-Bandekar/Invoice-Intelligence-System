import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_brands",
    "total_item_quantity",
    "days_po_to_invoice",
    "total_item_dollars",
]


def load_source_tables(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source invoice and purchase tables from SQLite."""
    conn = sqlite3.connect(db_path)
    vendor_df = pd.read_sql_query("SELECT * FROM vendor_invoice", conn)
    purchases_df = pd.read_sql_query("SELECT * FROM purchases", conn)
    conn.close()
    return vendor_df, purchases_df


def build_feature_frame(vendor_df: pd.DataFrame, purchases_df: pd.DataFrame) -> pd.DataFrame:
    """Create invoice-level ML features with pandas operations."""
    purchases = purchases_df.copy()
    purchases["PODate"] = pd.to_datetime(purchases["PODate"], errors="coerce")
    purchases["ReceivingDate"] = pd.to_datetime(purchases["ReceivingDate"], errors="coerce")
    purchases["receiving_delay"] = (purchases["ReceivingDate"] - purchases["PODate"]).dt.days

    purchase_agg_df = (
        purchases.groupby("PONumber", as_index=False)
        .agg(
            total_brands=("Brand", "nunique"),
            total_item_quantity=("Quantity", "sum"),
            total_item_dollars=("Dollars", "sum"),
            avg_receiving_delay=("receiving_delay", "mean"),
        )
    )

    invoice = vendor_df.copy()
    invoice["InvoiceDate"] = pd.to_datetime(invoice["InvoiceDate"], errors="coerce")
    invoice["PODate"] = pd.to_datetime(invoice["PODate"], errors="coerce")
    invoice["PayDate"] = pd.to_datetime(invoice["PayDate"], errors="coerce")
    invoice["days_po_to_invoice"] = (invoice["InvoiceDate"] - invoice["PODate"]).dt.days
    invoice["days_to_pay"] = (invoice["PayDate"] - invoice["InvoiceDate"]).dt.days

    df = invoice.merge(purchase_agg_df, on="PONumber", how="left").rename(
        columns={"Quantity": "invoice_quantity", "Dollars": "invoice_dollars"}
    )

    # Heuristic label aligned with the notebook approach.
    df["flag_invoice"] = (
        (df["invoice_dollars"].sub(df["total_item_dollars"]).abs() > 5)
        | (df["avg_receiving_delay"] > 10)
    ).astype(int)

    numeric_cols = FEATURE_COLUMNS + ["flag_invoice"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split feature matrix and target column."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["flag_invoice"].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/test split with class stratification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
