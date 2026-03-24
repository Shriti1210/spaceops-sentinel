from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# XGBoost is optional, but we expect it to be installed
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = PROCESSED_DIR / "feature_table.csv"
LABEL_PATH = PROCESSED_DIR / "label_table.csv"


def evaluate_model(name, model, X_test, y_test):
    """
    Train-prediction evaluation helper.
    Returns metrics and predictions.
    """
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name} results:")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2  : {r2:.6f}")

    return {
        "name": name,
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "preds": preds
    }


def main():
    if not FEATURE_PATH.exists():
        print("Feature table not found:", FEATURE_PATH)
        return

    if not LABEL_PATH.exists():
        print("Label table not found:", LABEL_PATH)
        return

    features_df = pd.read_csv(FEATURE_PATH)
    labels_df = pd.read_csv(LABEL_PATH)

    print("Feature table shape:", features_df.shape)
    print("Label table shape:", labels_df.shape)

    # Merge on city
    df = features_df.merge(labels_df[["city", "gt_change_ratio"]], on="city", how="inner")

    print("Merged dataset shape:", df.shape)
    print("\nMerged preview:")
    print(df.head())

    if df.shape[0] < 5:
        print("Not enough merged rows to train a meaningful model.")
        return

    # Prepare X and y
    drop_cols = ["city", "mask_file", "gt_change_ratio"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["gt_change_ratio"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print("\nTraining rows:", X_train.shape[0])
    print("Testing rows :", X_test.shape[0])
    print("Features     :", feature_cols)

    results = []

    # Model 1: Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_model("RandomForestRegressor", rf, X_test, y_test))

    # Model 2: XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror"
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate_model("XGBRegressor", xgb, X_test, y_test))
    else:
        print("\nXGBoost not available, skipping.")

    # Pick best model by lowest RMSE
    best = min(results, key=lambda x: x["rmse"])

    print(f"\nBest model: {best['name']}")
    print(f"Best RMSE : {best['rmse']:.6f}")

    # Save model
    model_path = MODELS_DIR / "best_change_model.joblib"
    joblib.dump(best["model"], model_path)

    # Save metadata
    metadata = {
        "model_name": best["name"],
        "feature_columns": feature_cols
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved model to:", model_path)
    print("Saved metadata to:", meta_path)


if __name__ == "__main__":
    main()