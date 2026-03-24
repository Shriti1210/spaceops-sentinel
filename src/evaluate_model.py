from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "data/processed/feature_table.csv"
LABEL_PATH = ROOT / "data/processed/label_table.csv"
MODEL_PATH = ROOT / "models/best_change_model.joblib"
META_PATH = ROOT / "models/model_metadata.json"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    features_df = pd.read_csv(FEATURE_PATH)
    labels_df = pd.read_csv(LABEL_PATH)

    df = features_df.merge(labels_df[["city", "gt_change_ratio"]], on="city", how="inner")

    if df.empty:
        print("Merged dataframe is empty. Cannot evaluate.")
        return

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]
    model = joblib.load(MODEL_PATH)

    X = df[feature_cols]
    y_true = df["gt_change_ratio"]
    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("Evaluation on merged training labels")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2  : {r2:.6f}")

    out_df = df[["city", "gt_change_ratio"]].copy()
    out_df["predicted_change_ratio"] = y_pred
    out_df["absolute_error"] = np.abs(out_df["gt_change_ratio"] - out_df["predicted_change_ratio"])

    out_csv = REPORTS_DIR / "predictions_report.csv"
    out_df.to_csv(out_csv, index=False)
    print("Saved prediction report to:", out_csv)

    # Scatter plot: actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
    plt.xlabel("Actual Change Ratio")
    plt.ylabel("Predicted Change Ratio")
    plt.title("Actual vs Predicted")
    scatter_path = REPORTS_DIR / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.show()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=10)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    resid_path = REPORTS_DIR / "residual_distribution.png"
    plt.tight_layout()
    plt.savefig(resid_path, dpi=150)
    plt.show()

    print("Saved plots to:", scatter_path)
    print("Saved plots to:", resid_path)


if __name__ == "__main__":
    main()