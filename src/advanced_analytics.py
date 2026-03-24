from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from inference import load_feature_table, load_artifacts


def get_feature_importance():
    """
    Returns feature importance for tree-based models like Random Forest.
    """
    model, metadata = load_artifacts()
    feature_cols = metadata["feature_columns"]

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def get_leaderboard(top_n=5):
    """
    Rank cities by predicted change severity and risk score.
    """
    features_df = load_feature_table()
    model, metadata = load_artifacts()
    feature_cols = metadata["feature_columns"]

    rows = []
    for _, row in features_df.iterrows():
        city = row["city"]
        X = pd.DataFrame([{col: row[col] for col in feature_cols}])
        predicted = float(model.predict(X)[0])

        change_ratio = float(row.get("change_ratio", 0.0))
        diff_std = float(row.get("diff_std", 0.0))

        risk_score = round(min(100.0, (predicted * 100) + (change_ratio * 50) + (diff_std * 0.4)), 2)

        rows.append({
            "city": city,
            "predicted_change_ratio": round(predicted, 6),
            "change_ratio": round(change_ratio, 6),
            "diff_std": round(diff_std, 6),
            "risk_score": risk_score
        })

    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df.head(top_n), df.tail(top_n)


def extract_regions(binary_mask, min_area=150):
    """
    Detect connected anomaly regions from a binary change mask.
    Returns region boxes, areas, and centroids.
    """
    labeled = label(binary_mask)
    regions = []

    for region in regionprops(labeled):
        if region.area < min_area:
            continue

        minr, minc, maxr, maxc = region.bbox
        regions.append({
            "area": int(region.area),
            "bbox": (int(minr), int(minc), int(maxr), int(maxc)),
            "centroid_row": float(region.centroid[0]),
            "centroid_col": float(region.centroid[1])
        })

    regions = sorted(regions, key=lambda x: x["area"], reverse=True)
    return regions