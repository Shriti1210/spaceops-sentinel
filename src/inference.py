from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image


# ==============================
# PROJECT ROOT
# ==============================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = ROOT / "data/processed/feature_table.csv"
LABEL_PATH = ROOT / "data/processed/label_table.csv"

MODEL_PATH = ROOT / "models/best_change_model.joblib"
META_PATH = ROOT / "models/model_metadata.json"

# ⭐ DEMO IMAGE DIRECTORY
DEMO_DIR = ROOT / "data/demo_pairs"


# ==============================
# NORMALIZER (VERY IMPORTANT)
# ==============================

def normalize_city_name(name: str):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
    )


# ==============================
# TABLE LOADERS
# ==============================

def load_feature_table():
    return pd.read_csv(FEATURE_PATH)


def load_label_table():
    return pd.read_csv(LABEL_PATH)


def load_artifacts():

    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    return model, metadata


# ==============================
# CITY UTILITIES
# ==============================

def list_cities():

    df = load_feature_table()

    cities = (
        df["city"]
        .dropna()
        .astype(str)
        .map(normalize_city_name)
        .unique()
        .tolist()
    )

    return sorted(cities)


# ⭐⭐⭐ VERY IMPORTANT FUNCTION ⭐⭐⭐
def find_city_pair(city: str):

    city_norm = normalize_city_name(city)

    # direct folder
    city_dir = DEMO_DIR / city_norm

    if city_dir.exists():

        for p in city_dir.iterdir():

            name = p.name.lower()

            if name.startswith("img1"):
                img1 = p

            if name.startswith("img2"):
                img2 = p

        if "img1" in locals() and "img2" in locals():
            return img1, img2

    # fallback recursive search
    for folder in DEMO_DIR.rglob("*"):

        if folder.is_dir():

            if normalize_city_name(folder.name) == city_norm:

                img1 = None
                img2 = None

                for f in folder.iterdir():

                    n = f.name.lower()

                    if n.startswith("img1"):
                        img1 = f

                    if n.startswith("img2"):
                        img2 = f

                if img1 and img2:
                    return img1, img2

    return None, None


def load_image(path: Path):
    return np.array(Image.open(path).convert("RGB"))


# ==============================
# CLASSICAL CHANGE MAP
# ==============================

def compute_change_map(img1, img2):

    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))

    diff_gray = diff.mean(axis=2)

    diff_norm = (diff_gray - diff_gray.min()) / (
        diff_gray.max() - diff_gray.min() + 1e-8
    )

    threshold = np.percentile(diff_norm, 95)

    binary_mask = (diff_norm >= threshold).astype(np.uint8)

    return diff_norm, binary_mask, threshold


# ==============================
# FEATURE / LABEL HELPERS
# ==============================

def get_city_row(city: str):

    df = load_feature_table()

    df["city"] = df["city"].astype(str).map(normalize_city_name)

    row = df[df["city"] == normalize_city_name(city)]

    if row.empty:
        return None

    return row.iloc[0].to_dict()


def get_city_label(city: str):

    df = load_label_table()

    df["city"] = df["city"].astype(str).map(normalize_city_name)

    row = df[df["city"] == normalize_city_name(city)]

    if row.empty:
        return None

    return row.iloc[0].to_dict()


# ==============================
# MODEL PREDICTION
# ==============================

def predict_city(city: str):

    model, metadata = load_artifacts()

    feature_cols = metadata["feature_columns"]

    row = get_city_row(city)

    if row is None:
        return None

    X = pd.DataFrame([{col: row[col] for col in feature_cols}])

    pred = float(model.predict(X)[0])

    return pred


# ==============================
# ADVANCED RISK SCORE
# ==============================

def compute_risk_score(city):

    row = get_city_row(city)

    if row is None:
        return 0

    base = row["change_ratio"] * 100
    variability = row["diff_std"]

    risk = base + variability * 0.5

    return round(min(100, risk), 2)