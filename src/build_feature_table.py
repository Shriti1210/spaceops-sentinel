from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

BASE = Path("data/raw/oscd/Onera Satellite Change Detection dataset - Images")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_image(path):
    return np.array(Image.open(path))


def extract_features(img1, img2):
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    diff_gray = diff.mean(axis=2)

    diff_norm = (diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min() + 1e-8)
    threshold = np.percentile(diff_norm, 95)
    change_mask = (diff_norm >= threshold).astype(np.uint8)

    features = {
        "img1_mean": float(img1.mean()),
        "img2_mean": float(img2.mean()),
        "img1_std": float(img1.std()),
        "img2_std": float(img2.std()),
        "diff_mean": float(diff_gray.mean()),
        "diff_std": float(diff_gray.std()),
        "change_ratio": float(change_mask.mean()),
        "diff_max": float(diff_gray.max()),
        "diff_min": float(diff_gray.min())
    }
    return features


def main():
    rows = []

    for city in sorted(BASE.iterdir()):
        pair_folder = city / "pair"
        img1_path = pair_folder / "img1.png"
        img2_path = pair_folder / "img2.png"

        if img1_path.exists() and img2_path.exists():
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)

            if img1.shape != img2.shape:
                print(f"Skipping {city.name}: shape mismatch")
                continue

            features = extract_features(img1, img2)
            features["city"] = city.name
            rows.append(features)

    df = pd.DataFrame(rows)
    out_path = PROCESSED / "feature_table.csv"
    df.to_csv(out_path, index=False)

    print("Saved feature table to:", out_path)
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()