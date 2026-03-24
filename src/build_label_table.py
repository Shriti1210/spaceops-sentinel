from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

TRAIN_LABELS_DIR = Path("data/raw/oscd/Onera Satellite Change Detection dataset - Train Labels")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def find_mask_file(city_folder: Path):
    """
    Find the first likely mask image inside a city folder.
    We search recursively because OSCD folders can contain nested files.
    """
    candidates = [
        p for p in city_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not candidates:
        return None

    # Prefer files that look like masks if possible
    def sort_key(p):
        name = p.name.lower()
        score = 0
        if "mask" in name:
            score -= 3
        if "cm" in name:
            score -= 2
        if p.suffix.lower() == ".png":
            score += 0
        return (score, len(str(p)))

    candidates.sort(key=sort_key)
    return candidates[0]


def load_mask(mask_path: Path):
    """
    Load a mask image and convert it into a binary array:
    0 = no change
    1 = change
    """
    mask = np.array(Image.open(mask_path))

    if mask.ndim == 3:
        mask = mask[..., 0]

    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask


def main():
    rows = []

    if not TRAIN_LABELS_DIR.exists():
        print("Train labels folder not found:", TRAIN_LABELS_DIR)
        return

    for city_folder in sorted(TRAIN_LABELS_DIR.iterdir()):
        if not city_folder.is_dir():
            continue

        mask_file = find_mask_file(city_folder)
        if mask_file is None:
            print(f"Skipping {city_folder.name}: no mask file found")
            continue

        mask = load_mask(mask_file)
        total_pixels = mask.size
        changed_pixels = int(mask.sum())
        gt_change_ratio = changed_pixels / total_pixels if total_pixels else 0

        rows.append({
            "city": city_folder.name,
            "mask_file": str(mask_file),
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "gt_change_ratio": gt_change_ratio
        })

        print(f"Processed {city_folder.name} -> change ratio: {gt_change_ratio:.6f}")

    df = pd.DataFrame(rows)
    out_path = PROCESSED_DIR / "label_table.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved label table to:", out_path)
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()