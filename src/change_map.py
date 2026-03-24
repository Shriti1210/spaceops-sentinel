from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("data/raw/oscd/Onera Satellite Change Detection dataset - Images")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def get_first_city_pair():
    for city in sorted(BASE.iterdir()):
        pair_folder = city / "pair"
        img1 = pair_folder / "img1.png"
        img2 = pair_folder / "img2.png"
        if img1.exists() and img2.exists():
            return city.name, img1, img2
    return None, None, None


def load_image(path):
    return np.array(Image.open(path))


def compute_change_map(img1, img2):
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    diff_gray = diff.mean(axis=2)
    diff_norm = (diff_gray - diff_gray.min()) / (diff_gray.max() - diff_gray.min() + 1e-8)

    threshold = np.percentile(diff_norm, 95)
    binary_mask = (diff_norm >= threshold).astype(np.uint8)

    return diff_norm, binary_mask, threshold


def main():
    city, img1_path, img2_path = get_first_city_pair()
    if city is None:
        print("No pair found.")
        return

    print("City:", city)
    print("Image 1:", img1_path)
    print("Image 2:", img2_path)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    print("Shape img1:", img1.shape)
    print("Shape img2:", img2.shape)

    diff_norm, binary_mask, threshold = compute_change_map(img1, img2)

    print("Change threshold:", threshold)
    print("Changed pixel ratio:", binary_mask.mean())

    # Save outputs
    np.save(PROCESSED / f"{city}_diff_map.npy", diff_norm)
    np.save(PROCESSED / f"{city}_change_mask.npy", binary_mask)

    # Visualize
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.title("Time 1")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(diff_norm, cmap="gray")
    plt.title("Change Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Binary Change Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()