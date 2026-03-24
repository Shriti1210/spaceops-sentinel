from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("data/raw/oscd/Onera Satellite Change Detection dataset - Images")

def get_first_city_pair():
    for city in sorted(BASE.iterdir()):
        pair_folder = city / "pair"
        if pair_folder.exists():
            img1 = pair_folder / "img1.png"
            img2 = pair_folder / "img2.png"
            if img1.exists() and img2.exists():
                return city.name, img1, img2
    return None, None, None


def load_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img


def main():
    city, img1_path, img2_path = get_first_city_pair()

    print("City:", city)
    print("Image1:", img1_path)
    print("Image2:", img2_path)

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    print("Shape img1:", img1.shape)
    print("Shape img2:", img2.shape)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title("Time 1")

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title("Time 2")

    plt.show()


if __name__ == "__main__":
    main()