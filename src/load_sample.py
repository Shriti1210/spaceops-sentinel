from pathlib import Path
import rasterio
import matplotlib.pyplot as plt

BASE_DIR = Path("data/raw/oscd")
IMAGES_DIR = BASE_DIR / "Onera Satellite Change Detection dataset - Images"
TRAIN_LABELS_DIR = BASE_DIR / "Onera Satellite Change Detection dataset - Train Labels"


def list_first_city(folder: Path):
    """Return the first city folder inside a given parent folder."""
    for item in sorted(folder.iterdir()):
        if item.is_dir():
            return item
    return None


def find_files(city_folder: Path):
    """List all files inside one city folder."""
    files = sorted([f for f in city_folder.iterdir() if f.is_file()])
    return files


def read_raster(path: Path):
    """Read a satellite image file using rasterio."""
    with rasterio.open(path) as src:
        img = src.read()  # shape: (bands, height, width)
    return img


def show_band_image(img, title="Image"):
    """
    Display first 3 bands if available.
    If the image has more than 3 bands, take the first 3 for visualization.
    """
    if img.ndim == 3:
        if img.shape[0] >= 3:
            vis = img[:3, :, :]
            vis = vis.transpose(1, 2, 0)
        else:
            vis = img[0, :, :]
    else:
        vis = img

    plt.figure(figsize=(8, 8))
    plt.imshow(vis.astype("uint8"))
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    city_folder = list_first_city(IMAGES_DIR)
    if city_folder is None:
        print("No city folders found in images directory.")
        return

    print("Using city folder:", city_folder.name)

    files = find_files(city_folder)
    print("\nFiles in this city folder:")
    for f in files[:10]:
        print("-", f.name)

    # Try to detect image files
    image_files = [f for f in files if f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]]

    if len(image_files) < 2:
        print("\nNot enough image files found in this city folder.")
        return

    img1_path = image_files[0]
    img2_path = image_files[1]

    print("\nReading:")
    print("Image 1:", img1_path.name)
    print("Image 2:", img2_path.name)

    img1 = read_raster(img1_path)
    img2 = read_raster(img2_path)

    print("\nImage 1 shape:", img1.shape)
    print("Image 2 shape:", img2.shape)

    # Display first image for now
    show_band_image(img1, title=f"{city_folder.name} - Image 1")


if __name__ == "__main__":
    main()