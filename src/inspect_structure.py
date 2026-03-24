from pathlib import Path

base = Path("data/raw/oscd")
images_dir = base / "Onera Satellite Change Detection dataset - Images"
train_labels_dir = base / "Onera Satellite Change Detection dataset - Train Labels"
test_labels_dir = base / "Onera Satellite Change Detection dataset - Test Labels"

folders = {
    "images": images_dir,
    "train_labels": train_labels_dir,
    "test_labels": test_labels_dir,
}

for name, folder in folders.items():
    print(f"\n{name.upper()}:")
    print("Exists:", folder.exists())
    if folder.exists():
        items = list(folder.iterdir())
        print("Number of items:", len(items))
        for item in items[:10]:
            print(" -", item.name)