from pathlib import Path

base = Path("data/raw/oscd")

print("Base folder exists:", base.exists())
print("\nTop-level folders/files:")

for item in base.iterdir():
    print("-", item.name)