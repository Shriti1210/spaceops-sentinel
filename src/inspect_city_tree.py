from pathlib import Path

base = Path("data/raw/oscd/Onera Satellite Change Detection dataset - Images/abudhabi")

print("City folder exists:", base.exists())
print("\nImmediate items:")

for item in base.iterdir():
    print("-", item.name, "(DIR)" if item.is_dir() else "(FILE)")
    
print("\nRecursive view of files:")
for path in base.rglob("*"):
    if path.is_file():
        print(path.relative_to(base))