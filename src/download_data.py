import kaggle
import json
import requests
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Clinical NER — Multi-Dataset Download")
print("=" * 60)

# ── Dataset 1: Already downloaded ─────────────────────────
print("\n[1/3] Corona2.json — already downloaded ✓")
with open(RAW / "Corona2.json") as f:
    d1 = json.load(f)
print(f"      Examples: {len(d1['examples'])}")

# ── Dataset 2: Kaggle NER Medical ─────────────────────────
print("\n[2/3] Downloading additional Kaggle NER dataset...")
try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "benroshan/medical-ner-datasets",
        path=str(RAW),
        unzip=True,
        quiet=False,
    )
    print("      ✓ Downloaded!")
except Exception as e:
    print(f"      Skipped: {e}")

# ── Dataset 3: MTSamples clinical notes ───────────────────
print("\n[3/3] Downloading MTSamples clinical NER...")
try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "tboyle10/medicaltranscriptions",
        path=str(RAW),
        unzip=True,
        quiet=False,
    )
    print("      ✓ Downloaded!")
except Exception as e:
    print(f"      Skipped: {e}")

# ── Check all files ────────────────────────────────────────
print(f"\n{'='*60}")
print("Files in data/raw/:")
for f in sorted(RAW.glob("*")):
    size = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size:.1f} KB")
print(f"{'='*60}")