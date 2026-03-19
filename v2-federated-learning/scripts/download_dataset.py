"""Download / copy the diabetes dataset into v2-federated-learning/data/raw/.

Priority order:
1. If the file already exists → skip.
2. If V1 project has the file → copy from there.
3. Print instructions for manual download from Kaggle.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATASET_PATH, V1_RAW_DATA_PATH, RAW_DATA_DIR


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATASET_PATH.exists():
        print(f"[✓] Dataset already present: {RAW_DATASET_PATH}")
        return

    if V1_RAW_DATA_PATH.exists():
        shutil.copy(V1_RAW_DATA_PATH, RAW_DATASET_PATH)
        print(f"[✓] Copied dataset from V1: {RAW_DATASET_PATH}")
        return

    print(
        "[!] Dataset not found.\n"
        "Please download 'diabetes_binary.csv' from Kaggle:\n"
        "  https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset\n"
        f"and place it at:\n  {RAW_DATASET_PATH}"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
