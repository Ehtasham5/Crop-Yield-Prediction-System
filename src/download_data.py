from pathlib import Path
from shutil import copy2

import kagglehub


DATASET_SLUG = "miadul/smart-crop-yield-predication-dataset"
RAW_DATA_DIR = Path("data/raw")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_files = list(dataset_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset: {dataset_path}")

    source = csv_files[0]
    destination = RAW_DATA_DIR / source.name
    copy2(source, destination)
    print(f"Dataset copied to {destination}")


if __name__ == "__main__":
    main()
