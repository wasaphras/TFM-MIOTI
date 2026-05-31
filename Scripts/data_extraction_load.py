"""
Data extraction and loading for the MultiEURLEX dataset.
Downloads from Hugging Face if not present, loads JSONL into DataFrame.
"""

import json
import random
import tarfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config


def extract_and_download() -> None:
    """
    Download MultiEURLEX from Hugging Face and save to Data/train.jsonl if not already present.
    Skips download if train.jsonl already exists.
    """
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    out_path = config.TRAIN_JSONL

    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"Data already exists at {out_path}. Using existing file ({count} documents).")
        return

    if not config.HF_TOKEN:
        print("HF_TOKEN is not set; set it if Hugging Face download fails.")

    print("Downloading MultiEURLEX archive from Hugging Face...")
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    tar_path = hf_hub_download(
        repo_id=config.HF_REPO_ID,
        filename="data/multi_eurlex.tar.gz",
        repo_type="dataset",
        token=config.HF_TOKEN or None,
    )

    summary = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            base_name = member.name.split("/")[-1]
            if base_name not in config.TAR_SPLIT_FILES:
                continue

            print(f"Scanning {base_name} to sample {config.DOC_LIMIT} random English documents...")
            valid_indices = []
            f = tar.extractfile(member)
            if f is None:
                continue

            all_lines = f.readlines()
            for i, line in enumerate(all_lines):
                data = json.loads(line.decode("utf-8"))
                if isinstance(data.get("text"), dict) and "en" in data["text"]:
                    valid_indices.append(i)

            sample_size = min(config.DOC_LIMIT, len(valid_indices))
            sampled_indices = set(random.sample(valid_indices, sample_size))

            out_path = config.DATA_DIR / config.TAR_SPLIT_FILES[base_name]
            count = 0

            with open(out_path, "w", encoding="utf-8") as out_f:
                pbar = tqdm(desc=f"Writing {base_name}", total=sample_size, unit=" docs")
                for i, line in enumerate(all_lines):
                    if i in sampled_indices:
                        data = json.loads(line.decode("utf-8"))
                        record = {
                            "celex_id": data.get("celex_id", ""),
                            "text": data["text"]["en"],
                            "labels": data.get("eurovoc_concepts", {}).get(
                                config.LABEL_LEVEL, data.get("labels", [])
                            ),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        pbar.update(1)
                pbar.close()

            summary[base_name.replace(".jsonl", "")] = count
            print(f"  {out_path.name}: {count} documents saved randomly.")

    summary_meta = {
        "dataset": config.HF_REPO_ID,
        "language": "en",
        "label_level": config.LABEL_LEVEL,
        "splits": summary,
    }
    with open(config.DATA_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    print("Random sampling and extraction complete.")


def load_data(limit: int | None = None) -> pd.DataFrame:
    """
    Load the train.jsonl file into a pandas DataFrame.
    Skips empty lines to avoid parser errors.
    If limit is set, returns only the first N rows (for testing).
    """
    with open(config.TRAIN_JSONL, "r", encoding="utf-8") as f:
        rows = []
        for line in tqdm(f, desc="Load train.jsonl", unit=" lines"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    if limit is not None:
        df = df.head(limit)
    return df


def main() -> pd.DataFrame:
    """Run extraction (if needed) and load data. Returns DataFrame."""
    extract_and_download()
    return load_data()


if __name__ == "__main__":
    df = main()
    print(f"Loaded {len(df)} documents. Shape: {df.shape}")
