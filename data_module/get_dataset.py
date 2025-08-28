#!/usr/bin/env python
"""
Download and convert the google/fleurs (en_us) dataset to CSV.
Usage:
    pip install datasets[soundfile] pandas
    huggingface-cli login
    python get_dataset.py
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Audio

def save_split_to_csv(dataset, out_path):
    rows = [{
        "audio_file": r["audio"]["path"],
        "transcription": r["transcription"],
        "duration": len(r["audio"]["array"]) / 16_000
    } for r in dataset]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def main():
    out_dir = Path("data/fleurs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "google/fleurs", "en_us",
        split={"train": "train", "validation": "validation", "test": "test"},
        trust_remote_code=True
    )

    for split in ds:
        ds[split] = ds[split].cast_column("audio", Audio(sampling_rate=16_000))
        save_split_to_csv(ds[split], out_dir / f"{split}.csv")

if __name__ == "__main__":
    main()
