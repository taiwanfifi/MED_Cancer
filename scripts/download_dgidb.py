#!/usr/bin/env python3
"""Download DGIdb drug-gene interactions."""
import pandas as pd
import requests
import io
from pathlib import Path

DATA_DIR = Path("/workspace/cancer_research/data/drug_repurpose")
DATA_DIR.mkdir(parents=True, exist_ok=True)

output_file = DATA_DIR / "dgidb_interactions.parquet"

urls = [
    "https://dgidb.org/data/monthly_tsvs/2024-Feb/interactions.tsv",
    "https://dgidb.org/data/monthly_tsvs/2023-Oct/interactions.tsv",
    "https://dgidb.org/data/monthly_tsvs/2023-Jan/interactions.tsv",
]

for url in urls:
    try:
        print(f"Trying: {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)
        df.to_parquet(output_file)
        print(f"Success! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        if "drug_name" in df.columns:
            print(f"Unique drugs: {df['drug_name'].nunique()}")
        if "gene_name" in df.columns:
            print(f"Unique genes: {df['gene_name'].nunique()}")
        if "interaction_types" in df.columns:
            print(f"Top interaction types:")
            print(df["interaction_types"].value_counts().head(10))
        break
    except Exception as e:
        print(f"Failed: {e}")

print("Done!")
