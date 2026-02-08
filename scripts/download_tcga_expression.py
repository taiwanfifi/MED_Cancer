#!/usr/bin/env python3
"""
Download TCGA Pan-Cancer Gene Expression Data via GDC API
=========================================================
Downloads RNA-Seq STAR-Counts data for multiple cancer types.
Uses the GDC API to query and download expression quantification files.

Supports checkpoint/resume - can be safely interrupted.
"""

import os
import json
import gzip
import logging
import argparse
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "tcga"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / "download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"


def get_expression_file_manifest(cancer_type):
    """Get list of expression files from GDC for a cancer type."""
    manifest_file = DATA_DIR / f"{cancer_type}_manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            return json.load(f)

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {
                "field": "cases.project.project_id",
                "value": [f"TCGA-{cancer_type}"]
            }},
            {"op": "in", "content": {
                "field": "data_category",
                "value": ["Transcriptome Profiling"]
            }},
            {"op": "in", "content": {
                "field": "data_type",
                "value": ["Gene Expression Quantification"]
            }},
            {"op": "in", "content": {
                "field": "analysis.workflow_type",
                "value": ["STAR - Counts"]
            }},
            {"op": "in", "content": {
                "field": "access",
                "value": ["open"]
            }},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "2000"
    }

    response = requests.get(GDC_FILES, params=params, timeout=60)
    data = response.json()

    manifest = []
    for hit in data["data"]["hits"]:
        case_id = "unknown"
        sample_type = "unknown"
        if hit.get("cases"):
            case_id = hit["cases"][0].get("submitter_id", "unknown")
            samples = hit["cases"][0].get("samples", [{}])
            if samples:
                sample_type = samples[0].get("sample_type", "unknown")

        manifest.append({
            "file_id": hit["file_id"],
            "file_name": hit.get("file_name", ""),
            "file_size": hit.get("file_size", 0),
            "case_id": case_id,
            "sample_type": sample_type,
        })

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[{cancer_type}] Manifest: {len(manifest)} files")
    return manifest


def download_single_expression_file(file_id):
    """Download a single expression file from GDC."""
    try:
        response = requests.get(f"{GDC_DATA}/{file_id}",
                               headers={"Content-Type": "application/json"},
                               timeout=120)
        if response.status_code == 200:
            content = response.content

            # Try to parse as gzipped TSV
            try:
                text = gzip.decompress(content).decode('utf-8')
            except:
                text = content.decode('utf-8')

            # Parse the STAR-Counts format
            lines = text.strip().split('\n')

            # Skip comment lines
            data_lines = [l for l in lines if not l.startswith('#') and not l.startswith('N_')]
            if not data_lines:
                return None

            # Parse header
            header_line = None
            data_start = 0
            for i, line in enumerate(data_lines):
                parts = line.split('\t')
                if parts[0] == 'gene_id' or parts[0].startswith('ENSG'):
                    if parts[0] == 'gene_id':
                        header_line = parts
                        data_start = i + 1
                    break

            if header_line:
                records = []
                for line in data_lines[data_start:]:
                    parts = line.split('\t')
                    if len(parts) >= 4 and parts[0].startswith('ENSG'):
                        records.append({
                            'gene_id': parts[0],
                            'gene_name': parts[1] if len(parts) > 1 else '',
                            'tpm': float(parts[6]) if len(parts) > 6 else 0,
                            'fpkm': float(parts[7]) if len(parts) > 7 else 0,
                            'counts': float(parts[3]) if len(parts) > 3 else 0,
                        })
                if records:
                    return pd.DataFrame(records)
            else:
                # Try parsing without header
                records = []
                for line in data_lines:
                    parts = line.split('\t')
                    if len(parts) >= 2 and parts[0].startswith('ENSG'):
                        records.append({
                            'gene_id': parts[0],
                            'gene_name': parts[1] if len(parts) > 1 else '',
                            'tpm': float(parts[6]) if len(parts) > 6 else 0,
                            'fpkm': float(parts[7]) if len(parts) > 7 else 0,
                            'counts': float(parts[3]) if len(parts) > 3 else 0,
                        })
                if records:
                    return pd.DataFrame(records)

    except Exception as e:
        logger.warning(f"Download error for {file_id}: {e}")

    return None


def download_expression_for_cancer(cancer_type, max_samples=None):
    """Download and aggregate expression data for a cancer type."""
    output_file = DATA_DIR / f"{cancer_type}_expression.parquet"
    progress_file = DATA_DIR / f"{cancer_type}_download_progress.json"

    if output_file.exists():
        logger.info(f"[{cancer_type}] Expression data already exists")
        return pd.read_parquet(output_file)

    # Get manifest
    manifest = get_expression_file_manifest(cancer_type)

    # Filter to primary tumor samples
    primary = [m for m in manifest if 'Primary' in m.get('sample_type', '')]
    if not primary:
        primary = manifest  # fallback: use all

    if max_samples:
        primary = primary[:max_samples]

    logger.info(f"[{cancer_type}] Downloading {len(primary)} expression files...")

    # Check progress
    completed = set()
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            completed = set(progress.get("completed", []))

    # Download each file
    all_tpm = {}  # gene_name -> {case_id: tpm_value}

    for entry in tqdm(primary, desc=f"{cancer_type}"):
        file_id = entry["file_id"]
        case_id = entry["case_id"][:12]  # TCGA barcode truncated

        if file_id in completed:
            continue

        df = download_single_expression_file(file_id)
        if df is not None and 'gene_name' in df.columns:
            for _, row in df.iterrows():
                gene = row['gene_name']
                if gene and gene != '':
                    if gene not in all_tpm:
                        all_tpm[gene] = {}
                    all_tpm[gene][case_id] = row['tpm']

            completed.add(file_id)

            # Save progress every 50 files
            if len(completed) % 50 == 0:
                with open(progress_file, 'w') as f:
                    json.dump({"completed": list(completed), "total": len(primary)}, f)
                logger.info(f"[{cancer_type}] Progress: {len(completed)}/{len(primary)}")

    # Build expression matrix
    if all_tpm:
        expr_df = pd.DataFrame(all_tpm).T  # genes x samples
        expr_df.index.name = 'gene'

        # Remove genes with all zeros
        expr_df = expr_df.loc[expr_df.sum(axis=1) > 0]

        expr_df.to_parquet(output_file)
        logger.info(f"[{cancer_type}] Expression matrix: {expr_df.shape} (genes x samples)")

        # Save final progress
        with open(progress_file, 'w') as f:
            json.dump({
                "completed": list(completed),
                "total": len(primary),
                "matrix_shape": list(expr_df.shape),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        return expr_df
    else:
        logger.warning(f"[{cancer_type}] No expression data extracted!")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", type=str, nargs='+',
                       default=["BRCA", "LUAD", "KIRC", "LGG", "STAD"],
                       help="Cancer types to download")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples per cancer type (None=all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TCGA Gene Expression Download")
    logger.info(f"Cancer types: {args.cancer}")
    logger.info("=" * 60)

    for ct in args.cancer:
        try:
            expr_df = download_expression_for_cancer(ct, args.max_samples)
            if expr_df is not None:
                # Quick stats
                logger.info(f"[{ct}] Stats:")
                logger.info(f"  Genes: {expr_df.shape[0]}")
                logger.info(f"  Samples: {expr_df.shape[1]}")
                logger.info(f"  Median TPM: {expr_df.median().median():.2f}")
                logger.info(f"  Top expressed: {expr_df.mean(axis=1).nlargest(5).index.tolist()}")
        except Exception as e:
            logger.error(f"[{ct}] Failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("Download complete!")


if __name__ == "__main__":
    main()
