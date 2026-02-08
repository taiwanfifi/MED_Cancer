#!/usr/bin/env python3
"""
Experiment 17: External Validation with Independent Cohorts
===========================================================
Validates key findings from TCGA analyses using independent datasets:
- BRCA: GSE20685 (n=327), METABRIC (n=1904) from cBioPortal
- LUAD: GSE31210 (n=226), GSE72094 (n=393)
- KIRC: E-MTAB-1980 (n=101)

Validation targets:
1. SL target expression-survival associations
2. Ferroptosis pathway signature → survival
3. Immune-survival associations
4. Multi-evidence target rankings

Target: Adds to ALL papers — critical for top-tier publication
"""

import json
import logging
import os
import time
import gzip
import tarfile
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
VAL_DIR = DATA_DIR / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE_DIR / "results" / "exp17_external_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Key genes to validate (from our TCGA findings)
SL_TARGETS = ['GPX4', 'FGFR1', 'MDM2', 'PTEN', 'BCL2', 'CDK6', 'MYC', 'CCND1',
              'CDS2', 'CHMP4B', 'GINS4', 'PSMA4']

FERROPTOSIS_GENES = {
    'anti': ['GPX4', 'SLC7A11', 'FSP1', 'GCLC', 'GSS', 'NFE2L2', 'FTH1'],
    'pro': ['ACSL4', 'LPCAT3', 'TFRC', 'NCOA4', 'HMOX1', 'ALOX15'],
}

IMMUNE_SIGNATURES = {
    'B_cells': ['CD19', 'CD79A', 'MS4A1', 'CD79B', 'BLNK'],
    'CD4_T': ['CD4', 'IL7R', 'CD40LG', 'FOXP3', 'IL2RA'],
    'CD8_T': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1'],
    'NK_cells': ['NKG7', 'GNLY', 'KLRB1', 'KLRD1', 'NCR1'],
    'Mast_cells': ['TPSAB1', 'TPSB2', 'CPA3', 'MS4A2', 'FCER1A'],
}


# ============================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================

def download_geo_series_matrix(gse_id):
    """Download GEO series matrix file."""
    import urllib.request

    output_dir = VAL_DIR / gse_id
    output_dir.mkdir(exist_ok=True)

    matrix_file = output_dir / f"{gse_id}_series_matrix.txt.gz"
    if matrix_file.exists():
        logger.info(f"  {gse_id} already downloaded")
        return matrix_file

    # Try standard URL pattern
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id[:-3]}nnn/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
    logger.info(f"  Downloading {gse_id} from GEO...")

    try:
        urllib.request.urlretrieve(url, str(matrix_file))
        logger.info(f"  Downloaded: {matrix_file}")
        return matrix_file
    except Exception as e:
        logger.warning(f"  Failed to download {gse_id}: {e}")
        return None


def parse_geo_series_matrix(matrix_file):
    """Parse GEO series matrix file into expression + clinical DataFrames."""
    if matrix_file is None or not matrix_file.exists():
        return None, None

    logger.info(f"  Parsing {matrix_file.name}...")

    raw_clinical = {}
    characteristics_rows = []
    expr_lines = []
    in_matrix = False
    sample_ids = None

    with gzip.open(str(matrix_file), 'rt') as f:
        for line in f:
            line = line.strip()

            if line.startswith('!Sample_'):
                parts = line.split('\t')
                key = parts[0].replace('!Sample_', '')
                values = [v.strip('"') for v in parts[1:]]

                # Collect characteristics rows separately for key:value parsing
                if 'characteristics' in key:
                    characteristics_rows.append(values)
                else:
                    raw_clinical[key] = values

            elif line.startswith('"ID_REF"'):
                in_matrix = True
                sample_ids = [s.strip('"') for s in line.split('\t')[1:]]
                continue

            elif line.startswith('!series_matrix_table_end'):
                break

            elif in_matrix and line:
                expr_lines.append(line)

    if not expr_lines or not sample_ids:
        return None, None

    # Build expression matrix
    genes = []
    values = []
    for line in expr_lines:
        parts = line.split('\t')
        genes.append(parts[0].strip('"'))
        values.append([float(v) if v and v != 'null' and v != 'NA' else np.nan
                       for v in parts[1:]])

    expr_df = pd.DataFrame(values, index=genes, columns=sample_ids)
    logger.info(f"  Expression: {expr_df.shape}")

    # Build clinical DataFrame from characteristics (key: value format)
    n_samples = len(sample_ids)
    parsed_clinical = {}

    for row_values in characteristics_rows:
        if not row_values or len(row_values) != n_samples:
            continue
        # Detect key from first non-empty value
        for val in row_values:
            if ':' in val:
                key = val.split(':')[0].strip().lower().replace(' ', '_')
                break
        else:
            continue

        # Parse all values for this key
        parsed_values = []
        for val in row_values:
            if ':' in val:
                parsed_values.append(':'.join(val.split(':')[1:]).strip())
            else:
                parsed_values.append(val.strip())
        parsed_clinical[key] = parsed_values

    # Add geo_accession
    if 'geo_accession' in raw_clinical:
        parsed_clinical['geo_accession'] = raw_clinical['geo_accession']

    clin_df = pd.DataFrame(parsed_clinical)
    if 'geo_accession' in clin_df.columns:
        clin_df = clin_df.set_index('geo_accession')
    logger.info(f"  Clinical columns: {list(clin_df.columns)}")
    logger.info(f"  Clinical: {clin_df.shape}")

    return expr_df, clin_df


def map_probes_to_genes(expr_df, gse_id):
    """Map Affymetrix probe IDs to gene symbols using GEO platform annotation.

    Supports GPL570 (Affy U133 Plus 2.0) and GPL15048 (Merck/Rosetta custom).
    For probes mapping to multiple genes, keeps first gene. Averages duplicates.
    """
    import urllib.request

    # Check if already gene symbols (if first index looks like a gene name, not a probe ID)
    first_idx = str(expr_df.index[0])
    if not first_idx.replace('_', '').replace('s', '').replace('x', '').replace('at', '').isdigit():
        # Looks like gene symbols already
        if any(c.isalpha() for c in first_idx) and '_at' not in first_idx:
            logger.info("  Expression already uses gene symbols")
            return expr_df

    annot_dir = VAL_DIR / "annotations"
    annot_dir.mkdir(exist_ok=True)

    # Detect platform: if probes start with "merck-", use GPL15048
    sample_probes = [str(p) for p in expr_df.index[:100]]
    is_merck = any(p.startswith('merck-') for p in sample_probes)

    if is_merck:
        logger.info("  Detected Merck/Rosetta probes → using GPL15048 annotation")
        annot_file = annot_dir / "GPL15048_probe2gene.tsv"
        if annot_file.exists():
            annot_df = pd.read_csv(annot_file, sep='\t')
            probe_to_gene = dict(zip(annot_df['ID'], annot_df['Gene Symbol']))
            # Remove None/empty
            probe_to_gene = {k: v for k, v in probe_to_gene.items()
                            if v and str(v) != 'None' and str(v) != 'nan'}
            logger.info(f"  GPL15048: {len(probe_to_gene)} probe-to-gene mappings")
        else:
            logger.warning("  GPL15048 annotation not found, skipping probe mapping")
            return expr_df
    else:
        # Standard GPL570 path
        annot_file = annot_dir / "GPL570_annot.txt.gz"

        if not annot_file.exists():
            logger.info("  Downloading GPL570 probe annotation...")
            url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
            try:
                urllib.request.urlretrieve(url, str(annot_file))
            except Exception:
                logger.warning("  Could not download annotation, skipping probe mapping")
                return expr_df

        # Parse GPL570 annotation file
        probe_to_gene = {}
        try:
            with gzip.open(str(annot_file), 'rt', errors='ignore') as f:
                header = None
                gene_col_idx = None
                for line in f:
                    if line.startswith('#') or line.startswith('^') or line.startswith('!'):
                        continue
                    if not line.strip():
                        continue

                    parts = line.strip().split('\t')

                    if header is None:
                        header = parts
                        for i, h in enumerate(header):
                            h_lower = h.lower().strip()
                            if h_lower in ['gene symbol', 'gene_symbol', 'genesymbol']:
                                gene_col_idx = i
                                break
                        if gene_col_idx is None:
                            for i, h in enumerate(header):
                                if 'symbol' in h.lower():
                                    gene_col_idx = i
                                    break
                        if gene_col_idx is None:
                            logger.warning(f"  No gene symbol column. Headers: {header[:15]}")
                            return expr_df
                        logger.info(f"  Gene Symbol at col {gene_col_idx}")
                        continue

                    if gene_col_idx is not None and len(parts) > gene_col_idx:
                        probe_id = parts[0].strip()
                        gene_symbol = parts[gene_col_idx].strip()
                        if gene_symbol and gene_symbol != '---' and gene_symbol != '':
                            gene_symbol = gene_symbol.split('///')[0].strip()
                            probe_to_gene[probe_id] = gene_symbol
        except Exception as e:
            logger.warning(f"  Error parsing annotation: {e}")
            return expr_df

    logger.info(f"  Loaded {len(probe_to_gene)} probe-to-gene mappings")

    # Map probes to genes
    mapped_genes = []
    for probe in expr_df.index:
        gene = probe_to_gene.get(probe, None)
        mapped_genes.append(gene)

    expr_df = expr_df.copy()
    expr_df['gene_symbol'] = mapped_genes
    expr_df = expr_df.dropna(subset=['gene_symbol'])
    expr_df = expr_df[expr_df['gene_symbol'] != '']

    # Average expression for duplicate genes
    expr_df = expr_df.groupby('gene_symbol').mean()

    logger.info(f"  After probe mapping: {expr_df.shape} (probes → gene symbols)")
    return expr_df


def download_metabric():
    """Download METABRIC data from cBioPortal."""
    import urllib.request

    output_dir = VAL_DIR / "metabric"
    output_dir.mkdir(exist_ok=True)

    tar_file = output_dir / "brca_metabric.tar.gz"

    if (output_dir / "brca_metabric").exists():
        logger.info("  METABRIC already downloaded")
        return output_dir / "brca_metabric"

    # Try multiple download URLs
    urls = [
        "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz",
        "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric.tar.gz",
    ]
    logger.info("  Downloading METABRIC from cBioPortal...")

    for url in urls:
        try:
            logger.info(f"    Trying: {url[:60]}...")
            urllib.request.urlretrieve(url, str(tar_file))
            with tarfile.open(str(tar_file), "r:gz") as tar:
                tar.extractall(str(output_dir))
            logger.info("  METABRIC downloaded and extracted")
            return output_dir / "brca_metabric"
        except Exception as e:
            logger.warning(f"    Failed: {e}")
            continue

    logger.warning("  Could not download METABRIC from any URL")
    return None


def load_metabric(metabric_dir):
    """Load METABRIC expression and clinical data."""
    if metabric_dir is None or not metabric_dir.exists():
        return None, None

    # Clinical data
    clin_file = metabric_dir / "data_clinical_patient.txt"
    if clin_file.exists():
        clin_df = pd.read_csv(clin_file, sep='\t', comment='#')
        clin_df = clin_df.set_index('PATIENT_ID')
        logger.info(f"  METABRIC clinical: {clin_df.shape}")
    else:
        clin_df = None

    # Expression data
    expr_file = metabric_dir / "data_mrna_illumina_microarray.txt"
    if not expr_file.exists():
        expr_file = metabric_dir / "data_mrna_agilent_microarray.txt"
    if not expr_file.exists():
        # Try any expression file
        for f in metabric_dir.glob("data_mrna*.txt"):
            expr_file = f
            break

    if expr_file.exists():
        expr_df = pd.read_csv(expr_file, sep='\t', comment='#')
        if 'Hugo_Symbol' in expr_df.columns:
            expr_df = expr_df.set_index('Hugo_Symbol')
            # Drop Entrez column if present
            if 'Entrez_Gene_Id' in expr_df.columns:
                expr_df = expr_df.drop('Entrez_Gene_Id', axis=1)
        logger.info(f"  METABRIC expression: {expr_df.shape}")
    else:
        expr_df = None

    return expr_df, clin_df


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def extract_survival_from_geo(clin_df, gse_id):
    """Extract survival time and event from GEO clinical data."""
    if clin_df is None:
        return None

    surv_data = pd.DataFrame(index=clin_df.index)

    # Map of keywords to find time and event columns (ordered by specificity)
    time_keywords = ['follow_up_duration', 'survival_time', 'days_before_death',
                     'os_time', 'os_months', 'os_years', 'time_to_death',
                     'time_to_event', 'overall_survival', 'rfs_time', 'dfs_time',
                     'followup', 'follow_up']
    event_keywords = ['event_death', 'vital_status', 'os_event', 'os_status',
                      'event_os', 'death']
    # Note: 'status' alone excluded to avoid matching 'smoking_status' etc.

    # Search columns for time
    time_col = None
    for col in clin_df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in time_keywords):
            time_col = col
            break

    # Search columns for event
    event_col = None
    for col in clin_df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in event_keywords):
            event_col = col
            break

    if time_col is None or event_col is None:
        logger.info(f"  [{gse_id}] Could not find survival columns. Available: {list(clin_df.columns)}")
        return None

    logger.info(f"  [{gse_id}] Using time={time_col}, event={event_col}")

    # Parse time values
    surv_data['time'] = pd.to_numeric(clin_df[time_col], errors='coerce')

    # Parse event values (handle 0/1, dead/alive, yes/no)
    event_vals = clin_df[event_col].astype(str).str.strip().str.lower()
    surv_data['event'] = event_vals.map(lambda x: 1 if x in ['1', 'dead', 'deceased', 'yes', 'true', 'death']
                                        else (0 if x in ['0', 'alive', 'living', 'no', 'false', 'censored', 'na', '']
                                              else np.nan))

    # If too few patients, try relapse-based survival as fallback
    valid_surv = surv_data.dropna(subset=['time', 'event'])
    if len(valid_surv[valid_surv['time'] > 0]) < 30:
        logger.info(f"  [{gse_id}] Too few OS patients, trying relapse/RFS...")
        rfs_time_kw = ['days_before_relapse', 'months_before_relapse', 'rfs_time', 'dfs_time']
        rfs_event_kw = ['relapse', 'recurrence', 'event_relapse', 'event_metastasis']

        for col in clin_df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in rfs_time_kw):
                surv_data['time'] = pd.to_numeric(clin_df[col], errors='coerce')
                break
        for col in clin_df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in rfs_event_kw):
                event_vals = clin_df[col].astype(str).str.strip().str.lower()
                surv_data['event'] = event_vals.map(
                    lambda x: 1 if x in ['1', 'yes', 'true', 'relapse', 'recurrence']
                    else (0 if x in ['0', 'no', 'false', 'na', ''] else np.nan))
                break

    surv_data = surv_data.dropna(subset=['time', 'event'])
    surv_data['event'] = surv_data['event'].astype(int)
    surv_data = surv_data[surv_data['time'] > 0]

    logger.info(f"  [{gse_id}] Survival data: {len(surv_data)} patients, {int(surv_data['event'].sum())} events")
    return surv_data if len(surv_data) > 20 else None


def extract_survival_metabric(clin_df):
    """Extract survival from METABRIC clinical data."""
    if clin_df is None:
        return None

    surv = pd.DataFrame(index=clin_df.index)

    # METABRIC uses OS_MONTHS and OS_STATUS
    if 'OS_MONTHS' in clin_df.columns:
        surv['time'] = pd.to_numeric(clin_df['OS_MONTHS'], errors='coerce')
    elif 'OVERALL_SURVIVAL_MONTHS' in clin_df.columns:
        surv['time'] = pd.to_numeric(clin_df['OVERALL_SURVIVAL_MONTHS'], errors='coerce')

    if 'OS_STATUS' in clin_df.columns:
        surv['event'] = clin_df['OS_STATUS'].map(
            lambda x: 1 if '1:DECEASED' in str(x) or str(x) == '1' else 0)
    elif 'VITAL_STATUS' in clin_df.columns:
        surv['event'] = clin_df['VITAL_STATUS'].map(
            lambda x: 1 if str(x).lower() in ['dead', 'deceased'] else 0)

    surv = surv.dropna()
    surv = surv[surv['time'] > 0]

    logger.info(f"  [METABRIC] Survival data: {len(surv)} patients, {int(surv['event'].sum())} events")
    return surv if len(surv) > 20 else None


def validate_gene_survival(expr_df, surv_df, genes, dataset_name):
    """Validate gene expression → survival association."""
    results = []

    if expr_df is None or surv_df is None:
        return results

    # Match samples
    common = list(set(expr_df.columns) & set(surv_df.index))
    if len(common) < 30:
        logger.info(f"  [{dataset_name}] Too few common samples: {len(common)}")
        return results

    # Build gene index map (case-insensitive)
    gene_map = {g.upper(): g for g in expr_df.index}

    for gene in genes:
        idx = gene_map.get(gene.upper())
        if idx is None:
            continue

        gene_expr = expr_df.loc[idx, common].astype(float)
        gene_surv = surv_df.loc[common]

        # Remove NaN
        valid = ~gene_expr.isna()
        gene_expr = gene_expr[valid]
        gene_surv = gene_surv.loc[gene_expr.index]

        if len(gene_expr) < 30:
            continue

        # Split by median
        median_val = gene_expr.median()
        high = gene_expr > median_val
        low = ~high

        if high.sum() < 10 or low.sum() < 10:
            continue

        # Log-rank test
        try:
            lr = logrank_test(
                gene_surv.loc[high.values, 'time'],
                gene_surv.loc[low.values, 'time'],
                gene_surv.loc[high.values, 'event'],
                gene_surv.loc[low.values, 'event'],
            )
            p_value = lr.p_value
        except Exception:
            p_value = 1.0

        # Cox regression
        try:
            cox_data = pd.DataFrame({
                'time': gene_surv['time'].values,
                'event': gene_surv['event'].values,
                'expr': gene_expr.values,
            })
            cph = CoxPHFitter()
            cph.fit(cox_data, 'time', 'event')
            hr = np.exp(cph.summary['coef'].values[0])
            cox_p = cph.summary['p'].values[0]
        except Exception:
            hr = 1.0
            cox_p = 1.0

        results.append({
            'dataset': dataset_name,
            'gene': gene,
            'n_patients': len(gene_expr),
            'logrank_p': p_value,
            'hazard_ratio': hr,
            'cox_p': cox_p,
            'high_median_expr': gene_expr[high].median(),
            'low_median_expr': gene_expr[low].median(),
            'significant': p_value < 0.05,
        })

    return results


def validate_ferroptosis_signature(expr_df, surv_df, dataset_name):
    """Validate ferroptosis vulnerability score → survival."""
    if expr_df is None or surv_df is None:
        return None

    common = list(set(expr_df.columns) & set(surv_df.index))
    if len(common) < 30:
        return None

    gene_map = {g.upper(): g for g in expr_df.index}

    # Compute ferroptosis score per sample
    scores = {}
    for sample in common:
        anti = 0
        pro = 0
        n_anti = 0
        n_pro = 0

        for gene in FERROPTOSIS_GENES['anti']:
            idx = gene_map.get(gene.upper())
            if idx is not None:
                val = float(expr_df.loc[idx, sample])
                if not np.isnan(val) and val > 0:
                    anti += val
                    n_anti += 1

        for gene in FERROPTOSIS_GENES['pro']:
            idx = gene_map.get(gene.upper())
            if idx is not None:
                val = float(expr_df.loc[idx, sample])
                if not np.isnan(val) and val > 0:
                    pro += val
                    n_pro += 1

        if n_anti > 0 and n_pro > 0:
            scores[sample] = (pro / n_pro) / (anti / n_anti + 0.01)

    if len(scores) < 30:
        return None

    score_series = pd.Series(scores)
    valid_samples = list(set(score_series.index) & set(surv_df.index))
    score_series = score_series[valid_samples]
    surv = surv_df.loc[valid_samples]

    # Split by median
    median_val = score_series.median()
    high = score_series > median_val

    try:
        lr = logrank_test(
            surv.loc[high.values, 'time'],
            surv.loc[~high.values, 'time'],
            surv.loc[high.values, 'event'],
            surv.loc[~high.values, 'event'],
        )
        p_value = lr.p_value
    except Exception:
        p_value = 1.0

    # Cox
    try:
        cox_data = pd.DataFrame({
            'time': surv['time'].values,
            'event': surv['event'].values,
            'ferro_score': score_series.values,
        })
        cph = CoxPHFitter()
        cph.fit(cox_data, 'time', 'event')
        hr = np.exp(cph.summary['coef'].values[0])
        cox_p = cph.summary['p'].values[0]
    except Exception:
        hr = 1.0
        cox_p = 1.0

    result = {
        'dataset': dataset_name,
        'n_patients': len(valid_samples),
        'logrank_p': p_value,
        'hazard_ratio': hr,
        'cox_p': cox_p,
        'significant': p_value < 0.05,
    }

    logger.info(f"  [{dataset_name}] Ferroptosis validation: n={len(valid_samples)}, "
                f"p={p_value:.4f}, HR={hr:.3f}")
    return result


def validate_immune_signature(expr_df, surv_df, dataset_name):
    """Validate immune cell signatures → survival."""
    if expr_df is None or surv_df is None:
        return []

    common = list(set(expr_df.columns) & set(surv_df.index))
    if len(common) < 30:
        return []

    gene_map = {g.upper(): g for g in expr_df.index}
    results = []

    for cell_type, genes in IMMUNE_SIGNATURES.items():
        # Compute mean z-score of signature genes
        sig_scores = {}
        for sample in common:
            vals = []
            for gene in genes:
                idx = gene_map.get(gene.upper())
                if idx is not None:
                    val = float(expr_df.loc[idx, sample])
                    if not np.isnan(val):
                        vals.append(val)
            if vals:
                sig_scores[sample] = np.mean(vals)

        if len(sig_scores) < 30:
            continue

        score_series = pd.Series(sig_scores)
        valid = list(set(score_series.index) & set(surv_df.index))
        score_series = score_series[valid]
        surv = surv_df.loc[valid]

        median_val = score_series.median()
        high = score_series > median_val

        try:
            lr = logrank_test(
                surv.loc[high.values, 'time'],
                surv.loc[~high.values, 'time'],
                surv.loc[high.values, 'event'],
                surv.loc[~high.values, 'event'],
            )
            p_value = lr.p_value
        except Exception:
            p_value = 1.0

        results.append({
            'dataset': dataset_name,
            'cell_type': cell_type,
            'n_patients': len(valid),
            'logrank_p': p_value,
            'significant': p_value < 0.05,
        })

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_validation_forest(gene_results, dataset_results):
    """Create forest plot comparing TCGA vs validation cohort results."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Panel A: Gene-survival validation forest plot
    ax = axes[0, 0]
    if gene_results:
        gene_df = pd.DataFrame(gene_results)
        # Group by gene, show HR across datasets
        genes_with_data = gene_df.groupby('gene').size()
        genes_with_data = genes_with_data[genes_with_data >= 2].index[:15]

        y_pos = 0
        for gene in genes_with_data:
            gene_data = gene_df[gene_df['gene'] == gene]
            for _, row in gene_data.iterrows():
                hr = row['hazard_ratio']
                color = '#2ecc71' if row['significant'] else '#e74c3c'
                marker = 'o' if 'TCGA' not in row['dataset'] else 's'
                ax.scatter(np.log2(hr), y_pos, c=color, marker=marker, s=80, zorder=5,
                          edgecolor='black', linewidth=0.5)
                ax.text(np.log2(hr) + 0.1, y_pos, row['dataset'][:10], fontsize=6, va='center')
                y_pos += 0.5
            y_pos += 0.5

        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('log2(Hazard Ratio)')
        ax.set_title('A) Gene Expression → Survival (Cross-Dataset)', fontweight='bold')

    # Panel B: Ferroptosis validation
    ax = axes[0, 1]
    if dataset_results.get('ferroptosis'):
        ferro_data = dataset_results['ferroptosis']
        datasets = [d['dataset'] for d in ferro_data]
        p_values = [-np.log10(d['logrank_p'] + 1e-10) for d in ferro_data]
        colors = ['#2ecc71' if d['significant'] else '#e74c3c' for d in ferro_data]

        bars = ax.barh(range(len(datasets)), p_values, color=colors, edgecolor='white')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=9)
        ax.set_xlabel('-log10(p-value)')
        ax.set_title('B) Ferroptosis Score → Survival Validation', fontweight='bold')
        ax.legend()

    # Panel C: Immune signature validation
    ax = axes[1, 0]
    if dataset_results.get('immune'):
        imm_df = pd.DataFrame(dataset_results['immune'])
        if len(imm_df) > 0:
            pivot = imm_df.pivot_table(index='cell_type', columns='dataset',
                                        values='logrank_p', fill_value=1.0)
            # -log10 transform
            pivot_log = -np.log10(pivot + 1e-10)
            sns.heatmap(pivot_log, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax,
                       linewidths=0.5, vmin=0, vmax=5)
            ax.set_title('C) Immune Signature → Survival (-log10 p)', fontweight='bold')

    # Panel D: Validation summary
    ax = axes[1, 1]
    if gene_results:
        gene_df = pd.DataFrame(gene_results)
        # Count validated vs not per dataset
        summary = gene_df.groupby('dataset').agg(
            total=('gene', 'count'),
            validated=('significant', 'sum'),
        ).reset_index()
        summary['validation_rate'] = summary['validated'] / summary['total']

        bars = ax.bar(range(len(summary)), summary['validation_rate'],
                     color='#3498db', alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary['dataset'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Validation Rate')
        ax.set_ylim(0, 1)
        for bar, rate, total in zip(bars, summary['validation_rate'], summary['total']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{rate:.0%}\n(n={total})', ha='center', fontsize=9)
        ax.set_title('D) Overall Validation Rate by Dataset', fontweight='bold')

    plt.suptitle('External Validation of TCGA Findings',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: validation_landscape.png")


def plot_km_validation(expr_df, surv_df, gene, dataset_name, ax):
    """Plot KM curve for a single gene in a validation dataset."""
    gene_map = {g.upper(): g for g in expr_df.index}
    idx = gene_map.get(gene.upper())
    if idx is None:
        return

    common = list(set(expr_df.columns) & set(surv_df.index))
    gene_expr = expr_df.loc[idx, common].astype(float)
    surv = surv_df.loc[common]

    valid = ~gene_expr.isna()
    gene_expr = gene_expr[valid]
    surv = surv.loc[gene_expr.index]

    median_val = gene_expr.median()
    high = gene_expr > median_val

    kmf = KaplanMeierFitter()

    kmf.fit(surv.loc[high.values, 'time'], surv.loc[high.values, 'event'],
           label=f'High {gene} (n={high.sum()})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

    kmf.fit(surv.loc[~high.values, 'time'], surv.loc[~high.values, 'event'],
           label=f'Low {gene} (n={(~high).sum()})')
    kmf.plot_survival_function(ax=ax, ci_show=True, color='blue')

    try:
        lr = logrank_test(
            surv.loc[high.values, 'time'], surv.loc[~high.values, 'time'],
            surv.loc[high.values, 'event'], surv.loc[~high.values, 'event'],
        )
        ax.text(0.5, 0.02, f'p={lr.p_value:.4f}', transform=ax.transAxes,
               ha='center', fontsize=10, fontweight='bold')
    except Exception:
        pass

    ax.set_title(f'{gene} ({dataset_name})', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')


# ============================================================
# TCGA BASELINE (for comparison)
# ============================================================

def compute_tcga_baseline():
    """Load our TCGA results as baseline for comparison."""
    tcga_results = []

    # Load from our existing results
    for cancer in ['BRCA', 'LUAD', 'KIRC']:
        expr_file = DATA_DIR / "tcga" / f"{cancer}_expression.parquet"
        if not expr_file.exists():
            continue

        expr = pd.read_parquet(expr_file)

        # Load clinical
        import json as json_mod
        clin_file = DATA_DIR / "tcga" / f"{cancer}_clinical.json"
        if not clin_file.exists():
            continue

        with open(clin_file) as f:
            clin_data = json_mod.load(f)
        clin_df = pd.DataFrame(clin_data)

        # Build survival data
        surv = pd.DataFrame(index=clin_df['case_id'])
        surv['time'] = clin_df.apply(
            lambda r: r['days_to_death'] if pd.notna(r.get('days_to_death')) and r.get('days_to_death', 0) > 0
            else r.get('days_to_follow_up', np.nan), axis=1).values
        surv['event'] = (clin_df['vital_status'].str.lower() == 'dead').astype(int).values
        surv = surv.dropna()
        surv['time'] = pd.to_numeric(surv['time'], errors='coerce')
        surv = surv[surv['time'] > 0]

        # Get tumor samples
        tumor_cols = [c for c in expr.columns if len(c.split('-')) <= 3 or not c.split('-')[-1].startswith('1')]
        expr_tumor = expr[tumor_cols]
        expr_tumor.columns = ['-'.join(c.split('-')[:3]) for c in expr_tumor.columns]

        # Validate genes
        gene_results = validate_gene_survival(expr_tumor, surv, SL_TARGETS, f'TCGA-{cancer}')
        tcga_results.extend(gene_results)

    return tcga_results


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("Experiment 17: External Validation")
    logger.info("=" * 60)

    all_gene_results = []
    dataset_results = {'ferroptosis': [], 'immune': []}

    # Step 0: TCGA baseline
    logger.info("\n--- TCGA Baseline ---")
    tcga_results = compute_tcga_baseline()
    all_gene_results.extend(tcga_results)
    logger.info(f"TCGA baseline: {len(tcga_results)} gene-survival tests")

    # Step 1: Download validation datasets
    logger.info("\n--- Downloading Validation Datasets ---")

    datasets = {}

    # BRCA: GSE20685
    logger.info("\n[GSE20685] Breast cancer validation")
    matrix_file = download_geo_series_matrix("GSE20685")
    expr, clin = parse_geo_series_matrix(matrix_file)
    if expr is not None:
        expr = map_probes_to_genes(expr, "GSE20685")
        surv = extract_survival_from_geo(clin, "GSE20685")
        datasets['GSE20685'] = {'expr': expr, 'surv': surv, 'cancer': 'BRCA'}

    # LUAD: GSE31210
    logger.info("\n[GSE31210] Lung adenocarcinoma validation")
    matrix_file = download_geo_series_matrix("GSE31210")
    expr, clin = parse_geo_series_matrix(matrix_file)
    if expr is not None:
        expr = map_probes_to_genes(expr, "GSE31210")
        surv = extract_survival_from_geo(clin, "GSE31210")
        datasets['GSE31210'] = {'expr': expr, 'surv': surv, 'cancer': 'LUAD'}

    # LUAD: GSE72094
    logger.info("\n[GSE72094] Lung adenocarcinoma validation (large)")
    matrix_file = download_geo_series_matrix("GSE72094")
    expr, clin = parse_geo_series_matrix(matrix_file)
    if expr is not None:
        expr = map_probes_to_genes(expr, "GSE72094")
        surv = extract_survival_from_geo(clin, "GSE72094")
        datasets['GSE72094'] = {'expr': expr, 'surv': surv, 'cancer': 'LUAD'}

    # BRCA: METABRIC
    logger.info("\n[METABRIC] Breast cancer gold standard")
    metabric_dir = download_metabric()
    expr, clin = load_metabric(metabric_dir)
    if expr is not None:
        surv = extract_survival_metabric(clin)
        datasets['METABRIC'] = {'expr': expr, 'surv': surv, 'cancer': 'BRCA'}

    logger.info(f"\nLoaded {len(datasets)} validation datasets")

    # Step 2: Validate gene-survival associations
    logger.info("\n--- Gene-Survival Validation ---")
    for ds_name, ds in datasets.items():
        if ds['surv'] is not None:
            results = validate_gene_survival(ds['expr'], ds['surv'], SL_TARGETS, ds_name)
            all_gene_results.extend(results)
            n_sig = sum(1 for r in results if r['significant'])
            logger.info(f"  [{ds_name}] {n_sig}/{len(results)} genes significant")

    # Step 3: Validate ferroptosis signature
    logger.info("\n--- Ferroptosis Signature Validation ---")
    for ds_name, ds in datasets.items():
        if ds['surv'] is not None:
            result = validate_ferroptosis_signature(ds['expr'], ds['surv'], ds_name)
            if result:
                dataset_results['ferroptosis'].append(result)

    # Step 4: Validate immune signatures
    logger.info("\n--- Immune Signature Validation ---")
    for ds_name, ds in datasets.items():
        if ds['surv'] is not None:
            results = validate_immune_signature(ds['expr'], ds['surv'], ds_name)
            dataset_results['immune'].extend(results)

    # Step 5: Visualization
    logger.info("\n--- Generating Figures ---")
    plot_validation_forest(all_gene_results, dataset_results)

    # KM curves for top genes
    if datasets:
        top_genes = ['GPX4', 'FGFR1', 'MDM2', 'BCL2']
        fig, axes = plt.subplots(len(top_genes), len(datasets), figsize=(6*len(datasets), 5*len(top_genes)))
        if len(datasets) == 1:
            axes = axes.reshape(-1, 1)
        if len(top_genes) == 1:
            axes = axes.reshape(1, -1)

        for j, (ds_name, ds) in enumerate(datasets.items()):
            if ds['surv'] is None:
                continue
            for i, gene in enumerate(top_genes):
                try:
                    plot_km_validation(ds['expr'], ds['surv'], gene, ds_name, axes[i, j])
                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error: {str(e)[:30]}',
                                   transform=axes[i, j].transAxes, ha='center')

        plt.suptitle('Key Gene Validation: KM Survival Curves',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "validation_km_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved: validation_km_curves.png")

    # Step 6: Summary
    gene_df = pd.DataFrame(all_gene_results)
    gene_df.to_csv(RESULTS_DIR / "validation_gene_survival.csv", index=False)

    ferro_df = pd.DataFrame(dataset_results['ferroptosis'])
    if len(ferro_df) > 0:
        ferro_df.to_csv(RESULTS_DIR / "validation_ferroptosis.csv", index=False)

    immune_df = pd.DataFrame(dataset_results['immune'])
    if len(immune_df) > 0:
        immune_df.to_csv(RESULTS_DIR / "validation_immune.csv", index=False)

    # Compute overall validation statistics
    summary = {
        'experiment': 'Exp 17: External Validation',
        'datasets_used': list(datasets.keys()),
        'n_datasets': len(datasets),
        'gene_survival': {
            'total_tests': len(gene_df),
            'significant': int(gene_df['significant'].sum()) if len(gene_df) > 0 else 0,
            'validation_rate': float(gene_df['significant'].mean()) if len(gene_df) > 0 else 0,
            'per_dataset': gene_df.groupby('dataset').agg({
                'gene': 'count',
                'significant': 'sum',
            }).to_dict() if len(gene_df) > 0 else {},
        },
        'ferroptosis': {
            'n_datasets_tested': len(ferro_df),
            'n_significant': int(ferro_df['significant'].sum()) if len(ferro_df) > 0 else 0,
        },
        'immune': {
            'n_tests': len(immune_df),
            'n_significant': int(immune_df['significant'].sum()) if len(immune_df) > 0 else 0,
        },
    }

    with open(RESULTS_DIR / "exp17_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 17 COMPLETE")
    logger.info("=" * 60)

    # Print validation summary
    if len(gene_df) > 0:
        for ds in gene_df['dataset'].unique():
            ds_data = gene_df[gene_df['dataset'] == ds]
            n_sig = ds_data['significant'].sum()
            logger.info(f"  {ds}: {n_sig}/{len(ds_data)} genes validated ({100*n_sig/len(ds_data):.0f}%)")


if __name__ == "__main__":
    main()
