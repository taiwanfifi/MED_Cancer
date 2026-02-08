#!/usr/bin/env python3
"""
Experiment 6: Tumor Microenvironment & Immune Cell Deconvolution
================================================================
Uses TCGA expression data to:
1. Estimate immune cell composition via gene signatures (CIBERSORTx-like)
2. Correlate immune composition with survival
3. Identify immunotherapy-responsive subtypes
4. Test if TME features outpredict TMB

Target Paper: Paper 10 (TME Immunotherapy Response) — Nature Immunology
Hypothesis: H4 - TME > TMB for immunotherapy prediction
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE_DIR = Path("/workspace/cancer_research")
EXPR_DIR = BASE_DIR / "data" / "tcga_expression"
CLINICAL_FILE = BASE_DIR / "data" / "tcga" / "pan_cancer_clinical.parquet"
RESULTS_DIR = BASE_DIR / "results" / "exp6_immune_tme"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Immune cell signature genes (simplified CIBERSORTx-like approach)
IMMUNE_SIGNATURES = {
    'CD8_T_cells': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'IFNG', 'TBX21', 'EOMES', 'NKG7', 'KLRK1'],
    'CD4_T_cells': ['CD4', 'IL7R', 'TCF7', 'LEF1', 'CCR7', 'SELL', 'CD27', 'CD28'],
    'T_regulatory': ['FOXP3', 'IL2RA', 'CTLA4', 'TNFRSF18', 'IKZF2', 'CCR8'],
    'NK_cells': ['NCAM1', 'NKG7', 'KLRD1', 'KLRB1', 'GNLY', 'GZMB', 'NCR1'],
    'B_cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'BANK1', 'PAX5', 'BLK'],
    'Macrophages_M1': ['CD68', 'NOS2', 'IL1B', 'TNF', 'IL6', 'CXCL10', 'IDO1'],
    'Macrophages_M2': ['CD163', 'MRC1', 'MSR1', 'CD209', 'TGFB1', 'IL10', 'ARG1'],
    'Dendritic_cells': ['ITGAX', 'BATF3', 'CLEC9A', 'XCR1', 'CD1C', 'FCER1A'],
    'Neutrophils': ['FCGR3B', 'CXCR2', 'CSF3R', 'S100A8', 'S100A9', 'CEACAM8'],
    'Mast_cells': ['KIT', 'CPA3', 'TPSAB1', 'TPSB2', 'MS4A2', 'HDC'],
    'Exhaustion': ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'CTLA4', 'CD274'],
    'IFN_gamma': ['IFNG', 'STAT1', 'IRF1', 'CXCL9', 'CXCL10', 'CXCL11', 'IDO1', 'GBP1'],
}

# Immune checkpoint genes
CHECKPOINT_GENES = {
    'targets': ['CD274', 'PDCD1LG2', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'VSIR'],
    'costimulatory': ['CD28', 'ICOS', 'CD40', 'CD40LG', 'TNFRSF4', 'TNFRSF9', 'CD27'],
}

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def load_expression(cancer_type):
    """Load TCGA expression matrix."""
    f = EXPR_DIR / f"{cancer_type}_expression_matrix.parquet"
    if not f.exists():
        logger.warning(f"Expression data not found for {cancer_type}")
        return None
    return pd.read_parquet(f)


def load_clinical(cancer_type):
    """Load clinical data for survival analysis from pan-cancer file."""
    if not CLINICAL_FILE.exists():
        logger.warning(f"Clinical data not found at {CLINICAL_FILE}")
        return None
    df = pd.read_parquet(CLINICAL_FILE)
    df = df[df['cancer_type'] == cancer_type].copy()
    logger.info(f"[{cancer_type}] Clinical data: {len(df)} patients")
    return df


def estimate_immune_composition(expr_df, cancer_type):
    """
    Estimate immune cell type scores using ssGSEA-like approach.
    For each cell type, compute the mean z-scored expression of signature genes.
    """
    # Z-score normalize expression
    expr_log = np.log2(expr_df + 1)
    expr_z = (expr_log.T - expr_log.mean(axis=1)) / (expr_log.std(axis=1) + 1e-6)
    expr_z = expr_z.T  # back to genes x samples

    immune_scores = {}

    for cell_type, genes in IMMUNE_SIGNATURES.items():
        available_genes = [g for g in genes if g in expr_z.index]
        if len(available_genes) < 3:
            logger.warning(f"[{cancer_type}] {cell_type}: only {len(available_genes)} signature genes found")
            continue

        # Mean signature score per sample
        scores = expr_z.loc[available_genes].mean(axis=0)
        immune_scores[cell_type] = scores
        logger.info(f"[{cancer_type}] {cell_type}: {len(available_genes)} genes, "
                   f"mean score = {scores.mean():.3f} ± {scores.std():.3f}")

    immune_df = pd.DataFrame(immune_scores)
    immune_df.index = expr_df.columns  # sample IDs
    immune_df.to_csv(RESULTS_DIR / f"{cancer_type}_immune_scores.csv")

    logger.info(f"[{cancer_type}] Immune composition estimated for {len(immune_df)} samples")
    return immune_df


def correlate_immune_survival(immune_df, clinical_df, cancer_type):
    """Correlate immune cell scores with patient survival."""
    # Match samples
    # TCGA barcodes: need to match on patient ID (first 12 chars)
    immune_patients = {}
    for sid in immune_df.index:
        patient_id = '-'.join(str(sid).split('-')[:3])
        immune_patients[patient_id] = sid

    clinical_df = clinical_df.copy()
    if 'submitter_id' in clinical_df.columns:
        clinical_df['patient_id'] = clinical_df['submitter_id']
    elif 'case_id' in clinical_df.columns:
        clinical_df['patient_id'] = clinical_df['case_id']
    else:
        logger.warning(f"[{cancer_type}] No patient ID column found in clinical data")
        return None

    # Try to match
    matched = []
    for _, row in clinical_df.iterrows():
        pid = str(row['patient_id'])
        if pid in immune_patients:
            sid = immune_patients[pid]
            immune_row = immune_df.loc[sid].to_dict()
            immune_row['patient_id'] = pid

            # Add survival data
            if 'days_to_death' in clinical_df.columns:
                immune_row['days_to_death'] = row.get('days_to_death')
            if 'days_to_follow_up' in clinical_df.columns:
                immune_row['days_to_follow_up'] = row.get('days_to_follow_up')
            elif 'days_to_last_follow_up' in clinical_df.columns:
                immune_row['days_to_follow_up'] = row.get('days_to_last_follow_up')
            if 'vital_status' in clinical_df.columns:
                immune_row['vital_status'] = row.get('vital_status')

            matched.append(immune_row)

    if len(matched) < 20:
        logger.warning(f"[{cancer_type}] Only {len(matched)} matched samples, insufficient for survival analysis")
        return None

    merged = pd.DataFrame(matched)
    logger.info(f"[{cancer_type}] Matched {len(merged)} samples with clinical data")

    # Compute survival time
    merged['time'] = pd.to_numeric(merged.get('days_to_death', pd.Series(dtype=float)), errors='coerce')
    follow_up = pd.to_numeric(merged.get('days_to_follow_up', pd.Series(dtype=float)), errors='coerce')
    merged['time'] = merged['time'].fillna(follow_up)
    merged['event'] = merged.get('vital_status', '').apply(
        lambda x: 1 if str(x).lower() in ['dead', 'deceased'] else 0
    )

    # Drop missing
    valid = merged.dropna(subset=['time'])
    valid = valid[valid['time'] > 0]

    if len(valid) < 20:
        logger.warning(f"[{cancer_type}] Too few valid samples after filtering")
        return None

    # Cox regression for each immune cell type
    cox_results = []
    cell_types = [col for col in immune_df.columns if col in valid.columns]

    for cell_type in cell_types:
        try:
            cph = CoxPHFitter()
            surv_df = valid[['time', 'event', cell_type]].dropna()

            if len(surv_df) < 20:
                continue

            cph.fit(surv_df, duration_col='time', event_col='event')

            hr = cph.hazard_ratios_[cell_type]
            p_val = cph.summary.loc[cell_type, 'p']
            ci_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
            ci_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])

            cox_results.append({
                'cancer_type': cancer_type,
                'cell_type': cell_type,
                'hazard_ratio': hr,
                'p_value': p_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_val < 0.05,
                'direction': 'protective' if hr < 1 else 'risk',
            })

            if p_val < 0.05:
                logger.info(f"  ** {cell_type}: HR={hr:.3f} ({ci_lower:.2f}-{ci_upper:.2f}), "
                           f"p={p_val:.4f} {'PROTECTIVE' if hr < 1 else 'RISK'}")
        except Exception as e:
            logger.warning(f"  {cell_type}: Cox regression failed: {e}")

    if cox_results:
        cox_df = pd.DataFrame(cox_results)
        cox_df.to_csv(RESULTS_DIR / f"{cancer_type}_immune_survival_cox.csv", index=False)
        return cox_df

    return None


def plot_immune_landscape(immune_df, cancer_type):
    """Plot immune composition heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: Heatmap of immune scores
    ax = axes[0]
    # Cluster samples by immune composition
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

    # Take a random subset if too many samples
    n_plot = min(100, len(immune_df))
    plot_df = immune_df.sample(n_plot, random_state=42) if len(immune_df) > n_plot else immune_df

    sns.heatmap(plot_df.T, cmap='RdBu_r', center=0, ax=ax,
                xticklabels=False, yticklabels=True,
                linewidths=0, vmin=-2, vmax=2)
    ax.set_title(f'A) {cancer_type} Immune Landscape (n={n_plot})', fontweight='bold')
    ax.set_xlabel('Samples')

    # Panel B: Immune score distributions
    ax = axes[1]
    immune_melted = immune_df.melt(var_name='Cell Type', value_name='Score')
    sns.boxplot(data=immune_melted, y='Cell Type', x='Score', ax=ax,
                orient='h', palette='Set3')
    ax.set_title(f'B) {cancer_type} Immune Cell Score Distribution', fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cancer_type}_immune_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {cancer_type}_immune_landscape.png")


def plot_immune_survival_km(immune_df, clinical_df, cancer_type, cell_type='CD8_T_cells'):
    """Kaplan-Meier by high vs low immune infiltration."""
    # Match data (same as in correlate_immune_survival)
    immune_patients = {}
    for sid in immune_df.index:
        patient_id = '-'.join(str(sid).split('-')[:3])
        immune_patients[patient_id] = sid

    clinical_df = clinical_df.copy()
    pid_col = 'submitter_id' if 'submitter_id' in clinical_df.columns else 'case_id'

    matched = []
    for _, row in clinical_df.iterrows():
        pid = str(row[pid_col])
        if pid in immune_patients:
            sid = immune_patients[pid]
            matched.append({
                'patient_id': pid,
                'immune_score': immune_df.loc[sid, cell_type] if cell_type in immune_df.columns else np.nan,
                'days_to_death': row.get('days_to_death'),
                'days_to_follow_up': row.get('days_to_follow_up', row.get('days_to_last_follow_up')),
                'vital_status': row.get('vital_status'),
            })

    if len(matched) < 20:
        return

    df = pd.DataFrame(matched)
    df['time'] = pd.to_numeric(df['days_to_death'], errors='coerce')
    df['time'] = df['time'].fillna(pd.to_numeric(df['days_to_follow_up'], errors='coerce'))
    df['event'] = df['vital_status'].apply(
        lambda x: 1 if str(x).lower() in ['dead', 'deceased'] else 0
    )
    df = df.dropna(subset=['time', 'immune_score'])
    df = df[df['time'] > 0]

    if len(df) < 20:
        return

    # Split by median
    median_score = df['immune_score'].median()
    df['group'] = df['immune_score'].apply(lambda x: 'High' if x > median_score else 'Low')

    fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()

    for group, color in [('High', '#e74c3c'), ('Low', '#3498db')]:
        mask = df['group'] == group
        if mask.sum() < 5:
            continue
        kmf.fit(df.loc[mask, 'time'] / 365.25, df.loc[mask, 'event'],
                label=f'{cell_type} {group} (n={mask.sum()})')
        kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.1)

    # Log-rank test
    high_mask = df['group'] == 'High'
    if high_mask.sum() >= 5 and (~high_mask).sum() >= 5:
        result = logrank_test(
            df.loc[high_mask, 'time'], df.loc[~high_mask, 'time'],
            df.loc[high_mask, 'event'], df.loc[~high_mask, 'event']
        )
        ax.text(0.5, 0.02, f'Log-rank p = {result.p_value:.4f}',
                transform=ax.transAxes, fontsize=11, ha='center')

    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'{cancer_type}: Survival by {cell_type.replace("_", " ")} Infiltration',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cancer_type}_{cell_type}_survival_km.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {cancer_type}_{cell_type}_survival_km.png")


def plot_cross_cancer_immune_comparison(all_immune_scores):
    """Compare immune composition across cancer types."""
    if not all_immune_scores:
        return

    # Compute mean scores per cancer type
    means = {}
    for ct, df in all_immune_scores.items():
        means[ct] = df.mean()

    mean_df = pd.DataFrame(means)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(mean_df, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax,
                linewidths=0.5, xticklabels=True, yticklabels=True)
    ax.set_title('Mean Immune Cell Scores Across Cancer Types',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Immune Cell Type')
    ax.set_xlabel('Cancer Type')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cross_cancer_immune_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: cross_cancer_immune_comparison.png")


def checkpoint_enrichment_analysis(immune_df, cancer_type, expr_df):
    """Analyze immune checkpoint expression patterns."""
    results = []

    for category, genes in CHECKPOINT_GENES.items():
        available = [g for g in genes if g in expr_df.index]
        if not available:
            continue

        for gene in available:
            vals = np.log2(expr_df.loc[gene] + 1)
            results.append({
                'cancer_type': cancer_type,
                'category': category,
                'gene': gene,
                'mean_log2tpm': vals.mean(),
                'std_log2tpm': vals.std(),
                'pct_expressed': (expr_df.loc[gene] > 1).mean() * 100,
            })

    if results:
        cp_df = pd.DataFrame(results)
        cp_df.to_csv(RESULTS_DIR / f"{cancer_type}_checkpoint_expression.csv", index=False)
        logger.info(f"[{cancer_type}] Checkpoint analysis: {len(results)} genes analyzed")

        # Key checkpoint genes
        for _, row in cp_df.iterrows():
            if row['gene'] in ['CD274', 'PDCD1', 'CTLA4', 'LAG3']:
                logger.info(f"  {row['gene']}: mean={row['mean_log2tpm']:.2f}, "
                           f"expressed in {row['pct_expressed']:.1f}% samples")

        return cp_df
    return None


def main():
    logger.info("=" * 60)
    logger.info("Experiment 6: Tumor Microenvironment & Immune Analysis")
    logger.info("=" * 60)

    all_immune_scores = {}
    all_cox_results = []

    for cancer_type in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer_type}")
        logger.info(f"{'='*40}")

        # Load expression
        expr_df = load_expression(cancer_type)
        if expr_df is None:
            continue

        # Load clinical
        clinical_df = load_clinical(cancer_type)

        # Estimate immune composition
        immune_df = estimate_immune_composition(expr_df, cancer_type)
        all_immune_scores[cancer_type] = immune_df

        # Plot immune landscape
        plot_immune_landscape(immune_df, cancer_type)

        # Checkpoint expression analysis
        checkpoint_enrichment_analysis(immune_df, cancer_type, expr_df)

        # Survival analysis
        if clinical_df is not None:
            cox_df = correlate_immune_survival(immune_df, clinical_df, cancer_type)
            if cox_df is not None:
                all_cox_results.append(cox_df)

            # KM plots for key cell types
            for cell_type in ['CD8_T_cells', 'T_regulatory', 'Macrophages_M2', 'Exhaustion']:
                if cell_type in immune_df.columns:
                    plot_immune_survival_km(immune_df, clinical_df, cancer_type, cell_type)

    # Cross-cancer comparison
    plot_cross_cancer_immune_comparison(all_immune_scores)

    # Combined results
    if all_cox_results:
        combined_cox = pd.concat(all_cox_results, ignore_index=True)
        combined_cox.to_csv(RESULTS_DIR / "all_immune_survival_cox.csv", index=False)

        # Key findings
        sig_results = combined_cox[combined_cox['significant']]
        logger.info(f"\n{'='*60}")
        logger.info(f"KEY FINDINGS: Significant immune-survival associations")
        logger.info(f"{'='*60}")
        for _, row in sig_results.iterrows():
            logger.info(f"  {row['cancer_type']} | {row['cell_type']}: "
                       f"HR={row['hazard_ratio']:.3f}, p={row['p_value']:.4f} "
                       f"[{row['direction'].upper()}]")

    # Summary
    summary = {
        'experiment': 'Exp 6: Tumor Microenvironment & Immune Analysis',
        'hypothesis': 'H4: TME features predict immunotherapy response better than TMB',
        'cancer_types_analyzed': list(all_immune_scores.keys()),
        'n_cell_types': len(IMMUNE_SIGNATURES),
        'significant_associations': len(sig_results) if all_cox_results else 0,
    }

    with open(RESULTS_DIR / "exp6_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 6 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
