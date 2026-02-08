#!/usr/bin/env python3
"""
Experiment 19: Immune-SL Combination Strategy
===============================================
Tests whether immune-cold tumors harbor specific SL vulnerabilities,
enabling rational immune + targeted therapy combinations.

Hypothesis: Immune-cold tumors (low T cell infiltration) have distinct
SL target profiles compared to immune-hot tumors. Identifying these
SL targets could enable combination strategies:
  - Immune-cold + SL-targeted therapy → sensitize to immunotherapy
  - Immune-hot + checkpoint inhibitor → standard approach

Key analyses:
1. Classify patients into immune-hot vs cold (ssGSEA immune scores)
2. Differential SL target expression: hot vs cold
3. Correlate immune composition with SL target dependency
4. Identify combination opportunities per cancer type
5. Survival analysis: immune group + SL target expression

Target: Paper on precision immuno-oncology via SL
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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp19_immune_sl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Immune cell gene signatures for ssGSEA-like scoring
IMMUNE_SIGNATURES = {
    'CD8_T': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10'],
    'CD4_T': ['CD4', 'IL7R', 'CD40LG', 'FOXP3', 'IL2RA', 'CCR7'],
    'B_cells': ['CD19', 'CD79A', 'MS4A1', 'CD79B', 'BLNK', 'PAX5'],
    'NK_cells': ['NKG7', 'GNLY', 'KLRB1', 'KLRD1', 'NCR1', 'KLRC1'],
    'Macrophage': ['CD68', 'CD163', 'CSF1R', 'MRC1', 'MSR1', 'MARCO'],
    'Dendritic': ['CLEC4C', 'NRP1', 'LILRA4', 'CD1C', 'FCER1A', 'BATF3'],
    'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'TNFRSF18', 'IKZF2'],
    'Exhaustion': ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'CTLA4'],
    'IFN_gamma': ['IFNG', 'STAT1', 'IRF1', 'IDO1', 'CXCL9', 'CXCL10', 'HLA-DRA'],
}

# Key SL targets from our analyses
SL_TARGETS = [
    'GPX4', 'FGFR1', 'MDM2', 'CDK6', 'BCL2', 'MYC', 'CCND1',
    'PTEN', 'STK11', 'BRCA1', 'BRCA2', 'EGFR', 'BRAF', 'TP53',
    'GCLM', 'GCLC', 'SLC7A11', 'ACSL4', 'DHODH', 'GSS',
]

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def load_expression(cancer_type):
    """Load expression matrix (genes in index, patients in columns)."""
    expr_file = DATA_DIR / "tcga_expression" / f"{cancer_type}_expression_matrix.parquet"
    if not expr_file.exists():
        expr_file = DATA_DIR / "tcga" / f"{cancer_type}_expression.parquet"
    if not expr_file.exists():
        logger.warning(f"Expression not found for {cancer_type}")
        return pd.DataFrame()
    expr = pd.read_parquet(expr_file)
    logger.info(f"  {cancer_type} expression: {expr.shape}")
    return expr


def load_clinical(cancer_type):
    """Load clinical data."""
    clin_file = DATA_DIR / "tcga" / f"{cancer_type}_clinical.json"
    if clin_file.exists():
        with open(clin_file) as f:
            data = json.load(f)
        clin = pd.DataFrame(data)
        if 'case_id' in clin.columns:
            clin = clin.set_index('case_id')
        return clin
    # Try parquet
    clin_file = DATA_DIR / "tcga_clinical" / f"TCGA-{cancer_type}_clinical.parquet"
    if clin_file.exists():
        return pd.read_parquet(clin_file)
    return pd.DataFrame()


def compute_immune_scores(expr):
    """
    Compute immune cell type scores using ssGSEA-like approach.
    Simple mean z-score of signature genes.
    """
    # Z-score expression per gene
    gene_index_map = {g.upper(): g for g in expr.index}

    scores = {}
    for cell_type, genes in IMMUNE_SIGNATURES.items():
        matched = []
        for g in genes:
            idx = gene_index_map.get(g.upper())
            if idx is not None:
                vals = expr.loc[idx].astype(float)
                # Z-score
                z = (vals - vals.mean()) / (vals.std() + 1e-8)
                matched.append(z)
        if matched:
            scores[cell_type] = pd.concat(matched, axis=1).mean(axis=1)

    score_df = pd.DataFrame(scores)
    logger.info(f"  Computed {len(scores)} immune scores for {len(score_df)} patients")
    return score_df


def classify_immune_status(immune_scores):
    """
    Classify patients as immune-hot, -intermediate, or -cold.
    Based on composite immune score (CD8 T + IFN gamma).
    """
    # Composite score
    composite_cols = ['CD8_T', 'IFN_gamma']
    available = [c for c in composite_cols if c in immune_scores.columns]
    if not available:
        available = list(immune_scores.columns)[:2]

    composite = immune_scores[available].mean(axis=1)

    # Tertile-based classification
    q33 = composite.quantile(0.33)
    q67 = composite.quantile(0.67)

    status = pd.Series('Intermediate', index=composite.index)
    status[composite <= q33] = 'Cold'
    status[composite >= q67] = 'Hot'

    logger.info(f"  Immune status: Hot={int((status=='Hot').sum())}, "
                f"Cold={int((status=='Cold').sum())}, "
                f"Intermediate={int((status=='Intermediate').sum())}")
    return status, composite


def differential_sl_expression(expr, immune_status, cancer_type):
    """
    Test if SL target expression differs between immune-hot and -cold tumors.
    """
    gene_index_map = {g.upper(): g for g in expr.index}

    # Get sample columns
    sample_cols = list(expr.columns)

    # Align immune status with expression columns
    # Shorten TCGA barcodes to match
    short_to_full = {}
    for col in sample_cols:
        short = '-'.join(str(col).split('-')[:3])
        short_to_full[short] = col

    results = []
    for target in SL_TARGETS:
        idx = gene_index_map.get(target.upper())
        if idx is None:
            continue

        gene_expr = expr.loc[idx].astype(float)

        hot_vals = []
        cold_vals = []
        for patient_id, status in immune_status.items():
            # Try to match patient ID to expression column
            col = short_to_full.get(patient_id) or short_to_full.get(str(patient_id))
            if col is not None and col in gene_expr.index:
                if status == 'Hot':
                    hot_vals.append(gene_expr[col])
                elif status == 'Cold':
                    cold_vals.append(gene_expr[col])

        if len(hot_vals) < 5 or len(cold_vals) < 5:
            continue

        # T-test
        t_stat, p_val = stats.ttest_ind(hot_vals, cold_vals)
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(hot_vals)**2 + np.std(cold_vals)**2) / 2)
        cohens_d = (np.mean(hot_vals) - np.mean(cold_vals)) / (pooled_std + 1e-8)

        results.append({
            'gene': target,
            'cancer_type': cancer_type,
            'mean_hot': float(np.mean(hot_vals)),
            'mean_cold': float(np.mean(cold_vals)),
            'log2fc': float(np.mean(hot_vals) - np.mean(cold_vals)),
            'cohens_d': float(cohens_d),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'n_hot': len(hot_vals),
            'n_cold': len(cold_vals),
            'upregulated_in': 'Hot' if np.mean(hot_vals) > np.mean(cold_vals) else 'Cold',
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['fdr'] = stats.false_discovery_control(results_df['p_value'])
        results_df = results_df.sort_values('p_value')

    return results_df


def immune_sl_survival(expr, clinical, immune_status, immune_composite, cancer_type):
    """
    4-group survival analysis: hot/cold × SL-target-high/low.
    Tests if immune status + SL target expression jointly predict survival.
    """
    gene_index_map = {g.upper(): g for g in expr.index}

    # Build survival data
    if 'vital_status' not in clinical.columns:
        return pd.DataFrame()

    survival_results = []

    for target in ['GPX4', 'FGFR1', 'MDM2', 'CDK6', 'BCL2']:
        idx = gene_index_map.get(target.upper())
        if idx is None:
            continue

        gene_expr = expr.loc[idx].astype(float)

        # Build combined DataFrame
        records = []
        for patient_id in immune_status.index:
            short = str(patient_id)
            # Find matching expression column
            match_col = None
            for col in expr.columns:
                if '-'.join(str(col).split('-')[:3]) == short:
                    match_col = col
                    break
            if match_col is None:
                continue

            # Get clinical data
            clin_match = None
            for cid in clinical.index:
                if short in str(cid):
                    clin_match = cid
                    break
            if clin_match is None:
                continue

            row = clinical.loc[clin_match]
            days_to_death = pd.to_numeric(row.get('days_to_death', np.nan), errors='coerce')
            days_to_fu = pd.to_numeric(row.get('days_to_last_follow_up', np.nan), errors='coerce')
            os_time = days_to_death if pd.notna(days_to_death) and days_to_death > 0 else days_to_fu
            os_event = 1 if str(row.get('vital_status', '')).lower() == 'dead' else 0

            if pd.isna(os_time) or os_time <= 0:
                continue

            records.append({
                'patient': patient_id,
                'immune_status': immune_status[patient_id],
                'target_expr': float(gene_expr[match_col]),
                'os_time': float(os_time),
                'os_event': int(os_event),
            })

        if len(records) < 30:
            continue

        df = pd.DataFrame(records)
        median_expr = df['target_expr'].median()
        df['target_group'] = np.where(df['target_expr'] >= median_expr, 'High', 'Low')

        # 4-group analysis: Hot-High, Hot-Low, Cold-High, Cold-Low
        df['combined_group'] = df['immune_status'] + '-' + df['target_group']

        # Filter to Hot and Cold only
        df = df[df['immune_status'].isin(['Hot', 'Cold'])]
        if len(df) < 20:
            continue

        groups = df['combined_group'].unique()
        if len(groups) < 3:
            continue

        # Overall 4-group log-rank
        try:
            from lifelines.statistics import multivariate_logrank_test
            result = multivariate_logrank_test(df['os_time'], df['combined_group'], df['os_event'])
            overall_p = result.p_value

            # Key comparison: Cold-High vs Hot-Low (potentially most different)
            cold_high = df[df['combined_group'] == 'Cold-High']
            hot_low = df[df['combined_group'] == 'Hot-Low']
            if len(cold_high) >= 5 and len(hot_low) >= 5:
                lr = logrank_test(cold_high['os_time'], hot_low['os_time'],
                                 cold_high['os_event'], hot_low['os_event'])
                contrast_p = lr.p_value
            else:
                contrast_p = np.nan

            survival_results.append({
                'gene': target,
                'cancer_type': cancer_type,
                'overall_4group_p': float(overall_p),
                'cold_high_vs_hot_low_p': float(contrast_p),
                'n_total': len(df),
                'group_sizes': df['combined_group'].value_counts().to_dict(),
            })
        except Exception as e:
            logger.warning(f"  Survival analysis failed for {target}: {e}")

    return pd.DataFrame(survival_results)


def plot_all_results(all_diff_expr, all_immune_scores, all_survival, output_dir):
    """Generate comprehensive figures."""
    fig = plt.figure(figsize=(20, 16))

    # 1. Heatmap: SL target differential expression (hot vs cold) across cancers
    ax1 = fig.add_subplot(2, 2, 1)
    if not all_diff_expr.empty:
        pivot = all_diff_expr.pivot_table(index='gene', columns='cancer_type',
                                          values='log2fc', aggfunc='first')
        pivot = pivot.dropna(how='all')
        if len(pivot) > 0:
            # Add significance stars
            annot = pivot.copy()
            for gene in pivot.index:
                for cancer in pivot.columns:
                    mask = (all_diff_expr['gene'] == gene) & (all_diff_expr['cancer_type'] == cancer)
                    if mask.any():
                        fdr = all_diff_expr.loc[mask, 'fdr'].values[0]
                        val = pivot.loc[gene, cancer]
                        if pd.notna(val):
                            star = '***' if fdr < 0.001 else '**' if fdr < 0.01 else '*' if fdr < 0.05 else ''
                            annot.loc[gene, cancer] = f'{val:.2f}{star}'

            sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax1,
                       annot=annot, fmt='', cbar_kws={'label': 'Log2 FC (Hot-Cold)'})
            ax1.set_title('SL Target Expression:\nImmune-Hot vs Cold')
            ax1.set_ylabel('SL Target Gene')

    # 2. Immune composition by cancer type
    ax2 = fig.add_subplot(2, 2, 2)
    if all_immune_scores:
        mean_scores = {}
        for cancer, scores_df in all_immune_scores.items():
            mean_scores[cancer] = scores_df.mean()
        mean_df = pd.DataFrame(mean_scores)
        if len(mean_df) > 0:
            mean_df.plot(kind='bar', ax=ax2, width=0.7)
            ax2.set_title('Mean Immune Cell Scores by Cancer Type')
            ax2.set_ylabel('Mean Z-score')
            ax2.set_xlabel('Cell Type')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(title='Cancer')

    # 3. Volcano plot: differential expression
    ax3 = fig.add_subplot(2, 2, 3)
    if not all_diff_expr.empty:
        for cancer in all_diff_expr['cancer_type'].unique():
            subset = all_diff_expr[all_diff_expr['cancer_type'] == cancer]
            sig = subset['fdr'] < 0.05
            ax3.scatter(subset.loc[sig, 'log2fc'], -np.log10(subset.loc[sig, 'p_value']),
                       alpha=0.7, s=50, label=f'{cancer} (sig)')
            ax3.scatter(subset.loc[~sig, 'log2fc'], -np.log10(subset.loc[~sig, 'p_value']),
                       alpha=0.3, s=30, marker='x')
            # Label top genes
            for _, row in subset[sig].head(3).iterrows():
                ax3.annotate(row['gene'], (row['log2fc'], -np.log10(row['p_value'])),
                           fontsize=7, ha='center')
        ax3.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Log2 FC (Hot - Cold)')
        ax3.set_ylabel('-Log10(P-value)')
        ax3.set_title('SL Target Expression: Hot vs Cold')
        ax3.legend(fontsize=8)

    # 4. Survival: 4-group results
    ax4 = fig.add_subplot(2, 2, 4)
    if not all_survival.empty:
        surv_plot = all_survival[['gene', 'cancer_type', 'overall_4group_p']].copy()
        surv_plot['neg_log10_p'] = -np.log10(surv_plot['overall_4group_p'].clip(1e-20))
        surv_pivot = surv_plot.pivot_table(index='gene', columns='cancer_type',
                                            values='neg_log10_p', aggfunc='first')
        if len(surv_pivot) > 0:
            sns.heatmap(surv_pivot, cmap='YlOrRd', ax=ax4,
                       annot=True, fmt='.1f',
                       cbar_kws={'label': '-Log10(P-value)'})
            ax4.axhline(y=0, color='white', linewidth=0)
            ax4.set_title('4-Group Survival Analysis\n(Immune × SL Target)')

    plt.suptitle('Experiment 19: Immune-SL Combination Strategy',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'immune_sl_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved immune_sl_landscape.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 19: Immune-SL Combination Strategy")
    logger.info("=" * 60)

    all_diff_expr = []
    all_immune_scores = {}
    all_survival = []
    all_classifications = {}

    for cancer in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer}")
        logger.info(f"{'='*40}")

        expr = load_expression(cancer)
        if expr.empty:
            continue

        clinical = load_clinical(cancer)

        # Step 1: Compute immune scores
        logger.info("Step 1: Computing immune scores...")
        immune_scores = compute_immune_scores(expr)
        all_immune_scores[cancer] = immune_scores

        # Step 2: Classify immune status
        logger.info("Step 2: Classifying immune status...")
        immune_status, composite = classify_immune_status(immune_scores)
        all_classifications[cancer] = {
            'n_hot': int((immune_status == 'Hot').sum()),
            'n_cold': int((immune_status == 'Cold').sum()),
            'n_intermediate': int((immune_status == 'Intermediate').sum()),
        }

        # Step 3: Differential SL target expression
        logger.info("Step 3: Differential SL target expression (hot vs cold)...")
        diff_expr = differential_sl_expression(expr, immune_status, cancer)
        if not diff_expr.empty:
            all_diff_expr.append(diff_expr)
            sig = diff_expr[diff_expr['fdr'] < 0.05]
            logger.info(f"  Significant (FDR<0.05): {len(sig)} / {len(diff_expr)} genes")
            if not sig.empty:
                logger.info(f"  Top hits:\n{sig[['gene','log2fc','fdr','upregulated_in']].head(5).to_string()}")

        # Step 4: Survival analysis
        logger.info("Step 4: 4-group survival analysis...")
        if not clinical.empty:
            survival = immune_sl_survival(expr, clinical, immune_status, composite, cancer)
            if not survival.empty:
                all_survival.append(survival)
                sig_surv = survival[survival['overall_4group_p'] < 0.05]
                logger.info(f"  Significant survival: {len(sig_surv)} / {len(survival)} genes")

    # Combine results
    all_diff_expr = pd.concat(all_diff_expr) if all_diff_expr else pd.DataFrame()
    all_survival = pd.concat(all_survival) if all_survival else pd.DataFrame()

    # Save results
    logger.info("\n--- Saving Results ---")
    if not all_diff_expr.empty:
        all_diff_expr.to_csv(RESULTS_DIR / 'immune_sl_diff_expression.csv', index=False)
    if not all_survival.empty:
        all_survival.to_csv(RESULTS_DIR / 'immune_sl_survival.csv', index=False)
    for cancer, scores in all_immune_scores.items():
        scores.to_csv(RESULTS_DIR / f'{cancer}_immune_scores.csv')

    # Plot
    plot_all_results(all_diff_expr, all_immune_scores, all_survival, RESULTS_DIR)

    # Summary
    summary = {
        'experiment': 'Exp 19: Immune-SL Combination Strategy',
        'cancer_types': CANCER_TYPES,
        'immune_classifications': all_classifications,
        'diff_expression': {
            'total_tests': len(all_diff_expr),
            'significant_fdr05': int((all_diff_expr['fdr'] < 0.05).sum()) if not all_diff_expr.empty else 0,
            'top_hits': all_diff_expr.nsmallest(5, 'fdr')[
                ['gene', 'cancer_type', 'log2fc', 'fdr', 'upregulated_in']
            ].to_dict('records') if not all_diff_expr.empty else [],
        },
        'survival_analysis': {
            'total_tests': len(all_survival),
            'significant_p05': int((all_survival['overall_4group_p'] < 0.05).sum()) if not all_survival.empty else 0,
        },
        'sl_targets_tested': SL_TARGETS,
        'immune_signatures': list(IMMUNE_SIGNATURES.keys()),
    }
    with open(RESULTS_DIR / 'exp19_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nExp 19 COMPLETE")
    logger.info(f"Differential expression: {len(all_diff_expr)} tests")
    if not all_diff_expr.empty:
        logger.info(f"  Significant (FDR<0.05): {(all_diff_expr['fdr'] < 0.05).sum()}")
    logger.info(f"Survival tests: {len(all_survival)}")


if __name__ == '__main__':
    main()
