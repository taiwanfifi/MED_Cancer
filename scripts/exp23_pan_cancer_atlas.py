#!/usr/bin/env python3
"""
Experiment 23: Pan-Cancer SL Atlas — Expanding to All 15 TCGA Cancer Types
============================================================================
Extends SL analysis beyond BRCA/LUAD/KIRC to all 15 available TCGA cohorts.
Creates comprehensive atlas of cancer-type-specific SL vulnerabilities.

Key analyses:
1. Pan-cancer expression landscape of top SL targets
2. Cancer-type-specific SL target expression profiles
3. Mutation frequency of SL driver genes across all 15 cancers
4. Survival association of SL targets per cancer type
5. Cancer similarity network based on SL expression patterns

Target: Comprehensive pan-cancer atlas paper
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from lifelines import CoxPHFitter

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp23_pan_cancer_atlas"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# All 15 TCGA cancer types with available data
ALL_CANCERS = ['BRCA', 'LUAD', 'KIRC', 'BLCA', 'CESC', 'COAD', 'HNSC',
               'LGG', 'LIHC', 'LUSC', 'OV', 'PRAD', 'STAD', 'THCA', 'UCEC']

# Key genes to profile across all cancers
KEY_GENES = {
    'SL_Targets': ['GPX4', 'FGFR1', 'MDM2', 'CDK6', 'BCL2', 'MYC', 'CCND1',
                   'PTEN', 'BRAF', 'EGFR', 'ERBB2', 'TP53'],
    'Ferroptosis': ['GPX4', 'SLC7A11', 'ACSL4', 'GCLC', 'GCLM', 'NFE2L2', 'DHODH'],
    'Immune': ['CD274', 'PDCD1', 'LAG3', 'HAVCR2', 'CTLA4', 'TIGIT'],
    'Novel_Targets': ['CDS2', 'CHMP4B', 'GINS4', 'PSMA4', 'MAD2L1', 'KIF14', 'ORC6'],
    'Epigenetic': ['EZH2', 'BRD4', 'DNMT1', 'HDAC1', 'ARID1A', 'SMARCA4'],
    'Prognostic_Sig': ['BRAF', 'KIF14', 'CDK6', 'ORC6', 'CHAF1B', 'MAD2L1'],
}

# Flatten unique genes
ALL_GENES = sorted(set(g for genes in KEY_GENES.values() for g in genes))


def load_expression(cancer_type):
    """Load expression matrix."""
    for path_pattern in [
        DATA_DIR / "tcga_expression" / f"{cancer_type}_expression_matrix.parquet",
        DATA_DIR / "tcga" / f"{cancer_type}_expression.parquet",
    ]:
        if path_pattern.exists():
            return pd.read_parquet(path_pattern)
    return pd.DataFrame()


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
    return pd.DataFrame()


def pan_cancer_expression_landscape():
    """
    Profile expression of key genes across all 15 TCGA cancer types.
    """
    results = {}
    for cancer in ALL_CANCERS:
        expr = load_expression(cancer)
        if expr.empty:
            logger.warning(f"  {cancer}: no expression data")
            continue

        gene_index_map = {g.upper(): g for g in expr.index}
        # Filter tumor samples only
        tumor_cols = []
        for col in expr.columns:
            parts = str(col).split('-')
            if len(parts) >= 4:
                sample_type = int(parts[3][:2]) if parts[3][:2].isdigit() else 0
                if sample_type < 10:
                    tumor_cols.append(col)
            else:
                tumor_cols.append(col)

        if not tumor_cols:
            tumor_cols = list(expr.columns)

        cancer_data = {'n_samples': len(tumor_cols)}
        for gene in ALL_GENES:
            idx = gene_index_map.get(gene.upper())
            if idx is not None:
                vals = expr.loc[idx, tumor_cols].astype(float)
                cancer_data[gene] = {
                    'median': float(vals.median()),
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'q25': float(vals.quantile(0.25)),
                    'q75': float(vals.quantile(0.75)),
                }

        results[cancer] = cancer_data
        logger.info(f"  {cancer}: {len(tumor_cols)} tumor samples profiled")

    return results


def build_expression_matrix(landscape):
    """Build cancer × gene median expression matrix."""
    rows = []
    for cancer, data in landscape.items():
        row = {'cancer': cancer, 'n_samples': data['n_samples']}
        for gene in ALL_GENES:
            if gene in data and isinstance(data[gene], dict):
                row[gene] = data[gene]['median']
            else:
                row[gene] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index('cancer')
    return df


def survival_across_cancers():
    """
    Test SL target → survival association across all cancer types.
    """
    results = []
    for cancer in ALL_CANCERS:
        expr = load_expression(cancer)
        clin = load_clinical(cancer)
        if expr.empty or clin.empty:
            continue

        gene_index_map = {g.upper(): g for g in expr.index}

        for gene in ALL_GENES:
            idx = gene_index_map.get(gene.upper())
            if idx is None:
                continue

            # Build survival data
            records = []
            for col in expr.columns:
                short = '-'.join(str(col).split('-')[:3])
                # Find clinical match
                clin_match = None
                for cid in clin.index:
                    if short in str(cid):
                        clin_match = cid
                        break
                if clin_match is None:
                    continue

                row = clin.loc[clin_match]
                days_to_death = pd.to_numeric(row.get('days_to_death', np.nan), errors='coerce')
                days_to_fu = pd.to_numeric(row.get('days_to_last_follow_up', np.nan), errors='coerce')
                os_time = days_to_death if pd.notna(days_to_death) and days_to_death > 0 else days_to_fu
                os_event = 1 if str(row.get('vital_status', '')).lower() == 'dead' else 0

                if pd.isna(os_time) or os_time <= 0:
                    continue

                gene_val = float(expr.loc[idx, col])
                records.append({
                    'os_time': os_time,
                    'os_event': os_event,
                    'gene_expr': gene_val,
                })

            if len(records) < 20:
                continue

            df = pd.DataFrame(records)
            if df['os_event'].sum() < 5:
                continue

            # Z-score gene expression
            df['gene_z'] = (df['gene_expr'] - df['gene_expr'].mean()) / (df['gene_expr'].std() + 1e-8)

            try:
                cph = CoxPHFitter()
                cph.fit(df[['os_time', 'os_event', 'gene_z']],
                       duration_col='os_time', event_col='os_event')
                hr = float(np.exp(cph.params_['gene_z']))
                p = float(cph.summary['p']['gene_z'])
                c = float(cph.concordance_index_)

                results.append({
                    'cancer': cancer,
                    'gene': gene,
                    'hr': hr,
                    'p_value': p,
                    'c_index': c,
                    'n_patients': len(df),
                    'n_events': int(df['os_event'].sum()),
                    'direction': 'Risk' if hr > 1 else 'Protective',
                })
            except Exception:
                continue

    surv_df = pd.DataFrame(results)
    if not surv_df.empty:
        surv_df['fdr'] = stats.false_discovery_control(surv_df['p_value'])
    return surv_df


def cancer_similarity_network(expr_matrix):
    """
    Cluster cancers by SL target expression similarity.
    """
    # Drop n_samples column, use only gene columns
    gene_cols = [c for c in expr_matrix.columns if c != 'n_samples']
    mat = expr_matrix[gene_cols].dropna(axis=1, how='all').fillna(0)

    if len(mat) < 3:
        return pd.DataFrame()

    # Z-score per gene across cancers
    mat_z = mat.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0)

    # Compute distance and cluster
    dist = pdist(mat_z.values, metric='correlation')
    Z = linkage(dist, method='ward')

    # Pairwise correlation matrix
    corr = mat_z.T.corr()

    return corr, Z, mat_z


def plot_results(expr_matrix, surv_df, corr_matrix, output_dir):
    """Generate comprehensive atlas figures."""
    fig = plt.figure(figsize=(24, 20))

    # 1. Pan-cancer expression heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    gene_cols = [c for c in expr_matrix.columns if c != 'n_samples' and c in ALL_GENES]
    plot_data = expr_matrix[gene_cols].dropna(axis=1, how='all')
    if len(plot_data) > 0:
        # Z-score for visualization
        plot_z = plot_data.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0)
        sns.heatmap(plot_z, cmap='RdBu_r', center=0, ax=ax1,
                   xticklabels=True, yticklabels=True,
                   cbar_kws={'label': 'Z-score'})
        ax1.set_title('Pan-Cancer SL Target Expression')
        ax1.tick_params(axis='x', rotation=45, labelsize=6)
        ax1.tick_params(axis='y', labelsize=8)

    # 2. Survival heatmap: gene × cancer
    ax2 = fig.add_subplot(2, 3, 2)
    if not surv_df.empty:
        surv_pivot = surv_df.pivot_table(index='gene', columns='cancer',
                                          values='p_value', aggfunc='first')
        surv_log = -np.log10(surv_pivot.clip(1e-20))
        # Only show genes with at least one significant result
        sig_genes = surv_df[surv_df['fdr'] < 0.1]['gene'].unique()
        if len(sig_genes) > 0:
            surv_log = surv_log.loc[surv_log.index.isin(sig_genes)]
        if len(surv_log) > 20:
            surv_log = surv_log.iloc[:20]
        if len(surv_log) > 0:
            sns.heatmap(surv_log, cmap='YlOrRd', ax=ax2,
                       annot=True, fmt='.1f',
                       cbar_kws={'label': '-Log10(P)'},
                       xticklabels=True, yticklabels=True)
            ax2.set_title('Survival Association\n(-Log10 P-value)')
            ax2.tick_params(axis='x', rotation=45, labelsize=7)
            ax2.tick_params(axis='y', labelsize=7)

    # 3. Cancer similarity dendrogram / heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    if corr_matrix is not None and len(corr_matrix) > 0:
        sns.heatmap(corr_matrix, cmap='RdYlBu_r', center=0, ax=ax3,
                   annot=True, fmt='.2f',
                   xticklabels=True, yticklabels=True,
                   cbar_kws={'label': 'Correlation'})
        ax3.set_title('Cancer Similarity\n(SL Target Expression)')
        ax3.tick_params(axis='x', rotation=45, labelsize=7)
        ax3.tick_params(axis='y', labelsize=7)

    # 4. Top prognostic genes across cancers
    ax4 = fig.add_subplot(2, 3, 4)
    if not surv_df.empty:
        top_genes = surv_df.groupby('gene')['fdr'].min().nsmallest(15).index
        top_surv = surv_df[surv_df['gene'].isin(top_genes)]
        if len(top_surv) > 0:
            for gene in top_genes:
                gene_data = top_surv[top_surv['gene'] == gene]
                ax4.scatter(gene_data['cancer'], [gene] * len(gene_data),
                           s=(-np.log10(gene_data['p_value'].clip(1e-20))) * 20,
                           c=['red' if hr > 1 else 'blue' for hr in gene_data['hr']],
                           alpha=0.7)
            ax4.set_xlabel('Cancer Type')
            ax4.set_title('Gene-Cancer Survival Map\n(Red=Risk, Blue=Protective)')
            ax4.tick_params(axis='x', rotation=45, labelsize=7)
            ax4.tick_params(axis='y', labelsize=7)

    # 5. Sample size overview
    ax5 = fig.add_subplot(2, 3, 5)
    if 'n_samples' in expr_matrix.columns:
        n_samples = expr_matrix['n_samples'].sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(n_samples)))
        ax5.barh(range(len(n_samples)), n_samples.values, color=colors)
        ax5.set_yticks(range(len(n_samples)))
        ax5.set_yticklabels(n_samples.index, fontsize=8)
        ax5.set_xlabel('Number of Tumor Samples')
        ax5.set_title('TCGA Cohort Sizes')

    # 6. Gene category expression by cancer
    ax6 = fig.add_subplot(2, 3, 6)
    category_medians = {}
    for category, genes in KEY_GENES.items():
        available = [g for g in genes if g in expr_matrix.columns]
        if available:
            category_medians[category] = expr_matrix[available].median(axis=1)
    if category_medians:
        cat_df = pd.DataFrame(category_medians)
        cat_z = cat_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0)
        sns.heatmap(cat_z, cmap='RdBu_r', center=0, ax=ax6,
                   annot=True, fmt='.1f',
                   xticklabels=True, yticklabels=True)
        ax6.set_title('Gene Category Expression\n(Z-score by cancer)')
        ax6.tick_params(axis='x', rotation=45, labelsize=7)
        ax6.tick_params(axis='y', labelsize=8)

    plt.suptitle('Experiment 23: Pan-Cancer SL Atlas (15 TCGA Cancer Types)',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'pan_cancer_atlas.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved pan_cancer_atlas.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 23: Pan-Cancer SL Atlas")
    logger.info("=" * 60)

    # Step 1: Pan-cancer expression landscape
    logger.info("\n--- Step 1: Pan-Cancer Expression Landscape ---")
    landscape = pan_cancer_expression_landscape()
    expr_matrix = build_expression_matrix(landscape)
    logger.info(f"Expression matrix: {expr_matrix.shape}")

    # Step 2: Survival across cancers
    logger.info("\n--- Step 2: Survival Association Across Cancers ---")
    surv_df = survival_across_cancers()
    if not surv_df.empty:
        sig = surv_df[surv_df['fdr'] < 0.05]
        logger.info(f"  Total tests: {len(surv_df)}, Significant (FDR<0.05): {len(sig)}")
        if not sig.empty:
            logger.info(f"  Top hits:\n{sig.nsmallest(10, 'fdr')[['cancer','gene','hr','fdr']].to_string()}")

    # Step 3: Cancer similarity
    logger.info("\n--- Step 3: Cancer Similarity Network ---")
    corr_matrix = None
    gene_cols = [c for c in expr_matrix.columns if c != 'n_samples']
    if len(gene_cols) > 3:
        try:
            corr_matrix, Z, mat_z = cancer_similarity_network(expr_matrix)
            logger.info(f"  Computed {len(corr_matrix)}×{len(corr_matrix)} correlation matrix")
        except Exception as e:
            logger.warning(f"  Similarity failed: {e}")

    # Save
    logger.info("\n--- Saving Results ---")
    expr_matrix.to_csv(RESULTS_DIR / 'pan_cancer_expression.csv')
    if not surv_df.empty:
        surv_df.to_csv(RESULTS_DIR / 'pan_cancer_survival.csv', index=False)
    if corr_matrix is not None:
        corr_matrix.to_csv(RESULTS_DIR / 'cancer_similarity.csv')

    with open(RESULTS_DIR / 'expression_landscape.json', 'w') as f:
        json.dump(landscape, f, indent=2, default=str)

    # Plot
    plot_results(expr_matrix, surv_df, corr_matrix, RESULTS_DIR)

    # Summary
    cancers_profiled = [c for c in ALL_CANCERS if c in landscape]
    summary = {
        'experiment': 'Exp 23: Pan-Cancer SL Atlas',
        'cancers_profiled': cancers_profiled,
        'n_cancers': len(cancers_profiled),
        'genes_profiled': len(ALL_GENES),
        'survival_tests': len(surv_df),
        'survival_significant_fdr05': int((surv_df['fdr'] < 0.05).sum()) if not surv_df.empty else 0,
        'top_pan_cancer_associations': surv_df.nsmallest(10, 'p_value')[
            ['cancer', 'gene', 'hr', 'p_value', 'fdr']
        ].to_dict('records') if not surv_df.empty else [],
    }
    with open(RESULTS_DIR / 'exp23_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nExp 23 COMPLETE")
    logger.info(f"Cancers profiled: {len(cancers_profiled)}")
    logger.info(f"Survival tests: {len(surv_df)}")


if __name__ == '__main__':
    main()
