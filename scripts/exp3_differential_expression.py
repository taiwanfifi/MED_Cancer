#!/usr/bin/env python3
"""
Experiment 3: Differential Gene Expression & Pathway Enrichment
================================================================
Uses TCGA expression data (BRCA, LUAD, KIRC) to:
1. Find differentially expressed genes (tumor vs normal)
2. Cross-reference with synthetic lethality targets from Exp 2
3. Pathway enrichment analysis (GO, KEGG via GSEApy)
4. Cancer-type-specific gene signatures

Target Papers: Paper 5 (Multi-Omics), Paper 4 (SL Map enrichment)
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

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
EXPR_DIR = DATA_DIR / "tcga_expression"
SL_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
RESULTS_DIR = BASE_DIR / "results" / "exp3_differential_expression"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def load_expression_data(cancer_type):
    """Load TCGA expression matrix for a cancer type."""
    expr_file = EXPR_DIR / f"{cancer_type}_expression_matrix.parquet"
    if not expr_file.exists():
        logger.warning(f"Expression data not found for {cancer_type}")
        return None

    df = pd.read_parquet(expr_file)
    logger.info(f"[{cancer_type}] Expression matrix: {df.shape}")
    return df


def identify_tumor_normal(sample_ids):
    """
    TCGA barcode convention:
    - Tumor samples: TCGA-XX-XXXX-01 (01-09 = tumor)
    - Normal samples: TCGA-XX-XXXX-11 (10-19 = normal)
    """
    tumor_mask = []
    normal_mask = []

    for sid in sample_ids:
        parts = str(sid).split('-')
        if len(parts) >= 4:
            sample_type = int(parts[3][:2]) if parts[3][:2].isdigit() else -1
            if 1 <= sample_type <= 9:
                tumor_mask.append(True)
                normal_mask.append(False)
            elif 10 <= sample_type <= 19:
                tumor_mask.append(False)
                normal_mask.append(True)
            else:
                tumor_mask.append(False)
                normal_mask.append(False)
        else:
            # If barcode format is different, assume tumor
            tumor_mask.append(True)
            normal_mask.append(False)

    return np.array(tumor_mask), np.array(normal_mask)


def run_differential_expression(expr_df, cancer_type, min_mean_tpm=1.0):
    """
    Simple differential expression: tumor vs normal using Welch's t-test.
    For proper DESeq2-style analysis, we'd need raw counts, but TPM works
    for a first pass with log-transform.
    """
    samples = expr_df.columns.tolist()
    tumor_mask, normal_mask = identify_tumor_normal(samples)

    n_tumor = tumor_mask.sum()
    n_normal = normal_mask.sum()

    logger.info(f"[{cancer_type}] Tumor samples: {n_tumor}, Normal samples: {n_normal}")

    if n_normal < 5:
        logger.warning(f"[{cancer_type}] Too few normal samples ({n_normal}), "
                       "using top-variance approach instead")
        return run_variance_analysis(expr_df, cancer_type)

    tumor_expr = expr_df.iloc[:, tumor_mask]
    normal_expr = expr_df.iloc[:, normal_mask]

    # Filter low-expression genes
    mean_expr = expr_df.mean(axis=1)
    expressed_genes = mean_expr[mean_expr >= min_mean_tpm].index
    logger.info(f"[{cancer_type}] Genes with mean TPM >= {min_mean_tpm}: {len(expressed_genes)}")

    tumor_expr = tumor_expr.loc[expressed_genes]
    normal_expr = normal_expr.loc[expressed_genes]

    # Log2 transform (add pseudocount)
    tumor_log = np.log2(tumor_expr + 1)
    normal_log = np.log2(normal_expr + 1)

    results = []
    for gene in expressed_genes:
        t_vals = tumor_log.loc[gene].values
        n_vals = normal_log.loc[gene].values

        if np.std(t_vals) == 0 and np.std(n_vals) == 0:
            continue

        t_stat, p_val = stats.ttest_ind(t_vals, n_vals, equal_var=False)
        log2fc = np.mean(t_vals) - np.mean(n_vals)

        results.append({
            'gene': gene,
            'log2FC': log2fc,
            'mean_tumor': np.mean(t_vals),
            'mean_normal': np.mean(n_vals),
            't_statistic': t_stat,
            'p_value': p_val,
            'n_tumor': len(t_vals),
            'n_normal': len(n_vals),
        })

    df = pd.DataFrame(results)

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(df['p_value'], method='fdr_bh')
    df['fdr'] = fdr

    # Sort by absolute log2FC
    df['abs_log2FC'] = df['log2FC'].abs()
    df = df.sort_values('abs_log2FC', ascending=False)

    # Classify
    df['regulation'] = 'NS'  # not significant
    df.loc[(df['fdr'] < 0.05) & (df['log2FC'] > 1), 'regulation'] = 'UP'
    df.loc[(df['fdr'] < 0.05) & (df['log2FC'] < -1), 'regulation'] = 'DOWN'

    n_up = (df['regulation'] == 'UP').sum()
    n_down = (df['regulation'] == 'DOWN').sum()
    logger.info(f"[{cancer_type}] DEGs: {n_up} up, {n_down} down (FDR<0.05, |log2FC|>1)")

    df.to_csv(RESULTS_DIR / f"{cancer_type}_deg_results.csv", index=False)
    return df


def run_variance_analysis(expr_df, cancer_type):
    """Fallback when no normal samples: find high-variance genes."""
    mean_expr = expr_df.mean(axis=1)
    expressed = mean_expr[mean_expr >= 1.0].index
    expr_sub = expr_df.loc[expressed]

    var_df = pd.DataFrame({
        'gene': expressed,
        'mean_tpm': expr_sub.mean(axis=1).values,
        'std_tpm': expr_sub.std(axis=1).values,
        'cv': (expr_sub.std(axis=1) / expr_sub.mean(axis=1)).values,
    })
    var_df = var_df.sort_values('cv', ascending=False)
    var_df.to_csv(RESULTS_DIR / f"{cancer_type}_variance_analysis.csv", index=False)

    logger.info(f"[{cancer_type}] Top variable genes: {var_df.head(10)['gene'].tolist()}")
    return var_df


def cross_reference_sl_targets(deg_results, cancer_type):
    """Cross-reference DEGs with synthetic lethality targets."""
    sl_file = SL_DIR / "synthetic_lethal_pairs.csv"
    if not sl_file.exists():
        logger.warning("SL results not found, skipping cross-reference")
        return None

    sl_df = pd.read_csv(sl_file)
    sl_targets = set(sl_df['target_gene'].unique())
    sl_drivers = set(sl_df['driver_gene'].unique())

    if 'regulation' in deg_results.columns:
        # Has proper DEG results
        up_genes = set(deg_results[deg_results['regulation'] == 'UP']['gene'])
        down_genes = set(deg_results[deg_results['regulation'] == 'DOWN']['gene'])
        all_degs = up_genes | down_genes

        # SL targets that are also DEGs
        sl_deg_overlap = sl_targets & all_degs
        sl_up = sl_targets & up_genes
        sl_down = sl_targets & down_genes

        logger.info(f"\n[{cancer_type}] SL-DEG Cross-reference:")
        logger.info(f"  SL targets: {len(sl_targets)}")
        logger.info(f"  DEGs: {len(all_degs)}")
        logger.info(f"  Overlap: {len(sl_deg_overlap)}")
        logger.info(f"  SL targets upregulated in tumor: {len(sl_up)}")
        logger.info(f"  SL targets downregulated in tumor: {len(sl_down)}")

        if sl_up:
            logger.info(f"  ** Upregulated SL targets (potential drug targets): {sorted(sl_up)}")

        # Also check which drivers are DE
        driver_deg = sl_drivers & all_degs
        logger.info(f"  Driver genes that are DEG: {sorted(driver_deg)}")

        cross_ref = {
            'cancer_type': cancer_type,
            'n_sl_targets': len(sl_targets),
            'n_degs': len(all_degs),
            'overlap': sorted(sl_deg_overlap),
            'sl_upregulated': sorted(sl_up),
            'sl_downregulated': sorted(sl_down),
            'driver_degs': sorted(driver_deg),
        }

        with open(RESULTS_DIR / f"{cancer_type}_sl_deg_crossref.json", 'w') as f:
            json.dump(cross_ref, f, indent=2)

        return cross_ref

    return None


def pathway_enrichment_manual(deg_df, cancer_type, top_n=200):
    """
    Manual pathway enrichment using cancer hallmark gene sets.
    (GSEApy would be better but may not be installed)
    """
    # Cancer hallmark gene sets (curated subset)
    hallmark_sets = {
        'Cell_Cycle': ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CCNA2', 'CCNB1', 'CCND1', 'CCNE1',
                       'E2F1', 'RB1', 'TP53', 'CDKN1A', 'CDKN2A', 'MYC', 'MCM2', 'MCM7',
                       'PCNA', 'BUB1', 'MAD2L1', 'AURKA', 'AURKB', 'PLK1', 'TOP2A', 'KIF11'],
        'DNA_Repair': ['BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'RAD51', 'PARP1',
                       'PARP2', 'MSH2', 'MLH1', 'XRCC1', 'ERCC1', 'ERCC3', 'RFC4', 'RPA1',
                       'FANCD2', 'FANCA', 'NBN', 'MRE11', 'RAD50'],
        'Apoptosis': ['BCL2', 'BAX', 'BAK1', 'BID', 'BCL2L1', 'MCL1', 'CASP3', 'CASP7',
                      'CASP8', 'CASP9', 'CFLAR', 'XIAP', 'BIRC5', 'CYCS', 'APAF1', 'CDKN1A'],
        'PI3K_AKT_mTOR': ['PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2', 'MTOR', 'PTEN',
                          'TSC1', 'TSC2', 'RHEB', 'RPTOR', 'RICTOR', 'EIF4E', 'RPS6KB1',
                          'IRS1', 'IRS2', 'GSK3B'],
        'RAS_MAPK': ['KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1',
                     'MAPK3', 'EGFR', 'ERBB2', 'SOS1', 'GRB2', 'NF1', 'PTPN11', 'DUSP6'],
        'Immune_Response': ['CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD47',
                           'CD8A', 'CD4', 'FOXP3', 'IFNG', 'TNF', 'IL6', 'IL10', 'TGFB1',
                           'CD19', 'CD3E', 'GZMA', 'PRF1'],
        'Angiogenesis': ['VEGFA', 'VEGFB', 'VEGFC', 'KDR', 'FLT1', 'ANGPT1', 'ANGPT2',
                        'TEK', 'HIF1A', 'EPAS1', 'VHL', 'PDGFA', 'PDGFB', 'FGF2'],
        'EMT': ['CDH1', 'CDH2', 'VIM', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2',
                'FN1', 'MMP2', 'MMP9', 'ITGAV', 'ITGB1', 'ACTA2', 'TGFB1'],
        'Metabolism': ['HK2', 'PKM', 'LDHA', 'GAPDH', 'SLC2A1', 'IDH1', 'IDH2', 'SDHB',
                      'SDHC', 'SDHD', 'FH', 'CS', 'ACO2', 'OGDH', 'DLST', 'GPX4',
                      'SLC7A11', 'HMGCR', 'FASN', 'ACLY'],
        'Epigenetics': ['EZH2', 'SUZ12', 'EED', 'DNMT1', 'DNMT3A', 'DNMT3B', 'TET1',
                       'TET2', 'KDM6A', 'KMT2A', 'ARID1A', 'SETD2', 'CREBBP', 'EP300',
                       'BRD4', 'HDAC1', 'HDAC2', 'MBD3', 'NAA10'],
        'Proteasome': ['PSMA2', 'PSMA4', 'PSMB5', 'PSMC2', 'PSMD11', 'UBE2D3', 'UBA1',
                      'NEDD8', 'CUL1', 'SKP1', 'FBXW7', 'VHL'],
        'Mitochondria': ['POLG', 'LARS2', 'MARS2', 'MTG2', 'MTPAP', 'MRPL17', 'MRPL20',
                        'MRPL23', 'MRPL43', 'HSD17B10', 'SDHC', 'SDHD', 'NDUFA13',
                        'UQCRC2', 'COX5B', 'ATP5F1B'],
    }

    if 'regulation' not in deg_df.columns:
        logger.info(f"[{cancer_type}] No DEG regulation column, skipping pathway enrichment")
        return None

    up_genes = set(deg_df[deg_df['regulation'] == 'UP']['gene'])
    down_genes = set(deg_df[deg_df['regulation'] == 'DOWN']['gene'])
    all_tested = set(deg_df['gene'])

    enrichment_results = []

    for pathway_name, pathway_genes in hallmark_sets.items():
        pathway_set = set(pathway_genes) & all_tested
        if len(pathway_set) < 3:
            continue

        # Up enrichment
        up_in_pathway = up_genes & pathway_set
        # Fisher exact test
        a = len(up_in_pathway)  # up AND in pathway
        b = len(up_genes - pathway_set)  # up but NOT in pathway
        c = len(pathway_set - up_genes)  # in pathway but NOT up
        d = len(all_tested - up_genes - pathway_set)  # neither

        _, p_up = stats.fisher_exact([[a, b], [c, d]], alternative='greater')

        # Down enrichment
        down_in_pathway = down_genes & pathway_set
        a2 = len(down_in_pathway)
        b2 = len(down_genes - pathway_set)
        c2 = len(pathway_set - down_genes)
        d2 = len(all_tested - down_genes - pathway_set)

        _, p_down = stats.fisher_exact([[a2, b2], [c2, d2]], alternative='greater')

        enrichment_results.append({
            'pathway': pathway_name,
            'pathway_size': len(pathway_set),
            'n_up_in_pathway': len(up_in_pathway),
            'up_genes': ', '.join(sorted(up_in_pathway)),
            'p_up': p_up,
            'n_down_in_pathway': len(down_in_pathway),
            'down_genes': ', '.join(sorted(down_in_pathway)),
            'p_down': p_down,
        })

    if enrichment_results:
        enrich_df = pd.DataFrame(enrichment_results)
        enrich_df = enrich_df.sort_values('p_up')
        enrich_df.to_csv(RESULTS_DIR / f"{cancer_type}_pathway_enrichment.csv", index=False)

        sig_up = enrich_df[enrich_df['p_up'] < 0.05]
        sig_down = enrich_df[enrich_df['p_down'] < 0.05]
        logger.info(f"[{cancer_type}] Pathways enriched in upregulated genes: "
                    f"{sig_up['pathway'].tolist()}")
        logger.info(f"[{cancer_type}] Pathways enriched in downregulated genes: "
                    f"{sig_down['pathway'].tolist()}")

        return enrich_df

    return None


def plot_volcano(deg_df, cancer_type):
    """Volcano plot of differential expression."""
    if 'regulation' not in deg_df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'UP': '#e74c3c', 'DOWN': '#3498db', 'NS': '#95a5a6'}

    for reg, color in colors.items():
        mask = deg_df['regulation'] == reg
        ax.scatter(deg_df.loc[mask, 'log2FC'],
                   -np.log10(deg_df.loc[mask, 'fdr'] + 1e-300),
                   c=color, alpha=0.4, s=8, label=f"{reg} ({mask.sum()})")

    # Label top genes
    top_up = deg_df[deg_df['regulation'] == 'UP'].head(10)
    top_down = deg_df[deg_df['regulation'] == 'DOWN'].head(10)

    for _, row in pd.concat([top_up, top_down]).iterrows():
        ax.annotate(row['gene'],
                    (row['log2FC'], -np.log10(row['fdr'] + 1e-300)),
                    fontsize=7, alpha=0.8)

    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('log2 Fold Change (Tumor / Normal)', fontsize=12)
    ax.set_ylabel('-log10(FDR)', fontsize=12)
    ax.set_title(f'{cancer_type}: Differential Gene Expression', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cancer_type}_volcano.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {cancer_type}_volcano.png")


def plot_pathway_heatmap(all_enrichment):
    """Heatmap of pathway enrichment across cancer types."""
    if not all_enrichment:
        return

    # Build matrix: -log10(p) for upregulated enrichment
    pathways = set()
    for cancer_type, df in all_enrichment.items():
        pathways.update(df['pathway'].tolist())

    matrix = pd.DataFrame(index=sorted(pathways), columns=list(all_enrichment.keys()))

    for cancer_type, df in all_enrichment.items():
        for _, row in df.iterrows():
            matrix.loc[row['pathway'], cancer_type] = -np.log10(row['p_up'] + 1e-300)

    matrix = matrix.astype(float).fillna(0)

    # Cap at 10 for visualization
    matrix = matrix.clip(upper=10)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap='YlOrRd', ax=ax, annot=True, fmt='.1f',
                linewidths=0.5, vmin=0, vmax=10,
                xticklabels=True, yticklabels=True)
    ax.set_title('Pathway Enrichment: -log10(p) for Upregulated Genes',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Pathway')
    ax.set_xlabel('Cancer Type')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"pathway_enrichment_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: pathway_enrichment_heatmap.png")


def plot_sl_deg_integration(cross_refs):
    """Visualize SL target + DEG overlap across cancer types."""
    if not cross_refs:
        return

    fig, axes = plt.subplots(1, len(cross_refs), figsize=(6*len(cross_refs), 6))
    if len(cross_refs) == 1:
        axes = [axes]

    for ax, cr in zip(axes, cross_refs):
        ct = cr['cancer_type']
        n_sl = cr['n_sl_targets']
        n_deg = cr['n_degs']
        overlap = len(cr['overlap'])

        # Simple bar chart
        categories = ['SL Targets', 'DEGs', 'Overlap', 'SL↑ in Tumor', 'SL↓ in Tumor']
        values = [n_sl, n_deg, overlap, len(cr['sl_upregulated']), len(cr['sl_downregulated'])]
        colors = ['#3498db', '#e74c3c', '#9b59b6', '#e67e22', '#1abc9c']

        bars = ax.bar(range(len(categories)), values, color=colors, edgecolor='white')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Number of Genes')
        ax.set_title(f'{ct}: SL-DEG Integration', fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sl_deg_integration.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: sl_deg_integration.png")


def generate_summary(all_deg_results, all_enrichment, cross_refs):
    """Generate comprehensive summary."""
    summary = {
        'experiment': 'Exp 3: Differential Expression & Pathway Enrichment',
        'cancer_types_analyzed': [],
        'key_findings': [],
        'cross_references': cross_refs,
    }

    for cancer_type, deg_df in all_deg_results.items():
        ct_summary = {'cancer_type': cancer_type}

        if 'regulation' in deg_df.columns:
            ct_summary['n_up'] = int((deg_df['regulation'] == 'UP').sum())
            ct_summary['n_down'] = int((deg_df['regulation'] == 'DOWN').sum())
            ct_summary['top_up_genes'] = deg_df[deg_df['regulation'] == 'UP'].head(20)['gene'].tolist()
            ct_summary['top_down_genes'] = deg_df[deg_df['regulation'] == 'DOWN'].head(20)['gene'].tolist()
        else:
            ct_summary['analysis_type'] = 'variance'
            ct_summary['top_variable_genes'] = deg_df.head(20)['gene'].tolist()

        summary['cancer_types_analyzed'].append(ct_summary)

    # Key discoveries
    for cr in cross_refs:
        if cr and cr.get('sl_upregulated'):
            summary['key_findings'].append(
                f"{cr['cancer_type']}: SL targets upregulated in tumor: {cr['sl_upregulated']} — "
                "These are prime drug target candidates (essential when driver is mutated AND overexpressed in tumor)"
            )

    with open(RESULTS_DIR / "exp3_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nSummary saved to exp3_summary.json")
    return summary


def main():
    logger.info("=" * 60)
    logger.info("Experiment 3: Differential Expression & Pathway Enrichment")
    logger.info("=" * 60)

    all_deg_results = {}
    all_enrichment = {}
    cross_refs = []

    for cancer_type in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer_type}")
        logger.info(f"{'='*40}")

        # Load expression data
        expr_df = load_expression_data(cancer_type)
        if expr_df is None:
            continue

        # Run differential expression
        deg_df = run_differential_expression(expr_df, cancer_type)
        all_deg_results[cancer_type] = deg_df

        # Volcano plot
        plot_volcano(deg_df, cancer_type)

        # Cross-reference with SL targets
        cr = cross_reference_sl_targets(deg_df, cancer_type)
        if cr:
            cross_refs.append(cr)

        # Pathway enrichment
        enrich = pathway_enrichment_manual(deg_df, cancer_type)
        if enrich is not None:
            all_enrichment[cancer_type] = enrich

    # Cross-cancer pathway comparison
    plot_pathway_heatmap(all_enrichment)

    # SL-DEG integration visualization
    plot_sl_deg_integration(cross_refs)

    # Summary
    summary = generate_summary(all_deg_results, all_enrichment, cross_refs)

    # Print key findings
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)
    for finding in summary.get('key_findings', []):
        logger.info(f"  ** {finding}")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 3 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
