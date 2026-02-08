#!/usr/bin/env python3
"""
Experiment 2: Synthetic Lethality Discovery from DepMap CRISPR Data
====================================================================
Uses DepMap CRISPR gene effect scores to identify synthetic lethal
gene pairs across cancer cell lines.

Target Paper: Paper 4 (Pan-Cancer Synthetic Lethality Map)
Hypothesis: H3 - Synthetic lethality is cancer-type transferable

Key Concept: If Gene A is mutated in a cancer cell line, and knocking
out Gene B kills that cell line but NOT cell lines where Gene A is
intact, then A-B is a synthetic lethal pair.

This is one of the most promising directions for novel cancer therapies.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "drug_repurpose"
RESULTS_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Known driver genes with frequent mutations
DRIVER_GENES = [
    'TP53', 'KRAS', 'BRAF', 'PIK3CA', 'PTEN', 'RB1', 'CDKN2A',
    'EGFR', 'ERBB2', 'MYC', 'APC', 'VHL', 'NF1', 'STK11',
    'ARID1A', 'KEAP1', 'SMAD4', 'BRCA1', 'BRCA2', 'ATM',
    'NRAS', 'IDH1', 'IDH2', 'CTNNB1', 'FBXW7',
]


def load_depmap_data():
    """Load DepMap gene effect data."""
    gene_effect_file = DATA_DIR / "depmap_gene_effect.parquet"
    if not gene_effect_file.exists():
        logger.error("DepMap data not found! Run download_drug_data.py first.")
        return None, None

    logger.info("Loading DepMap gene effect data...")
    gene_effect = pd.read_parquet(gene_effect_file)
    logger.info(f"Gene effect matrix: {gene_effect.shape} (cell lines x genes)")

    # Clean column names (format: "GENE (ENTREZ_ID)")
    clean_cols = {}
    for col in gene_effect.columns:
        gene_name = col.split(' ')[0] if ' ' in col else col
        clean_cols[col] = gene_name

    gene_effect = gene_effect.rename(columns=clean_cols)

    # Remove duplicate columns (keep first)
    gene_effect = gene_effect.loc[:, ~gene_effect.columns.duplicated()]

    logger.info(f"After cleaning: {gene_effect.shape}")
    return gene_effect


def identify_mutated_lines(gene_effect, driver_gene, threshold=-0.5):
    """
    Identify cell lines where a driver gene is likely mutated/lost.
    We use the gene effect score itself as a proxy:
    - If a gene has very negative effect (essential), it's likely functional
    - If effect is near 0 or positive, the gene may be already lost/mutated
    """
    if driver_gene not in gene_effect.columns:
        return None, None

    scores = gene_effect[driver_gene]

    # For tumor suppressors: mutated lines have LESS dependency (higher scores)
    # because the gene is already non-functional
    # Threshold: if gene effect > -0.2, likely already mutated
    mutated = scores > -0.2  # cell lines where gene is NOT essential (likely mutated)
    wildtype = scores <= -0.5  # cell lines where gene IS essential (likely intact)

    return mutated, wildtype


def find_synthetic_lethal_pairs(gene_effect, driver_gene, n_top=50):
    """
    Find genes that are selectively essential in cells where driver_gene is mutated.
    This identifies synthetic lethal partners.
    """
    mutated, wildtype = identify_mutated_lines(gene_effect, driver_gene)

    if mutated is None or mutated.sum() < 20 or wildtype.sum() < 20:
        logger.warning(f"Insufficient data for {driver_gene}: "
                      f"mutated={mutated.sum() if mutated is not None else 0}, "
                      f"wildtype={wildtype.sum() if wildtype is not None else 0}")
        return None

    results = []

    for target_gene in tqdm(gene_effect.columns, desc=f"SL scan for {driver_gene}", leave=False):
        if target_gene == driver_gene:
            continue

        scores_mut = gene_effect.loc[mutated, target_gene].dropna()
        scores_wt = gene_effect.loc[wildtype, target_gene].dropna()

        if len(scores_mut) < 10 or len(scores_wt) < 10:
            continue

        # t-test: is the target gene more essential (more negative) in mutated lines?
        t_stat, p_value = stats.ttest_ind(scores_mut, scores_wt, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((scores_mut.std()**2 + scores_wt.std()**2) / 2)
        cohens_d = (scores_mut.mean() - scores_wt.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            'driver_gene': driver_gene,
            'target_gene': target_gene,
            'mean_effect_mutated': scores_mut.mean(),
            'mean_effect_wildtype': scores_wt.mean(),
            'delta_effect': scores_mut.mean() - scores_wt.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n_mutated': len(scores_mut),
            'n_wildtype': len(scores_wt),
        })

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # Multiple testing correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['fdr'] = fdr

    # Sort by delta effect (most selectively essential in mutated lines)
    results_df = results_df.sort_values('delta_effect')

    return results_df


def run_pan_cancer_sl_analysis(gene_effect):
    """Run SL analysis for all driver genes."""
    all_results = []

    for driver in DRIVER_GENES:
        logger.info(f"Analyzing SL partners for {driver}...")
        sl_df = find_synthetic_lethal_pairs(gene_effect, driver)

        if sl_df is not None:
            # Keep top SL candidates (most negative delta = more essential in mutated)
            top_sl = sl_df[
                (sl_df['fdr'] < 0.05) &
                (sl_df['delta_effect'] < -0.2) &
                (sl_df['cohens_d'] < -0.3)
            ].head(50)

            if len(top_sl) > 0:
                logger.info(f"  {driver}: {len(top_sl)} significant SL partners")
                logger.info(f"  Top 5: {top_sl['target_gene'].head(5).tolist()}")
                all_results.append(top_sl)
            else:
                logger.info(f"  {driver}: No significant SL partners found")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "synthetic_lethal_pairs.csv", index=False)
        logger.info(f"\nTotal SL pairs found: {len(combined)}")
        return combined

    return None


def plot_sl_results(sl_df):
    """Generate publication-quality figures for SL analysis."""
    if sl_df is None or len(sl_df) == 0:
        return

    # Figure 1: SL partner count by driver gene
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: Number of SL partners per driver
    ax = axes[0]
    sl_counts = sl_df.groupby('driver_gene').size().sort_values(ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sl_counts)))
    sl_counts.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Number of Synthetic Lethal Partners (FDR < 0.05)')
    ax.set_title('A) SL Partners per Driver Gene', fontweight='bold')

    # Panel B: Top SL pairs by effect size
    ax = axes[1]
    top_pairs = sl_df.nsmallest(20, 'delta_effect')
    labels = [f"{r['driver_gene']}-{r['target_gene']}" for _, r in top_pairs.iterrows()]
    ax.barh(range(len(top_pairs)), top_pairs['delta_effect'].values,
            color=plt.cm.RdBu(np.linspace(0, 0.4, len(top_pairs))))
    ax.set_yticks(range(len(top_pairs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Differential Gene Effect (Mutated - WT)')
    ax.set_title('B) Top 20 Synthetic Lethal Pairs', fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sl_discovery_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: sl_discovery_overview.png")

    # Figure 2: Heatmap of SL relationships
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create driver x target matrix
    pivot = sl_df.pivot_table(
        values='delta_effect',
        index='driver_gene',
        columns='target_gene',
        aggfunc='first'
    )

    # Keep top 30 targets by average effect
    top_targets = sl_df.groupby('target_gene')['delta_effect'].mean().nsmallest(30).index
    pivot_sub = pivot[pivot.columns.intersection(top_targets)]

    if len(pivot_sub.columns) > 0:
        sns.heatmap(pivot_sub, cmap='RdBu', center=0, ax=ax,
                   xticklabels=True, yticklabels=True,
                   linewidths=0.5, vmin=-1, vmax=0.5)
        ax.set_title('Synthetic Lethality Map: Driver Gene x Target Gene',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Gene (Potential Drug Target)')
        ax.set_ylabel('Driver Gene (Mutated)')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "sl_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved: sl_heatmap.png")

    # Figure 3: Volcano plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for driver in sl_df['driver_gene'].unique():
        driver_data = sl_df[sl_df['driver_gene'] == driver]
        ax.scatter(
            driver_data['delta_effect'],
            -np.log10(driver_data['fdr'] + 1e-300),
            alpha=0.3, s=10, label=driver
        )

    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='FDR=0.05')
    ax.axvline(x=-0.2, color='blue', linestyle='--', alpha=0.5, label='Effect=-0.2')
    ax.set_xlabel('Differential Gene Effect (Mutated - WT)')
    ax.set_ylabel('-log10(FDR)')
    ax.set_title('Synthetic Lethality Volcano Plot', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sl_volcano.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: sl_volcano.png")


def analyze_druggability(sl_df):
    """Check which SL targets are druggable."""
    # Load DGIdb data if available
    dgidb_file = DATA_DIR / "dgidb_interactions.parquet"
    if not dgidb_file.exists():
        logger.info("DGIdb data not available, skipping druggability analysis")
        return

    dgidb = pd.read_parquet(dgidb_file)

    # Get unique SL targets
    sl_targets = set(sl_df['target_gene'].unique())

    # Check which are druggable
    druggable_genes = set()
    if 'gene' in dgidb.columns:
        druggable_genes = set(dgidb['gene'].unique())
    elif 'gene_name' in dgidb.columns:
        druggable_genes = set(dgidb['gene_name'].unique())

    druggable_sl = sl_targets.intersection(druggable_genes)
    novel_sl = sl_targets - druggable_genes

    logger.info(f"\nDruggability Analysis:")
    logger.info(f"  Total SL targets: {len(sl_targets)}")
    logger.info(f"  Already druggable: {len(druggable_sl)} ({100*len(druggable_sl)/len(sl_targets):.1f}%)")
    logger.info(f"  Novel (undrugged): {len(novel_sl)}")

    # Save druggability report
    druggability = {
        'total_sl_targets': len(sl_targets),
        'druggable': sorted(list(druggable_sl)),
        'novel': sorted(list(novel_sl))[:100],  # Top 100
        'druggable_pct': len(druggable_sl) / len(sl_targets) * 100 if sl_targets else 0,
    }

    with open(RESULTS_DIR / "druggability_report.json", 'w') as f:
        json.dump(druggability, f, indent=2)


def main():
    logger.info("=" * 60)
    logger.info("Experiment 2: Synthetic Lethality Discovery")
    logger.info("=" * 60)

    # Load data
    gene_effect = load_depmap_data()
    if gene_effect is None:
        return

    # Check which driver genes are in the data
    available_drivers = [g for g in DRIVER_GENES if g in gene_effect.columns]
    logger.info(f"Available driver genes: {len(available_drivers)}/{len(DRIVER_GENES)}")
    logger.info(f"Drivers: {available_drivers}")

    # Run SL analysis
    sl_df = run_pan_cancer_sl_analysis(gene_effect)

    if sl_df is not None and len(sl_df) > 0:
        # Generate figures
        plot_sl_results(sl_df)

        # Druggability analysis
        analyze_druggability(sl_df)

        # Summary stats
        logger.info("\n=== Summary ===")
        logger.info(f"Total SL pairs: {len(sl_df)}")
        logger.info(f"Driver genes with SL partners: {sl_df['driver_gene'].nunique()}")
        logger.info(f"Unique SL targets: {sl_df['target_gene'].nunique()}")

        # Most connected drivers
        top_drivers = sl_df.groupby('driver_gene').size().sort_values(ascending=False)
        logger.info(f"\nMost SL-connected drivers:\n{top_drivers.head(10).to_string()}")

        # Most commonly targeted genes
        top_targets = sl_df.groupby('target_gene').size().sort_values(ascending=False)
        logger.info(f"\nMost common SL targets:\n{top_targets.head(10).to_string()}")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
