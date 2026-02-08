#!/usr/bin/env python3
"""
Experiment 13: Literature Validation via PubMed Mining
======================================================
Validates top therapeutic targets from Exp 12 against PubMed literature.
Searches for each gene + cancer/therapy/synthetic lethality terms.

Pipeline:
1. Load top genes from Exp 12 evidence matrix
2. Query PubMed E-utilities API for publication counts
3. Score literature support for each gene
4. Compare computational vs literature ranking
5. Identify under-studied genes with high computational evidence (novel opportunities)

Target: All papers — provides literature validation layer
"""

import json
import logging
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path("/workspace/cancer_research")
RESULTS_DIR = BASE_DIR / "results" / "exp13_literature"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def search_pubmed(query, retmax=0):
    """Search PubMed and return count of results."""
    import requests

    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': retmax,
        'rettype': 'count',
    }

    try:
        r = requests.get(PUBMED_SEARCH_URL, params=params, timeout=30)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            count = root.find('.//Count')
            if count is not None:
                return int(count.text)
    except Exception as e:
        logger.debug(f"PubMed search failed for '{query}': {e}")

    return 0


def mine_literature(genes_df, top_n=50):
    """Mine PubMed for evidence supporting top genes."""
    top_genes = genes_df.head(top_n)

    search_terms = {
        'cancer': '{gene} cancer',
        'therapeutic_target': '{gene} therapeutic target cancer',
        'synthetic_lethality': '{gene} synthetic lethality',
        'drug_resistance': '{gene} drug resistance cancer',
        'biomarker': '{gene} biomarker cancer prognosis',
        'clinical_trial': '{gene} clinical trial cancer',
        'immunotherapy': '{gene} immunotherapy',
    }

    results = []

    for idx, (_, row) in enumerate(top_genes.iterrows()):
        gene = row['gene']
        gene_result = {'gene': gene, 'composite_score': row['composite_score']}

        total_pubs = 0
        for term_name, term_template in search_terms.items():
            query = term_template.format(gene=gene)
            count = search_pubmed(query)
            gene_result[f'pub_{term_name}'] = count
            total_pubs += count

            # Rate limit: 3 requests/sec for NCBI without API key
            time.sleep(0.35)

        gene_result['total_publications'] = total_pubs
        gene_result['pub_cancer_count'] = gene_result['pub_cancer']

        results.append(gene_result)
        logger.info(f"  [{idx+1}/{top_n}] {gene}: cancer={gene_result['pub_cancer']}, "
                    f"SL={gene_result['pub_synthetic_lethality']}, "
                    f"trial={gene_result['pub_clinical_trial']}, "
                    f"total={total_pubs}")

    return pd.DataFrame(results)


def identify_novel_opportunities(lit_df, genes_df):
    """Find genes with high computational evidence but low literature coverage."""
    merged = lit_df.merge(genes_df[['gene', 'composite_score', 'sl_n_drivers',
                                      'n_approved_drugs', 'dep_auc']],
                           on='gene', how='left', suffixes=('', '_orig'))

    # Normalize both scores
    if merged['pub_cancer'].max() > 0:
        merged['lit_score'] = merged['pub_cancer'] / merged['pub_cancer'].max()
    else:
        merged['lit_score'] = 0

    merged['comp_score_norm'] = merged['composite_score'] / merged['composite_score'].max()

    # Gap = computational evidence - literature evidence
    merged['novelty_gap'] = merged['comp_score_norm'] - merged['lit_score']

    # Under-studied high-evidence genes
    novel = merged[merged['novelty_gap'] > 0.3].sort_values('novelty_gap', ascending=False)
    well_studied = merged[merged['lit_score'] > 0.5].sort_values('composite_score', ascending=False)

    return merged, novel, well_studied


def plot_literature_validation(lit_df, merged_df, novel_df):
    """Create publication-ready literature validation figures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: Computational vs Literature ranking
    ax = axes[0, 0]
    ax.scatter(merged_df['comp_score_norm'], merged_df['lit_score'],
              s=60, c='#3498db', alpha=0.6, edgecolor='white')
    for _, row in merged_df.head(15).iterrows():
        ax.annotate(row['gene'], (row['comp_score_norm'], row['lit_score']),
                   fontsize=7, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Perfect correlation')
    ax.set_xlabel('Computational Evidence Score (normalized)')
    ax.set_ylabel('Literature Evidence Score (normalized)')
    ax.set_title('A) Computational vs Literature Validation', fontweight='bold')
    ax.legend()

    # Highlight novel opportunities (high comp, low lit)
    if len(novel_df) > 0:
        ax.scatter(novel_df['comp_score_norm'], novel_df['lit_score'],
                  s=100, c='red', marker='*', zorder=5, label='Novel opportunities')
        ax.legend()

    # Panel B: Publication counts by category for top 20 genes
    ax = axes[0, 1]
    top20 = lit_df.head(20)
    categories = ['pub_cancer', 'pub_therapeutic_target', 'pub_synthetic_lethality',
                  'pub_clinical_trial', 'pub_immunotherapy']
    cat_labels = ['Cancer', 'Target', 'SL', 'Trial', 'Immuno']

    x = np.arange(len(top20))
    width = 0.15
    for i, (cat, label) in enumerate(zip(categories, cat_labels)):
        vals = np.log10(top20[cat].clip(lower=1))
        ax.bar(x + i * width, vals, width, label=label, alpha=0.8)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(top20['gene'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('log10(Publication Count)')
    ax.set_title('B) Literature Support by Category (Top 20)', fontweight='bold')
    ax.legend(fontsize=8, ncol=3)

    # Panel C: Novel targets (high computational, low literature)
    ax = axes[1, 0]
    if len(novel_df) > 0:
        novel_top = novel_df.head(15)
        y_pos = range(len(novel_top))
        ax.barh(y_pos, novel_top['novelty_gap'], color='#e74c3c', alpha=0.8, edgecolor='white')
        ax.set_yticks(y_pos)
        labels = [f"{r['gene']} (comp={r['comp_score_norm']:.2f}, lit={r['lit_score']:.2f})"
                 for _, r in novel_top.iterrows()]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Novelty Gap (Comp. Score - Lit. Score)')
        ax.set_title('C) Under-Studied Targets with High Evidence', fontweight='bold')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No novel opportunities\n(all genes well-studied)',
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('C) Under-Studied Targets', fontweight='bold')

    # Panel D: Correlation matrix
    ax = axes[1, 1]
    corr_cols = ['composite_score', 'pub_cancer', 'pub_therapeutic_target',
                 'pub_synthetic_lethality', 'pub_clinical_trial']
    corr_labels = ['Comp. Score', 'Cancer Pubs', 'Target Pubs', 'SL Pubs', 'Trial Pubs']
    corr_data = merged_df[corr_cols].corr()
    corr_data.index = corr_labels
    corr_data.columns = corr_labels
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('D) Evidence Correlation Matrix', fontweight='bold')

    plt.suptitle('Literature Validation of Computational Therapeutic Targets',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "literature_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: literature_validation.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 13: Literature Validation via PubMed")
    logger.info("=" * 60)

    # Load gene evidence matrix from Exp 12
    evidence_file = BASE_DIR / "results" / "exp12_integration" / "gene_evidence_matrix.csv"
    if not evidence_file.exists():
        logger.error("Gene evidence matrix not found. Run Exp 12 first.")
        return

    genes_df = pd.read_csv(evidence_file)
    logger.info(f"Loaded {len(genes_df)} genes from evidence matrix")

    # Mine PubMed for top 50 genes
    logger.info("\n--- Mining PubMed for top 50 genes ---")
    lit_df = mine_literature(genes_df, top_n=50)
    lit_df.to_csv(RESULTS_DIR / "pubmed_literature_counts.csv", index=False)

    # Identify novel opportunities
    logger.info("\n--- Identifying novel opportunities ---")
    merged_df, novel_df, well_studied_df = identify_novel_opportunities(lit_df, genes_df)
    merged_df.to_csv(RESULTS_DIR / "literature_computational_comparison.csv", index=False)

    if len(novel_df) > 0:
        novel_df.to_csv(RESULTS_DIR / "novel_understudied_targets.csv", index=False)
        logger.info(f"\nNovel under-studied targets: {len(novel_df)}")
        for _, row in novel_df.head(10).iterrows():
            logger.info(f"  {row['gene']}: comp={row['comp_score_norm']:.3f}, "
                       f"lit={row['lit_score']:.3f}, gap={row['novelty_gap']:.3f}")

    logger.info(f"\nWell-studied targets: {len(well_studied_df)}")
    for _, row in well_studied_df.head(10).iterrows():
        logger.info(f"  {row['gene']}: {int(row['pub_cancer'])} cancer pubs, "
                   f"comp={row['composite_score']:.3f}")

    # Compute Spearman correlation between computational and literature ranking
    from scipy.stats import spearmanr
    rho, pval = spearmanr(merged_df['comp_score_norm'], merged_df['lit_score'])
    logger.info(f"\nSpearman rho (comp vs lit): {rho:.3f} (p={pval:.4f})")

    # Plot
    logger.info("\n--- Generating visualizations ---")
    plot_literature_validation(lit_df, merged_df, novel_df)

    # Summary
    summary = {
        'experiment': 'Exp 13: Literature Validation',
        'genes_queried': len(lit_df),
        'total_queries': len(lit_df) * 7,
        'spearman_rho': float(rho),
        'spearman_pval': float(pval),
        'novel_targets': len(novel_df),
        'well_studied_targets': len(well_studied_df),
        'top_novel': novel_df.head(10)[['gene', 'novelty_gap', 'pub_cancer']].to_dict('records') if len(novel_df) > 0 else [],
        'top_studied': well_studied_df.head(10)[['gene', 'pub_cancer', 'composite_score']].to_dict('records'),
    }

    with open(RESULTS_DIR / "exp13_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 13 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
