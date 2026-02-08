#!/usr/bin/env python3
"""
Experiment 21: Network Medicine — SL Pathway Enrichment & Cross-Talk
=====================================================================
Performs pathway-level analysis of synthetic lethality interactions
to identify targetable biological processes rather than individual genes.

Key analyses:
1. Map SL pairs to KEGG/Reactome pathways
2. Identify pathway-level SL interactions (e.g., DNA repair ↔ cell cycle)
3. Network propagation: which pathways are central SL hubs?
4. Cross-cancer pathway vulnerability comparison
5. Druggable pathway opportunities

Target: Network medicine paper on pathway-level SL
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp21_network_medicine"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Curated cancer-relevant pathway definitions (KEGG-inspired)
PATHWAYS = {
    'Cell_Cycle': ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2',
                   'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNE2', 'RB1', 'E2F1', 'E2F3',
                   'CDC20', 'CDC25A', 'CDC25B', 'CDK7', 'CDKN1A', 'CDKN1B', 'CDKN2A'],
    'DNA_Repair': ['BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'RAD51', 'PARP1',
                   'PARP2', 'MLH1', 'MSH2', 'MSH6', 'XRCC1', 'XRCC4', 'FANCA', 'FANCD2',
                   'POLQ', 'REV3L', 'LIG4', 'NBN'],
    'PI3K_AKT_mTOR': ['PIK3CA', 'PIK3CB', 'PIK3R1', 'AKT1', 'AKT2', 'AKT3', 'MTOR',
                       'PTEN', 'TSC1', 'TSC2', 'RPTOR', 'RICTOR', 'STK11', 'RHEB'],
    'RAS_MAPK': ['KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'MAP2K1', 'MAP2K2',
                 'MAPK1', 'MAPK3', 'NF1', 'SOS1', 'GRB2', 'EGFR', 'ERBB2'],
    'p53_Apoptosis': ['TP53', 'MDM2', 'MDM4', 'BCL2', 'BCL2L1', 'BAX', 'BAK1', 'BIM',
                      'CASP3', 'CASP8', 'CASP9', 'CYCS', 'APAF1', 'BIRC5', 'MCL1'],
    'Wnt_Signaling': ['CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'GSK3B', 'WNT1', 'WNT3A',
                      'FZD1', 'LRP5', 'LRP6', 'TCF7L2', 'LEF1', 'DVL1', 'DVL2'],
    'Chromatin_Remodeling': ['SMARCA4', 'SMARCA2', 'ARID1A', 'ARID1B', 'PBRM1',
                              'EZH2', 'SUZ12', 'EED', 'BRD4', 'HDAC1', 'HDAC2',
                              'KDM1A', 'KDM6A', 'KMT2A', 'KMT2D', 'DNMT1', 'DNMT3A'],
    'Ferroptosis': ['GPX4', 'SLC7A11', 'ACSL4', 'LPCAT3', 'FSP1', 'GCLC', 'GCLM',
                    'GSS', 'TFRC', 'NCOA4', 'NFE2L2', 'HMOX1', 'FTH1', 'DHODH'],
    'Immune_Checkpoint': ['CD274', 'PDCD1LG2', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2',
                          'TIGIT', 'CD80', 'CD86', 'IDO1', 'VSIR', 'BTLA'],
    'RTK_Signaling': ['EGFR', 'ERBB2', 'ERBB3', 'FGFR1', 'FGFR2', 'FGFR3',
                      'MET', 'RET', 'ALK', 'ROS1', 'VEGFA', 'KDR', 'PDGFRA', 'KIT'],
    'Metabolism': ['HK2', 'PKM', 'LDHA', 'SLC2A1', 'IDH1', 'IDH2', 'FASN',
                   'ACLY', 'GLS', 'OGDH', 'CS', 'SDHA', 'FH', 'VHL'],
    'MYC_Network': ['MYC', 'MYCN', 'MAX', 'MXD1', 'MXI1', 'AURKA', 'AURKB',
                    'BRD4', 'CDK9', 'WDR5', 'KAT2A'],
}


def load_sl_pairs():
    """Load SL pairs from multiple experiments."""
    sl_files = [
        BASE_DIR / "results" / "exp2_synthetic_lethality" / "synthetic_lethal_pairs.csv",
        BASE_DIR / "results" / "exp4_sl_transferability" / "universal_sl_pairs.csv",
    ]
    all_sl = []
    for f in sl_files:
        if f.exists():
            df = pd.read_csv(f)
            all_sl.append(df)
            logger.info(f"  Loaded {len(df)} pairs from {f.name}")
    if all_sl:
        combined = pd.concat(all_sl, ignore_index=True)
        # Deduplicate
        gene_cols = [c for c in combined.columns if 'gene' in c.lower()][:2]
        if len(gene_cols) >= 2:
            combined['pair_key'] = combined.apply(
                lambda r: tuple(sorted([str(r[gene_cols[0]]), str(r[gene_cols[1]])])), axis=1)
            combined = combined.drop_duplicates(subset='pair_key')
        logger.info(f"  Total unique SL pairs: {len(combined)}")
        return combined
    return pd.DataFrame()


def map_genes_to_pathways():
    """Create gene → pathway mapping."""
    gene_to_pathways = defaultdict(list)
    for pathway, genes in PATHWAYS.items():
        for gene in genes:
            gene_to_pathways[gene].append(pathway)
    return gene_to_pathways


def pathway_sl_enrichment(sl_pairs):
    """
    Map SL pairs to pathway-pathway interactions.
    Test: are certain pathway combinations enriched in SL pairs?
    """
    gene_to_pw = map_genes_to_pathways()
    gene_cols = [c for c in sl_pairs.columns if 'gene' in c.lower()][:2]

    # Count pathway-pathway interactions
    pw_interactions = Counter()
    pw_gene_pairs = defaultdict(list)
    mapped_pairs = 0

    for _, row in sl_pairs.iterrows():
        a, b = str(row[gene_cols[0]]), str(row[gene_cols[1]])
        pws_a = gene_to_pw.get(a, [])
        pws_b = gene_to_pw.get(b, [])

        if pws_a and pws_b:
            mapped_pairs += 1
            for pa in pws_a:
                for pb in pws_b:
                    key = tuple(sorted([pa, pb]))
                    pw_interactions[key] += 1
                    pw_gene_pairs[key].append((a, b))

    logger.info(f"  Mapped {mapped_pairs}/{len(sl_pairs)} SL pairs to pathway interactions")

    # Build pathway interaction matrix
    all_pathways = sorted(PATHWAYS.keys())
    matrix = pd.DataFrame(0, index=all_pathways, columns=all_pathways, dtype=int)
    for (pa, pb), count in pw_interactions.items():
        matrix.loc[pa, pb] = count
        matrix.loc[pb, pa] = count

    # Statistical test: are cross-pathway interactions enriched?
    results = []
    for key, count in pw_interactions.most_common(50):
        pa, pb = key
        n_a = len(PATHWAYS[pa])
        n_b = len(PATHWAYS[pb])
        n_sl = len(sl_pairs)
        n_genes = len(gene_to_pw)

        # Expected under independence: (n_a * n_b) / total_possible_pairs * n_sl
        total_possible = n_genes * (n_genes - 1) / 2
        expected = (n_a * n_b) / total_possible * n_sl if total_possible > 0 else 0

        enrichment = count / expected if expected > 0 else float('inf')

        example_pairs = pw_gene_pairs[key][:3]

        results.append({
            'pathway_a': pa,
            'pathway_b': pb,
            'n_sl_pairs': count,
            'expected': round(expected, 2),
            'enrichment_ratio': round(enrichment, 2),
            'genes_a': n_a,
            'genes_b': n_b,
            'example_pairs': str(example_pairs),
        })

    enrichment_df = pd.DataFrame(results).sort_values('n_sl_pairs', ascending=False)
    return matrix, enrichment_df


def pathway_hub_analysis(sl_pairs):
    """
    Identify which pathways are central SL hubs.
    A hub pathway has many SL connections to diverse other pathways.
    """
    gene_to_pw = map_genes_to_pathways()
    gene_cols = [c for c in sl_pairs.columns if 'gene' in c.lower()][:2]

    # Count pathway-level degree
    pw_degree = Counter()
    pw_partners = defaultdict(set)

    for _, row in sl_pairs.iterrows():
        a, b = str(row[gene_cols[0]]), str(row[gene_cols[1]])
        pws_a = gene_to_pw.get(a, [])
        pws_b = gene_to_pw.get(b, [])

        for pa in pws_a:
            pw_degree[pa] += 1
            for pb in pws_b:
                if pa != pb:
                    pw_partners[pa].add(pb)
        for pb in pws_b:
            pw_degree[pb] += 1
            for pa in pws_a:
                if pa != pb:
                    pw_partners[pb].add(pa)

    hub_results = []
    for pw in sorted(PATHWAYS.keys()):
        hub_results.append({
            'pathway': pw,
            'n_genes': len(PATHWAYS[pw]),
            'sl_connections': pw_degree.get(pw, 0),
            'connected_pathways': len(pw_partners.get(pw, set())),
            'partner_pathways': ', '.join(sorted(pw_partners.get(pw, set()))),
            'hub_score': pw_degree.get(pw, 0) * len(pw_partners.get(pw, set())),
        })

    hub_df = pd.DataFrame(hub_results).sort_values('hub_score', ascending=False)
    return hub_df


def cross_cancer_pathway_vulnerability(gene_effect, cell_line_info):
    """
    Compare pathway-level vulnerability across cancer lineages.
    """
    if 'lineage' in cell_line_info.columns:
        lineage_col = 'lineage'
    elif 'primary_disease' in cell_line_info.columns:
        lineage_col = 'primary_disease'
    else:
        lineage_col = cell_line_info.columns[1]

    lineages = cell_line_info[lineage_col].value_counts()
    top_lineages = lineages[lineages >= 15].index.tolist()[:10]

    pw_vulnerability = {}
    for lineage in top_lineages:
        lines = cell_line_info[cell_line_info[lineage_col] == lineage].index
        lines = lines.intersection(gene_effect.index)
        if len(lines) < 10:
            continue

        ge_sub = gene_effect.loc[lines]
        pw_scores = {}
        for pw, genes in PATHWAYS.items():
            matched = [g for g in genes if g in ge_sub.columns]
            if matched:
                pw_scores[pw] = float(ge_sub[matched].mean().mean())
        pw_vulnerability[lineage] = pw_scores

    vuln_df = pd.DataFrame(pw_vulnerability)
    return vuln_df


def drug_pathway_opportunities(sl_pairs, drug_interactions):
    """
    Identify druggable pathway-level SL opportunities.
    """
    gene_to_pw = map_genes_to_pathways()

    # Get druggable genes
    if drug_interactions.empty:
        return pd.DataFrame()

    drug_gene_col = 'gene_name' if 'gene_name' in drug_interactions.columns else drug_interactions.columns[0]
    drug_col = 'drug_name' if 'drug_name' in drug_interactions.columns else [c for c in drug_interactions.columns if 'drug' in c.lower()][0]

    druggable = set(drug_interactions[drug_gene_col].dropna().unique())
    gene_drugs = drug_interactions.groupby(drug_gene_col)[drug_col].apply(
        lambda x: list(x.dropna().unique()[:5])
    ).to_dict()

    # For each pathway, count druggable genes and their SL partners
    results = []
    for pw, genes in PATHWAYS.items():
        druggable_in_pw = [g for g in genes if g in druggable]
        n_drugs = sum(len(gene_drugs.get(g, [])) for g in druggable_in_pw)

        results.append({
            'pathway': pw,
            'total_genes': len(genes),
            'druggable_genes': len(druggable_in_pw),
            'druggable_fraction': len(druggable_in_pw) / len(genes) if genes else 0,
            'total_drugs': n_drugs,
            'druggable_gene_list': ', '.join(druggable_in_pw[:10]),
        })

    return pd.DataFrame(results).sort_values('druggable_genes', ascending=False)


def plot_results(pw_matrix, enrichment, hub_df, vuln_df, drug_pw, output_dir):
    """Generate figures."""
    fig = plt.figure(figsize=(22, 18))

    # 1. Pathway-pathway SL interaction heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    if pw_matrix is not None and len(pw_matrix) > 0:
        mask = np.triu(np.ones_like(pw_matrix, dtype=bool), k=1)
        sns.heatmap(pw_matrix, mask=mask, cmap='YlOrRd', ax=ax1,
                   annot=True, fmt='d', cbar_kws={'label': 'SL Pairs'},
                   xticklabels=True, yticklabels=True)
        ax1.set_title('Pathway-Pathway SL Interactions')
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.tick_params(axis='y', labelsize=7)

    # 2. Top enriched pathway interactions
    ax2 = fig.add_subplot(2, 3, 2)
    if not enrichment.empty:
        top = enrichment.head(15)
        labels = [f"{r['pathway_a'][:12]}↔{r['pathway_b'][:12]}" for _, r in top.iterrows()]
        ax2.barh(range(len(labels)), top['n_sl_pairs'], color='steelblue')
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.set_xlabel('Number of SL Pairs')
        ax2.set_title('Top Pathway SL Interactions')
        ax2.invert_yaxis()

    # 3. Pathway hub scores
    ax3 = fig.add_subplot(2, 3, 3)
    if not hub_df.empty:
        top_hubs = hub_df.head(12)
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_hubs)))
        ax3.barh(range(len(top_hubs)), top_hubs['hub_score'], color=colors)
        ax3.set_yticks(range(len(top_hubs)))
        ax3.set_yticklabels(top_hubs['pathway'], fontsize=8)
        ax3.set_xlabel('Hub Score (connections × diversity)')
        ax3.set_title('Pathway SL Hub Analysis')
        ax3.invert_yaxis()

    # 4. Cross-cancer pathway vulnerability
    ax4 = fig.add_subplot(2, 3, 4)
    if vuln_df is not None and len(vuln_df) > 0:
        sns.heatmap(vuln_df, cmap='RdBu_r', center=0, ax=ax4,
                   annot=True, fmt='.2f', cbar_kws={'label': 'Mean Gene Effect'},
                   xticklabels=True, yticklabels=True)
        ax4.set_title('Pathway Vulnerability by Cancer')
        ax4.tick_params(axis='x', rotation=45, labelsize=7)
        ax4.tick_params(axis='y', labelsize=7)

    # 5. Druggable pathways
    ax5 = fig.add_subplot(2, 3, 5)
    if not drug_pw.empty:
        pw_plot = drug_pw.head(12)
        ax5.bar(range(len(pw_plot)), pw_plot['druggable_genes'],
                color='coral', label='Druggable')
        ax5.bar(range(len(pw_plot)),
                pw_plot['total_genes'] - pw_plot['druggable_genes'],
                bottom=pw_plot['druggable_genes'],
                color='lightgray', label='Not druggable')
        ax5.set_xticks(range(len(pw_plot)))
        ax5.set_xticklabels(pw_plot['pathway'], rotation=45, ha='right', fontsize=7)
        ax5.set_ylabel('Number of Genes')
        ax5.set_title('Pathway Druggability')
        ax5.legend(fontsize=8)

    # 6. Network diagram (simplified)
    ax6 = fig.add_subplot(2, 3, 6)
    if not hub_df.empty:
        # Scatter: hub score vs connected pathways
        ax6.scatter(hub_df['sl_connections'], hub_df['connected_pathways'],
                   s=hub_df['n_genes'] * 10, alpha=0.6, c='steelblue', edgecolors='black')
        for _, row in hub_df.iterrows():
            ax6.annotate(row['pathway'][:15], (row['sl_connections'], row['connected_pathways']),
                        fontsize=6, ha='center')
        ax6.set_xlabel('SL Connections (edges)')
        ax6.set_ylabel('Connected Pathways (diversity)')
        ax6.set_title('Pathway Network Centrality')

    plt.suptitle('Experiment 21: Network Medicine — Pathway-Level SL',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'network_medicine_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved network_medicine_landscape.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 21: Network Medicine")
    logger.info("=" * 60)

    # Load data
    sl_pairs = load_sl_pairs()
    if sl_pairs.empty:
        logger.error("No SL pairs. Cannot proceed.")
        return

    # Load DepMap for vulnerability
    ge_file = DATA_DIR / "drug_repurpose" / "depmap_gene_effect.parquet"
    gene_effect = pd.DataFrame()
    cell_line_info = pd.DataFrame()
    if ge_file.exists():
        gene_effect = pd.read_parquet(ge_file)
        gene_effect.columns = [c.split(' (')[0] if ' (' in str(c) else c for c in gene_effect.columns]
    cli_file = DATA_DIR / "drug_repurpose" / "depmap_cell_line_info.parquet"
    if cli_file.exists():
        cell_line_info = pd.read_parquet(cli_file)

    # Load drug interactions
    dgi_file = DATA_DIR / "drug_repurpose" / "dgidb_interactions.parquet"
    drug_interactions = pd.read_parquet(dgi_file) if dgi_file.exists() else pd.DataFrame()

    # Step 1: Pathway SL enrichment
    logger.info("\n--- Step 1: Pathway SL Enrichment ---")
    pw_matrix, enrichment = pathway_sl_enrichment(sl_pairs)
    if not enrichment.empty:
        logger.info(f"Top pathway interactions:\n{enrichment.head(10).to_string()}")

    # Step 2: Hub analysis
    logger.info("\n--- Step 2: Pathway Hub Analysis ---")
    hub_df = pathway_hub_analysis(sl_pairs)
    logger.info(f"Top pathway hubs:\n{hub_df.head(10).to_string()}")

    # Step 3: Cross-cancer vulnerability
    logger.info("\n--- Step 3: Cross-Cancer Pathway Vulnerability ---")
    vuln_df = pd.DataFrame()
    if not gene_effect.empty and not cell_line_info.empty:
        vuln_df = cross_cancer_pathway_vulnerability(gene_effect, cell_line_info)

    # Step 4: Druggable pathways
    logger.info("\n--- Step 4: Druggable Pathway Opportunities ---")
    drug_pw = drug_pathway_opportunities(sl_pairs, drug_interactions)
    if not drug_pw.empty:
        logger.info(f"Pathway druggability:\n{drug_pw.to_string()}")

    # Save results
    logger.info("\n--- Saving Results ---")
    pw_matrix.to_csv(RESULTS_DIR / 'pathway_sl_matrix.csv')
    enrichment.to_csv(RESULTS_DIR / 'pathway_sl_enrichment.csv', index=False)
    hub_df.to_csv(RESULTS_DIR / 'pathway_hub_analysis.csv', index=False)
    if not vuln_df.empty:
        vuln_df.to_csv(RESULTS_DIR / 'pathway_vulnerability.csv')
    if not drug_pw.empty:
        drug_pw.to_csv(RESULTS_DIR / 'pathway_druggability.csv', index=False)

    # Plot
    plot_results(pw_matrix, enrichment, hub_df, vuln_df, drug_pw, RESULTS_DIR)

    # Summary
    summary = {
        'experiment': 'Exp 21: Network Medicine — Pathway SL',
        'total_sl_pairs': len(sl_pairs),
        'pathways_defined': len(PATHWAYS),
        'pathway_interactions': len(enrichment),
        'top_pathway_hubs': hub_df.head(5)[['pathway', 'hub_score', 'connected_pathways']].to_dict('records'),
        'top_enriched_interactions': enrichment.head(5)[['pathway_a', 'pathway_b', 'n_sl_pairs', 'enrichment_ratio']].to_dict('records'),
        'druggable_pathways': drug_pw.head(5)[['pathway', 'druggable_genes', 'total_drugs']].to_dict('records') if not drug_pw.empty else [],
    }
    with open(RESULTS_DIR / 'exp21_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nExp 21 COMPLETE")
    logger.info(f"Pathway interactions: {len(enrichment)}")
    logger.info(f"Top hub: {hub_df.iloc[0]['pathway']} (score={hub_df.iloc[0]['hub_score']})")


if __name__ == '__main__':
    main()
