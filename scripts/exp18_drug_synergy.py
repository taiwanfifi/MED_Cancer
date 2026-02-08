#!/usr/bin/env python3
"""
Experiment 18: Drug Synergy Prediction via SL Network Topology
===============================================================
Predicts synergistic drug combinations by leveraging synthetic lethality
network structure and DepMap gene dependency data.

Hypothesis: If genes A and B are synthetic lethal partners, and both have
approved drugs, then co-targeting A+B should show synergistic cell killing
beyond what individual drugs achieve. We quantify this using DepMap
co-dependency patterns and network topology metrics.

Key analyses:
1. Build SL drug-pair network from Exp 2 SL pairs + DGIdb drug mappings
2. Compute co-dependency scores (do cells dependent on A also depend on B?)
3. Network topology: hub genes, betweenness centrality → prioritize targets
4. Rank drug pairs by synergy potential score
5. Validate against known synergy databases (literature)

Target: Paper on computational drug synergy prediction
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp18_drug_synergy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_sl_pairs():
    """Load SL pairs from Exp 2 results."""
    sl_file = BASE_DIR / "results" / "exp2_synthetic_lethality" / "synthetic_lethal_pairs.csv"
    if not sl_file.exists():
        logger.error(f"SL pairs file not found: {sl_file}")
        return pd.DataFrame()
    sl = pd.read_csv(sl_file)
    logger.info(f"Loaded {len(sl)} SL pairs")
    return sl


def load_drug_interactions():
    """Load DGIdb drug-gene interactions."""
    dgidb_file = DATA_DIR / "drug_repurpose" / "dgidb_interactions.parquet"
    if not dgidb_file.exists():
        logger.warning("DGIdb interactions not found")
        return pd.DataFrame()
    dgi = pd.read_parquet(dgidb_file)
    logger.info(f"Loaded {len(dgi)} drug-gene interactions")
    return dgi


def load_gene_effect():
    """Load DepMap gene effect (CRISPR dependency) data."""
    ge_file = DATA_DIR / "drug_repurpose" / "depmap_gene_effect.parquet"
    if not ge_file.exists():
        logger.warning("DepMap gene effect not found")
        return pd.DataFrame()
    ge = pd.read_parquet(ge_file)
    # Parse gene names from column format "GENE (ENTREZ_ID)"
    ge.columns = [c.split(' (')[0] if ' (' in str(c) else c for c in ge.columns]
    logger.info(f"DepMap gene effect: {ge.shape[0]} cell lines x {ge.shape[1]} genes")
    return ge


def build_sl_drug_network(sl_pairs, drug_interactions):
    """
    Build network of druggable SL pairs.
    Returns pairs where BOTH genes have approved drugs.
    """
    # Get set of druggable genes
    if 'gene_name' in drug_interactions.columns:
        drug_gene_col = 'gene_name'
    elif 'gene' in drug_interactions.columns:
        drug_gene_col = 'gene'
    else:
        # Try first column that looks like gene names
        for col in drug_interactions.columns:
            if drug_interactions[col].dtype == object:
                drug_gene_col = col
                break

    druggable_genes = set(drug_interactions[drug_gene_col].dropna().unique())
    logger.info(f"Druggable genes: {len(druggable_genes)}")

    # Get drug mappings per gene
    if 'drug_name' in drug_interactions.columns:
        drug_col = 'drug_name'
    elif 'drug' in drug_interactions.columns:
        drug_col = 'drug'
    else:
        drug_col = [c for c in drug_interactions.columns if 'drug' in c.lower()][0]

    gene_drugs = drug_interactions.groupby(drug_gene_col)[drug_col].apply(
        lambda x: list(x.dropna().unique())
    ).to_dict()

    # Identify SL pair columns
    gene_a_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][0]
    gene_b_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][1]

    # Filter SL pairs where both genes are druggable
    druggable_pairs = []
    for _, row in sl_pairs.iterrows():
        gene_a = row[gene_a_col]
        gene_b = row[gene_b_col]
        if gene_a in druggable_genes and gene_b in druggable_genes:
            drugs_a = gene_drugs.get(gene_a, [])
            drugs_b = gene_drugs.get(gene_b, [])
            if drugs_a and drugs_b:
                druggable_pairs.append({
                    'gene_a': gene_a,
                    'gene_b': gene_b,
                    'drugs_a': ', '.join(drugs_a[:5]),
                    'drugs_b': ', '.join(drugs_b[:5]),
                    'n_drugs_a': len(drugs_a),
                    'n_drugs_b': len(drugs_b),
                })

    druggable_df = pd.DataFrame(druggable_pairs)
    logger.info(f"Druggable SL pairs: {len(druggable_df)}")
    return druggable_df


def compute_codependency(gene_effect, sl_pairs):
    """
    Compute co-dependency scores between SL gene pairs.

    Co-dependency = Pearson correlation of gene effect scores across cell lines.
    Negative co-dependency in SL pairs suggests synthetic lethality:
    if knocking out gene A makes cells dependent on gene B.
    """
    gene_a_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][0]
    gene_b_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][1]

    available_genes = set(gene_effect.columns)
    results = []

    for _, row in sl_pairs.iterrows():
        gene_a = row[gene_a_col]
        gene_b = row[gene_b_col]

        if gene_a not in available_genes or gene_b not in available_genes:
            continue

        effect_a = gene_effect[gene_a].dropna()
        effect_b = gene_effect[gene_b].dropna()

        # Align indices
        common = effect_a.index.intersection(effect_b.index)
        if len(common) < 20:
            continue

        ea = effect_a.loc[common]
        eb = effect_b.loc[common]

        corr, pval = stats.pearsonr(ea, eb)

        # Compute co-essentiality score
        # Both essential (effect < -0.5) in same cell lines
        both_essential = ((ea < -0.5) & (eb < -0.5)).sum()
        either_essential = ((ea < -0.5) | (eb < -0.5)).sum()
        jaccard = both_essential / either_essential if either_essential > 0 else 0

        # Complementary lethality: A essential → B more essential
        a_dep = ea < -0.5
        if a_dep.sum() >= 5:
            b_when_a_dep = eb[a_dep].mean()
            b_when_a_nodep = eb[~a_dep].mean()
            conditional_effect = b_when_a_dep - b_when_a_nodep
        else:
            conditional_effect = np.nan

        results.append({
            'gene_a': gene_a,
            'gene_b': gene_b,
            'codependency_corr': corr,
            'codependency_pval': pval,
            'co_essential_jaccard': jaccard,
            'both_essential_count': int(both_essential),
            'conditional_effect': conditional_effect,
            'n_cell_lines': len(common),
        })

    codep_df = pd.DataFrame(results)
    logger.info(f"Computed co-dependency for {len(codep_df)} SL pairs")
    return codep_df


def network_topology_analysis(sl_pairs):
    """
    Analyze SL network topology to identify hub targets.
    Uses degree centrality and betweenness approximation.
    """
    gene_a_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][0]
    gene_b_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][1]

    # Build adjacency
    from collections import defaultdict
    adjacency = defaultdict(set)
    for _, row in sl_pairs.iterrows():
        a, b = row[gene_a_col], row[gene_b_col]
        adjacency[a].add(b)
        adjacency[b].add(a)

    # Degree centrality
    genes = list(adjacency.keys())
    n_genes = len(genes)
    degrees = {g: len(neighbors) for g, neighbors in adjacency.items()}

    # Normalized degree
    max_degree = max(degrees.values()) if degrees else 1
    norm_degrees = {g: d / max_degree for g, d in degrees.items()}

    # Clustering coefficient
    clustering = {}
    for g in genes:
        neighbors = list(adjacency[g])
        if len(neighbors) < 2:
            clustering[g] = 0
            continue
        # Count edges between neighbors
        n_edges = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adjacency[neighbors[i]]:
                    n_edges += 1
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2
        clustering[g] = n_edges / max_edges if max_edges > 0 else 0

    topology = pd.DataFrame({
        'gene': genes,
        'degree': [degrees[g] for g in genes],
        'norm_degree': [norm_degrees[g] for g in genes],
        'clustering_coeff': [clustering[g] for g in genes],
    }).sort_values('degree', ascending=False)

    logger.info(f"Network: {n_genes} genes, {len(sl_pairs)} edges")
    logger.info(f"Top hub genes:\n{topology.head(10).to_string()}")
    return topology


def compute_synergy_score(druggable_pairs, codep_df, topology):
    """
    Compute composite synergy prediction score.

    Score = weighted combination of:
    - Co-dependency correlation (negative = good for SL)
    - Conditional effect (negative = gene B more essential when A is knocked out)
    - Network topology (hub genes = more impactful)
    - Drug availability (more drugs = more options)
    """
    # Merge all features
    merged = druggable_pairs.copy()

    if not codep_df.empty:
        merged = merged.merge(
            codep_df[['gene_a', 'gene_b', 'codependency_corr', 'conditional_effect',
                       'co_essential_jaccard']],
            on=['gene_a', 'gene_b'], how='left'
        )

    if not topology.empty:
        # Add topology for both genes
        topo_a = topology[['gene', 'degree', 'norm_degree']].rename(
            columns={'gene': 'gene_a', 'degree': 'degree_a', 'norm_degree': 'norm_degree_a'})
        topo_b = topology[['gene', 'degree', 'norm_degree']].rename(
            columns={'gene': 'gene_b', 'degree': 'degree_b', 'norm_degree': 'norm_degree_b'})
        merged = merged.merge(topo_a, on='gene_a', how='left')
        merged = merged.merge(topo_b, on='gene_b', how='left')

    # Compute synergy score
    # Higher score = more promising synergy candidate
    scores = []
    for _, row in merged.iterrows():
        score = 0

        # Co-dependency: negative correlation in SL pairs is good
        codep = row.get('codependency_corr', np.nan)
        if pd.notna(codep):
            score += (1 - codep) * 0.3  # Range: 0-0.6, negative corr gets higher score

        # Conditional effect: negative = B more essential when A knocked out
        cond = row.get('conditional_effect', np.nan)
        if pd.notna(cond):
            score += max(0, -cond) * 0.3  # More negative = better

        # Network centrality: hub genes are better targets
        deg_a = row.get('norm_degree_a', 0)
        deg_b = row.get('norm_degree_b', 0)
        if pd.notna(deg_a) and pd.notna(deg_b):
            score += (deg_a + deg_b) * 0.2

        # Drug availability: more drugs = more combination options
        n_a = row.get('n_drugs_a', 0)
        n_b = row.get('n_drugs_b', 0)
        if n_a and n_b:
            score += min(1, (n_a + n_b) / 20) * 0.2

        scores.append(score)

    merged['synergy_score'] = scores
    merged = merged.sort_values('synergy_score', ascending=False)
    return merged


def cancer_specific_synergy(gene_effect, cell_line_info, sl_pairs, cancer_types=None):
    """
    Analyze synergy predictions per cancer lineage.
    """
    if cancer_types is None:
        cancer_types = ['Breast Cancer', 'Lung Cancer', 'Kidney Cancer']

    gene_a_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][0]
    gene_b_col = [c for c in sl_pairs.columns if 'gene' in c.lower()][1]

    # Map cell lines to cancer types
    if 'lineage' in cell_line_info.columns:
        lineage_col = 'lineage'
    elif 'primary_disease' in cell_line_info.columns:
        lineage_col = 'primary_disease'
    else:
        lineage_col = cell_line_info.columns[1]

    results_by_cancer = {}
    available_genes = set(gene_effect.columns)

    for cancer in cancer_types:
        cancer_lines = cell_line_info[
            cell_line_info[lineage_col].str.contains(cancer.split()[0], case=False, na=False)
        ].index
        cancer_lines = cancer_lines.intersection(gene_effect.index)

        if len(cancer_lines) < 10:
            logger.warning(f"{cancer}: only {len(cancer_lines)} cell lines, skipping")
            continue

        ge_cancer = gene_effect.loc[cancer_lines]
        logger.info(f"{cancer}: {len(cancer_lines)} cell lines")

        pair_results = []
        for _, row in sl_pairs.head(200).iterrows():  # Top 200 pairs
            a, b = row[gene_a_col], row[gene_b_col]
            if a not in available_genes or b not in available_genes:
                continue

            ea = ge_cancer[a].dropna()
            eb = ge_cancer[b].dropna()
            common = ea.index.intersection(eb.index)
            if len(common) < 5:
                continue

            ea, eb = ea.loc[common], eb.loc[common]

            # Combined essentiality: geometric mean of effects
            combined = np.sqrt(ea.values ** 2 + eb.values ** 2)

            # Synergy indicator: combined effect > sum of individual
            mean_a = ea.mean()
            mean_b = eb.mean()
            mean_combined = -np.mean(combined)  # negative = more essential

            pair_results.append({
                'gene_a': a,
                'gene_b': b,
                'mean_effect_a': float(mean_a),
                'mean_effect_b': float(mean_b),
                'mean_combined': float(mean_combined),
                'n_both_essential': int(((ea < -0.5) & (eb < -0.5)).sum()),
                'n_cell_lines': len(common),
            })

        if pair_results:
            results_by_cancer[cancer] = pd.DataFrame(pair_results)
            logger.info(f"  {len(pair_results)} pairs analyzed")

    return results_by_cancer


def plot_results(synergy_df, topology, codep_df, cancer_results, output_dir):
    """Generate all figures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Top synergy pairs
    ax = axes[0, 0]
    top = synergy_df.head(20)
    if len(top) > 0:
        labels = [f"{r['gene_a']}-{r['gene_b']}" for _, r in top.iterrows()]
        ax.barh(range(len(labels)), top['synergy_score'].values, color='steelblue')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Synergy Prediction Score')
        ax.set_title('Top 20 Predicted Drug Synergy Pairs')
        ax.invert_yaxis()

    # 2. Network degree distribution
    ax = axes[0, 1]
    if not topology.empty:
        ax.hist(topology['degree'], bins=30, color='coral', edgecolor='black')
        ax.set_xlabel('Node Degree (SL partners)')
        ax.set_ylabel('Number of Genes')
        ax.set_title('SL Network Degree Distribution')
        # Mark top hubs
        hubs = topology.head(5)
        for _, hub in hubs.iterrows():
            ax.axvline(hub['degree'], color='red', linestyle='--', alpha=0.5)
            ax.text(hub['degree'], ax.get_ylim()[1] * 0.9, hub['gene'],
                    rotation=45, fontsize=7, ha='right')

    # 3. Co-dependency distribution
    ax = axes[1, 0]
    if not codep_df.empty:
        ax.hist(codep_df['codependency_corr'].dropna(), bins=40, color='mediumpurple',
                edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Co-dependency Correlation')
        ax.set_ylabel('Number of SL Pairs')
        ax.set_title('SL Pair Co-dependency Distribution')
        neg = (codep_df['codependency_corr'] < 0).sum()
        pos = (codep_df['codependency_corr'] >= 0).sum()
        ax.text(0.05, 0.95, f'Negative: {neg}\nPositive: {pos}',
                transform=ax.transAxes, va='top', fontsize=10)

    # 4. Cancer-specific top pairs heatmap
    ax = axes[1, 1]
    if cancer_results:
        # Get union of top genes across cancers
        all_genes = set()
        for cancer, df in cancer_results.items():
            top_genes = set(df.nlargest(10, 'n_both_essential')['gene_a'].tolist() +
                           df.nlargest(10, 'n_both_essential')['gene_b'].tolist())
            all_genes.update(top_genes)

        if all_genes:
            heatmap_data = {}
            for cancer, df in cancer_results.items():
                gene_effects = {}
                for gene in all_genes:
                    mask_a = df['gene_a'] == gene
                    mask_b = df['gene_b'] == gene
                    effects = pd.concat([df.loc[mask_a, 'mean_effect_a'],
                                        df.loc[mask_b, 'mean_effect_b']])
                    gene_effects[gene] = effects.mean() if len(effects) > 0 else 0
                heatmap_data[cancer.split()[0]] = gene_effects

            hm = pd.DataFrame(heatmap_data)
            hm = hm.loc[hm.abs().max(axis=1).nlargest(15).index]
            if len(hm) > 0:
                sns.heatmap(hm, cmap='RdBu_r', center=0, ax=ax, annot=True,
                           fmt='.2f', cbar_kws={'label': 'Mean Gene Effect'})
                ax.set_title('Top Target Essentiality by Cancer Type')

    plt.suptitle('Experiment 18: Drug Synergy Prediction via SL Network',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'drug_synergy_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved drug_synergy_landscape.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 18: Drug Synergy Prediction")
    logger.info("=" * 60)

    # Load data
    sl_pairs = load_sl_pairs()
    if sl_pairs.empty:
        logger.error("No SL pairs available. Cannot proceed.")
        return

    drug_interactions = load_drug_interactions()
    gene_effect = load_gene_effect()

    # Load cell line info
    cli_file = DATA_DIR / "drug_repurpose" / "depmap_cell_line_info.parquet"
    cell_line_info = pd.read_parquet(cli_file) if cli_file.exists() else pd.DataFrame()

    # Step 1: Build druggable SL network
    logger.info("\n--- Step 1: Build Druggable SL Network ---")
    druggable_pairs = pd.DataFrame()
    if not drug_interactions.empty:
        druggable_pairs = build_sl_drug_network(sl_pairs, drug_interactions)

    # Step 2: Co-dependency analysis
    logger.info("\n--- Step 2: Co-dependency Analysis ---")
    codep_df = pd.DataFrame()
    if not gene_effect.empty:
        codep_df = compute_codependency(gene_effect, sl_pairs)

    # Step 3: Network topology
    logger.info("\n--- Step 3: Network Topology Analysis ---")
    topology = network_topology_analysis(sl_pairs)

    # Step 4: Synergy scoring
    logger.info("\n--- Step 4: Synergy Prediction Scoring ---")
    if druggable_pairs.empty and not codep_df.empty:
        # Use codep pairs even without drug mapping
        druggable_pairs = codep_df[['gene_a', 'gene_b']].copy()
        druggable_pairs['n_drugs_a'] = 0
        druggable_pairs['n_drugs_b'] = 0
        druggable_pairs['drugs_a'] = ''
        druggable_pairs['drugs_b'] = ''

    synergy_df = pd.DataFrame()
    if not druggable_pairs.empty:
        synergy_df = compute_synergy_score(druggable_pairs, codep_df, topology)
        logger.info(f"\nTop 10 synergy predictions:")
        logger.info(synergy_df.head(10).to_string())

    # Step 5: Cancer-specific analysis
    logger.info("\n--- Step 5: Cancer-Specific Synergy ---")
    cancer_results = {}
    if not gene_effect.empty and not cell_line_info.empty:
        cancer_results = cancer_specific_synergy(gene_effect, cell_line_info, sl_pairs)

    # Save results
    logger.info("\n--- Saving Results ---")
    if not synergy_df.empty:
        synergy_df.to_csv(RESULTS_DIR / 'synergy_predictions.csv', index=False)
    if not codep_df.empty:
        codep_df.to_csv(RESULTS_DIR / 'codependency_scores.csv', index=False)
    topology.to_csv(RESULTS_DIR / 'network_topology.csv', index=False)
    for cancer, df in cancer_results.items():
        safe_name = cancer.replace(' ', '_').lower()
        df.to_csv(RESULTS_DIR / f'synergy_{safe_name}.csv', index=False)

    # Plot
    plot_results(synergy_df, topology, codep_df, cancer_results, RESULTS_DIR)

    # Summary
    summary = {
        'experiment': 'Exp 18: Drug Synergy Prediction',
        'total_sl_pairs': len(sl_pairs),
        'druggable_sl_pairs': len(druggable_pairs),
        'codependency_pairs_tested': len(codep_df),
        'negative_codep_pairs': int((codep_df['codependency_corr'] < 0).sum()) if not codep_df.empty else 0,
        'network_genes': len(topology),
        'top_hub_genes': topology.head(5)[['gene', 'degree']].to_dict('records') if not topology.empty else [],
        'top_synergy_pairs': synergy_df.head(10)[['gene_a', 'gene_b', 'synergy_score']].to_dict('records') if not synergy_df.empty else [],
        'cancer_types_analyzed': list(cancer_results.keys()),
    }
    with open(RESULTS_DIR / 'exp18_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nExp 18 COMPLETE")
    logger.info(f"SL pairs: {len(sl_pairs)}, Druggable: {len(druggable_pairs)}")
    logger.info(f"Co-dependency tested: {len(codep_df)}")
    logger.info(f"Network genes: {len(topology)}")
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
