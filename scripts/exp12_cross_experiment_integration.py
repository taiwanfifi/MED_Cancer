#!/usr/bin/env python3
"""
Experiment 12: Cross-Experiment Integration & Pan-Cancer Gene Scoring
=====================================================================
Aggregates evidence from ALL previous experiments to create a comprehensive
gene-level scoring system for therapeutic target prioritization.

Integrates:
- Exp 2: Synthetic lethality pairs (SL score)
- Exp 3: Differential expression (DEG score)
- Exp 4: SL transferability across lineages
- Exp 5: GNN drug repurposing predictions
- Exp 6: Immune microenvironment associations
- Exp 7: Multi-omics actionability scores
- Exp 10: Gene dependency prediction (AUC)
- Exp 11: Outlier survival associations

Output: Comprehensive gene evidence matrix + radar plots + heatmaps
Target: Paper 5 (Multi-Omics Precision Oncology) — Nature Medicine
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
RESULTS_DIR = BASE_DIR / "results" / "exp12_integration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_sl_data():
    """Load synthetic lethality pairs from Exp 2."""
    sl_file = BASE_DIR / "results" / "exp2_synthetic_lethality" / "synthetic_lethal_pairs.csv"
    if not sl_file.exists():
        logger.warning("SL pairs not found")
        return {}

    df = pd.read_csv(sl_file)
    logger.info(f"SL pairs loaded: {len(df)}")

    # Count how many drivers each target is SL with
    gene_sl = {}
    for gene in df['target_gene'].unique():
        gene_data = df[df['target_gene'] == gene]
        gene_sl[gene] = {
            'n_drivers': len(gene_data['driver_gene'].unique()),
            'drivers': gene_data['driver_gene'].unique().tolist(),
            'mean_effect': gene_data['delta_effect'].mean() if 'delta_effect' in df.columns else -1.0,
        }

    return gene_sl


def load_deg_data():
    """Load differential expression from Exp 3."""
    deg_scores = {}

    for cancer in ['BRCA', 'LUAD', 'KIRC']:
        deg_file = BASE_DIR / "results" / "exp3_differential_expression" / f"{cancer}_deg_results.csv"
        if not deg_file.exists():
            continue

        df = pd.read_csv(deg_file)
        logger.info(f"[{cancer}] DEGs loaded: {len(df)}")

        for _, row in df.iterrows():
            gene = row.get('gene', row.get('gene_id', ''))
            if gene not in deg_scores:
                deg_scores[gene] = {'cancers_up': [], 'cancers_down': [], 'max_log2fc': 0, 'min_fdr': 1.0}

            log2fc = row.get('log2FC', row.get('log2FoldChange', 0))
            fdr = row.get('fdr', row.get('padj', 1.0))

            if fdr < 0.05 and abs(log2fc) > 1:
                if log2fc > 0:
                    deg_scores[gene]['cancers_up'].append(cancer)
                else:
                    deg_scores[gene]['cancers_down'].append(cancer)
                deg_scores[gene]['max_log2fc'] = max(deg_scores[gene]['max_log2fc'], abs(log2fc))
                deg_scores[gene]['min_fdr'] = min(deg_scores[gene]['min_fdr'], fdr)

    return deg_scores


def load_sl_transferability():
    """Load SL transferability from Exp 4."""
    # Use universal_sl_pairs.csv which has per-pair lineage counts
    transfer_file = BASE_DIR / "results" / "exp4_sl_transferability" / "universal_sl_pairs.csv"
    if not transfer_file.exists():
        return {}

    df = pd.read_csv(transfer_file)
    logger.info(f"Universal SL pairs loaded: {len(df)}")

    gene_transfer = {}
    for _, row in df.iterrows():
        gene = row['target_gene']
        if gene not in gene_transfer:
            gene_transfer[gene] = {'max_lineages': 0, 'is_universal': False}
        gene_transfer[gene]['max_lineages'] = max(
            gene_transfer[gene]['max_lineages'], row['n_lineages'])
        if row['n_lineages'] >= 3:
            gene_transfer[gene]['is_universal'] = True

    return gene_transfer


def load_drug_interactions():
    """Load drug interaction data from DGIdb."""
    dgidb_file = BASE_DIR / "data" / "drug_repurpose" / "dgidb_interactions.parquet"
    if not dgidb_file.exists():
        return {}

    df = pd.read_parquet(dgidb_file)
    logger.info(f"DGIdb interactions: {len(df)}")

    gene_drugs = {}
    for gene in df['gene'].unique():
        gene_data = df[df['gene'] == gene]
        gene_drugs[gene] = {
            'n_total_drugs': len(gene_data),
            'n_approved_drugs': len(gene_data[gene_data['approved']]),
            'approved_drugs': gene_data[gene_data['approved']]['drug'].unique().tolist()[:5],
            'is_druggable': gene_data['approved'].any(),
        }

    return gene_drugs


def load_immune_associations():
    """Load immune-survival associations from Exp 6."""
    immune_scores = {}

    for cancer in ['BRCA', 'LUAD', 'KIRC']:
        immune_file = BASE_DIR / "results" / "exp6_immune_tme" / f"{cancer}_immune_survival_cox.csv"
        if not immune_file.exists():
            continue

        df = pd.read_csv(immune_file)
        for _, row in df.iterrows():
            cell_type = row.get('cell_type', row.get('immune_cell', ''))
            key = f"immune_{cell_type}"
            if key not in immune_scores:
                immune_scores[key] = {'cancers_protective': [], 'cancers_risk': []}

            hr = row.get('hazard_ratio', row.get('HR', 1.0))
            pval = row.get('p_value', row.get('pval', 1.0))

            if pval < 0.05:
                if hr < 1:
                    immune_scores[key]['cancers_protective'].append(cancer)
                else:
                    immune_scores[key]['cancers_risk'].append(cancer)

    return immune_scores


def load_dependency_prediction():
    """Load gene dependency prediction from Exp 10."""
    dep_file = BASE_DIR / "results" / "exp10_drug_sensitivity" / "gene_dependency_prediction.csv"
    if not dep_file.exists():
        return {}

    df = pd.read_csv(dep_file)
    logger.info(f"Gene dependency predictions: {len(df)}")

    gene_dep = {}
    for _, row in df.iterrows():
        gene_dep[row['gene']] = {
            'dep_auc': row['auc_mean'],
            'dep_auc_std': row.get('auc_std', 0),
            'n_dependent_lines': row.get('n_dependent', 0),
            'is_sl_target': row.get('is_sl_target', False),
        }

    return gene_dep


def load_outlier_immune():
    """Load outlier immune associations from Exp 11."""
    outlier_immune = {}

    for cancer in ['BRCA', 'LUAD', 'KIRC']:
        immune_file = BASE_DIR / "results" / "exp11_outlier_analysis" / f"{cancer}_outlier_immune.csv"
        if not immune_file.exists():
            continue

        df = pd.read_csv(immune_file)
        for _, row in df.iterrows():
            cell_type = row.get('cell_type', row.get('immune_cell', ''))
            key = f"outlier_immune_{cell_type}"
            pval = row.get('p_value', row.get('pval', 1.0))
            if pval < 0.1:
                if key not in outlier_immune:
                    outlier_immune[key] = []
                outlier_immune[key].append({
                    'cancer': cancer,
                    'p_value': pval,
                    'survivor_mean': row.get('survivor_mean', 0),
                    'death_mean': row.get('death_mean', 0),
                })

    return outlier_immune


def build_gene_evidence_matrix(sl_data, deg_data, gene_drugs, gene_dep, sl_transfer):
    """Build comprehensive gene evidence matrix."""
    all_genes = set()
    all_genes.update(sl_data.keys())
    all_genes.update(g for g, v in deg_data.items() if v['cancers_up'] or v['cancers_down'])
    all_genes.update(gene_drugs.keys())
    all_genes.update(gene_dep.keys())

    logger.info(f"Total unique genes across all experiments: {len(all_genes)}")

    rows = []
    for gene in sorted(all_genes):
        row = {'gene': gene}

        # SL evidence
        if gene in sl_data:
            row['sl_n_drivers'] = sl_data[gene]['n_drivers']
            row['sl_drivers'] = ', '.join(sl_data[gene]['drivers'][:5])
        else:
            row['sl_n_drivers'] = 0
            row['sl_drivers'] = ''

        # DEG evidence
        if gene in deg_data:
            row['deg_cancers_up'] = len(deg_data[gene]['cancers_up'])
            row['deg_cancers_down'] = len(deg_data[gene]['cancers_down'])
            row['deg_max_log2fc'] = deg_data[gene]['max_log2fc']
            row['deg_min_fdr'] = deg_data[gene]['min_fdr']
        else:
            row['deg_cancers_up'] = 0
            row['deg_cancers_down'] = 0
            row['deg_max_log2fc'] = 0
            row['deg_min_fdr'] = 1.0

        # Drug evidence
        if gene in gene_drugs:
            row['n_approved_drugs'] = gene_drugs[gene]['n_approved_drugs']
            row['n_total_drugs'] = gene_drugs[gene]['n_total_drugs']
            row['is_druggable'] = gene_drugs[gene]['is_druggable']
            row['top_drugs'] = ', '.join(gene_drugs[gene]['approved_drugs'][:3])
        else:
            row['n_approved_drugs'] = 0
            row['n_total_drugs'] = 0
            row['is_druggable'] = False
            row['top_drugs'] = ''

        # Gene dependency prediction
        if gene in gene_dep:
            row['dep_auc'] = gene_dep[gene]['dep_auc']
            row['n_dependent_lines'] = gene_dep[gene]['n_dependent_lines']
        else:
            row['dep_auc'] = 0.5
            row['n_dependent_lines'] = 0

        # SL transferability
        if gene in sl_transfer:
            row['sl_max_lineages'] = sl_transfer[gene]['max_lineages']
            row['sl_is_universal'] = sl_transfer[gene]['is_universal']
        else:
            row['sl_max_lineages'] = 0
            row['sl_is_universal'] = False

        # Composite score (weighted multi-evidence)
        # Normalize each dimension to 0-1
        sl_score = min(row['sl_n_drivers'] / 5.0, 1.0)  # max 5 drivers
        deg_score = min((row['deg_cancers_up'] + row['deg_cancers_down']) / 3.0, 1.0)  # max 3 cancers
        drug_score = min(row['n_approved_drugs'] / 10.0, 1.0)  # max 10 drugs
        dep_score = max(0, (row['dep_auc'] - 0.5) / 0.5)  # AUC 0.5-1.0 → 0-1
        transfer_score = min(row['sl_max_lineages'] / 4.0, 1.0)  # max 4 lineages

        # Weighted composite
        row['composite_score'] = (
            0.25 * sl_score +
            0.20 * deg_score +
            0.20 * drug_score +
            0.20 * dep_score +
            0.15 * transfer_score
        )

        rows.append(row)

    df = pd.DataFrame(rows).sort_values('composite_score', ascending=False)
    return df


def plot_top_genes_radar(evidence_df, top_n=10):
    """Create radar plots for top therapeutic targets."""
    top_genes = evidence_df.head(top_n)

    categories = ['SL Partners', 'DEG Evidence', 'Druggability', 'Dependency\nPrediction', 'Pan-Cancer\nTransfer']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 5, figsize=(25, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, top_n))

    for idx, (_, row) in enumerate(top_genes.iterrows()):
        ax = axes[idx]
        values = [
            min(row['sl_n_drivers'] / 5.0, 1.0),
            min((row['deg_cancers_up'] + row['deg_cancers_down']) / 3.0, 1.0),
            min(row['n_approved_drugs'] / 10.0, 1.0),
            max(0, (row['dep_auc'] - 0.5) / 0.5),
            min(row['sl_max_lineages'] / 4.0, 1.0),
        ]
        values += values[:1]

        ax.fill(angles, values, color=colors[idx], alpha=0.25)
        ax.plot(angles, values, color=colors[idx], linewidth=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{row['gene']}\n(Score: {row['composite_score']:.3f})",
                    fontsize=10, fontweight='bold', pad=15)

    plt.suptitle('Top 10 Therapeutic Target Profiles — Multi-Evidence Radar',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top10_radar_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: top10_radar_profiles.png")


def plot_evidence_heatmap(evidence_df, top_n=30):
    """Create heatmap of evidence across top genes."""
    top = evidence_df.head(top_n).copy()

    # Normalize columns for heatmap
    heatmap_data = pd.DataFrame({
        'SL Drivers': top['sl_n_drivers'] / top['sl_n_drivers'].max() if top['sl_n_drivers'].max() > 0 else 0,
        'DEGs (Up)': top['deg_cancers_up'] / 3.0,
        'DEGs (Down)': top['deg_cancers_down'] / 3.0,
        'Approved Drugs': top['n_approved_drugs'].clip(upper=20) / 20.0,
        'Dep. AUC': (top['dep_auc'] - 0.5).clip(lower=0) / 0.5,
        'Pan-Cancer SL': top['sl_max_lineages'] / top['sl_max_lineages'].max() if top['sl_max_lineages'].max() > 0 else 0,
        'Composite': top['composite_score'] / top['composite_score'].max() if top['composite_score'].max() > 0 else 0,
    })
    heatmap_data.index = top['gene'].values

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.2f',
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Normalized Evidence Score'})
    ax.set_title('Multi-Evidence Therapeutic Target Heatmap (Top 30)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Evidence Type')
    ax.set_ylabel('Gene')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "evidence_heatmap_top30.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: evidence_heatmap_top30.png")


def plot_drug_combination_opportunities(evidence_df, sl_data, gene_drugs):
    """Identify and plot drug combination opportunities from SL + drug data."""
    combos = []

    for gene, info in sl_data.items():
        if gene not in gene_drugs or not gene_drugs[gene]['is_druggable']:
            continue

        for driver in info['drivers']:
            if driver in gene_drugs and gene_drugs[driver]['is_druggable']:
                combos.append({
                    'driver': driver,
                    'sl_target': gene,
                    'driver_drugs': ', '.join(gene_drugs[driver]['approved_drugs'][:2]),
                    'target_drugs': ', '.join(gene_drugs[gene]['approved_drugs'][:2]),
                    'n_driver_drugs': gene_drugs[driver]['n_approved_drugs'],
                    'n_target_drugs': gene_drugs[gene]['n_approved_drugs'],
                    'total_drugs': gene_drugs[driver]['n_approved_drugs'] + gene_drugs[gene]['n_approved_drugs'],
                })

    if not combos:
        logger.info("No drug combination opportunities found")
        return pd.DataFrame()

    combo_df = pd.DataFrame(combos).sort_values('total_drugs', ascending=False)
    combo_df = combo_df.drop_duplicates(subset=['driver', 'sl_target'])
    logger.info(f"Drug combination opportunities: {len(combo_df)}")

    # Plot top combinations
    fig, ax = plt.subplots(figsize=(14, 8))
    top_combos = combo_df.head(20)

    labels = [f"{r['driver']} → {r['sl_target']}" for _, r in top_combos.iterrows()]
    x = range(len(labels))
    width = 0.35

    ax.barh([i - width/2 for i in x], top_combos['n_driver_drugs'], width,
            label='Driver Gene Drugs', color='#e74c3c', alpha=0.8)
    ax.barh([i + width/2 for i in x], top_combos['n_target_drugs'], width,
            label='SL Target Drugs', color='#3498db', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Number of Approved Drugs')
    ax.set_title('Top Drug Combination Opportunities\n(Driver Gene + Synthetically Lethal Target)',
                fontweight='bold', fontsize=13)
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "drug_combination_opportunities.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: drug_combination_opportunities.png")

    combo_df.to_csv(RESULTS_DIR / "drug_combinations.csv", index=False)
    return combo_df


def plot_summary_dashboard(evidence_df, sl_data, gene_drugs, gene_dep):
    """Create a publication-ready summary dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # Panel A: Composite score distribution
    ax = axes[0, 0]
    scores = evidence_df['composite_score']
    ax.hist(scores[scores > 0], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(x=scores.quantile(0.95), color='red', linestyle='--',
              label=f'95th %ile: {scores.quantile(0.95):.3f}')
    ax.set_xlabel('Composite Evidence Score')
    ax.set_ylabel('Number of Genes')
    ax.set_title('A) Distribution of Multi-Evidence Scores', fontweight='bold')
    ax.legend()

    # Panel B: SL partners vs druggability
    ax = axes[0, 1]
    plot_df = evidence_df[evidence_df['sl_n_drivers'] > 0].copy()
    colors = ['#2ecc71' if d else '#e74c3c' for d in plot_df['is_druggable']]
    ax.scatter(plot_df['sl_n_drivers'], plot_df['n_approved_drugs'],
              c=colors, s=60, alpha=0.6, edgecolor='white')
    for _, row in plot_df[plot_df['composite_score'] > plot_df['composite_score'].quantile(0.9)].iterrows():
        ax.annotate(row['gene'], (row['sl_n_drivers'], row['n_approved_drugs']),
                   fontsize=7, alpha=0.8)
    ax.set_xlabel('Number of SL Driver Partners')
    ax.set_ylabel('Number of Approved Drugs')
    ax.set_title('B) SL Vulnerability vs Druggability', fontweight='bold')

    # Panel C: Dependency AUC distribution
    ax = axes[0, 2]
    dep_genes = evidence_df[evidence_df['dep_auc'] > 0.5]
    if len(dep_genes) > 0:
        dep_sorted = dep_genes.sort_values('dep_auc', ascending=True).tail(20)
        colors = ['#2ecc71' if auc > 0.7 else '#f1c40f' if auc > 0.6 else '#e74c3c'
                 for auc in dep_sorted['dep_auc']]
        ax.barh(range(len(dep_sorted)), dep_sorted['dep_auc'],
               color=colors, edgecolor='white')
        ax.set_yticks(range(len(dep_sorted)))
        ax.set_yticklabels(dep_sorted['gene'], fontsize=8)
        ax.set_xlabel('Dependency Prediction AUC')
        ax.set_title('C) Gene Dependency Predictability', fontweight='bold')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Panel D: Evidence count per gene (how many evidence types)
    ax = axes[1, 0]
    evidence_df_tmp = evidence_df.copy()
    evidence_df_tmp['n_evidence'] = (
        (evidence_df_tmp['sl_n_drivers'] > 0).astype(int) +
        (evidence_df_tmp['deg_cancers_up'] > 0).astype(int) +
        (evidence_df_tmp['deg_cancers_down'] > 0).astype(int) +
        (evidence_df_tmp['n_approved_drugs'] > 0).astype(int) +
        (evidence_df_tmp['dep_auc'] > 0.6).astype(int) +
        (evidence_df_tmp['sl_max_lineages'] > 0).astype(int)
    )
    counts = evidence_df_tmp['n_evidence'].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color='#9b59b6', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Number of Evidence Types')
    ax.set_ylabel('Number of Genes')
    ax.set_title('D) Multi-Evidence Support per Gene', fontweight='bold')

    # Panel E: Top 15 genes bar chart
    ax = axes[1, 1]
    top15 = evidence_df.head(15)
    y_pos = range(len(top15))
    bars = ax.barh(y_pos, top15['composite_score'], color='#e67e22', alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15['gene'], fontsize=9)
    ax.set_xlabel('Composite Score')
    ax.set_title('E) Top 15 Therapeutic Targets', fontweight='bold')
    ax.invert_yaxis()

    # Add drug markers
    for i, (_, row) in enumerate(top15.iterrows()):
        if row['is_druggable']:
            ax.text(row['composite_score'] + 0.005, i, ' [D]', va='center', fontsize=7, color='green')
        if row['sl_n_drivers'] > 0:
            ax.text(row['composite_score'] + 0.005, i + 0.3, f' SL:{int(row["sl_n_drivers"])}',
                   va='center', fontsize=6, color='blue')

    # Panel F: Novel targets (SL targets with NO drugs)
    ax = axes[1, 2]
    novel = evidence_df[(evidence_df['sl_n_drivers'] > 0) & (evidence_df['n_approved_drugs'] == 0)].copy()
    novel = novel.sort_values('composite_score', ascending=False).head(15)
    if len(novel) > 0:
        y_pos = range(len(novel))
        colors_novel = plt.cm.Reds(np.linspace(0.3, 0.9, len(novel)))
        ax.barh(y_pos, novel['composite_score'], color=colors_novel, edgecolor='white')
        ax.set_yticks(y_pos)
        labels = [f"{r['gene']} (SL w/ {r['sl_drivers'][:20]})" for _, r in novel.iterrows()]
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Composite Score')
        ax.set_title('F) Novel Targets (No Approved Drugs)', fontweight='bold')
        ax.invert_yaxis()

    plt.suptitle('Pan-Cancer Therapeutic Target Discovery — Multi-Evidence Integration',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "integration_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: integration_dashboard.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 12: Cross-Experiment Integration")
    logger.info("=" * 60)

    # Load all evidence
    logger.info("\n--- Loading evidence from all experiments ---")
    sl_data = load_sl_data()
    deg_data = load_deg_data()
    sl_transfer = load_sl_transferability()
    gene_drugs = load_drug_interactions()
    gene_dep = load_dependency_prediction()
    immune_data = load_immune_associations()
    outlier_immune = load_outlier_immune()

    # Build gene evidence matrix
    logger.info("\n--- Building multi-evidence gene matrix ---")
    evidence_df = build_gene_evidence_matrix(sl_data, deg_data, gene_drugs, gene_dep, sl_transfer)
    evidence_df.to_csv(RESULTS_DIR / "gene_evidence_matrix.csv", index=False)
    logger.info(f"Gene evidence matrix: {len(evidence_df)} genes")

    # Top genes summary
    logger.info("\n--- Top 20 Therapeutic Targets ---")
    for i, (_, row) in enumerate(evidence_df.head(20).iterrows()):
        drugs_tag = f"[{int(row['n_approved_drugs'])} drugs]" if row['n_approved_drugs'] > 0 else "[NO DRUGS]"
        sl_tag = f"SL:{int(row['sl_n_drivers'])}" if row['sl_n_drivers'] > 0 else ""
        dep_tag = f"AUC:{row['dep_auc']:.2f}" if row['dep_auc'] > 0.5 else ""
        logger.info(f"  {i+1:2d}. {row['gene']:10s} Score={row['composite_score']:.3f} "
                    f"{drugs_tag} {sl_tag} {dep_tag}")

    # Plots
    logger.info("\n--- Generating visualizations ---")
    plot_evidence_heatmap(evidence_df)
    plot_top_genes_radar(evidence_df)
    plot_summary_dashboard(evidence_df, sl_data, gene_drugs, gene_dep)

    # Drug combinations
    logger.info("\n--- Identifying drug combination opportunities ---")
    combo_df = plot_drug_combination_opportunities(evidence_df, sl_data, gene_drugs)

    # Summary statistics
    n_sl = len([g for g in evidence_df.itertuples() if g.sl_n_drivers > 0])
    n_druggable = len([g for g in evidence_df.itertuples() if g.is_druggable])
    n_novel = len([g for g in evidence_df.itertuples() if g.sl_n_drivers > 0 and g.n_approved_drugs == 0])
    n_dep = len([g for g in evidence_df.itertuples() if g.dep_auc > 0.7])

    summary = {
        'experiment': 'Exp 12: Cross-Experiment Integration',
        'total_genes': len(evidence_df),
        'genes_with_sl': n_sl,
        'druggable_genes': n_druggable,
        'novel_targets': n_novel,
        'highly_predictable_dep': n_dep,
        'drug_combinations': len(combo_df) if isinstance(combo_df, pd.DataFrame) else 0,
        'top_10_targets': evidence_df.head(10)[['gene', 'composite_score', 'sl_n_drivers',
                                                  'n_approved_drugs', 'dep_auc']].to_dict('records'),
        'immune_associations': immune_data,
        'outlier_immune_findings': {k: len(v) for k, v in outlier_immune.items()},
    }

    with open(RESULTS_DIR / "exp12_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSaved summary with {len(evidence_df)} genes")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 12 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
