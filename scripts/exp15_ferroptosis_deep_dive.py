#!/usr/bin/env python3
"""
Experiment 15: Ferroptosis Pathway as Pan-Cancer Vulnerability
==============================================================
Deep analysis of the ferroptosis pathway across cancers.
GPX4 was identified as SL with 5 major drivers (Exp 2/7).
Now we test: is the ENTIRE ferroptosis pathway vulnerable?

Pipeline:
1. Define comprehensive ferroptosis gene set (20+ genes)
2. Test each ferroptosis gene for SL with common cancer drivers
3. Expression analysis of ferroptosis genes in tumors vs normal
4. Survival analysis stratified by ferroptosis pathway activity
5. Drug mapping for ferroptosis regulators
6. Build ferroptosis vulnerability map across cancer types

Target: Paper 6 (GPX4 Ferroptosis) → Cancer Cell
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
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp15_ferroptosis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Comprehensive ferroptosis gene set
# Based on: Dixon et al. 2012, Stockwell et al. 2017, Jiang et al. 2021
FERROPTOSIS_GENES = {
    # Core regulators
    'GPX4': 'glutathione peroxidase 4 (key anti-ferroptosis)',
    'SLC7A11': 'cystine/glutamate transporter (xCT, imports cystine)',
    'FSP1': 'ferroptosis suppressor protein 1 (CoQ10-dependent)',
    'DHODH': 'dihydroorotate dehydrogenase (mitochondrial anti-ferroptosis)',
    'GCH1': 'GTP cyclohydrolase 1 (BH4 synthesis, anti-ferroptosis)',

    # Glutathione synthesis pathway
    'GCLC': 'glutamate-cysteine ligase catalytic (GSH synthesis)',
    'GCLM': 'glutamate-cysteine ligase modifier',
    'GSS': 'glutathione synthetase',
    'GSR': 'glutathione reductase',

    # Iron metabolism (pro-ferroptosis when dysregulated)
    'TFRC': 'transferrin receptor (iron import)',
    'FTH1': 'ferritin heavy chain 1 (iron storage)',
    'FTL': 'ferritin light chain',
    'SLC40A1': 'ferroportin (iron export)',
    'HMOX1': 'heme oxygenase 1 (releases iron from heme)',
    'NCOA4': 'nuclear receptor coactivator 4 (ferritinophagy)',

    # Lipid peroxidation (pro-ferroptosis)
    'ACSL4': 'acyl-CoA synthetase 4 (PUFA activation, pro-ferroptosis)',
    'LPCAT3': 'lysophosphatidylcholine acyltransferase 3',
    'ALOX15': 'arachidonate 15-lipoxygenase (lipid peroxidation)',
    'ALOX12': 'arachidonate 12-lipoxygenase',
    'POR': 'cytochrome P450 oxidoreductase',

    # Transcription factor
    'NFE2L2': 'NRF2 (master antioxidant regulator)',
    'KEAP1': 'kelch-like ECH-associated protein 1 (NRF2 inhibitor)',

    # Other regulators
    'VDAC2': 'voltage-dependent anion channel 2',
    'VDAC3': 'voltage-dependent anion channel 3',
    'CISD1': 'CDGSH iron sulfur domain 1 (mitoNEET)',
    'ATG5': 'autophagy related 5 (ferritinophagy)',
    'ATG7': 'autophagy related 7',
}

CANCER_DRIVERS = [
    'TP53', 'PIK3CA', 'KRAS', 'BRAF', 'EGFR', 'ERBB2', 'PTEN',
    'STK11', 'BRCA1', 'BRCA2', 'VHL', 'CTNNB1', 'APC', 'NF1',
    'RB1', 'CDH1', 'ARID1A', 'CDKN2A', 'MYC',
]


def test_ferroptosis_sl(gene_effect_file):
    """Test each ferroptosis gene for synthetic lethality with cancer drivers."""
    if not gene_effect_file.exists():
        logger.warning("Gene effect file not found")
        return None

    ge = pd.read_parquet(gene_effect_file)
    logger.info(f"Gene effect: {ge.shape[0]} cell lines × {ge.shape[1]} genes")

    # Parse column names: "GENE (ENTREZ_ID)" → gene symbol
    col_to_gene = {col: col.split(' (')[0] for col in ge.columns}
    gene_to_col = {v: k for k, v in col_to_gene.items()}

    # Load cell line mutation data if available
    cell_info_file = DATA_DIR / "drug_repurpose" / "depmap_cell_line_info.parquet"
    cell_info = pd.read_parquet(cell_info_file) if cell_info_file.exists() else None

    sl_results = []

    for ferro_gene, description in FERROPTOSIS_GENES.items():
        if ferro_gene not in gene_to_col:
            logger.debug(f"  {ferro_gene} not in DepMap")
            continue

        ferro_col = gene_to_col[ferro_gene]
        ferro_effect = ge[ferro_col]

        for driver in CANCER_DRIVERS:
            if driver not in gene_to_col:
                continue

            driver_col = gene_to_col[driver]
            driver_effect = ge[driver_col]

            # Define "mutant" as cell lines with high dependency on driver
            # (gene_effect < -0.5 means the cell line depends on the driver)
            # For SL: we want cells where driver is essential AND ferroptosis gene is also essential
            # Alternative: use driver dependency as proxy for driver mutation
            driver_dep = driver_effect < -0.5
            driver_nodep = driver_effect >= -0.5

            n_dep = driver_dep.sum()
            n_nodep = driver_nodep.sum()

            if n_dep < 10 or n_nodep < 10:
                continue

            # Compare ferroptosis gene effect in driver-dependent vs not
            ferro_in_dep = ferro_effect[driver_dep].dropna()
            ferro_in_nodep = ferro_effect[driver_nodep].dropna()

            if len(ferro_in_dep) < 10 or len(ferro_in_nodep) < 10:
                continue

            t_stat, p_val = stats.ttest_ind(ferro_in_dep, ferro_in_nodep)
            delta = ferro_in_dep.mean() - ferro_in_nodep.mean()
            cohens_d = delta / np.sqrt((ferro_in_dep.var() + ferro_in_nodep.var()) / 2)

            sl_results.append({
                'ferroptosis_gene': ferro_gene,
                'description': description,
                'driver_gene': driver,
                'mean_effect_driver_dep': ferro_in_dep.mean(),
                'mean_effect_driver_nodep': ferro_in_nodep.mean(),
                'delta_effect': delta,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'n_driver_dep': len(ferro_in_dep),
                'n_driver_nodep': len(ferro_in_nodep),
            })

    if sl_results:
        df = pd.DataFrame(sl_results)
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(df['p_value'], method='fdr_bh')
        df['fdr'] = fdr
        df = df.sort_values('p_value')
        return df

    return None


def ferroptosis_expression_analysis():
    """Analyze ferroptosis gene expression in tumors vs normal."""
    results = []

    for cancer in ['BRCA', 'LUAD', 'KIRC']:
        expr_file = DATA_DIR / "tcga" / f"{cancer}_expression.parquet"
        if not expr_file.exists():
            continue

        expr = pd.read_parquet(expr_file)
        logger.info(f"[{cancer}] Expression: {expr.shape}")

        # Identify tumor vs normal (genes are in index, columns are patient IDs)
        sample_cols = list(expr.columns)
        tumor_cols = [c for c in sample_cols if len(c.split('-')) <= 3 or not c.split('-')[-1].startswith('1')]
        normal_cols = [c for c in sample_cols if len(c.split('-')) >= 4 and c.split('-')[-1].startswith('1')]

        if len(normal_cols) < 5:
            logger.info(f"  [{cancer}] Not enough normals ({len(normal_cols)})")
            continue

        logger.info(f"  [{cancer}] Tumor: {len(tumor_cols)}, Normal: {len(normal_cols)}")

        # Build case-insensitive gene index lookup
        gene_index_map = {g.upper(): g for g in expr.index}

        for ferro_gene in FERROPTOSIS_GENES:
            idx_name = gene_index_map.get(ferro_gene.upper())
            if idx_name is None:
                continue
            gene_row = expr.loc[[idx_name]]

            tumor_vals = gene_row[tumor_cols].values.flatten()
            normal_vals = gene_row[normal_cols].values.flatten()

            # Remove zeros and log2 transform
            tumor_vals = tumor_vals[tumor_vals > 0]
            normal_vals = normal_vals[normal_vals > 0]

            if len(tumor_vals) < 5 or len(normal_vals) < 5:
                continue

            tumor_log = np.log2(tumor_vals + 1)
            normal_log = np.log2(normal_vals + 1)

            t_stat, p_val = stats.ttest_ind(tumor_log, normal_log, equal_var=False)
            log2fc = tumor_log.mean() - normal_log.mean()

            results.append({
                'cancer_type': cancer,
                'gene': ferro_gene,
                'description': FERROPTOSIS_GENES[ferro_gene],
                'tumor_mean': tumor_log.mean(),
                'normal_mean': normal_log.mean(),
                'log2FC': log2fc,
                'p_value': p_val,
                'direction': 'UP' if log2fc > 0 else 'DOWN',
            })

    if results:
        df = pd.DataFrame(results)
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(df['p_value'], method='fdr_bh')
        df['fdr'] = fdr
        return df

    return None


def ferroptosis_survival_analysis():
    """Stratify patients by ferroptosis pathway activity and test survival."""
    clinical_file = DATA_DIR / "tcga" / "pan_cancer_clinical.parquet"
    if not clinical_file.exists():
        return None

    clinical = pd.read_parquet(clinical_file)
    results = []

    for cancer in ['BRCA', 'LUAD', 'KIRC']:
        expr_file = DATA_DIR / "tcga" / f"{cancer}_expression.parquet"
        if not expr_file.exists():
            continue

        expr = pd.read_parquet(expr_file)
        cancer_clinical = clinical[clinical['cancer_type'] == cancer].copy()

        # Get ferroptosis pathway score (mean z-score of anti-ferroptosis genes)
        anti_ferro = ['GPX4', 'SLC7A11', 'FSP1', 'GCLC', 'GSS', 'NFE2L2', 'FTH1']
        pro_ferro = ['ACSL4', 'LPCAT3', 'TFRC', 'NCOA4', 'HMOX1', 'ALOX15']

        sample_cols = list(expr.columns)
        tumor_cols = [c for c in sample_cols if len(c.split('-')) <= 3 or not c.split('-')[-1].startswith('1')]

        # Build case-insensitive gene index lookup
        gene_index_map = {g.upper(): g for g in expr.index}

        # Build ferroptosis score per patient
        ferro_scores = {}
        for patient_col in tumor_cols:
            patient_id = '-'.join(patient_col.split('-')[:3])
            anti_score = 0
            pro_score = 0
            n_anti = 0
            n_pro = 0

            for gene in anti_ferro:
                idx_name = gene_index_map.get(gene.upper())
                if idx_name is not None and idx_name in expr.index:
                    val = expr.loc[idx_name, patient_col]
                    if val > 0:
                        anti_score += np.log2(val + 1)
                        n_anti += 1

            for gene in pro_ferro:
                idx_name = gene_index_map.get(gene.upper())
                if idx_name is not None and idx_name in expr.index:
                    val = expr.loc[idx_name, patient_col]
                    if val > 0:
                        pro_score += np.log2(val + 1)
                        n_pro += 1

            if n_anti > 0 and n_pro > 0:
                # Ferroptosis vulnerability = pro / anti (higher = more vulnerable)
                ferro_scores[patient_id] = (pro_score / n_pro) / (anti_score / n_anti + 0.01)

        if len(ferro_scores) < 20:
            continue

        # Merge with clinical
        score_df = pd.DataFrame.from_dict(ferro_scores, orient='index', columns=['ferro_score'])
        score_df.index.name = 'case_id'
        score_df = score_df.reset_index()

        merged = cancer_clinical.merge(score_df, on='case_id', how='inner')
        logger.info(f"  [{cancer}] Merged survival+score: {len(merged)} patients")

        # Need survival columns - compute OS time
        if 'days_to_death' in merged.columns and 'days_to_follow_up' in merged.columns:
            merged['os_time'] = merged.apply(
                lambda r: r['days_to_death'] if pd.notna(r['days_to_death']) and r['days_to_death'] > 0
                else r['days_to_follow_up'], axis=1)
            time_col = 'os_time'
        else:
            time_col = None
            for col in ['days_to_follow_up', 'days_to_last_follow_up', 'days_to_death']:
                if col in merged.columns:
                    time_col = col
                    break

        status_col = 'vital_status' if 'vital_status' in merged.columns else None

        if time_col is None or status_col is None:
            continue

        merged = merged.dropna(subset=[time_col, status_col, 'ferro_score'])
        merged[time_col] = pd.to_numeric(merged[time_col], errors='coerce')
        merged = merged[merged[time_col] > 0]
        merged['event'] = (merged[status_col].str.lower() == 'dead').astype(int)

        if len(merged) < 30 or merged['event'].sum() < 5:
            continue

        # Split into high/low ferroptosis vulnerability
        median_score = merged['ferro_score'].median()
        merged['ferro_group'] = np.where(merged['ferro_score'] > median_score, 'High Vulnerability', 'Low Vulnerability')

        # Log-rank test
        high = merged[merged['ferro_group'] == 'High Vulnerability']
        low = merged[merged['ferro_group'] == 'Low Vulnerability']

        try:
            lr = logrank_test(high[time_col], low[time_col], high['event'], low['event'])
            p_val = lr.p_value

            # Cox regression
            cox_data = merged[[time_col, 'event', 'ferro_score']].dropna()
            cox_data.columns = ['T', 'E', 'ferro_score']
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='T', event_col='E')
            hr = np.exp(cph.params_['ferro_score'])
            hr_p = cph.summary.loc['ferro_score', 'p']

            results.append({
                'cancer_type': cancer,
                'n_patients': len(merged),
                'n_events': int(merged['event'].sum()),
                'logrank_p': p_val,
                'cox_hr': hr,
                'cox_p': hr_p,
                'median_score': median_score,
                'high_n': len(high),
                'low_n': len(low),
            })

            logger.info(f"  [{cancer}] Ferroptosis survival: HR={hr:.3f}, p={hr_p:.4f}, "
                       f"logrank p={p_val:.4f}")

            # KM plot
            fig, ax = plt.subplots(figsize=(8, 6))
            kmf = KaplanMeierFitter()

            for label, group in [('High Vulnerability', high), ('Low Vulnerability', low)]:
                kmf.fit(group[time_col], group['event'], label=label)
                kmf.plot_survival_function(ax=ax)

            ax.set_xlabel('Days')
            ax.set_ylabel('Survival Probability')
            ax.set_title(f'{cancer}: Ferroptosis Vulnerability & Survival\n'
                        f'HR={hr:.2f}, p={hr_p:.4f} (Log-rank p={p_val:.4f})',
                        fontweight='bold')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"{cancer}_ferroptosis_survival_km.png", dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"  [{cancer}] Survival analysis failed: {e}")

    return pd.DataFrame(results) if results else None


def plot_ferroptosis_landscape(sl_df, expr_df, survival_df):
    """Create comprehensive ferroptosis vulnerability landscape."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Panel A: SL significance heatmap (ferroptosis genes × drivers)
    ax = axes[0, 0]
    if sl_df is not None and len(sl_df) > 0:
        sig = sl_df[sl_df['fdr'] < 0.1].copy()
        if len(sig) > 0:
            pivot = sig.pivot_table(index='ferroptosis_gene', columns='driver_gene',
                                     values='delta_effect', aggfunc='mean')
            if pivot.shape[0] > 0 and pivot.shape[1] > 0:
                sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                           ax=ax, linewidths=0.5, cbar_kws={'label': 'Delta Effect'})
                ax.set_title('A) Ferroptosis Gene SL with Drivers\n(FDR < 0.1)', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No significant SL pairs', transform=ax.transAxes, ha='center')
                ax.set_title('A) Ferroptosis SL (None significant)', fontweight='bold')
        else:
            # Show top pairs even if not significant
            top = sl_df.head(20)
            pivot = top.pivot_table(index='ferroptosis_gene', columns='driver_gene',
                                     values='delta_effect', aggfunc='mean')
            sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                       ax=ax, linewidths=0.5)
            ax.set_title('A) Top Ferroptosis-Driver Interactions\n(nominal)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No SL data', transform=ax.transAxes, ha='center')

    # Panel B: Expression changes in tumors
    ax = axes[0, 1]
    if expr_df is not None and len(expr_df) > 0:
        sig_expr = expr_df[expr_df['fdr'] < 0.05].copy()
        if len(sig_expr) > 0:
            pivot_expr = sig_expr.pivot_table(index='gene', columns='cancer_type',
                                               values='log2FC', aggfunc='mean')
            sns.heatmap(pivot_expr, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                       ax=ax, linewidths=0.5, cbar_kws={'label': 'log2(Tumor/Normal)'})
            ax.set_title('B) Ferroptosis Gene Expression Changes\n(FDR < 0.05)', fontweight='bold')
        else:
            # Show all anyway
            pivot_expr = expr_df.pivot_table(index='gene', columns='cancer_type',
                                              values='log2FC', aggfunc='mean')
            sns.heatmap(pivot_expr, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                       ax=ax, linewidths=0.5)
            ax.set_title('B) Ferroptosis Gene Expression Changes', fontweight='bold')

    # Panel C: Volcano plot of SL effects
    ax = axes[0, 2]
    if sl_df is not None and len(sl_df) > 0:
        sl_df['neg_log10_p'] = -np.log10(sl_df['p_value'].clip(lower=1e-300))
        colors = ['red' if f < 0.05 else 'gray' for f in sl_df['fdr']]
        ax.scatter(sl_df['delta_effect'], sl_df['neg_log10_p'],
                  c=colors, s=30, alpha=0.6, edgecolor='white')
        # Label significant ones
        for _, row in sl_df[sl_df['fdr'] < 0.05].iterrows():
            ax.annotate(f"{row['ferroptosis_gene']}-{row['driver_gene']}",
                       (row['delta_effect'], row['neg_log10_p']),
                       fontsize=6, alpha=0.8)
        ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Delta Effect (driver-dep vs no-dep)')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('C) Ferroptosis SL Volcano Plot', fontweight='bold')

    # Panel D: Ferroptosis pathway diagram (simplified bar chart)
    ax = axes[1, 0]
    if sl_df is not None:
        gene_sig = sl_df.groupby('ferroptosis_gene').agg({
            'p_value': 'min',
            'delta_effect': lambda x: x.loc[x.abs().idxmax()] if len(x) > 0 else 0,
            'driver_gene': 'count',
        }).sort_values('p_value')

        top_genes = gene_sig.head(15)
        colors = ['#e74c3c' if p < 0.05 else '#f1c40f' if p < 0.1 else '#bdc3c7'
                 for p in top_genes['p_value']]
        ax.barh(range(len(top_genes)), -np.log10(top_genes['p_value']),
               color=colors, edgecolor='white')
        ax.set_yticks(range(len(top_genes)))
        ax.set_yticklabels(top_genes.index, fontsize=9)
        ax.set_xlabel('-log10(best p-value)')
        ax.set_title('D) Ferroptosis Gene SL Significance', fontweight='bold')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax.legend()
        ax.invert_yaxis()

    # Panel E: Survival results
    ax = axes[1, 1]
    if survival_df is not None and len(survival_df) > 0:
        x_pos = range(len(survival_df))
        colors = ['#e74c3c' if p < 0.05 else '#f1c40f' if p < 0.1 else '#bdc3c7'
                 for p in survival_df['cox_p']]
        ax.bar(x_pos, survival_df['cox_hr'], color=colors, edgecolor='white', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(survival_df['cancer_type'])
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Hazard Ratio (High vs Low Ferroptosis Vulnerability)')
        ax.set_title('E) Ferroptosis Vulnerability & Survival', fontweight='bold')
        for i, (_, row) in enumerate(survival_df.iterrows()):
            ax.text(i, row['cox_hr'] + 0.05, f'p={row["cox_p"]:.3f}', ha='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No survival data', transform=ax.transAxes, ha='center')

    # Panel F: Drug opportunities
    ax = axes[1, 2]
    dgidb_file = DATA_DIR / "drug_repurpose" / "dgidb_interactions.parquet"
    if dgidb_file.exists():
        dgidb = pd.read_parquet(dgidb_file)
        ferro_drugs = {}
        for gene in FERROPTOSIS_GENES:
            gene_drugs = dgidb[(dgidb['gene'] == gene) & (dgidb['approved'])]
            if len(gene_drugs) > 0:
                ferro_drugs[gene] = len(gene_drugs['drug'].unique())

        if ferro_drugs:
            genes = list(ferro_drugs.keys())
            counts = list(ferro_drugs.values())
            sorted_idx = np.argsort(counts)[::-1]
            genes = [genes[i] for i in sorted_idx]
            counts = [counts[i] for i in sorted_idx]

            ax.barh(range(len(genes)), counts, color='#2ecc71', alpha=0.8, edgecolor='white')
            ax.set_yticks(range(len(genes)))
            ax.set_yticklabels(genes, fontsize=9)
            ax.set_xlabel('Number of Approved Drugs')
            ax.set_title('F) Druggable Ferroptosis Regulators', fontweight='bold')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No druggable ferroptosis genes', transform=ax.transAxes, ha='center')
    else:
        ax.text(0.5, 0.5, 'No DGIdb data', transform=ax.transAxes, ha='center')

    plt.suptitle('Ferroptosis as Pan-Cancer Therapeutic Vulnerability',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ferroptosis_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: ferroptosis_landscape.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 15: Ferroptosis Pathway Deep Dive")
    logger.info("=" * 60)

    # Step 1: Test ferroptosis genes for SL
    logger.info("\n--- Testing ferroptosis genes for synthetic lethality ---")
    gene_effect_file = DATA_DIR / "drug_repurpose" / "depmap_gene_effect.parquet"
    sl_df = test_ferroptosis_sl(gene_effect_file)

    if sl_df is not None:
        sl_df.to_csv(RESULTS_DIR / "ferroptosis_sl_results.csv", index=False)
        sig = sl_df[sl_df['fdr'] < 0.1]
        logger.info(f"Total tests: {len(sl_df)}, Significant (FDR<0.1): {len(sig)}")

        for _, row in sl_df.head(20).iterrows():
            sig_tag = "***" if row['fdr'] < 0.05 else "**" if row['fdr'] < 0.1 else "*" if row['p_value'] < 0.05 else ""
            logger.info(f"  {row['ferroptosis_gene']} × {row['driver_gene']}: "
                       f"Δ={row['delta_effect']:.3f}, p={row['p_value']:.4f}, "
                       f"FDR={row['fdr']:.4f} {sig_tag}")

    # Step 2: Expression analysis
    logger.info("\n--- Ferroptosis gene expression in tumors vs normal ---")
    expr_df = ferroptosis_expression_analysis()

    if expr_df is not None:
        expr_df.to_csv(RESULTS_DIR / "ferroptosis_expression.csv", index=False)
        sig_expr = expr_df[expr_df['fdr'] < 0.05]
        logger.info(f"Expression tests: {len(expr_df)}, Significant (FDR<0.05): {len(sig_expr)}")

        for _, row in sig_expr.iterrows():
            logger.info(f"  [{row['cancer_type']}] {row['gene']}: "
                       f"log2FC={row['log2FC']:.3f} ({row['direction']}), FDR={row['fdr']:.4f}")

    # Step 3: Survival analysis
    logger.info("\n--- Ferroptosis vulnerability & survival ---")
    survival_df = ferroptosis_survival_analysis()

    if survival_df is not None:
        survival_df.to_csv(RESULTS_DIR / "ferroptosis_survival.csv", index=False)

    # Step 4: Comprehensive visualization
    logger.info("\n--- Generating ferroptosis landscape ---")
    plot_ferroptosis_landscape(sl_df, expr_df, survival_df)

    # Summary
    summary = {
        'experiment': 'Exp 15: Ferroptosis Deep Dive',
        'ferroptosis_genes_tested': len(FERROPTOSIS_GENES),
        'sl_tests': len(sl_df) if sl_df is not None else 0,
        'sl_significant_fdr01': len(sl_df[sl_df['fdr'] < 0.1]) if sl_df is not None else 0,
        'sl_significant_fdr005': len(sl_df[sl_df['fdr'] < 0.05]) if sl_df is not None else 0,
        'expression_significant': len(expr_df[expr_df['fdr'] < 0.05]) if expr_df is not None else 0,
        'survival_results': survival_df.to_dict('records') if survival_df is not None else [],
        'top_sl_pairs': sl_df.head(10).to_dict('records') if sl_df is not None else [],
    }

    with open(RESULTS_DIR / "exp15_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 15 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
