#!/usr/bin/env python3
"""
Experiment 9: Pan-Cancer Biomarker Panel Discovery
====================================================
Uses LASSO-regularized Cox regression + stability selection to identify
a minimal set of genes that predict survival across cancer types.

Approach:
1. For each cancer type: LASSO Cox → identify prognostic genes
2. Cross-validate stability: bootstrap LASSO (stability selection)
3. Find pan-cancer consensus biomarkers
4. Validate: compare panel vs known signatures (Oncotype DX, MammaPrint)

Target Paper: Paper 6 (AI Biomarker Panel) — JCO / Nature Medicine
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
EXPR_DIR = BASE_DIR / "data" / "tcga_expression"
CLINICAL_FILE = BASE_DIR / "data" / "tcga" / "pan_cancer_clinical.parquet"
SL_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
RESULTS_DIR = BASE_DIR / "results" / "exp9_biomarker_panel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']

# Known cancer gene panels for comparison
KNOWN_PANELS = {
    'Oncotype_DX_21': ['MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'ERBB2', 'GRB7',
                       'ESR1', 'PGR', 'BCL2', 'SCUBE2', 'MMP11', 'CTSL2', 'CD68',
                       'GSTM1', 'BAG1', 'ACTB', 'GAPDH', 'RPLP0', 'GUS', 'TFRC'],
    'MammaPrint_70': ['CCNE2', 'MMP9', 'CXCL1', 'IGFBP5', 'MELK', 'FLT1', 'TGFB3',
                      'ESM1', 'MCM6', 'KNTC2', 'CDC42BPA', 'PITRM1', 'CENPA', 'DTL',
                      'ECT2', 'EXT1', 'GPR180', 'IGFBP5', 'LIN9', 'LOC100288906'],
    'Immune_Checkpoint': ['CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD8A',
                          'IFNG', 'GZMA', 'PRF1'],
}


def prepare_survival_data(cancer_type):
    """Prepare expression + survival data for one cancer type."""
    # Expression
    expr_file = EXPR_DIR / f"{cancer_type}_expression_matrix.parquet"
    if not expr_file.exists():
        return None, None

    expr_df = pd.read_parquet(expr_file)
    expr_log = np.log2(expr_df + 1)

    # Filter to tumor samples
    tumor_samples = []
    for s in expr_df.columns:
        parts = s.split('-')
        if len(parts) >= 4:
            st = parts[3][:2]
            if st.isdigit() and 10 <= int(st) <= 19:
                continue  # Skip normal
        tumor_samples.append(s)

    expr_tumor = expr_log[tumor_samples]

    # Clinical
    clinical = pd.read_parquet(CLINICAL_FILE)
    clinical = clinical[clinical['cancer_type'] == cancer_type]

    # Match samples
    rows = []
    for sample_id in expr_tumor.columns:
        patient_id = '-'.join(str(sample_id).split('-')[:3])
        clin_match = clinical[clinical['case_id'] == patient_id]
        if len(clin_match) == 0:
            continue

        cr = clin_match.iloc[0]
        time = cr.get('days_to_death')
        if pd.isna(time):
            time = cr.get('days_to_follow_up')
        if pd.isna(time) or time is None or float(time) <= 0:
            continue

        event = 1 if str(cr.get('vital_status', '')).lower() in ['dead', 'deceased'] else 0

        rows.append({
            'sample_id': sample_id,
            'time': float(time),
            'event': event,
        })

    surv_df = pd.DataFrame(rows)
    if len(surv_df) < 30:
        return None, None

    # Filter to medium-high variance genes
    gene_var = expr_tumor.var(axis=1)
    gene_mean = expr_tumor.mean(axis=1)
    selected_genes = gene_var[(gene_var > gene_var.quantile(0.5)) & (gene_mean > 1.0)].index

    expr_selected = expr_tumor.loc[selected_genes, surv_df['sample_id']].T
    surv_df = surv_df.set_index('sample_id')

    logger.info(f"[{cancer_type}] Survival data: {len(surv_df)} patients, {len(selected_genes)} genes")
    logger.info(f"  Events: {surv_df['event'].sum()}, Median time: {surv_df['time'].median():.0f} days")

    return expr_selected, surv_df


def lasso_cox_selection(X, surv_df, n_bootstrap=50, alpha_range=None):
    """
    Stability selection via bootstrapped LASSO Cox regression.
    Returns genes selected in >50% of bootstrap iterations.
    """
    from lifelines import CoxPHFitter

    n_samples, n_genes = X.shape
    gene_names = X.columns.tolist()
    selection_counts = np.zeros(n_genes)

    if alpha_range is None:
        alpha_range = [0.5, 1.0, 2.0, 5.0]

    for boot in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X.iloc[idx].copy()
        surv_boot = surv_df.iloc[idx].copy()

        # Try different penalty strengths
        for alpha in alpha_range:
            try:
                # Use top 200 genes by univariate Cox p-value (pre-filter for speed)
                if n_genes > 200:
                    p_values = []
                    for gene in gene_names[:500]:  # Check first 500
                        try:
                            cph = CoxPHFitter()
                            df = pd.DataFrame({
                                'time': surv_boot['time'].values,
                                'event': surv_boot['event'].values,
                                gene: X_boot[gene].values,
                            })
                            cph.fit(df, duration_col='time', event_col='event')
                            p_values.append((gene, cph.summary.iloc[0]['p']))
                        except:
                            p_values.append((gene, 1.0))

                    p_values.sort(key=lambda x: x[1])
                    top_genes = [g for g, _ in p_values[:100]]
                else:
                    top_genes = gene_names

                # LASSO Cox
                df = surv_boot[['time', 'event']].copy()
                for g in top_genes:
                    df[g] = X_boot[g].values

                cph = CoxPHFitter(penalizer=alpha, l1_ratio=1.0)
                cph.fit(df, duration_col='time', event_col='event')

                # Count selected (non-zero coefficients)
                coefs = cph.params_
                for gene in top_genes:
                    if gene in coefs.index and abs(coefs[gene]) > 1e-6:
                        gene_idx = gene_names.index(gene) if gene in gene_names else -1
                        if gene_idx >= 0:
                            selection_counts[gene_idx] += 1

                break  # Success, move to next bootstrap
            except Exception:
                continue

        if (boot + 1) % 10 == 0:
            logger.info(f"  Bootstrap {boot+1}/{n_bootstrap}")

    # Stability scores
    stability_scores = selection_counts / (n_bootstrap * len(alpha_range))

    results = pd.DataFrame({
        'gene': gene_names,
        'stability_score': stability_scores,
        'n_selected': selection_counts.astype(int),
    })
    results = results.sort_values('stability_score', ascending=False)

    return results


def evaluate_panel(X, surv_df, panel_genes, panel_name):
    """Evaluate a gene panel for survival prediction (C-index)."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    available = [g for g in panel_genes if g in X.columns]
    if len(available) < 3:
        return None

    df = surv_df[['time', 'event']].copy()
    for g in available:
        df[g] = X[g].values

    df = df.dropna()
    if len(df) < 30:
        return None

    # 5-fold cross-validation
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)
    fold_size = n // 5
    c_indices = []

    for fold in range(5):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        try:
            cph = CoxPHFitter(penalizer=0.5)
            cph.fit(df.iloc[train_idx], duration_col='time', event_col='event')
            c_idx = cph.score(df.iloc[test_idx], scoring_method='concordance_index')
            c_indices.append(c_idx)
        except:
            pass

    if c_indices:
        return {
            'panel': panel_name,
            'n_genes': len(available),
            'mean_c_index': np.mean(c_indices),
            'std_c_index': np.std(c_indices),
            'genes': ', '.join(available[:10]),
        }
    return None


def plot_biomarker_results(all_stability, all_panel_results):
    """Visualize biomarker panel discovery results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: Top stable genes per cancer type
    ax = axes[0, 0]
    for i, (ct, stab_df) in enumerate(all_stability.items()):
        top20 = stab_df.head(20)
        ax.barh([f"{g} ({ct})" for g in top20['gene'][:10]],
                top20['stability_score'][:10],
                color=plt.cm.Set2(i), alpha=0.8, label=ct)
    ax.set_xlabel('Stability Score')
    ax.set_title('A) Top Stable Prognostic Genes', fontweight='bold')
    ax.legend()
    ax.invert_yaxis()

    # Panel B: Consensus genes (appear in multiple cancer types)
    ax = axes[0, 1]
    gene_ct_count = {}
    gene_mean_stability = {}
    for ct, stab_df in all_stability.items():
        top50 = stab_df[stab_df['stability_score'] > 0.1]
        for _, row in top50.iterrows():
            g = row['gene']
            if g not in gene_ct_count:
                gene_ct_count[g] = 0
                gene_mean_stability[g] = []
            gene_ct_count[g] += 1
            gene_mean_stability[g].append(row['stability_score'])

    consensus = {g: np.mean(s) for g, s in gene_mean_stability.items()
                 if gene_ct_count[g] >= 2}
    if consensus:
        consensus_sorted = sorted(consensus.items(), key=lambda x: -x[1])[:20]
        genes_c = [g for g, _ in consensus_sorted]
        scores_c = [s for _, s in consensus_sorted]
        ax.barh(genes_c, scores_c, color='#e74c3c', alpha=0.8)
        ax.set_xlabel('Mean Stability Score')
        ax.set_title(f'B) Pan-Cancer Consensus Genes (≥2 types)', fontweight='bold')
        ax.invert_yaxis()

    # Panel C: Panel comparison (our panel vs known panels)
    ax = axes[1, 0]
    if all_panel_results:
        panel_df = pd.DataFrame(all_panel_results)
        panel_pivot = panel_df.pivot_table(index='panel', columns='cancer_type',
                                          values='mean_c_index', aggfunc='max')
        panel_pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='white')
        ax.set_ylabel('C-index')
        ax.set_title('C) Gene Panel Comparison', fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0.4, 0.8)
        ax.tick_params(axis='x', rotation=45)

    # Panel D: Number of genes vs C-index
    ax = axes[1, 1]
    if all_panel_results:
        panel_df = pd.DataFrame(all_panel_results)
        ax.scatter(panel_df['n_genes'], panel_df['mean_c_index'],
                  c=panel_df['cancer_type'].map({'BRCA': 0, 'LUAD': 1, 'KIRC': 2}),
                  cmap='Set1', s=100, alpha=0.7, edgecolor='white', linewidth=1.5)
        for _, row in panel_df.iterrows():
            ax.annotate(f"{row['panel'][:10]}", (row['n_genes'], row['mean_c_index']),
                       fontsize=7, alpha=0.8)
        ax.set_xlabel('Number of Genes')
        ax.set_ylabel('C-index')
        ax.set_title('D) Panel Size vs Predictive Power', fontweight='bold')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Pan-Cancer Biomarker Panel Discovery', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "biomarker_panel_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: biomarker_panel_results.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 9: Pan-Cancer Biomarker Panel Discovery")
    logger.info("=" * 60)

    all_stability = {}
    all_panel_results = []

    for cancer_type in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer_type}")
        logger.info(f"{'='*40}")

        X, surv_df = prepare_survival_data(cancer_type)
        if X is None:
            continue

        # Step 1: LASSO Cox stability selection
        logger.info(f"Running stability selection...")
        stability = lasso_cox_selection(X, surv_df, n_bootstrap=30)
        all_stability[cancer_type] = stability

        n_stable = (stability['stability_score'] > 0.2).sum()
        logger.info(f"[{cancer_type}] Stable genes (score > 0.2): {n_stable}")
        logger.info(f"  Top 10: {stability.head(10)['gene'].tolist()}")

        stability.to_csv(RESULTS_DIR / f"{cancer_type}_stability_selection.csv", index=False)

        # Step 2: Evaluate our panel
        our_panel = stability[stability['stability_score'] > 0.15]['gene'].tolist()[:30]
        if our_panel:
            r = evaluate_panel(X, surv_df, our_panel, f'Our-{cancer_type}')
            if r:
                r['cancer_type'] = cancer_type
                all_panel_results.append(r)
                logger.info(f"  Our panel ({len(our_panel)} genes): C-index = {r['mean_c_index']:.4f}")

        # Step 3: Evaluate known panels
        for panel_name, panel_genes in KNOWN_PANELS.items():
            r = evaluate_panel(X, surv_df, panel_genes, panel_name)
            if r:
                r['cancer_type'] = cancer_type
                all_panel_results.append(r)
                logger.info(f"  {panel_name}: C-index = {r['mean_c_index']:.4f} ({r['n_genes']} genes)")

    # Visualize
    plot_biomarker_results(all_stability, all_panel_results)

    # Find consensus genes
    consensus_genes = {}
    for ct, stab in all_stability.items():
        for _, row in stab[stab['stability_score'] > 0.1].iterrows():
            g = row['gene']
            if g not in consensus_genes:
                consensus_genes[g] = {'cancer_types': [], 'scores': []}
            consensus_genes[g]['cancer_types'].append(ct)
            consensus_genes[g]['scores'].append(row['stability_score'])

    pan_cancer_panel = {g: info for g, info in consensus_genes.items()
                       if len(info['cancer_types']) >= 2}

    logger.info(f"\nPan-cancer consensus biomarkers (≥2 cancer types): {len(pan_cancer_panel)}")
    for gene, info in sorted(pan_cancer_panel.items(), key=lambda x: -np.mean(x[1]['scores']))[:20]:
        logger.info(f"  {gene}: {info['cancer_types']} (mean stability = {np.mean(info['scores']):.3f})")

    # Check SL overlap
    sl_file = SL_DIR / "synthetic_lethal_pairs.csv"
    if sl_file.exists():
        sl_df = pd.read_csv(sl_file)
        sl_targets = set(sl_df['target_gene'].unique())
        panel_sl_overlap = set(pan_cancer_panel.keys()) & sl_targets
        if panel_sl_overlap:
            logger.info(f"\n*** Biomarker-SL overlap: {sorted(panel_sl_overlap)} ***")
            logger.info("These genes are both prognostic AND synthetic lethal targets!")

    # Summary
    summary = {
        'experiment': 'Exp 9: Pan-Cancer Biomarker Panel',
        'cancer_types': CANCER_TYPES,
        'pan_cancer_consensus_genes': len(pan_cancer_panel),
        'consensus_gene_list': list(pan_cancer_panel.keys())[:50],
        'panel_evaluations': all_panel_results,
    }

    with open(RESULTS_DIR / "exp9_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 9 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
