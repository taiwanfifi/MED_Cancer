#!/usr/bin/env python3
"""
Experiment 11: Survival Outlier Analysis — Finding the "Golden Keys"
=====================================================================
Inspired by: "Why do some Stage IV patients survive 10 years?
Why do some Stage I patients die in 3 months?"

This experiment:
1. Identifies extreme survival outliers (top/bottom 10% by expected vs actual)
2. Compares their gene expression profiles
3. Identifies protective/lethal gene signatures
4. Cross-references with SL targets, immune scores, and drug targets
5. Proposes mechanisms for exceptional outcomes

Target Paper: Paper 5 (Multi-Omics) or standalone in Nature Medicine
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

BASE_DIR = Path("/workspace/cancer_research")
EXPR_DIR = BASE_DIR / "data" / "tcga_expression"
CLINICAL_FILE = BASE_DIR / "data" / "tcga" / "pan_cancer_clinical.parquet"
SL_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
IMM_DIR = BASE_DIR / "results" / "exp6_immune_tme"
RESULTS_DIR = BASE_DIR / "results" / "exp11_outlier_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def identify_outliers(cancer_type):
    """
    Find survival outliers: patients whose survival deviates most
    from what their stage/age would predict.
    """
    clinical = pd.read_parquet(CLINICAL_FILE)
    ct_clin = clinical[clinical['cancer_type'] == cancer_type].copy()

    # Compute survival time
    ct_clin['time'] = pd.to_numeric(ct_clin['days_to_death'], errors='coerce')
    follow_up = pd.to_numeric(ct_clin['days_to_follow_up'], errors='coerce')
    ct_clin['time'] = ct_clin['time'].fillna(follow_up)
    ct_clin['event'] = ct_clin['vital_status'].apply(
        lambda x: 1 if str(x).lower() in ['dead', 'deceased'] else 0
    )

    # Filter valid
    ct_clin = ct_clin.dropna(subset=['time'])
    ct_clin = ct_clin[ct_clin['time'] > 0]

    # Encode stage
    def parse_stage(s):
        s = str(s)
        if 'IV' in s: return 4
        if 'III' in s: return 3
        if 'II' in s: return 2
        if 'I' in s: return 1
        return np.nan

    ct_clin['stage_num'] = ct_clin['stage'].apply(parse_stage)
    ct_clin['age_num'] = pd.to_numeric(ct_clin['age'], errors='coerce')

    # Fit Cox model on stage + age to get expected risk
    valid = ct_clin.dropna(subset=['stage_num', 'age_num', 'time', 'event']).copy()

    if len(valid) < 30:
        logger.warning(f"[{cancer_type}] Only {len(valid)} patients with complete data")
        return None, None

    try:
        cph = CoxPHFitter()
        cox_df = valid[['time', 'event', 'stage_num', 'age_num']].copy()
        cph.fit(cox_df, duration_col='time', event_col='event')

        # Get predicted risk scores
        valid['risk_score'] = cph.predict_partial_hazard(cox_df).values

        # Residuals: actual survival vs expected
        # Positive residual = survived longer than expected (GOOD outlier)
        # Negative residual = died sooner than expected (BAD outlier)
        median_time = valid['time'].median()
        valid['survival_ratio'] = valid['time'] / median_time
        valid['risk_residual'] = valid['survival_ratio'] / (valid['risk_score'] + 1e-6)

        # Identify outliers
        q_low = valid['risk_residual'].quantile(0.1)
        q_high = valid['risk_residual'].quantile(0.9)

        valid['outlier_type'] = 'normal'
        valid.loc[valid['risk_residual'] >= q_high, 'outlier_type'] = 'exceptional_survivor'
        valid.loc[valid['risk_residual'] <= q_low, 'outlier_type'] = 'unexpected_death'

        n_good = (valid['outlier_type'] == 'exceptional_survivor').sum()
        n_bad = (valid['outlier_type'] == 'unexpected_death').sum()

        logger.info(f"[{cancer_type}] Outlier analysis: {len(valid)} patients")
        logger.info(f"  Exceptional survivors (top 10%): {n_good}")
        logger.info(f"  Unexpected deaths (bottom 10%): {n_bad}")

        # Print some examples
        good_ex = valid[valid['outlier_type'] == 'exceptional_survivor'].nlargest(5, 'risk_residual')
        bad_ex = valid[valid['outlier_type'] == 'unexpected_death'].nsmallest(5, 'risk_residual')

        logger.info(f"  Top exceptional survivors:")
        for _, p in good_ex.iterrows():
            logger.info(f"    {p['case_id']}: Stage {p['stage']}, Age {p['age_num']:.0f}, "
                       f"Survived {p['time']:.0f} days, Status: {p['vital_status']}")

        logger.info(f"  Top unexpected deaths:")
        for _, p in bad_ex.iterrows():
            logger.info(f"    {p['case_id']}: Stage {p['stage']}, Age {p['age_num']:.0f}, "
                       f"Survived {p['time']:.0f} days, Status: {p['vital_status']}")

        return valid, cph

    except Exception as e:
        logger.error(f"[{cancer_type}] Cox model failed: {e}")
        return None, None


def compare_outlier_expression(outlier_df, cancer_type):
    """Compare gene expression between exceptional survivors and unexpected deaths."""
    expr_file = EXPR_DIR / f"{cancer_type}_expression_matrix.parquet"
    if not expr_file.exists():
        return None

    expr_df = pd.read_parquet(expr_file)
    expr_log = np.log2(expr_df + 1)

    # Map patients to expression samples
    good_patients = set(outlier_df[outlier_df['outlier_type'] == 'exceptional_survivor']['case_id'])
    bad_patients = set(outlier_df[outlier_df['outlier_type'] == 'unexpected_death']['case_id'])

    good_samples = []
    bad_samples = []
    for s in expr_df.columns:
        pid = '-'.join(str(s).split('-')[:3])
        if pid in good_patients:
            good_samples.append(s)
        elif pid in bad_patients:
            bad_samples.append(s)

    logger.info(f"[{cancer_type}] Expression matched: {len(good_samples)} good, {len(bad_samples)} bad")

    if len(good_samples) < 3 or len(bad_samples) < 3:
        logger.warning(f"[{cancer_type}] Too few matched samples")
        return None

    # Differential expression: good vs bad
    good_expr = expr_log[good_samples]
    bad_expr = expr_log[bad_samples]

    # Filter expressed genes
    mean_expr = expr_log.mean(axis=1)
    expressed = mean_expr[mean_expr >= 1.0].index

    results = []
    for gene in expressed:
        g_vals = good_expr.loc[gene].values
        b_vals = bad_expr.loc[gene].values

        if np.std(g_vals) == 0 and np.std(b_vals) == 0:
            continue

        t_stat, p_val = stats.ttest_ind(g_vals, b_vals, equal_var=False)
        log2fc = np.mean(g_vals) - np.mean(b_vals)

        results.append({
            'gene': gene,
            'log2FC_good_vs_bad': log2fc,
            'mean_good': np.mean(g_vals),
            'mean_bad': np.mean(b_vals),
            't_statistic': t_stat,
            'p_value': p_val,
        })

    if not results:
        return None

    deg_df = pd.DataFrame(results)

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(deg_df['p_value'], method='fdr_bh')
    deg_df['fdr'] = fdr
    deg_df['abs_log2FC'] = deg_df['log2FC_good_vs_bad'].abs()
    deg_df = deg_df.sort_values('abs_log2FC', ascending=False)

    # Classify
    deg_df['direction'] = 'NS'
    deg_df.loc[(deg_df['fdr'] < 0.05) & (deg_df['log2FC_good_vs_bad'] > 0.5), 'direction'] = 'UP_in_survivors'
    deg_df.loc[(deg_df['fdr'] < 0.05) & (deg_df['log2FC_good_vs_bad'] < -0.5), 'direction'] = 'DOWN_in_survivors'

    n_up = (deg_df['direction'] == 'UP_in_survivors').sum()
    n_down = (deg_df['direction'] == 'DOWN_in_survivors').sum()

    logger.info(f"[{cancer_type}] Outlier DEGs: {n_up} up in survivors, {n_down} down in survivors")

    # Top genes
    logger.info(f"  Top genes UP in exceptional survivors (protective?):")
    for _, row in deg_df[deg_df['direction'] == 'UP_in_survivors'].head(10).iterrows():
        logger.info(f"    {row['gene']}: log2FC={row['log2FC_good_vs_bad']:.2f}, FDR={row['fdr']:.4f}")

    logger.info(f"  Top genes DOWN in exceptional survivors (lethal when high?):")
    for _, row in deg_df[deg_df['direction'] == 'DOWN_in_survivors'].head(10).iterrows():
        logger.info(f"    {row['gene']}: log2FC={row['log2FC_good_vs_bad']:.2f}, FDR={row['fdr']:.4f}")

    deg_df.to_csv(RESULTS_DIR / f"{cancer_type}_outlier_deg.csv", index=False)

    return deg_df


def compare_outlier_immune(outlier_df, cancer_type):
    """Compare immune profiles of outliers."""
    immune_file = IMM_DIR / f"{cancer_type}_immune_scores.csv"
    if not immune_file.exists():
        return None

    immune_df = pd.read_csv(immune_file, index_col=0)

    # Map patients
    good_patients = set(outlier_df[outlier_df['outlier_type'] == 'exceptional_survivor']['case_id'])
    bad_patients = set(outlier_df[outlier_df['outlier_type'] == 'unexpected_death']['case_id'])

    good_scores = []
    bad_scores = []
    for sid in immune_df.index:
        pid = '-'.join(str(sid).split('-')[:3])
        if pid in good_patients:
            good_scores.append(immune_df.loc[sid])
        elif pid in bad_patients:
            bad_scores.append(immune_df.loc[sid])

    if len(good_scores) < 3 or len(bad_scores) < 3:
        return None

    good_df = pd.DataFrame(good_scores)
    bad_df = pd.DataFrame(bad_scores)

    # Compare each immune cell type
    results = []
    for cell_type in immune_df.columns:
        g = good_df[cell_type].values
        b = bad_df[cell_type].values
        t_stat, p_val = stats.ttest_ind(g, b, equal_var=False)
        diff = np.mean(g) - np.mean(b)

        results.append({
            'cell_type': cell_type,
            'mean_survivor': np.mean(g),
            'mean_death': np.mean(b),
            'difference': diff,
            'p_value': p_val,
            'direction': 'Higher in survivors' if diff > 0 else 'Higher in deaths',
        })

        if p_val < 0.1:
            logger.info(f"  {cell_type}: survivors={np.mean(g):.3f} vs deaths={np.mean(b):.3f}, "
                       f"p={p_val:.4f} {'***' if p_val < 0.05 else '*'}")

    immune_results = pd.DataFrame(results)
    immune_results.to_csv(RESULTS_DIR / f"{cancer_type}_outlier_immune.csv", index=False)

    return immune_results


def plot_outlier_analysis(outlier_df, cancer_type, deg_df=None, immune_results=None):
    """Visualize outlier analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: Survival plot colored by outlier type
    ax = axes[0, 0]
    colors = {'exceptional_survivor': '#2ecc71', 'unexpected_death': '#e74c3c', 'normal': '#95a5a6'}
    for otype, color in colors.items():
        mask = outlier_df['outlier_type'] == otype
        ax.scatter(outlier_df.loc[mask, 'age_num'],
                  outlier_df.loc[mask, 'time'] / 365.25,
                  c=color, alpha=0.6, s=40, label=f"{otype} (n={mask.sum()})")
    ax.set_xlabel('Age at Diagnosis')
    ax.set_ylabel('Survival Time (Years)')
    ax.set_title(f'A) {cancer_type}: Survival Outliers', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel B: Stage distribution of outliers
    ax = axes[0, 1]
    stage_data = []
    for otype in ['exceptional_survivor', 'unexpected_death', 'normal']:
        sub = outlier_df[outlier_df['outlier_type'] == otype]
        for stage in [1, 2, 3, 4]:
            n = (sub['stage_num'] == stage).sum()
            stage_data.append({'type': otype, 'stage': f'Stage {stage}', 'count': n})

    stage_df = pd.DataFrame(stage_data)
    stage_pivot = stage_df.pivot(index='stage', columns='type', values='count').fillna(0)
    stage_pivot.plot(kind='bar', ax=ax, color=['#2ecc71', '#95a5a6', '#e74c3c'])
    ax.set_title(f'B) Stage Distribution by Outlier Type', fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.tick_params(axis='x', rotation=0)

    # Panel C: Top protective genes (if available)
    ax = axes[1, 0]
    if deg_df is not None:
        up_genes = deg_df[deg_df['direction'] == 'UP_in_survivors'].head(15)
        down_genes = deg_df[deg_df['direction'] == 'DOWN_in_survivors'].head(15)
        combined = pd.concat([up_genes.head(10), down_genes.head(10)])
        if len(combined) > 0:
            colors_gene = ['#2ecc71' if d == 'UP_in_survivors' else '#e74c3c'
                          for d in combined['direction']]
            ax.barh(range(len(combined)), combined['log2FC_good_vs_bad'], color=colors_gene)
            ax.set_yticks(range(len(combined)))
            ax.set_yticklabels(combined['gene'], fontsize=8)
            ax.set_xlabel('log2FC (Survivor vs Death)')
            ax.set_title(f'C) Protective/Lethal Gene Signatures', fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--')
            ax.invert_yaxis()

    # Panel D: Immune composition comparison
    ax = axes[1, 1]
    if immune_results is not None:
        sig = immune_results[immune_results['p_value'] < 0.2].sort_values('difference')
        if len(sig) > 0:
            colors_imm = ['#2ecc71' if d > 0 else '#e74c3c' for d in sig['difference']]
            ax.barh(range(len(sig)), sig['difference'], color=colors_imm)
            ax.set_yticks(range(len(sig)))
            ax.set_yticklabels(sig['cell_type'], fontsize=9)
            ax.set_xlabel('Score Difference (Survivor - Death)')
            ax.set_title(f'D) Immune Profile: Survivors vs Deaths', fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--')
            ax.invert_yaxis()

    plt.suptitle(f'{cancer_type}: Survival Outlier Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cancer_type}_outlier_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {cancer_type}_outlier_analysis.png")


def cross_reference_with_sl(deg_df, cancer_type):
    """Check if outlier-specific genes overlap with SL targets."""
    sl_file = SL_DIR / "synthetic_lethal_pairs.csv"
    if not sl_file.exists() or deg_df is None:
        return

    sl_df = pd.read_csv(sl_file)
    sl_targets = set(sl_df['target_gene'].unique())

    protective = set(deg_df[deg_df['direction'] == 'UP_in_survivors']['gene'])
    lethal = set(deg_df[deg_df['direction'] == 'DOWN_in_survivors']['gene'])

    protective_sl = protective & sl_targets
    lethal_sl = lethal & sl_targets

    if protective_sl:
        logger.info(f"\n  *** PROTECTIVE genes that are also SL targets: {sorted(protective_sl)} ***")
        logger.info("  → These genes protect against cancer AND are vulnerable when specific drivers are mutated")

    if lethal_sl:
        logger.info(f"\n  *** LETHAL genes that are also SL targets: {sorted(lethal_sl)} ***")
        logger.info("  → High expression of these genes kills patients AND they are SL targets")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 11: Survival Outlier Analysis")
    logger.info("=" * 60)

    all_findings = []

    for cancer_type in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer_type}")
        logger.info(f"{'='*40}")

        # Step 1: Identify outliers
        outlier_df, cph = identify_outliers(cancer_type)
        if outlier_df is None:
            continue

        # Step 2: Compare expression
        logger.info(f"\nComparing gene expression...")
        deg_df = compare_outlier_expression(outlier_df, cancer_type)

        # Step 3: Compare immune profiles
        logger.info(f"\nComparing immune profiles...")
        immune_results = compare_outlier_immune(outlier_df, cancer_type)

        # Step 4: Cross-reference with SL targets
        logger.info(f"\nCross-referencing with SL targets...")
        cross_reference_with_sl(deg_df, cancer_type)

        # Step 5: Visualize
        plot_outlier_analysis(outlier_df, cancer_type, deg_df, immune_results)

        # Collect findings
        finding = {
            'cancer_type': cancer_type,
            'n_patients': len(outlier_df),
            'n_exceptional_survivors': (outlier_df['outlier_type'] == 'exceptional_survivor').sum(),
            'n_unexpected_deaths': (outlier_df['outlier_type'] == 'unexpected_death').sum(),
        }
        if deg_df is not None:
            finding['n_protective_genes'] = (deg_df['direction'] == 'UP_in_survivors').sum()
            finding['n_lethal_genes'] = (deg_df['direction'] == 'DOWN_in_survivors').sum()
            finding['top_protective'] = deg_df[deg_df['direction'] == 'UP_in_survivors'].head(5)['gene'].tolist()
            finding['top_lethal'] = deg_df[deg_df['direction'] == 'DOWN_in_survivors'].head(5)['gene'].tolist()

        all_findings.append(finding)

    # Summary
    summary = {
        'experiment': 'Exp 11: Survival Outlier Analysis',
        'rationale': 'Why do some Stage IV patients survive years while some Stage I patients die quickly?',
        'findings': all_findings,
    }

    with open(RESULTS_DIR / "exp11_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 11 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
