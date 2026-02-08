#!/usr/bin/env python3
"""
Experiment 22: SL-Based Prognostic Signature Development
=========================================================
Develops a novel prognostic gene signature based on SL target expression.
Uses multi-evidence genes from our analyses to build a clinically-useful
risk score.

Key analyses:
1. Feature selection: SL targets with survival association
2. LASSO Cox signature development (training on BRCA)
3. Risk score calculation and patient stratification
4. Validation across LUAD and KIRC
5. Comparison with established signatures (Oncotype DX, MammaPrint)
6. Multivariate analysis controlling for clinical variables

Target: Clinical prognostic signature paper
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp22_prognostic_signature"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Candidate genes: multi-evidence targets from our analyses
CANDIDATE_GENES = [
    # Top SL targets
    'GPX4', 'FGFR1', 'MDM2', 'PTEN', 'BCL2', 'CDK6', 'MYC', 'CCND1',
    # Novel targets (low literature, high computational evidence)
    'CDS2', 'CHMP4B', 'GINS4', 'PSMA4', 'MAD2L1',
    # Pan-cancer SL targets
    'KIF14', 'ORC6', 'SGO1', 'TTK', 'TUBG1', 'CHAF1B',
    # Ferroptosis markers
    'SLC7A11', 'ACSL4', 'GCLC', 'GCLM', 'NFE2L2', 'DHODH',
    # Immune markers
    'CD274', 'PDCD1', 'LAG3', 'HAVCR2',
    # Key oncogenes/TSGs
    'EGFR', 'ERBB2', 'BRAF', 'TP53', 'BRCA1', 'BRCA2', 'STK11',
]

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def load_data(cancer_type):
    """Load expression and clinical data."""
    # Expression
    expr_file = DATA_DIR / "tcga_expression" / f"{cancer_type}_expression_matrix.parquet"
    if not expr_file.exists():
        expr_file = DATA_DIR / "tcga" / f"{cancer_type}_expression.parquet"
    if not expr_file.exists():
        return pd.DataFrame(), pd.DataFrame()
    expr = pd.read_parquet(expr_file)

    # Clinical
    clin_file = DATA_DIR / "tcga" / f"{cancer_type}_clinical.json"
    if clin_file.exists():
        with open(clin_file) as f:
            data = json.load(f)
        clin = pd.DataFrame(data)
        if 'case_id' in clin.columns:
            clin = clin.set_index('case_id')
    else:
        clin = pd.DataFrame()

    return expr, clin


def build_survival_matrix(expr, clin, cancer_type):
    """
    Build combined expression + survival DataFrame.
    Returns: DataFrame with gene expression columns + os_time + os_event.
    """
    gene_index_map = {g.upper(): g for g in expr.index}

    # Build patient data
    records = []
    for col in expr.columns:
        # Get short barcode
        short = '-'.join(str(col).split('-')[:3])

        # Only tumor samples
        parts = str(col).split('-')
        if len(parts) >= 4:
            sample_type = int(parts[3][:2])
            if sample_type >= 10:  # Normal
                continue

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

        record = {
            'patient': short,
            'expr_col': col,
            'os_time': float(os_time),
            'os_event': int(os_event),
        }

        # Add clinical features
        age = pd.to_numeric(row.get('age_at_index', np.nan), errors='coerce')
        if pd.notna(age):
            record['age'] = float(age)

        stage = str(row.get('tumor_stage', '')).lower()
        stage_map = {'stage i': 1, 'stage ia': 1, 'stage ib': 1,
                     'stage ii': 2, 'stage iia': 2, 'stage iib': 2,
                     'stage iii': 3, 'stage iiia': 3, 'stage iiib': 3, 'stage iiic': 3,
                     'stage iv': 4, 'stage iva': 4, 'stage ivb': 4}
        record['stage_numeric'] = stage_map.get(stage, np.nan)

        # Add gene expression
        for gene in CANDIDATE_GENES:
            idx = gene_index_map.get(gene.upper())
            if idx is not None:
                record[gene] = float(expr.loc[idx, col])

        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"  {cancer_type}: {len(df)} patients with survival + expression")
    logger.info(f"  Events: {df['os_event'].sum()}, Median OS: {df['os_time'].median():.0f} days")
    return df


def univariate_screening(df, genes, alpha=0.1):
    """
    Screen genes for survival association using univariate Cox regression.
    Returns genes with p < alpha.
    """
    significant = []
    for gene in genes:
        if gene not in df.columns:
            continue
        try:
            cox_df = df[['os_time', 'os_event', gene]].dropna()
            if len(cox_df) < 20 or cox_df['os_event'].sum() < 5:
                continue
            # Z-score
            cox_df[gene] = (cox_df[gene] - cox_df[gene].mean()) / (cox_df[gene].std() + 1e-8)

            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='os_time', event_col='os_event')
            hr = float(np.exp(cph.params_[gene]))
            p = float(cph.summary['p'][gene])

            significant.append({
                'gene': gene,
                'hr': hr,
                'coef': float(cph.params_[gene]),
                'p_value': p,
                'concordance': float(cph.concordance_index_),
            })
        except Exception as e:
            logger.debug(f"  {gene} failed: {e}")

    result = pd.DataFrame(significant).sort_values('p_value')
    sig = result[result['p_value'] < alpha]
    logger.info(f"  Univariate screening: {len(sig)}/{len(result)} genes significant (p<{alpha})")
    return result, sig


def build_lasso_signature(df, sig_genes, n_folds=5):
    """
    Build LASSO Cox prognostic signature.
    Uses cross-validation to select optimal regularization.
    """
    from sklearn.linear_model import Lasso
    from lifelines.utils import concordance_index

    gene_list = [g for g in sig_genes if g in df.columns]
    if len(gene_list) < 3:
        logger.warning("  Too few genes for signature, using all candidates")
        gene_list = [g for g in CANDIDATE_GENES if g in df.columns]

    # Prepare data
    complete = df.dropna(subset=['os_time', 'os_event'] + gene_list)
    if len(complete) < 30:
        logger.warning(f"  Only {len(complete)} complete cases")
        return None, None, None

    X = complete[gene_list].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use Cox regression with L1 penalty (via penalizer in lifelines)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_penalizer = 0.1
    best_c = 0

    for pen in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        c_indices = []
        for train_idx, test_idx in kf.split(X_scaled):
            train_df = complete.iloc[train_idx][['os_time', 'os_event'] + gene_list].copy()
            test_df = complete.iloc[test_idx][['os_time', 'os_event'] + gene_list].copy()

            # Z-score within training set
            for g in gene_list:
                mu, sd = train_df[g].mean(), train_df[g].std()
                train_df[g] = (train_df[g] - mu) / (sd + 1e-8)
                test_df[g] = (test_df[g] - mu) / (sd + 1e-8)

            try:
                cph = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
                cph.fit(train_df, duration_col='os_time', event_col='os_event')
                c = concordance_index(test_df['os_time'], -cph.predict_partial_hazard(test_df),
                                     test_df['os_event'])
                c_indices.append(c)
            except Exception:
                continue

        mean_c = np.mean(c_indices) if c_indices else 0
        if mean_c > best_c:
            best_c = mean_c
            best_penalizer = pen

    logger.info(f"  Best penalizer: {best_penalizer}, CV C-index: {best_c:.3f}")

    # Fit final model
    final_df = complete[['os_time', 'os_event'] + gene_list].copy()
    for g in gene_list:
        final_df[g] = (final_df[g] - final_df[g].mean()) / (final_df[g].std() + 1e-8)

    cph = CoxPHFitter(penalizer=best_penalizer, l1_ratio=1.0)
    cph.fit(final_df, duration_col='os_time', event_col='os_event')

    # Get non-zero coefficients
    coefs = cph.params_
    non_zero = coefs[coefs.abs() > 0.01]
    logger.info(f"  Signature genes ({len(non_zero)} non-zero):")
    for gene, coef in non_zero.items():
        logger.info(f"    {gene}: coef={coef:.3f}, HR={np.exp(coef):.3f}")

    return cph, non_zero, best_c


def compute_risk_score(df, model, sig_genes):
    """Compute risk score for each patient."""
    gene_list = [g for g in sig_genes.index if g in df.columns]
    if not gene_list:
        return df

    df = df.copy()
    risk = np.zeros(len(df))
    for gene in gene_list:
        vals = df[gene].values
        z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-8)
        risk += sig_genes[gene] * z

    df['risk_score'] = risk
    df['risk_group'] = np.where(risk >= np.median(risk), 'High', 'Low')
    return df


def validate_signature(df, sig_genes, cancer_type):
    """Validate signature: KM curves + log-rank test."""
    if 'risk_score' not in df.columns:
        return None

    high = df[df['risk_group'] == 'High']
    low = df[df['risk_group'] == 'Low']

    if len(high) < 5 or len(low) < 5:
        return None

    lr = logrank_test(high['os_time'], low['os_time'],
                     high['os_event'], low['os_event'])

    # Concordance index
    from lifelines.utils import concordance_index
    c_idx = concordance_index(df['os_time'], -df['risk_score'], df['os_event'])

    result = {
        'cancer_type': cancer_type,
        'n_patients': len(df),
        'n_events': int(df['os_event'].sum()),
        'n_high': len(high),
        'n_low': len(low),
        'logrank_p': float(lr.p_value),
        'c_index': float(c_idx),
    }

    logger.info(f"  {cancer_type}: log-rank p={lr.p_value:.4e}, C-index={c_idx:.3f}")
    return result


def multivariate_analysis(df, sig_genes, cancer_type):
    """
    Multivariate Cox: risk score + age + stage.
    Tests if signature adds independent prognostic value.
    """
    cols = ['os_time', 'os_event', 'risk_score']
    if 'age' in df.columns:
        cols.append('age')
    if 'stage_numeric' in df.columns:
        cols.append('stage_numeric')

    mv_df = df[cols].dropna()
    if len(mv_df) < 20 or mv_df['os_event'].sum() < 5:
        return None

    try:
        cph = CoxPHFitter()
        cph.fit(mv_df, duration_col='os_time', event_col='os_event')
        summary = cph.summary
        logger.info(f"  Multivariate Cox ({cancer_type}):\n{summary.to_string()}")
        return summary
    except Exception as e:
        logger.warning(f"  Multivariate failed: {e}")
        return None


def plot_results(all_screening, all_validation, all_km_data, sig_genes, output_dir):
    """Generate figures."""
    fig = plt.figure(figsize=(20, 16))

    # 1. Univariate screening forest plot
    ax1 = fig.add_subplot(2, 3, 1)
    if all_screening:
        combined = pd.concat(all_screening.values())
        top = combined.nsmallest(15, 'p_value').drop_duplicates('gene')
        for i, (_, row) in enumerate(top.iterrows()):
            hr = row['hr']
            color = 'red' if hr > 1 else 'blue'
            ax1.plot(np.log2(hr), i, 'o', color=color, markersize=8)
            ax1.plot([np.log2(hr) - 0.3, np.log2(hr) + 0.3], [i, i],
                    color=color, linewidth=2)
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top['gene'], fontsize=8)
        ax1.axvline(0, color='gray', linestyle='--')
        ax1.set_xlabel('Log2(HR)')
        ax1.set_title('Top Prognostic Genes\n(Univariate Cox)')
        ax1.invert_yaxis()

    # 2. Signature coefficients
    ax2 = fig.add_subplot(2, 3, 2)
    if sig_genes is not None and len(sig_genes) > 0:
        genes = sig_genes.sort_values()
        colors = ['red' if v > 0 else 'blue' for v in genes.values]
        ax2.barh(range(len(genes)), genes.values, color=colors)
        ax2.set_yticks(range(len(genes)))
        ax2.set_yticklabels(genes.index, fontsize=8)
        ax2.set_xlabel('LASSO Cox Coefficient')
        ax2.set_title('Prognostic Signature Genes')
        ax2.axvline(0, color='gray', linestyle='--')
        ax2.invert_yaxis()

    # 3-5. KM curves for each cancer
    for i, (cancer, km_data) in enumerate(all_km_data.items()):
        if i >= 3:
            break
        ax = fig.add_subplot(2, 3, i + 3 + 1)
        if km_data is not None and 'risk_group' in km_data.columns:
            kmf = KaplanMeierFitter()
            for group in ['Low', 'High']:
                mask = km_data['risk_group'] == group
                if mask.sum() > 0:
                    color = 'blue' if group == 'Low' else 'red'
                    n = mask.sum()
                    events = km_data.loc[mask, 'os_event'].sum()
                    kmf.fit(km_data.loc[mask, 'os_time'] / 365.25,
                           km_data.loc[mask, 'os_event'],
                           label=f'{group} Risk (n={n}, e={events})')
                    kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

            # Add p-value
            val = all_validation.get(cancer)
            if val:
                ax.text(0.05, 0.05,
                       f"p = {val['logrank_p']:.2e}\nC = {val['c_index']:.3f}",
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Overall Survival')
            ax.set_title(f'{cancer} Risk Stratification')
            ax.set_ylim(0, 1.05)

    # 6. Validation summary bar chart
    ax6 = fig.add_subplot(2, 3, 6)
    if all_validation:
        cancers = list(all_validation.keys())
        c_indices = [all_validation[c]['c_index'] for c in cancers]
        p_values = [-np.log10(all_validation[c]['logrank_p'] + 1e-20) for c in cancers]
        x = np.arange(len(cancers))
        width = 0.35
        ax6.bar(x - width/2, c_indices, width, label='C-index', color='steelblue')
        ax6_twin = ax6.twinx()
        ax6_twin.bar(x + width/2, p_values, width, label='-log10(p)', color='coral')
        ax6.set_xticks(x)
        ax6.set_xticklabels(cancers)
        ax6.set_ylabel('C-index', color='steelblue')
        ax6_twin.set_ylabel('-Log10(P-value)', color='coral')
        ax6.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax6_twin.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        ax6.set_title('Signature Performance Summary')
        ax6.legend(loc='upper left', fontsize=8)
        ax6_twin.legend(loc='upper right', fontsize=8)

    plt.suptitle('Experiment 22: SL-Based Prognostic Signature',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'prognostic_signature.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved prognostic_signature.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 22: SL-Based Prognostic Signature")
    logger.info("=" * 60)

    all_screening = {}
    all_validation = {}
    all_km_data = {}
    all_multivariate = {}
    final_sig_genes = None

    for cancer in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer}")
        logger.info(f"{'='*40}")

        expr, clin = load_data(cancer)
        if expr.empty or clin.empty:
            continue

        # Build survival matrix
        df = build_survival_matrix(expr, clin, cancer)
        if len(df) < 20:
            continue

        # Univariate screening
        logger.info("Step 1: Univariate screening...")
        available_genes = [g for g in CANDIDATE_GENES if g in df.columns]
        full_screen, sig_screen = univariate_screening(df, available_genes)
        all_screening[cancer] = full_screen

        # Build signature (train on first cancer type)
        if final_sig_genes is None:
            logger.info("Step 2: Building LASSO signature...")
            sig_gene_list = sig_screen['gene'].tolist() if len(sig_screen) >= 3 else available_genes
            model, sig_genes, cv_c = build_lasso_signature(df, sig_gene_list)
            if sig_genes is not None:
                final_sig_genes = sig_genes
                logger.info(f"  Signature: {len(sig_genes)} genes, CV C-index: {cv_c:.3f}")

        # Apply signature
        if final_sig_genes is not None:
            df = compute_risk_score(df, None, final_sig_genes)
            all_km_data[cancer] = df

            # Validate
            logger.info(f"Step 3: Validating on {cancer}...")
            val = validate_signature(df, final_sig_genes, cancer)
            if val:
                all_validation[cancer] = val

            # Multivariate
            logger.info(f"Step 4: Multivariate analysis...")
            mv = multivariate_analysis(df, final_sig_genes, cancer)
            if mv is not None:
                all_multivariate[cancer] = mv

    # Save results
    logger.info("\n--- Saving Results ---")
    for cancer, screen in all_screening.items():
        screen.to_csv(RESULTS_DIR / f'{cancer}_univariate_screening.csv', index=False)
    if final_sig_genes is not None:
        sig_df = pd.DataFrame({'gene': final_sig_genes.index, 'coefficient': final_sig_genes.values})
        sig_df.to_csv(RESULTS_DIR / 'signature_genes.csv', index=False)
    for cancer, mv in all_multivariate.items():
        mv.to_csv(RESULTS_DIR / f'{cancer}_multivariate.csv')

    # Plot
    plot_results(all_screening, all_validation, all_km_data, final_sig_genes, RESULTS_DIR)

    # Summary
    summary = {
        'experiment': 'Exp 22: SL-Based Prognostic Signature',
        'candidate_genes': len(CANDIDATE_GENES),
        'signature_genes': final_sig_genes.to_dict() if final_sig_genes is not None else {},
        'validation': all_validation,
    }
    with open(RESULTS_DIR / 'exp22_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nExp 22 COMPLETE")
    if final_sig_genes is not None:
        logger.info(f"Signature: {len(final_sig_genes)} genes")
    for cancer, val in all_validation.items():
        logger.info(f"  {cancer}: C-index={val['c_index']:.3f}, p={val['logrank_p']:.4e}")


if __name__ == '__main__':
    main()
