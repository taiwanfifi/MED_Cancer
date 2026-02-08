#!/usr/bin/env python3
"""
Experiment 1: Pan-Cancer Survival Analysis & Biomarker Discovery
=================================================================
Quick-win analysis using TCGA clinical data already downloaded.
Identifies prognostic factors across 15 cancer types.

This produces publishable figures and tables immediately.
Connects to Paper 5 (Multi-Omics Subtyping) and Paper 9 (Biomarkers).
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
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "tcga"
RESULTS_DIR = BASE_DIR / "results" / "exp1_survival"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_clinical_data():
    """Load pan-cancer clinical data."""
    parquet_file = DATA_DIR / "pan_cancer_clinical.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    else:
        # Load from individual JSON files
        all_records = []
        for f in DATA_DIR.glob("*_clinical.json"):
            with open(f) as fh:
                records = json.load(fh)
                all_records.extend(records)
        df = pd.DataFrame(all_records)

    # Process survival data
    df['event'] = (df['vital_status'] == 'Dead').astype(int)
    df['time'] = df.apply(
        lambda r: r['days_to_death'] if pd.notna(r['days_to_death']) and r['days_to_death'] > 0
        else r.get('days_to_follow_up', 0) if pd.notna(r.get('days_to_follow_up', None)) else 0,
        axis=1
    )
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df[df['time'] > 0]  # Remove invalid entries

    # Process age
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 65, 75, 100],
                             labels=['<40', '40-55', '55-65', '65-75', '>75'])

    # Process stage
    df['stage_simple'] = df['stage'].apply(lambda x: simplify_stage(x) if pd.notna(x) else 'Unknown')

    logger.info(f"Clinical data: {len(df)} patients, {df['cancer_type'].nunique()} cancer types")
    return df


def simplify_stage(stage_str):
    """Simplify AJCC stage to I/II/III/IV."""
    stage_str = str(stage_str).upper().strip()
    if 'IV' in stage_str:
        return 'IV'
    elif 'III' in stage_str:
        return 'III'
    elif 'II' in stage_str:
        return 'II'
    elif 'I' in stage_str and 'II' not in stage_str and 'IV' not in stage_str:
        return 'I'
    return 'Unknown'


def plot_pan_cancer_km(df):
    """Kaplan-Meier curves for all cancer types."""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.flatten()

    cancer_types = sorted(df['cancer_type'].unique())

    for idx, ct in enumerate(cancer_types):
        if idx >= 15:
            break
        ax = axes[idx]
        ct_df = df[df['cancer_type'] == ct]

        kmf = KaplanMeierFitter()
        kmf.fit(ct_df['time'] / 365.25, ct_df['event'], label=ct)
        kmf.plot_survival_function(ax=ax, ci_show=True)

        ax.set_title(f'{ct} (n={len(ct_df)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Years')
        ax.set_ylabel('Survival Probability')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pan_cancer_km_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: pan_cancer_km_curves.png")


def plot_stage_km(df):
    """KM curves stratified by stage for each cancer type."""
    # Select cancers with good stage data
    stage_counts = df.groupby('cancer_type')['stage_simple'].apply(
        lambda x: (x != 'Unknown').sum()
    ).sort_values(ascending=False)

    top_cancers = stage_counts[stage_counts > 100].index[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, ct in enumerate(top_cancers):
        ax = axes[idx]
        ct_df = df[(df['cancer_type'] == ct) & (df['stage_simple'] != 'Unknown')]

        colors = {'I': '#2ecc71', 'II': '#f39c12', 'III': '#e74c3c', 'IV': '#8e44ad'}

        for stage in ['I', 'II', 'III', 'IV']:
            stage_df = ct_df[ct_df['stage_simple'] == stage]
            if len(stage_df) >= 10:
                kmf = KaplanMeierFitter()
                kmf.fit(stage_df['time'] / 365.25, stage_df['event'],
                       label=f'Stage {stage} (n={len(stage_df)})')
                kmf.plot_survival_function(ax=ax, ci_show=False,
                                          color=colors.get(stage, 'gray'))

        ax.set_title(f'{ct}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Years')
        ax.set_ylabel('Survival Probability')
        ax.set_xlim(0, 10)
        ax.legend(loc='lower left', fontsize=9)

    plt.suptitle('Stage-Stratified Survival Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stage_stratified_km.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: stage_stratified_km.png")


def plot_age_km(df):
    """KM curves stratified by age group."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Select high-mortality cancers
    high_mort = ['OV', 'LUSC', 'BLCA', 'HNSC', 'STAD', 'LIHC']

    colors = {'<40': '#3498db', '40-55': '#2ecc71', '55-65': '#f39c12',
              '65-75': '#e74c3c', '>75': '#8e44ad'}

    for idx, ct in enumerate(high_mort):
        ax = axes[idx]
        ct_df = df[(df['cancer_type'] == ct) & (df['age_group'].notna())]

        for age_grp in ['<40', '40-55', '55-65', '65-75', '>75']:
            grp_df = ct_df[ct_df['age_group'] == age_grp]
            if len(grp_df) >= 10:
                kmf = KaplanMeierFitter()
                kmf.fit(grp_df['time'] / 365.25, grp_df['event'],
                       label=f'{age_grp} (n={len(grp_df)})')
                kmf.plot_survival_function(ax=ax, ci_show=False,
                                          color=colors.get(age_grp, 'gray'))

        ax.set_title(f'{ct}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Years')
        ax.set_ylabel('Survival Probability')
        ax.set_xlim(0, 10)
        ax.legend(loc='lower left', fontsize=9)

    plt.suptitle('Age-Stratified Survival Analysis (High-Mortality Cancers)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "age_stratified_km.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: age_stratified_km.png")


def plot_gender_km(df):
    """Gender-stratified survival for cancers affecting both sexes."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    both_gender = ['LUAD', 'LUSC', 'KIRC', 'STAD', 'BLCA', 'HNSC', 'LIHC', 'COAD']

    for idx, ct in enumerate(both_gender):
        ax = axes[idx]
        ct_df = df[(df['cancer_type'] == ct) & (df['gender'].isin(['male', 'female']))]

        for gender, color in [('male', '#3498db'), ('female', '#e74c3c')]:
            g_df = ct_df[ct_df['gender'] == gender]
            if len(g_df) >= 10:
                kmf = KaplanMeierFitter()
                kmf.fit(g_df['time'] / 365.25, g_df['event'],
                       label=f'{gender.capitalize()} (n={len(g_df)})')
                kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

        # Log-rank test
        male_df = ct_df[ct_df['gender'] == 'male']
        female_df = ct_df[ct_df['gender'] == 'female']
        if len(male_df) >= 10 and len(female_df) >= 10:
            result = logrank_test(male_df['time'], female_df['time'],
                                 male_df['event'], female_df['event'])
            ax.set_title(f'{ct} (p={result.p_value:.4f})', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{ct}', fontsize=12, fontweight='bold')

        ax.set_xlabel('Years')
        ax.set_ylabel('Survival Probability')
        ax.set_xlim(0, 10)
        ax.legend(loc='lower left', fontsize=9)

    plt.suptitle('Gender-Stratified Survival Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gender_stratified_km.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: gender_stratified_km.png")


def cox_multivariate_analysis(df):
    """Multivariate Cox regression for each cancer type."""
    results = []

    for ct in sorted(df['cancer_type'].unique()):
        ct_df = df[df['cancer_type'] == ct].copy()

        # Prepare covariates
        ct_df['male'] = (ct_df['gender'] == 'male').astype(int)
        ct_df['age_scaled'] = (ct_df['age'] - ct_df['age'].mean()) / ct_df['age'].std()

        stage_dummies = pd.get_dummies(ct_df['stage_simple'], prefix='stage')
        if 'stage_Unknown' in stage_dummies.columns:
            stage_dummies = stage_dummies.drop('stage_Unknown', axis=1)
        if 'stage_I' in stage_dummies.columns:
            stage_dummies = stage_dummies.drop('stage_I', axis=1)  # reference

        cox_df = pd.concat([ct_df[['time', 'event', 'male', 'age_scaled']], stage_dummies], axis=1)
        cox_df = cox_df.dropna()

        if len(cox_df) < 50:
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='time', event_col='event')

            for var in cph.summary.index:
                results.append({
                    'cancer_type': ct,
                    'variable': var,
                    'coef': cph.summary.loc[var, 'coef'],
                    'hr': np.exp(cph.summary.loc[var, 'coef']),
                    'hr_lower': np.exp(cph.summary.loc[var, 'coef lower 95%']),
                    'hr_upper': np.exp(cph.summary.loc[var, 'coef upper 95%']),
                    'p_value': cph.summary.loc[var, 'p'],
                    'significant': cph.summary.loc[var, 'p'] < 0.05,
                })
        except Exception as e:
            logger.warning(f"Cox regression failed for {ct}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "cox_multivariate_results.csv", index=False)

    # Summary heatmap
    if len(results_df) > 0:
        # Pivot for heatmap
        sig_results = results_df[results_df['significant']].copy()
        sig_results['log_hr'] = np.log2(sig_results['hr'])

        # Plot significant hazard ratios
        fig, ax = plt.subplots(figsize=(14, 8))

        pivot = results_df.pivot_table(
            values='hr', index='cancer_type', columns='variable'
        ).fillna(1)

        # Log2 transform for visualization
        pivot_log = np.log2(pivot)

        sns.heatmap(pivot_log, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                   ax=ax, linewidths=0.5, vmin=-2, vmax=2)
        ax.set_title('Hazard Ratios (log2) from Multivariate Cox Regression',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Cancer Type')
        ax.set_xlabel('Covariate')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "cox_hazard_ratio_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved: cox_hazard_ratio_heatmap.png")

    return results_df


def mortality_landscape(df):
    """Create a comprehensive mortality landscape figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: 5-year survival rate by cancer type
    ax = axes[0, 0]
    survival_5yr = {}
    for ct in sorted(df['cancer_type'].unique()):
        ct_df = df[df['cancer_type'] == ct]
        kmf = KaplanMeierFitter()
        kmf.fit(ct_df['time'] / 365.25, ct_df['event'])
        try:
            s5 = kmf.predict(5.0)
            survival_5yr[ct] = float(s5)
        except:
            survival_5yr[ct] = float(kmf.survival_function_.iloc[-1, 0])

    s5_df = pd.Series(survival_5yr).sort_values()
    colors = ['#e74c3c' if v < 0.5 else '#f39c12' if v < 0.7 else '#2ecc71' for v in s5_df.values]
    s5_df.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('5-Year Survival Rate')
    ax.set_title('A) 5-Year Survival Rate by Cancer Type', fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

    # Panel B: Median survival time
    ax = axes[0, 1]
    median_surv = {}
    for ct in sorted(df['cancer_type'].unique()):
        ct_df = df[df['cancer_type'] == ct]
        kmf = KaplanMeierFitter()
        kmf.fit(ct_df['time'] / 365.25, ct_df['event'])
        med = kmf.median_survival_time_
        median_surv[ct] = float(med) if not np.isinf(med) else 15.0

    ms_df = pd.Series(median_surv).sort_values()
    colors = ['#e74c3c' if v < 3 else '#f39c12' if v < 5 else '#2ecc71' for v in ms_df.values]
    ms_df.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Median Survival (Years)')
    ax.set_title('B) Median Survival Time', fontweight='bold')

    # Panel C: Age distribution
    ax = axes[1, 0]
    order = df.groupby('cancer_type')['age'].median().sort_values().index
    sns.boxplot(data=df, y='cancer_type', x='age', order=order, ax=ax,
               palette='coolwarm', flierprops={'markersize': 2})
    ax.set_xlabel('Age at Diagnosis')
    ax.set_title('C) Age Distribution by Cancer Type', fontweight='bold')

    # Panel D: Gender distribution
    ax = axes[1, 1]
    gender_counts = df.groupby(['cancer_type', 'gender']).size().unstack(fill_value=0)
    if 'male' in gender_counts.columns and 'female' in gender_counts.columns:
        gender_pct = gender_counts.div(gender_counts.sum(axis=1), axis=0)
        gender_pct[['male', 'female']].plot(kind='barh', stacked=True, ax=ax,
                                              color=['#3498db', '#e74c3c'])
        ax.set_xlabel('Proportion')
        ax.set_title('D) Gender Distribution', fontweight='bold')
        ax.legend(['Male', 'Female'])

    plt.suptitle('Pan-Cancer Clinical Landscape (TCGA, n=7,943)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pan_cancer_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: pan_cancer_landscape.png")

    return survival_5yr, median_surv


def main():
    logger.info("=" * 60)
    logger.info("Experiment 1: Pan-Cancer Survival Analysis")
    logger.info("=" * 60)

    # Load data
    df = load_clinical_data()
    logger.info(f"Loaded {len(df)} patients")
    logger.info(f"Cancer types: {sorted(df['cancer_type'].unique())}")

    # 1. Pan-cancer KM curves
    logger.info("\n--- Kaplan-Meier Analysis ---")
    plot_pan_cancer_km(df)

    # 2. Stage-stratified KM
    plot_stage_km(df)

    # 3. Age-stratified KM
    plot_age_km(df)

    # 4. Gender-stratified KM
    plot_gender_km(df)

    # 5. Cox regression
    logger.info("\n--- Cox Regression ---")
    cox_results = cox_multivariate_analysis(df)
    if cox_results is not None:
        sig = cox_results[cox_results['significant']]
        logger.info(f"Significant associations: {len(sig)}")
        logger.info(f"Top hazard ratios:\n{sig.nlargest(10, 'hr')[['cancer_type', 'variable', 'hr', 'p_value']].to_string()}")

    # 6. Mortality landscape
    s5yr, med_surv = mortality_landscape(df)

    # Save summary
    summary = {
        "n_patients": len(df),
        "n_cancer_types": df['cancer_type'].nunique(),
        "5yr_survival": s5yr,
        "median_survival_years": med_surv,
        "worst_prognosis": [ct for ct, s in sorted(s5yr.items(), key=lambda x: x[1])[:3]],
        "best_prognosis": [ct for ct, s in sorted(s5yr.items(), key=lambda x: x[1], reverse=True)[:3]],
    }

    with open(RESULTS_DIR / "exp1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 1 COMPLETE")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
