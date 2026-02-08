#!/usr/bin/env python3
"""
Experiment 8: Deep Learning Survival Prediction
=================================================
Multi-modal survival prediction combining:
- Gene expression features (top variable genes)
- Immune microenvironment scores
- Clinical features (age, stage, gender)

Tests whether multi-modal ML outperforms single-feature survival models.
Compares: Cox-PH, Random Survival Forest, Neural Network (DeepSurv).

Target Paper: Paper 7 (AI-Enhanced Prognosis) — Lancet Digital Health
Hypothesis: Multi-modal deep learning > single-omics for survival prediction
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
IMM_DIR = BASE_DIR / "results" / "exp6_immune_tme"
RESULTS_DIR = BASE_DIR / "results" / "exp8_dl_survival"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def load_clinical():
    """Load pan-cancer clinical data."""
    if not CLINICAL_FILE.exists():
        logger.error("Clinical data not found")
        return None
    df = pd.read_parquet(CLINICAL_FILE)
    logger.info(f"Clinical data: {len(df)} patients")
    return df


def prepare_features(cancer_type, clinical_df):
    """Prepare multi-modal feature matrix for one cancer type."""
    # 1. Expression features (top 500 most variable genes)
    expr_file = EXPR_DIR / f"{cancer_type}_expression_matrix.parquet"
    if not expr_file.exists():
        logger.warning(f"[{cancer_type}] Expression data not found")
        return None

    expr_df = pd.read_parquet(expr_file)
    expr_log = np.log2(expr_df + 1)

    # Get tumor samples only
    tumor_samples = []
    for s in expr_df.columns:
        parts = s.split('-')
        if len(parts) >= 4:
            st = parts[3][:2]
            if st.isdigit() and 1 <= int(st) <= 9:
                tumor_samples.append(s)
        else:
            tumor_samples.append(s)  # Default to tumor

    expr_tumor = expr_log[tumor_samples]

    # Top variable genes
    gene_var = expr_tumor.var(axis=1)
    top_genes = gene_var.nlargest(500).index
    expr_features = expr_tumor.loc[top_genes].T  # samples x genes
    expr_features.columns = [f"EXPR_{g}" for g in expr_features.columns]

    # 2. Immune scores
    immune_file = IMM_DIR / f"{cancer_type}_immune_scores.csv"
    immune_df = None
    if immune_file.exists():
        immune_df = pd.read_csv(immune_file, index_col=0)
        immune_df.columns = [f"IMM_{c}" for c in immune_df.columns]

    # 3. Clinical features
    ct_clinical = clinical_df[clinical_df['cancer_type'] == cancer_type].copy()
    ct_clinical['patient_id'] = ct_clinical['case_id']

    # Match by patient ID (first 3 parts of TCGA barcode)
    feature_rows = []
    for sample_id in expr_features.index:
        patient_id = '-'.join(str(sample_id).split('-')[:3])

        clin_match = ct_clinical[ct_clinical['patient_id'] == patient_id]
        if len(clin_match) == 0:
            continue

        clin_row = clin_match.iloc[0]

        # Survival data
        time = clin_row.get('days_to_death')
        if pd.isna(time):
            time = clin_row.get('days_to_follow_up')
        if pd.isna(time) or time is None or float(time) <= 0:
            continue

        event = 1 if str(clin_row.get('vital_status', '')).lower() in ['dead', 'deceased'] else 0

        row = {
            'sample_id': sample_id,
            'patient_id': patient_id,
            'time': float(time),
            'event': event,
            'CLIN_age': float(clin_row.get('age', np.nan)),
            'CLIN_gender': 1 if str(clin_row.get('gender', '')).lower() == 'male' else 0,
        }

        # Stage encoding
        stage = str(clin_row.get('stage', ''))
        if 'IV' in stage:
            row['CLIN_stage'] = 4
        elif 'III' in stage:
            row['CLIN_stage'] = 3
        elif 'II' in stage:
            row['CLIN_stage'] = 2
        elif 'I' in stage:
            row['CLIN_stage'] = 1
        else:
            row['CLIN_stage'] = np.nan

        # Add expression features
        for col in expr_features.columns[:100]:  # Use top 100 for speed
            row[col] = expr_features.loc[sample_id, col]

        # Add immune features
        if immune_df is not None and sample_id in immune_df.index:
            for col in immune_df.columns:
                row[col] = immune_df.loc[sample_id, col]

        feature_rows.append(row)

    if len(feature_rows) < 30:
        logger.warning(f"[{cancer_type}] Only {len(feature_rows)} matched samples")
        return None

    feature_df = pd.DataFrame(feature_rows)
    logger.info(f"[{cancer_type}] Feature matrix: {feature_df.shape}")
    logger.info(f"  Events: {feature_df['event'].sum()}/{len(feature_df)}")
    logger.info(f"  Median time: {feature_df['time'].median():.0f} days")

    return feature_df


def train_cox_ph(feature_df, feature_prefix, cancer_type, model_name):
    """Train Cox PH model on a subset of features."""
    from lifelines import CoxPHFitter

    # Select features by prefix
    feature_cols = [c for c in feature_df.columns if c.startswith(feature_prefix)]
    if not feature_cols:
        return None

    surv_df = feature_df[['time', 'event'] + feature_cols].dropna()

    if len(surv_df) < 30:
        return None

    # Train/test split
    np.random.seed(42)
    n = len(surv_df)
    train_idx = np.random.choice(n, int(0.7 * n), replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    train_df = surv_df.iloc[train_idx]
    test_df = surv_df.iloc[test_idx]

    try:
        cph = CoxPHFitter(penalizer=0.1)  # L2 regularization
        cph.fit(train_df, duration_col='time', event_col='event')

        # Concordance index on test set
        c_index = cph.score(test_df, scoring_method='concordance_index')

        logger.info(f"  {model_name}: C-index = {c_index:.4f} "
                    f"(train={len(train_df)}, test={len(test_df)}, features={len(feature_cols)})")

        return {
            'model': model_name,
            'cancer_type': cancer_type,
            'c_index': c_index,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_features': len(feature_cols),
            'method': 'Cox-PH',
        }
    except Exception as e:
        logger.warning(f"  {model_name}: failed - {e}")
        return None


def train_random_survival_forest(feature_df, feature_cols, cancer_type, model_name):
    """Train Random Survival Forest."""
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        logger.warning("scikit-survival not available, skipping RSF")
        return None

    if not feature_cols:
        return None

    valid = feature_df[['time', 'event'] + feature_cols].dropna()
    if len(valid) < 30:
        return None

    X = valid[feature_cols].values
    y = np.array([(bool(e), t) for e, t in zip(valid['event'], valid['time'])],
                 dtype=[('event', bool), ('time', float)])

    # Train/test split
    np.random.seed(42)
    n = len(valid)
    train_idx = np.random.choice(n, int(0.7 * n), replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    try:
        rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                   random_state=42, n_jobs=-1)
        rsf.fit(X[train_idx], y[train_idx])

        pred = rsf.predict(X[test_idx])
        c_index = concordance_index_censored(y[test_idx]['event'], y[test_idx]['time'], pred)[0]

        logger.info(f"  {model_name} (RSF): C-index = {c_index:.4f}")

        return {
            'model': model_name,
            'cancer_type': cancer_type,
            'c_index': c_index,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_features': len(feature_cols),
            'method': 'RSF',
        }
    except Exception as e:
        logger.warning(f"  {model_name} (RSF): failed - {e}")
        return None


def train_neural_survival(feature_df, feature_cols, cancer_type, model_name):
    """Train a simple DeepSurv-like neural network for survival."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.warning("PyTorch not available, skipping neural survival")
        return None

    valid = feature_df[['time', 'event'] + feature_cols].dropna()
    if len(valid) < 30:
        return None

    X = torch.tensor(valid[feature_cols].values, dtype=torch.float32)
    time = torch.tensor(valid['time'].values, dtype=torch.float32)
    event = torch.tensor(valid['event'].values, dtype=torch.float32)

    # Standardize
    X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)

    np.random.seed(42)
    n = len(valid)
    train_idx = np.random.choice(n, int(0.7 * n), replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    class DeepSurv(nn.Module):
        def __init__(self, in_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def cox_ph_loss(risk_pred, time, event):
        """Negative partial log-likelihood for Cox PH."""
        sorted_idx = torch.argsort(time, descending=True)
        risk_pred = risk_pred[sorted_idx]
        event = event[sorted_idx]

        hazard_ratio = torch.exp(risk_pred)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred - log_risk
        censored_likelihood = uncensored_likelihood * event

        return -torch.mean(censored_likelihood)

    model = DeepSurv(len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X[train_idx])
        loss = cox_ph_loss(pred, time[train_idx], event[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X[test_idx]).numpy()

    # Concordance index
    from lifelines.utils import concordance_index
    try:
        c_index = concordance_index(
            valid.iloc[test_idx]['time'].values,
            -test_pred,  # Negative because higher risk = lower survival
            valid.iloc[test_idx]['event'].values
        )

        logger.info(f"  {model_name} (DeepSurv): C-index = {c_index:.4f}")

        return {
            'model': model_name,
            'cancer_type': cancer_type,
            'c_index': c_index,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_features': len(feature_cols),
            'method': 'DeepSurv',
        }
    except Exception as e:
        logger.warning(f"  {model_name} (DeepSurv): eval failed - {e}")
        return None


def plot_model_comparison(all_results):
    """Compare survival prediction models across cancer types and modalities."""
    if not all_results:
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Grouped bar chart by cancer type and model
    ax = axes[0]
    pivot = results_df.pivot_table(index='model', columns='cancer_type', values='c_index', aggfunc='max')
    pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='white', linewidth=1)
    ax.set_ylabel('Concordance Index (C-index)')
    ax.set_title('A) Model Comparison by Cancer Type', fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.set_ylim(0.4, 0.85)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=45)

    # Panel B: Modality comparison (aggregate across cancer types)
    ax = axes[1]
    modality_scores = results_df.groupby('model')['c_index'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(modality_scores)))
    bars = ax.barh(range(len(modality_scores)), modality_scores['mean'], xerr=modality_scores['std'],
                   color=colors, edgecolor='white', capsize=5)
    ax.set_yticks(range(len(modality_scores)))
    ax.set_yticklabels(modality_scores.index, fontsize=9)
    ax.set_xlabel('Mean C-index (± std)')
    ax.set_title('B) Average Performance by Model', fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: model_comparison.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 8: Deep Learning Survival Prediction")
    logger.info("=" * 60)

    clinical_df = load_clinical()
    if clinical_df is None:
        return

    all_results = []

    for cancer_type in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer_type}")
        logger.info(f"{'='*40}")

        feature_df = prepare_features(cancer_type, clinical_df)
        if feature_df is None:
            continue

        # Define feature groups
        expr_cols = [c for c in feature_df.columns if c.startswith('EXPR_')]
        imm_cols = [c for c in feature_df.columns if c.startswith('IMM_')]
        clin_cols = [c for c in feature_df.columns if c.startswith('CLIN_')]
        all_cols = expr_cols + imm_cols + clin_cols

        logger.info(f"  Features: {len(expr_cols)} expression, {len(imm_cols)} immune, {len(clin_cols)} clinical")

        # Model 1: Clinical only (Cox-PH)
        r = train_cox_ph(feature_df, 'CLIN_', cancer_type, 'Clinical-only')
        if r:
            all_results.append(r)

        # Model 2: Expression only (Cox-PH)
        # Use top 20 PCA components instead of raw genes
        r = train_cox_ph(feature_df, 'EXPR_', cancer_type, 'Expression-only')
        if r:
            all_results.append(r)

        # Model 3: Immune only (Cox-PH)
        r = train_cox_ph(feature_df, 'IMM_', cancer_type, 'Immune-only')
        if r:
            all_results.append(r)

        # Model 4: Multi-modal Cox-PH (immune + clinical)
        multi_basic = imm_cols + clin_cols
        r = train_cox_ph(feature_df, ('IMM_', 'CLIN_'), cancer_type, 'Immune+Clinical')
        # Custom: select both prefixes
        surv_cols = imm_cols + clin_cols
        if surv_cols:
            from lifelines import CoxPHFitter
            surv_df = feature_df[['time', 'event'] + surv_cols].dropna()
            if len(surv_df) >= 30:
                np.random.seed(42)
                n = len(surv_df)
                train_idx = np.random.choice(n, int(0.7 * n), replace=False)
                test_idx = np.setdiff1d(np.arange(n), train_idx)
                try:
                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(surv_df.iloc[train_idx], duration_col='time', event_col='event')
                    c_index = cph.score(surv_df.iloc[test_idx], scoring_method='concordance_index')
                    all_results.append({
                        'model': 'Immune+Clinical', 'cancer_type': cancer_type,
                        'c_index': c_index, 'n_train': len(train_idx),
                        'n_test': len(test_idx), 'n_features': len(surv_cols),
                        'method': 'Cox-PH',
                    })
                    logger.info(f"  Immune+Clinical: C-index = {c_index:.4f}")
                except Exception as e:
                    logger.warning(f"  Immune+Clinical: failed - {e}")

        # Model 5: DeepSurv multi-modal
        r = train_neural_survival(feature_df, all_cols, cancer_type, 'Multi-modal-DeepSurv')
        if r:
            all_results.append(r)

        # Model 6: DeepSurv immune+clinical
        r = train_neural_survival(feature_df, imm_cols + clin_cols, cancer_type, 'Immune+Clin-DeepSurv')
        if r:
            all_results.append(r)

        # Model 7: RSF multi-modal (if available)
        r = train_random_survival_forest(feature_df, all_cols[:50], cancer_type, 'Multi-modal-RSF')
        if r:
            all_results.append(r)

    # Compare all models
    if all_results:
        plot_model_comparison(all_results)

        results_df = pd.DataFrame(all_results)
        logger.info(f"\n{'='*60}")
        logger.info("RESULTS SUMMARY")
        logger.info(f"{'='*60}")

        # Best model per cancer type
        for ct in CANCER_TYPES:
            ct_results = results_df[results_df['cancer_type'] == ct]
            if len(ct_results) > 0:
                best = ct_results.loc[ct_results['c_index'].idxmax()]
                logger.info(f"  {ct}: Best = {best['model']} (C-index={best['c_index']:.4f})")

        # Overall best
        best_overall = results_df.groupby('model')['c_index'].mean().sort_values(ascending=False)
        logger.info(f"\nOverall ranking:")
        for model, score in best_overall.items():
            logger.info(f"  {model}: mean C-index = {score:.4f}")

    # Summary
    summary = {
        'experiment': 'Exp 8: Deep Learning Survival Prediction',
        'hypothesis': 'Multi-modal > single-omics for survival prediction',
        'cancer_types': CANCER_TYPES,
        'models_tested': len(set(r['model'] for r in all_results)) if all_results else 0,
        'results': all_results,
    }

    with open(RESULTS_DIR / "exp8_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 8 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
