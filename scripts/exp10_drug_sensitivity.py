#!/usr/bin/env python3
"""
Experiment 10: Drug Sensitivity Prediction from Gene Expression
================================================================
Downloads GDSC (Genomics of Drug Sensitivity in Cancer) data and trains
ML models to predict drug sensitivity from gene expression.

Pipeline:
1. Download GDSC IC50 data + cell line expression
2. Map to DepMap cell lines (already have gene effect data)
3. Train: expression → IC50 prediction for key cancer drugs
4. Feature importance → which genes predict drug response?
5. Cross-reference with SL targets → synthetic lethal-informed drug response

Target Paper: Paper 1 (Drug Repurposing & Sensitivity) — Nature Communications
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
DATA_DIR = BASE_DIR / "data" / "drug_repurpose"
EXPR_DIR = BASE_DIR / "data" / "tcga_expression"
SL_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
RESULTS_DIR = BASE_DIR / "results" / "exp10_drug_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def download_gdsc_data():
    """Download GDSC drug sensitivity data."""
    import requests
    import io

    gdsc_file = DATA_DIR / "gdsc_ic50.parquet"
    if gdsc_file.exists():
        logger.info("GDSC data already exists, loading...")
        return pd.read_parquet(gdsc_file)

    logger.info("Downloading GDSC drug sensitivity data...")

    # Try multiple GDSC URLs
    urls = [
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_fitted_dose_response_27Oct23.xlsx",
        "https://www.cancerrxgene.org/gdsc1000/GDSC2_fitted_dose_response_24Jul22.xlsx",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                if url.endswith('.xlsx'):
                    df = pd.read_excel(io.BytesIO(r.content))
                else:
                    df = pd.read_csv(io.StringIO(r.text))
                logger.info(f"GDSC downloaded: {df.shape} from {url}")
                df.to_parquet(gdsc_file)
                return df
        except Exception as e:
            logger.warning(f"GDSC download failed ({url}): {e}")

    logger.warning("All GDSC download attempts failed. Proceeding with DepMap-only analysis.")
    return None


def download_gdsc_expression():
    """Download GDSC cell line expression data."""
    import requests
    import io

    expr_file = DATA_DIR / "gdsc_expression.parquet"
    if expr_file.exists():
        logger.info("GDSC expression already exists, loading...")
        return pd.read_parquet(expr_file)

    # Use DepMap expression as proxy (already mapped cell lines)
    logger.info("Using DepMap gene effect as proxy for cell line genomic features...")
    depmap_file = DATA_DIR / "depmap_gene_effect.parquet"
    if depmap_file.exists():
        return pd.read_parquet(depmap_file)

    return None


def prepare_drug_response_dataset(gdsc_df, gene_effect_df):
    """
    Prepare ML-ready dataset: cell line features → drug IC50.
    Uses DepMap gene effect as cell line genomic features.
    """
    if gdsc_df is None or gene_effect_df is None:
        return None

    # GDSC uses COSMIC_ID or cell line names
    # DepMap uses ACH-XXXXXX IDs
    # We need cell line metadata to map between them

    cell_info_file = DATA_DIR / "depmap_cell_line_info.parquet"
    if cell_info_file.exists():
        cell_info = pd.read_parquet(cell_info_file)
        # Create mapping: cell line name → DepMap_ID
        name_to_depmap = {}
        if 'stripped_cell_line_name' in cell_info.columns:
            for _, row in cell_info.iterrows():
                name_to_depmap[row['stripped_cell_line_name']] = row['DepMap_ID']
        elif 'cell_line_name' in cell_info.columns:
            for _, row in cell_info.iterrows():
                name_to_depmap[str(row['cell_line_name']).upper()] = row['DepMap_ID']
    else:
        logger.warning("No cell line info for mapping")
        return None

    # Identify common columns
    depmap_ids = set(gene_effect_df.columns)
    logger.info(f"DepMap cell lines: {len(depmap_ids)}")

    # Map GDSC cell lines to DepMap IDs
    gdsc_col = None
    for col in ['CELL_LINE_NAME', 'Cell line Name', 'SANGER_MODEL_ID']:
        if col in gdsc_df.columns:
            gdsc_col = col
            break

    if gdsc_col is None:
        # Try to identify the right column
        logger.info(f"GDSC columns: {list(gdsc_df.columns[:10])}")
        return None

    # Build dataset
    logger.info(f"Building drug response dataset using GDSC column: {gdsc_col}")

    # Get top drugs (most data points)
    drug_col = None
    for col in ['DRUG_NAME', 'Drug Name', 'DRUG_ID']:
        if col in gdsc_df.columns:
            drug_col = col
            break

    ic50_col = None
    for col in ['LN_IC50', 'IC50', 'AUC']:
        if col in gdsc_df.columns:
            ic50_col = col
            break

    if drug_col is None or ic50_col is None:
        logger.info(f"Available GDSC columns: {list(gdsc_df.columns)}")
        return None

    # Top 20 drugs by sample count
    drug_counts = gdsc_df[drug_col].value_counts()
    top_drugs = drug_counts.head(20).index.tolist()
    logger.info(f"Top drugs: {top_drugs[:5]}")

    datasets = {}
    for drug in top_drugs:
        drug_data = gdsc_df[gdsc_df[drug_col] == drug]

        X_rows = []
        y_vals = []
        cell_lines_used = []

        for _, row in drug_data.iterrows():
            cell_name = str(row[gdsc_col]).upper().replace('-', '').replace(' ', '')
            depmap_id = name_to_depmap.get(cell_name)

            if depmap_id is None:
                # Try fuzzy match
                for name, did in name_to_depmap.items():
                    if cell_name in str(name).upper().replace('-', '').replace(' ', ''):
                        depmap_id = did
                        break

            if depmap_id and depmap_id in gene_effect_df.columns:
                X_rows.append(gene_effect_df[depmap_id].values)
                y_vals.append(float(row[ic50_col]))
                cell_lines_used.append(depmap_id)

        if len(X_rows) >= 20:
            X = np.array(X_rows)
            y = np.array(y_vals)
            datasets[drug] = {
                'X': X,
                'y': y,
                'cell_lines': cell_lines_used,
                'gene_names': gene_effect_df.index.tolist(),
                'n_samples': len(y),
            }
            logger.info(f"  {drug}: {len(y)} cell lines matched")

    return datasets


def train_drug_response_models(datasets):
    """Train ML models for each drug."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    results = []

    for drug, data in datasets.items():
        X, y = data['X'], data['y']
        gene_names = data['gene_names']

        # Remove NaN features
        nan_mask = np.isnan(X).any(axis=0)
        X_clean = X[:, ~nan_mask]
        clean_genes = [g for g, m in zip(gene_names, ~nan_mask) if m]

        if X_clean.shape[1] < 10:
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Top 500 features by variance
        if X_scaled.shape[1] > 500:
            var = np.var(X_scaled, axis=0)
            top_idx = np.argsort(var)[-500:]
            X_scaled = X_scaled[:, top_idx]
            clean_genes = [clean_genes[i] for i in top_idx]

        # Models
        models = {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
            'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        }

        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
                r2_mean = scores.mean()
                r2_std = scores.std()

                results.append({
                    'drug': drug,
                    'model': model_name,
                    'r2_mean': r2_mean,
                    'r2_std': r2_std,
                    'n_samples': len(y),
                    'n_features': X_scaled.shape[1],
                })

                if r2_mean > 0.1:
                    # Get feature importance
                    model.fit(X_scaled, y)
                    if hasattr(model, 'feature_importances_'):
                        imp = model.feature_importances_
                        top_idx = np.argsort(imp)[-10:][::-1]
                        top_genes = [clean_genes[i] for i in top_idx]
                        logger.info(f"  {drug} ({model_name}): R²={r2_mean:.3f}±{r2_std:.3f}, "
                                   f"top genes: {top_genes[:5]}")
                    elif hasattr(model, 'coef_'):
                        coef = np.abs(model.coef_)
                        top_idx = np.argsort(coef)[-10:][::-1]
                        top_genes = [clean_genes[i] for i in top_idx]
                        logger.info(f"  {drug} ({model_name}): R²={r2_mean:.3f}±{r2_std:.3f}, "
                                   f"top genes: {top_genes[:5]}")
            except Exception as e:
                logger.warning(f"  {drug} ({model_name}): failed - {e}")

    return pd.DataFrame(results)


def plot_drug_sensitivity_results(results_df, datasets):
    """Visualize drug sensitivity prediction results."""
    if results_df is None or len(results_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel A: Best R² per drug
    ax = axes[0, 0]
    best_per_drug = results_df.groupby('drug')['r2_mean'].max().sort_values(ascending=True)
    colors = ['#2ecc71' if v > 0.2 else '#e74c3c' if v < 0 else '#f1c40f' for v in best_per_drug]
    ax.barh(range(len(best_per_drug)), best_per_drug.values, color=colors)
    ax.set_yticks(range(len(best_per_drug)))
    ax.set_yticklabels(best_per_drug.index, fontsize=8)
    ax.set_xlabel('Best R² (5-fold CV)')
    ax.set_title('A) Drug Sensitivity Prediction Accuracy', fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Panel B: Model comparison
    ax = axes[0, 1]
    model_scores = results_df.groupby('model')['r2_mean'].agg(['mean', 'std'])
    model_scores = model_scores.sort_values('mean', ascending=True)
    ax.barh(model_scores.index, model_scores['mean'], xerr=model_scores['std'],
            color=['#3498db', '#e74c3c', '#2ecc71'][:len(model_scores)],
            capsize=5, edgecolor='white')
    ax.set_xlabel('Mean R² across drugs')
    ax.set_title('B) Model Comparison', fontweight='bold')

    # Panel C: Sample size vs R²
    ax = axes[1, 0]
    best_results = results_df.loc[results_df.groupby('drug')['r2_mean'].idxmax()]
    ax.scatter(best_results['n_samples'], best_results['r2_mean'],
              s=80, c='#3498db', alpha=0.7, edgecolor='white')
    for _, row in best_results.iterrows():
        if row['r2_mean'] > 0.15:
            ax.annotate(row['drug'][:15], (row['n_samples'], row['r2_mean']),
                       fontsize=7, alpha=0.8)
    ax.set_xlabel('Number of Cell Lines')
    ax.set_ylabel('Best R²')
    ax.set_title('C) Sample Size vs Prediction Accuracy', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel D: Distribution of R² scores
    ax = axes[1, 1]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]['r2_mean']
        ax.hist(model_data, bins=20, alpha=0.5, label=model, edgecolor='white')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Count')
    ax.set_title('D) Distribution of Prediction Accuracy', fontweight='bold')
    ax.legend()
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Drug Sensitivity Prediction from Genomic Features',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "drug_sensitivity_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: drug_sensitivity_results.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 10: Drug Sensitivity Prediction")
    logger.info("=" * 60)

    # Step 1: Download/load GDSC data
    gdsc_df = download_gdsc_data()

    # Step 2: Load cell line genomic features (DepMap gene effect)
    gene_effect = download_gdsc_expression()

    if gdsc_df is None:
        logger.warning("GDSC data not available. Using DepMap gene dependency prediction instead.")

        # Use DepMap gene effect to predict drug-gene interactions from DGIdb
        dgidb_file = DATA_DIR / "dgidb_interactions.parquet"
        if dgidb_file.exists() and gene_effect is not None:
            dgidb = pd.read_parquet(dgidb_file)

            logger.info("\nPredicting gene dependency from CRISPR gene effect")
            logger.info("Using DGIdb druggable genes as targets")

            # gene_effect: rows=cell lines (ACH-*), columns=GENE (ENTREZ_ID)
            # Parse gene symbols from column names
            col_to_gene = {}
            gene_to_col = {}
            for col in gene_effect.columns:
                gene_sym = col.split(' (')[0] if ' (' in col else col
                col_to_gene[col] = gene_sym
                gene_to_col[gene_sym] = col

            logger.info(f"Gene effect: {gene_effect.shape[0]} cell lines, {gene_effect.shape[1]} genes")
            logger.info(f"Parsed gene symbols: {list(gene_to_col.keys())[:5]}...")

            druggable_genes = dgidb[dgidb['approved']]['gene'].unique()
            overlap = [g for g in druggable_genes if g in gene_to_col]
            logger.info(f"Druggable genes with approved drugs: {len(druggable_genes)}")
            logger.info(f"Overlap with DepMap: {len(overlap)}")

            # Also add SL target genes from Exp 2
            sl_file = SL_DIR / "sl_pairs.csv"
            sl_genes = set()
            if sl_file.exists():
                sl_df = pd.read_csv(sl_file)
                if 'target_gene' in sl_df.columns:
                    sl_genes = set(sl_df['target_gene'].unique())
                elif 'gene_B' in sl_df.columns:
                    sl_genes = set(sl_df['gene_B'].unique())
                sl_overlap = [g for g in sl_genes if g in gene_to_col]
                logger.info(f"SL target genes in DepMap: {len(sl_overlap)}")

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            # Test all druggable + SL genes
            test_genes = list(set(overlap) | (sl_genes & set(gene_to_col.keys())))
            logger.info(f"Total genes to test: {len(test_genes)}")

            results = []
            for i, gene in enumerate(test_genes):
                col = gene_to_col[gene]

                # Target: is cell line dependent on this gene? (gene effect < -0.5)
                gene_vals = gene_effect[col]
                y = (gene_vals < -0.5).astype(int)

                if y.sum() < 10 or y.sum() > len(y) - 10:
                    continue

                # Features: 500 random other genes' effects
                other_cols = [c for c in gene_effect.columns if c != col]
                np.random.seed(42 + i)
                feat_cols = np.random.choice(other_cols, min(500, len(other_cols)), replace=False)
                X = gene_effect[feat_cols].values
                valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(gene_vals.values)
                X = X[valid_mask]
                y = y.values[valid_mask]

                if len(y) < 50:
                    continue

                try:
                    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
                    auc_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc', n_jobs=-1)

                    n_drugs = len(dgidb[dgidb['gene'] == gene])
                    approved_list = dgidb[(dgidb['gene'] == gene) & (dgidb['approved'])]['drug'].unique()
                    is_sl = gene in sl_genes

                    results.append({
                        'gene': gene,
                        'auc_mean': auc_scores.mean(),
                        'auc_std': auc_scores.std(),
                        'n_dependent': int(y.sum()),
                        'n_total': len(y),
                        'n_drugs': n_drugs,
                        'approved_drugs': ', '.join(approved_list[:3]),
                        'is_sl_target': is_sl,
                    })

                    if auc_scores.mean() > 0.6:
                        sl_tag = " [SL]" if is_sl else ""
                        logger.info(f"  {gene}{sl_tag}: AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}, "
                                   f"{y.sum()}/{len(y)} dependent, drugs: {', '.join(approved_list[:2])}")
                except Exception as e:
                    logger.debug(f"  {gene}: failed - {e}")

                if (i + 1) % 20 == 0:
                    logger.info(f"  Progress: {i+1}/{len(test_genes)} genes tested")

            if results:
                results_df = pd.DataFrame(results).sort_values('auc_mean', ascending=False)
                results_df.to_csv(RESULTS_DIR / "gene_dependency_prediction.csv", index=False)

                # Plot
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))

                ax = axes[0]
                top20 = results_df.head(20)
                colors = ['#2ecc71' if v > 0.7 else '#f1c40f' if v > 0.6 else '#e74c3c'
                         for v in top20['auc_mean']]
                ax.barh(range(len(top20)), top20['auc_mean'],
                       xerr=top20['auc_std'], color=colors, capsize=3)
                ax.set_yticks(range(len(top20)))
                ax.set_yticklabels(top20['gene'], fontsize=9)
                ax.set_xlabel('AUC-ROC (5-fold CV)')
                ax.set_title('Gene Dependency Prediction', fontweight='bold')
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.invert_yaxis()

                ax = axes[1]
                ax.scatter(results_df['n_drugs'], results_df['auc_mean'],
                          s=results_df['n_dependent'] * 2, c='#3498db', alpha=0.6)
                for _, row in results_df[results_df['auc_mean'] > 0.65].iterrows():
                    ax.annotate(row['gene'], (row['n_drugs'], row['auc_mean']),
                               fontsize=8)
                ax.set_xlabel('Number of Known Drugs')
                ax.set_ylabel('Dependency Prediction AUC')
                ax.set_title('Druggability vs Predictability', fontweight='bold')
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

                plt.tight_layout()
                plt.savefig(RESULTS_DIR / "gene_dependency_prediction.png", dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("Saved: gene_dependency_prediction.png")

                summary = {
                    'experiment': 'Exp 10: Drug Sensitivity / Gene Dependency Prediction',
                    'approach': 'DepMap CRISPR gene effect + RF classification',
                    'n_genes_tested': len(results_df),
                    'n_predictable': int((results_df['auc_mean'] > 0.6).sum()),
                    'top_genes': results_df.head(10).to_dict('records'),
                }

                with open(RESULTS_DIR / "exp10_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2, default=str)

        logger.info("\n" + "=" * 60)
        logger.info("Experiment 10 COMPLETE")
        logger.info("=" * 60)
        return

    # If GDSC data is available
    datasets = prepare_drug_response_dataset(gdsc_df, gene_effect)
    if datasets:
        results_df = train_drug_response_models(datasets)
        results_df.to_csv(RESULTS_DIR / "drug_sensitivity_models.csv", index=False)
        plot_drug_sensitivity_results(results_df, datasets)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 10 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
