#!/usr/bin/env python3
"""
Experiment 16: Multi-Omics Patient Clustering & Subtype Discovery
==================================================================
Integrates expression, immune, and clinical features to discover
clinically meaningful patient subgroups via consensus clustering.

Pipeline:
1. Load expression (top variable genes), immune scores, clinical data
2. Feature normalization & integration
3. Consensus clustering (k=2..8) with stability analysis
4. Survival analysis per cluster (log-rank test)
5. Pathway enrichment per cluster
6. Treatment implications per subtype

Target: Paper 10 (Patient Stratification) — Journal of Clinical Oncology
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "tcga"
EXPR_DIR = BASE_DIR / "results" / "exp3_differential_expression"
IMM_DIR = BASE_DIR / "results" / "exp6_immune_tme"
SL_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
RESULTS_DIR = BASE_DIR / "results" / "exp16_patient_clustering"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CANCER_TYPES = ['BRCA', 'LUAD', 'KIRC']


def load_expression_features(cancer_type, n_top_genes=500):
    """Load top variable genes from expression data."""
    expr_file = DATA_DIR / f"{cancer_type}_expression.parquet"
    if not expr_file.exists():
        return None, None

    expr = pd.read_parquet(expr_file)
    logger.info(f"[{cancer_type}] Expression: {expr.shape}")

    # Get tumor samples only (genes in index, patients in columns)
    sample_cols = list(expr.columns)
    tumor_cols = [c for c in sample_cols if len(c.split('-')) <= 3 or not c.split('-')[-1].startswith('1')]

    # Filter out non-coding RNA genes (MIR*, RNU*, SNOR*, LINC*, LOC*, etc.)
    non_coding_prefixes = ('MIR', 'RNU', 'SNOR', 'LINC', 'LOC', 'MT-', 'MTRNR',
                           'RNA5S', 'RNA18S', 'RNA28S', 'RNA45S', 'SCARNA', 'SNHG')
    protein_coding = [g for g in expr.index
                      if not any(g.upper().startswith(p) for p in non_coding_prefixes)
                      and not g.startswith('.')
                      and len(g) > 1]
    expr_filtered = expr.loc[protein_coding]
    logger.info(f"  Filtered to {len(protein_coding)} protein-coding genes (from {len(expr.index)})")

    # Select top variable genes (by median absolute deviation - more robust)
    tumor_data = expr_filtered[tumor_cols]
    gene_medians = tumor_data.median(axis=1)
    gene_mad = (tumor_data.subtract(gene_medians, axis=0)).abs().median(axis=1)
    # Also require reasonable expression (filter near-zero genes)
    expressed = gene_medians > 1.0
    gene_mad_filtered = gene_mad[expressed]
    top_genes = gene_mad_filtered.nlargest(n_top_genes).index.tolist()

    # Transpose: patients x genes
    expr_features = tumor_data.loc[top_genes].T
    expr_features.index = ['-'.join(c.split('-')[:3]) for c in expr_features.index]
    # Drop duplicate patient IDs (keep first sample per patient)
    expr_features = expr_features[~expr_features.index.duplicated(keep='first')]

    logger.info(f"  Expression features: {expr_features.shape} (top {n_top_genes} variable genes)")
    return expr_features, top_genes


def load_immune_features(cancer_type):
    """Load immune deconvolution scores."""
    imm_file = IMM_DIR / f"{cancer_type}_immune_scores.csv"
    if not imm_file.exists():
        return None

    imm = pd.read_csv(imm_file, index_col=0)
    # Normalize patient IDs to 3-part and drop duplicates
    imm.index = ['-'.join(str(idx).split('-')[:3]) for idx in imm.index]
    imm = imm[~imm.index.duplicated(keep='first')]
    logger.info(f"  Immune features: {imm.shape}")
    return imm


def load_clinical_features(cancer_type):
    """Load clinical data (JSON format)."""
    import json as json_mod

    clin_file = DATA_DIR / f"{cancer_type}_clinical.json"
    if not clin_file.exists():
        # Try parquet fallback
        parquet_file = DATA_DIR / "pan_cancer_clinical.parquet"
        if parquet_file.exists():
            pan = pd.read_parquet(parquet_file)
            clin = pan[pan['cancer_type'] == cancer_type].copy()
            if 'case_id' in clin.columns:
                clin = clin.set_index('case_id')
            logger.info(f"  Clinical data (from pan-cancer): {clin.shape}")
            return clin
        return None

    with open(clin_file) as f:
        clin_data = json_mod.load(f)
    clin = pd.DataFrame(clin_data)

    if 'case_id' in clin.columns:
        clin = clin.set_index('case_id')
    clin.index = ['-'.join(str(idx).split('-')[:3]) for idx in clin.index]

    logger.info(f"  Clinical data: {clin.shape}")
    return clin


def consensus_clustering(features_df, k_range=range(2, 9), n_iterations=50):
    """
    Perform consensus clustering to find optimal k.

    For each k, run KMeans n_iterations times with different seeds,
    build a consensus matrix, then evaluate stability.
    """
    n_samples = len(features_df)
    results = {}

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.fillna(0))

    # PCA for dimensionality reduction
    n_components = min(50, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"  PCA: {n_components} components, {var_explained:.1%} variance explained")

    for k in k_range:
        # Consensus matrix: how often are pairs of samples in same cluster?
        consensus = np.zeros((n_samples, n_samples))

        for i in range(n_iterations):
            # Subsample 80% of samples
            np.random.seed(i)
            sample_idx = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)

            km = KMeans(n_clusters=k, random_state=i, n_init=10, max_iter=300)
            labels = km.fit_predict(X_pca[sample_idx])

            # Update consensus matrix
            for a_idx in range(len(sample_idx)):
                for b_idx in range(a_idx + 1, len(sample_idx)):
                    sa, sb = sample_idx[a_idx], sample_idx[b_idx]
                    consensus[sa, sb] += (labels[a_idx] == labels[b_idx])
                    consensus[sb, sa] = consensus[sa, sb]

        consensus /= n_iterations

        # Final clustering on consensus matrix
        agg = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        final_labels = agg.fit_predict(1 - consensus)

        # Metrics
        sil = silhouette_score(X_pca, final_labels) if k < n_samples else 0
        ch = calinski_harabasz_score(X_pca, final_labels) if k < n_samples else 0

        # Cophenetic correlation (consensus matrix quality)
        # Higher = more stable clusters
        within = []
        between = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if final_labels[i] == final_labels[j]:
                    within.append(consensus[i, j])
                else:
                    between.append(consensus[i, j])

        cpi = np.mean(within) - np.mean(between) if within and between else 0

        results[k] = {
            'silhouette': sil,
            'calinski_harabasz': ch,
            'cpi': cpi,
            'labels': final_labels,
            'consensus': consensus,
        }

        logger.info(f"  k={k}: Silhouette={sil:.3f}, CH={ch:.1f}, CPI={cpi:.3f}")

    # Select optimal k (highest silhouette)
    best_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
    logger.info(f"  Optimal k={best_k} (silhouette={results[best_k]['silhouette']:.3f})")

    return results, best_k, X_pca


def survival_analysis_by_cluster(labels, clinical_df, cancer_type):
    """Run survival analysis per cluster."""
    # Match patients
    common_patients = list(set(clinical_df.index) & set(labels.index))
    if len(common_patients) < 20:
        logger.info(f"  Too few patients for survival analysis ({len(common_patients)})")
        return None

    clin_matched = clinical_df.loc[common_patients]
    labels_matched = labels.loc[common_patients]

    # Get survival columns
    time_col = None
    event_col = None

    for col in clin_matched.columns:
        col_lower = col.lower()
        if 'overall_survival' in col_lower or 'os_time' in col_lower or col_lower == 'os.time':
            if 'status' not in col_lower and 'event' not in col_lower:
                time_col = col
        if 'vital_status' in col_lower or 'os_status' in col_lower or col_lower == 'os':
            event_col = col

    # Try alternate column names
    if time_col is None:
        for col in clin_matched.columns:
            if 'time' in col.lower() and 'surviv' in col.lower():
                time_col = col
                break
            if col.lower() in ['days_to_death', 'days_to_last_followup', 'os_months']:
                time_col = col
                break

    if event_col is None:
        for col in clin_matched.columns:
            if 'status' in col.lower() or 'vital' in col.lower():
                event_col = col
                break

    if time_col is None or event_col is None:
        logger.info(f"  Could not find survival columns. Available: {list(clin_matched.columns)}")
        return None

    # Parse survival data
    surv_data = pd.DataFrame({
        'time': pd.to_numeric(clin_matched[time_col], errors='coerce'),
        'event': clin_matched[event_col],
        'cluster': labels_matched,
    }).dropna(subset=['time'])

    # Convert event to binary
    if surv_data['event'].dtype == object:
        surv_data['event'] = surv_data['event'].map(
            lambda x: 1 if str(x).lower() in ['dead', '1', 'deceased', 'true'] else 0
        )
    surv_data['event'] = pd.to_numeric(surv_data['event'], errors='coerce').fillna(0).astype(int)
    surv_data = surv_data[surv_data['time'] > 0]

    if len(surv_data) < 20:
        return None

    # Log-rank test
    try:
        result = multivariate_logrank_test(
            surv_data['time'], surv_data['cluster'], surv_data['event']
        )
        p_value = result.p_value
    except Exception:
        p_value = 1.0

    logger.info(f"  [{cancer_type}] Survival analysis: n={len(surv_data)}, log-rank p={p_value:.4f}")

    return {
        'cancer_type': cancer_type,
        'n_patients': len(surv_data),
        'n_events': int(surv_data['event'].sum()),
        'log_rank_p': p_value,
        'surv_data': surv_data,
    }


def cluster_characterization(features_df, labels, top_genes, cancer_type):
    """Characterize each cluster by distinguishing features."""
    features_with_labels = features_df.copy()
    features_with_labels['cluster'] = labels

    cluster_profiles = {}
    for k in sorted(labels.unique()):
        cluster_data = features_with_labels[features_with_labels['cluster'] == k]
        other_data = features_with_labels[features_with_labels['cluster'] != k]

        profile = {
            'n_patients': len(cluster_data),
            'distinguishing_genes': [],
        }

        # Find top distinguishing genes (by t-test)
        gene_cols = [c for c in features_df.columns if c != 'cluster']
        p_values = []
        for gene in gene_cols[:200]:  # Test top 200 genes
            try:
                t, p = stats.ttest_ind(
                    cluster_data[gene].dropna(),
                    other_data[gene].dropna(),
                    equal_var=False
                )
                fc = cluster_data[gene].mean() - other_data[gene].mean()
                p_values.append((gene, p, fc, t))
            except Exception:
                continue

        p_values.sort(key=lambda x: x[1])
        profile['distinguishing_genes'] = [
            {'gene': g, 'p_value': p, 'fold_change': fc}
            for g, p, fc, t in p_values[:20]
        ]

        cluster_profiles[int(k)] = profile
        logger.info(f"  Cluster {k}: {profile['n_patients']} patients, "
                    f"top gene: {p_values[0][0] if p_values else 'N/A'}")

    return cluster_profiles


def plot_clustering_results(X_pca, labels, surv_result, cancer_type, consensus_results, best_k):
    """Create comprehensive clustering visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Panel A: PCA scatter plot colored by cluster
    ax = axes[0, 0]
    unique_labels = sorted(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    for i, k in enumerate(unique_labels):
        mask = labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], label=f'Cluster {k+1}',
                  s=30, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'A) Patient Clusters ({cancer_type})', fontweight='bold')
    ax.legend()

    # Panel B: Silhouette scores across k values
    ax = axes[0, 1]
    k_values = sorted(consensus_results.keys())
    sil_values = [consensus_results[k]['silhouette'] for k in k_values]
    cpi_values = [consensus_results[k]['cpi'] for k in k_values]
    ax.plot(k_values, sil_values, 'bo-', label='Silhouette', linewidth=2)
    ax.plot(k_values, cpi_values, 'rs--', label='CPI', linewidth=2)
    ax.axvline(x=best_k, color='green', linestyle=':', label=f'Best k={best_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Score')
    ax.set_title('B) Cluster Stability Metrics', fontweight='bold')
    ax.legend()

    # Panel C: Consensus heatmap
    ax = axes[0, 2]
    consensus = consensus_results[best_k]['consensus']
    # Sort by cluster assignment for better visualization
    sort_idx = np.argsort(labels)
    consensus_sorted = consensus[sort_idx][:, sort_idx]
    im = ax.imshow(consensus_sorted, cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Co-clustering frequency')
    ax.set_title('C) Consensus Matrix', fontweight='bold')
    ax.set_xlabel('Patients (sorted by cluster)')
    ax.set_ylabel('Patients (sorted by cluster)')

    # Panel D: KM survival curves by cluster
    ax = axes[1, 0]
    if surv_result is not None:
        surv_data = surv_result['surv_data']
        kmf = KaplanMeierFitter()
        for k in sorted(surv_data['cluster'].unique()):
            mask = surv_data['cluster'] == k
            kmf.fit(surv_data.loc[mask, 'time'], surv_data.loc[mask, 'event'],
                   label=f'Cluster {k+1} (n={mask.sum()})')
            kmf.plot_survival_function(ax=ax)

        p_val = surv_result['log_rank_p']
        ax.set_title(f'D) Survival by Cluster (p={p_val:.4f})', fontweight='bold')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
    else:
        ax.text(0.5, 0.5, 'No survival data', transform=ax.transAxes,
               ha='center', va='center', fontsize=14)
        ax.set_title('D) Survival by Cluster', fontweight='bold')

    # Panel E: Cluster size distribution
    ax = axes[1, 1]
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    bars = ax.bar(range(len(cluster_sizes)),
                 cluster_sizes.values,
                 color=[colors[i] for i in range(len(cluster_sizes))],
                 edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(cluster_sizes)))
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(len(cluster_sizes))])
    for bar, size in zip(bars, cluster_sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               str(size), ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.set_title('E) Cluster Sizes', fontweight='bold')

    # Panel F: PCA variance explained
    ax = axes[1, 2]
    # Re-do PCA to get variance explained
    from sklearn.decomposition import PCA as PCA2
    pca_full = PCA2(n_components=min(20, X_pca.shape[1]), random_state=42)
    pca_full.fit(X_pca)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax.bar(range(1, len(cumvar)+1), pca_full.explained_variance_ratio_,
          color='#3498db', alpha=0.7, label='Individual')
    ax.plot(range(1, len(cumvar)+1), cumvar, 'ro-', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title('F) PCA Variance Explained', fontweight='bold')
    ax.legend()

    plt.suptitle(f'Multi-Omics Patient Subtyping: {cancer_type}',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{cancer_type}_clustering.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {cancer_type}_clustering.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 16: Multi-Omics Patient Clustering")
    logger.info("=" * 60)

    all_results = {}

    for cancer in CANCER_TYPES:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {cancer}")
        logger.info(f"{'='*40}")

        # 1. Load features
        expr_features, top_genes = load_expression_features(cancer, n_top_genes=500)
        if expr_features is None:
            logger.info(f"  [{cancer}] No expression data, skipping")
            continue

        immune_features = load_immune_features(cancer)
        clinical_df = load_clinical_features(cancer)

        # 2. Integrate features
        # Start with expression
        integrated = expr_features.copy()

        # Add immune features
        if immune_features is not None:
            common = list(set(integrated.index) & set(immune_features.index))
            if len(common) > 20:
                # Prefix immune columns to distinguish
                imm_cols = {col: f'IMM_{col}' for col in immune_features.columns}
                immune_renamed = immune_features.rename(columns=imm_cols)
                integrated = integrated.loc[common].join(immune_renamed.loc[common])
                logger.info(f"  Integrated with immune: {integrated.shape}")

        # Add clinical features (encoded)
        if clinical_df is not None:
            common = list(set(integrated.index) & set(clinical_df.index))
            if len(common) > 20:
                # Only add numeric clinical columns
                numeric_clin = clinical_df.select_dtypes(include=[np.number])
                if len(numeric_clin.columns) > 0:
                    clin_cols = {col: f'CLIN_{col}' for col in numeric_clin.columns}
                    clin_renamed = numeric_clin.rename(columns=clin_cols)
                    integrated = integrated.loc[common].join(clin_renamed.loc[common], how='left')
                    logger.info(f"  Integrated with clinical: {integrated.shape}")

        # Drop columns with too many NaN
        integrated = integrated.dropna(axis=1, thresh=int(0.5 * len(integrated)))
        integrated = integrated.fillna(integrated.median())
        logger.info(f"  Final feature matrix: {integrated.shape}")

        if len(integrated) < 30:
            logger.info(f"  [{cancer}] Too few samples ({len(integrated)}), skipping")
            continue

        # 3. Consensus clustering
        consensus_results, best_k, X_pca = consensus_clustering(integrated)
        labels = consensus_results[best_k]['labels']

        # Create labels series with patient IDs
        labels_series = pd.Series(labels, index=integrated.index)

        # 4. Survival analysis
        surv_result = None
        if clinical_df is not None:
            surv_result = survival_analysis_by_cluster(labels_series, clinical_df, cancer)

        # 5. Cluster characterization
        cluster_profiles = cluster_characterization(integrated, labels_series, top_genes, cancer)

        # 6. Visualize
        plot_clustering_results(X_pca, labels, surv_result, cancer, consensus_results, best_k)

        # Save results
        cancer_result = {
            'cancer_type': cancer,
            'n_patients': len(integrated),
            'n_features': integrated.shape[1],
            'best_k': best_k,
            'silhouette': consensus_results[best_k]['silhouette'],
            'cpi': consensus_results[best_k]['cpi'],
            'cluster_sizes': {int(k): int(v) for k, v in
                            pd.Series(labels).value_counts().to_dict().items()},
            'survival_p': surv_result['log_rank_p'] if surv_result else None,
            'cluster_profiles': cluster_profiles,
            'k_metrics': {int(k): {
                'silhouette': v['silhouette'],
                'calinski_harabasz': v['calinski_harabasz'],
                'cpi': v['cpi'],
            } for k, v in consensus_results.items()},
        }

        all_results[cancer] = cancer_result

        # Save per-cancer cluster assignments
        labels_df = pd.DataFrame({
            'patient_id': integrated.index,
            'cluster': labels,
        })
        labels_df.to_csv(RESULTS_DIR / f"{cancer}_cluster_assignments.csv", index=False)

    # Save summary
    summary = {
        'experiment': 'Exp 16: Multi-Omics Patient Clustering',
        'method': 'Consensus clustering (KMeans x50 iterations, 80% subsampling)',
        'features': 'Expression (top 500 var genes) + Immune + Clinical',
        'results': {cancer: {k: v for k, v in r.items() if k != 'cluster_profiles'}
                   for cancer, r in all_results.items()},
    }

    with open(RESULTS_DIR / "exp16_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 16 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
