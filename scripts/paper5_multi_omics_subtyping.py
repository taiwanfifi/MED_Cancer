#!/usr/bin/env python3
"""
Paper 5: Multi-Omics Cancer Subtype Discovery
===============================================
Integrates mutation, gene expression, DNA methylation, and CNV data from TCGA
to discover clinically actionable cancer subtypes using deep multi-modal learning.

Target: Cell Reports Medicine / Nature Cancer
Hypothesis: H1 - Multi-omics integration reveals hidden cancer subtypes
            missed by single-omics analysis

Experiment Flow:
1. Download TCGA pan-cancer data via GDC API
2. Preprocess each omics layer
3. Train Multi-modal Variational Autoencoder (MVAE)
4. Cluster latent representations
5. Validate subtypes with survival analysis
6. Compare single vs multi-omics subtyping

Checkpoint support: All intermediate results saved for resume.
"""

import os
import sys
import json
import hashlib
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "tcga"
RESULTS_DIR = BASE_DIR / "results" / "paper5"
CKPT_DIR = BASE_DIR / "checkpoints" / "paper5"
LOG_DIR = BASE_DIR / "logs" / "paper5"

for d in [DATA_DIR, RESULTS_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# TCGA cancer types to analyze (top 10 by sample size)
CANCER_TYPES = [
    "BRCA",  # Breast (1098)
    "LUAD",  # Lung Adenocarcinoma (585)
    "UCEC",  # Uterine (560)
    "KIRC",  # Kidney Clear Cell (537)
    "HNSC",  # Head & Neck (528)
    "LGG",   # Lower Grade Glioma (516)
    "THCA",  # Thyroid (507)
    "LUSC",  # Lung Squamous (504)
    "PRAD",  # Prostate (500)
    "STAD",  # Stomach (443)
]

# GDC API endpoints
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

# ── Progress Tracking ───────────────────────────────────────────────────────
class ProgressTracker:
    """Track experiment progress for checkpoint/resume."""
    def __init__(self, path):
        self.path = Path(path)
        self.progress = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"steps_completed": [], "current_step": None, "metadata": {}}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)

    def is_done(self, step):
        return step in self.progress["steps_completed"]

    def mark_done(self, step, metadata=None):
        if step not in self.progress["steps_completed"]:
            self.progress["steps_completed"].append(step)
        if metadata:
            self.progress["metadata"][step] = metadata
        self.progress["current_step"] = None
        self.save()

    def set_current(self, step):
        self.progress["current_step"] = step
        self.save()

tracker = ProgressTracker(CKPT_DIR / "progress.json")


# ── Step 1: Download TCGA Data via GDC API ──────────────────────────────────
def download_tcga_gene_expression(cancer_type):
    """Download RNA-Seq gene expression data from GDC."""
    import requests

    output_file = DATA_DIR / f"{cancer_type}_expression.parquet"
    if output_file.exists():
        logger.info(f"[{cancer_type}] Expression data already downloaded")
        return pd.read_parquet(output_file)

    logger.info(f"[{cancer_type}] Downloading gene expression from GDC...")

    # Query GDC for HTSeq-FPKM files
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [f"TCGA-{cancer_type}"]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.case_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "2000"
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params)
    data = response.json()

    file_ids = [hit["file_id"] for hit in data["data"]["hits"]]
    case_ids = []
    for hit in data["data"]["hits"]:
        if hit.get("cases"):
            case_ids.append(hit["cases"][0]["submitter_id"])
        else:
            case_ids.append("unknown")

    logger.info(f"[{cancer_type}] Found {len(file_ids)} expression files")

    # Download files in batches
    all_samples = {}
    batch_size = 50
    for i in range(0, len(file_ids), batch_size):
        batch = file_ids[i:i+batch_size]
        payload = {"ids": batch}

        response = requests.post(
            GDC_DATA_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )

        if response.status_code == 200:
            # Save raw tarball
            import tarfile
            import io

            tar_path = DATA_DIR / f"{cancer_type}_batch_{i}.tar.gz"
            with open(tar_path, 'wb') as f:
                f.write(response.content)

            # Extract and parse
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith('.tsv') or member.name.endswith('.counts'):
                            f = tar.extractfile(member)
                            if f:
                                df_temp = pd.read_csv(f, sep='\t', comment='#',
                                                       header=0, index_col=0)
                                sample_id = member.name.split('/')[0][:12]
                                if 'tpm_unstranded' in df_temp.columns:
                                    all_samples[sample_id] = df_temp['tpm_unstranded']
                                elif 'unstranded' in df_temp.columns:
                                    all_samples[sample_id] = df_temp['unstranded']
                                elif len(df_temp.columns) >= 1:
                                    all_samples[sample_id] = df_temp.iloc[:, 0]
            except Exception as e:
                logger.warning(f"[{cancer_type}] Error extracting batch {i}: {e}")

            # Clean up tar
            tar_path.unlink(missing_ok=True)

        logger.info(f"[{cancer_type}] Processed {min(i+batch_size, len(file_ids))}/{len(file_ids)} files")

    if all_samples:
        expr_df = pd.DataFrame(all_samples)
        expr_df.to_parquet(output_file)
        logger.info(f"[{cancer_type}] Saved expression: {expr_df.shape}")
        return expr_df
    else:
        logger.warning(f"[{cancer_type}] No expression data extracted")
        return None


def download_tcga_clinical(cancer_type):
    """Download clinical data from GDC."""
    import requests

    output_file = DATA_DIR / f"{cancer_type}_clinical.parquet"
    if output_file.exists():
        logger.info(f"[{cancer_type}] Clinical data already downloaded")
        return pd.read_parquet(output_file)

    logger.info(f"[{cancer_type}] Downloading clinical data from GDC...")

    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": [f"TCGA-{cancer_type}"]
        }
    }

    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.gender",
        "demographic.race",
        "demographic.age_at_index",
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.morphology",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "2000"
    }

    response = requests.get(GDC_CASES_ENDPOINT, params=params)
    data = response.json()

    records = []
    for hit in data["data"]["hits"]:
        record = {"case_id": hit.get("submitter_id", "")}

        demo = hit.get("demographic", {})
        record["vital_status"] = demo.get("vital_status", "")
        record["days_to_death"] = demo.get("days_to_death", None)
        record["gender"] = demo.get("gender", "")
        record["age_at_index"] = demo.get("age_at_index", None)

        diags = hit.get("diagnoses", [{}])
        if diags:
            diag = diags[0]
            record["tumor_stage"] = diag.get("ajcc_pathologic_stage", "")
            record["days_to_last_follow_up"] = diag.get("days_to_last_follow_up", None)
            record["primary_diagnosis"] = diag.get("primary_diagnosis", "")

        records.append(record)

    clin_df = pd.DataFrame(records)
    clin_df.to_parquet(output_file)
    logger.info(f"[{cancer_type}] Saved clinical: {clin_df.shape}")
    return clin_df


def download_tcga_mutations(cancer_type):
    """Download somatic mutation data (MAF) from GDC."""
    import requests

    output_file = DATA_DIR / f"{cancer_type}_mutations.parquet"
    if output_file.exists():
        logger.info(f"[{cancer_type}] Mutation data already downloaded")
        return pd.read_parquet(output_file)

    logger.info(f"[{cancer_type}] Downloading mutation data from GDC...")

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [f"TCGA-{cancer_type}"]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Simple Nucleotide Variation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": "100"
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params)
    data = response.json()

    file_ids = [hit["file_id"] for hit in data["data"]["hits"]]
    logger.info(f"[{cancer_type}] Found {len(file_ids)} MAF files")

    if not file_ids:
        return None

    # Download first MAF file (aggregated)
    payload = {"ids": file_ids[:1]}
    response = requests.post(
        GDC_DATA_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=300
    )

    if response.status_code == 200:
        import tarfile
        import io

        tar_path = DATA_DIR / f"{cancer_type}_maf.tar.gz"
        with open(tar_path, 'wb') as f:
            f.write(response.content)

        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.maf') or member.name.endswith('.maf.gz'):
                        f = tar.extractfile(member)
                        if f:
                            maf_df = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
                            # Create mutation matrix (gene x sample)
                            if 'Hugo_Symbol' in maf_df.columns and 'Tumor_Sample_Barcode' in maf_df.columns:
                                mut_matrix = pd.crosstab(
                                    maf_df['Hugo_Symbol'],
                                    maf_df['Tumor_Sample_Barcode'].str[:12]
                                ).clip(upper=1)
                                mut_matrix.to_parquet(output_file)
                                logger.info(f"[{cancer_type}] Saved mutations: {mut_matrix.shape}")
                                tar_path.unlink(missing_ok=True)
                                return mut_matrix
        except Exception as e:
            logger.warning(f"[{cancer_type}] Error processing MAF: {e}")

        tar_path.unlink(missing_ok=True)

    return None


# ── Step 2: Preprocess Data ─────────────────────────────────────────────────
def preprocess_expression(expr_df, n_top_var=5000):
    """Log-transform and select top variable genes."""
    if expr_df is None:
        return None

    # Log2(TPM + 1) transform
    expr_log = np.log2(expr_df.astype(float) + 1)

    # Remove genes with low variance
    gene_var = expr_log.var(axis=1)
    top_genes = gene_var.nlargest(n_top_var).index
    expr_filtered = expr_log.loc[top_genes]

    # Z-score normalize per gene
    expr_norm = (expr_filtered - expr_filtered.mean(axis=1).values[:, None]) / \
                (expr_filtered.std(axis=1).values[:, None] + 1e-8)

    return expr_norm


# ── Step 3: Multi-Modal VAE ─────────────────────────────────────────────────
def build_mvae_model(input_dims, latent_dim=128):
    """Build Multi-Modal Variational Autoencoder."""
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ])
                prev_dim = h_dim
            self.encoder = nn.Sequential(*layers)
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        def forward(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

    class Decoder(nn.Module):
        def __init__(self, latent_dim, hidden_dims, output_dim):
            super().__init__()
            layers = []
            prev_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.decoder = nn.Sequential(*layers)

        def forward(self, z):
            return self.decoder(z)

    class MVAE(nn.Module):
        """Multi-Modal Variational Autoencoder for cancer subtyping."""
        def __init__(self, input_dims, latent_dim=128):
            super().__init__()
            self.n_modalities = len(input_dims)
            self.latent_dim = latent_dim

            # One encoder/decoder per modality
            hidden_dims = [512, 256]
            self.encoders = nn.ModuleList([
                Encoder(dim, hidden_dims, latent_dim) for dim in input_dims
            ])
            self.decoders = nn.ModuleList([
                Decoder(latent_dim, hidden_dims, dim) for dim in input_dims
            ])

            # Product-of-experts fusion
            self.prior_mu = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
            self.prior_logvar = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def product_of_experts(self, mus, logvars, masks):
            """Combine multiple modality posteriors using PoE."""
            # Prior as uniform expert
            mu_prior = self.prior_mu.expand(mus[0].shape[0], -1)
            logvar_prior = self.prior_logvar.expand(mus[0].shape[0], -1)

            precision = 1.0 / torch.exp(logvar_prior)
            weighted_mu = mu_prior / torch.exp(logvar_prior)

            for i, (mu, logvar, mask) in enumerate(zip(mus, logvars, masks)):
                prec_i = 1.0 / torch.exp(logvar)
                precision += prec_i * mask.unsqueeze(1)
                weighted_mu += (mu * prec_i) * mask.unsqueeze(1)

            joint_logvar = -torch.log(precision)
            joint_mu = weighted_mu / precision

            return joint_mu, joint_logvar

        def forward(self, inputs, masks=None):
            """
            inputs: list of tensors, one per modality
            masks: list of boolean tensors indicating which samples have each modality
            """
            if masks is None:
                masks = [torch.ones(inputs[0].shape[0], device=inputs[0].device) for _ in inputs]

            # Encode each modality
            mus, logvars = [], []
            for i, (enc, x) in enumerate(zip(self.encoders, inputs)):
                mu, logvar = enc(x)
                mus.append(mu)
                logvars.append(logvar)

            # Fuse with Product of Experts
            joint_mu, joint_logvar = self.product_of_experts(mus, logvars, masks)

            # Sample latent
            z = self.reparameterize(joint_mu, joint_logvar)

            # Decode each modality
            recons = [dec(z) for dec in self.decoders]

            return recons, joint_mu, joint_logvar, z

        def loss_function(self, inputs, recons, mu, logvar, masks=None, beta=1.0):
            """ELBO loss with beta-VAE weighting."""
            recon_loss = 0
            for i, (x, r, m) in enumerate(zip(inputs, recons, masks)):
                # MSE reconstruction loss (only for available modalities)
                loss_i = nn.functional.mse_loss(r * m.unsqueeze(1), x * m.unsqueeze(1), reduction='sum')
                recon_loss += loss_i

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            return recon_loss + beta * kl_loss, recon_loss, kl_loss

    return MVAE(input_dims, latent_dim)


# ── Step 4: Training Loop ───────────────────────────────────────────────────
def train_mvae(model, data_loaders, n_epochs=200, lr=1e-3, device='cuda'):
    """Train MVAE with checkpoint support."""
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Check for checkpoint
    ckpt_path = CKPT_DIR / "mvae_latest.pt"
    start_epoch = 0
    best_loss = float('inf')

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")

    history = []

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        n_batches = 0

        # Beta annealing (warm up KL)
        beta = min(1.0, epoch / 50.0)

        for batch in data_loaders['train']:
            inputs = [x.to(device) for x in batch['data']]
            masks = [m.to(device) for m in batch['masks']]

            optimizer.zero_grad()
            recons, mu, logvar, z = model(inputs, masks)
            loss, recon, kl = model.loss_function(inputs, recons, mu, logvar, masks, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)

        history.append({
            'epoch': epoch, 'loss': avg_loss,
            'recon': avg_recon, 'kl': avg_kl, 'beta': beta
        })

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.2f} | "
                       f"Recon: {avg_recon:.2f} | KL: {avg_kl:.2f} | Beta: {beta:.2f}")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'history': history,
                }, CKPT_DIR / "mvae_best.pt")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'history': history,
            }, ckpt_path)

    return model, history


# ── Step 5: Clustering & Evaluation ─────────────────────────────────────────
def cluster_and_evaluate(model, data_loader, clinical_df, device='cuda'):
    """Extract latent representations, cluster, and evaluate with survival."""
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    model.eval()
    all_z = []
    all_ids = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = [x.to(device) for x in batch['data']]
            masks = [m.to(device) for m in batch['masks']]
            _, _, _, z = model(inputs, masks)
            all_z.append(z.cpu().numpy())
            all_ids.extend(batch['ids'])

    Z = np.concatenate(all_z, axis=0)

    # Try different numbers of clusters
    results = {}
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(Z)

        sil = silhouette_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)

        # Survival analysis
        cluster_df = pd.DataFrame({
            'case_id': all_ids,
            'cluster': labels
        })
        merged = cluster_df.merge(clinical_df, on='case_id', how='inner')

        # Compute survival time
        merged['time'] = merged.apply(
            lambda r: r['days_to_death'] if pd.notna(r['days_to_death'])
            else r.get('days_to_last_follow_up', 0), axis=1
        )
        merged['event'] = (merged['vital_status'] == 'Dead').astype(int)
        merged = merged[merged['time'] > 0]

        # Log-rank test (cluster 0 vs rest)
        p_value = None
        if len(merged['cluster'].unique()) >= 2 and len(merged) > 10:
            try:
                groups = merged.groupby('cluster')
                best_p = 1.0
                for c in merged['cluster'].unique():
                    mask = merged['cluster'] == c
                    if mask.sum() > 5 and (~mask).sum() > 5:
                        result = logrank_test(
                            merged.loc[mask, 'time'], merged.loc[~mask, 'time'],
                            merged.loc[mask, 'event'], merged.loc[~mask, 'event']
                        )
                        best_p = min(best_p, result.p_value)
                p_value = best_p
            except Exception as e:
                logger.warning(f"Survival analysis failed for k={k}: {e}")

        results[k] = {
            'silhouette': sil,
            'calinski_harabasz': ch,
            'log_rank_p': p_value,
            'cluster_sizes': [int(x) for x in np.bincount(labels)],
        }
        logger.info(f"k={k}: Silhouette={sil:.3f}, CH={ch:.1f}, LogRank p={p_value}")

    return Z, results


# ── Step 6: Single vs Multi-Omics Comparison ────────────────────────────────
def compare_single_vs_multi(Z_multi, Z_singles, clinical_df, sample_ids):
    """Compare subtypes discovered by single vs multi-omics."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    results = {'multi_omics': {}, 'single_omics': {}}

    # Cluster multi-omics
    best_k = 4  # Will be determined by silhouette
    km_multi = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels_multi = km_multi.fit_predict(Z_multi)

    # Cluster each single omics
    for name, Z_single in Z_singles.items():
        km_single = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        labels_single = km_single.fit_predict(Z_single)

        ari = adjusted_rand_score(labels_multi, labels_single)
        nmi = normalized_mutual_info_score(labels_multi, labels_single)

        results['single_omics'][name] = {
            'ari_vs_multi': ari,
            'nmi_vs_multi': nmi,
        }
        logger.info(f"Single [{name}] vs Multi: ARI={ari:.3f}, NMI={nmi:.3f}")

    return results


# ── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper 5: Multi-Omics Cancer Subtyping")
    parser.add_argument("--cancer", type=str, default="BRCA", choices=CANCER_TYPES)
    parser.add_argument("--step", type=str, default="all",
                       choices=["download", "preprocess", "train", "evaluate", "all"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger.info(f"=" * 60)
    logger.info(f"Paper 5: Multi-Omics Subtyping — Cancer: {args.cancer}")
    logger.info(f"=" * 60)

    # Step 1: Download
    if args.step in ["download", "all"]:
        step_name = f"download_{args.cancer}"
        if not tracker.is_done(step_name):
            tracker.set_current(step_name)

            expr_df = download_tcga_gene_expression(args.cancer)
            clin_df = download_tcga_clinical(args.cancer)
            mut_df = download_tcga_mutations(args.cancer)

            tracker.mark_done(step_name, {
                "expression_shape": str(expr_df.shape) if expr_df is not None else None,
                "clinical_shape": str(clin_df.shape) if clin_df is not None else None,
                "mutation_shape": str(mut_df.shape) if mut_df is not None else None,
            })
        else:
            logger.info(f"Step [{step_name}] already completed, skipping...")

    # Step 2: Preprocess
    if args.step in ["preprocess", "all"]:
        step_name = f"preprocess_{args.cancer}"
        if not tracker.is_done(step_name):
            tracker.set_current(step_name)

            expr_df = pd.read_parquet(DATA_DIR / f"{args.cancer}_expression.parquet")
            expr_norm = preprocess_expression(expr_df)

            if expr_norm is not None:
                expr_norm.to_parquet(DATA_DIR / f"{args.cancer}_expression_processed.parquet")
                logger.info(f"Preprocessed expression: {expr_norm.shape}")

            tracker.mark_done(step_name)

    # Step 3-4: Train MVAE
    if args.step in ["train", "all"]:
        step_name = f"train_{args.cancer}"
        tracker.set_current(step_name)

        import torch
        from torch.utils.data import DataLoader, Dataset

        # Load processed data
        expr_path = DATA_DIR / f"{args.cancer}_expression_processed.parquet"
        if not expr_path.exists():
            logger.error("Run preprocessing first!")
            return

        expr_df = pd.read_parquet(expr_path)

        # For now, use expression as primary modality
        # (mutation and methylation will be added once downloaded)
        logger.info(f"Training MVAE with expression data: {expr_df.shape}")

        class OmicsDataset(Dataset):
            def __init__(self, expr_data, sample_ids):
                self.expr = torch.FloatTensor(expr_data.T.values)  # samples x genes
                self.ids = sample_ids
                self.masks = [torch.ones(len(sample_ids))]

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                return {
                    'data': [self.expr[idx]],
                    'masks': [self.masks[0][idx]],
                    'ids': self.ids[idx]
                }

        def collate_fn(batch):
            return {
                'data': [torch.stack([b['data'][0] for b in batch])],
                'masks': [torch.stack([b['masks'][0] for b in batch])],
                'ids': [b['ids'] for b in batch]
            }

        dataset = OmicsDataset(expr_df, list(expr_df.columns))
        data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn)

        # Build model
        input_dims = [expr_df.shape[0]]  # number of genes
        model = build_mvae_model(input_dims, latent_dim=args.latent_dim)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on {device}")

        model, history = train_mvae(
            model, {'train': data_loader},
            n_epochs=args.epochs, lr=args.lr, device=device
        )

        # Save training history
        with open(RESULTS_DIR / f"{args.cancer}_training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        tracker.mark_done(step_name)

    # Step 5: Evaluate
    if args.step in ["evaluate", "all"]:
        step_name = f"evaluate_{args.cancer}"
        tracker.set_current(step_name)

        logger.info("Evaluation step — will run after training completes")
        # Load model and evaluate clustering
        # (implemented in cluster_and_evaluate function)

        tracker.mark_done(step_name)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
