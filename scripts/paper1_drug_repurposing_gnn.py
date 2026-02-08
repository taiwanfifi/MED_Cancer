#!/usr/bin/env python3
"""
Paper 1: Pan-Cancer Drug Repurposing via Graph Neural Networks
===============================================================
Builds a heterogeneous graph (gene-drug-disease) and uses GNN to predict
which existing drugs could be repurposed for specific cancer types.

Target: Nature Communications / npj Precision Oncology
Hypothesis: H5 - GNNs on drug-gene interaction networks can identify
            repurposable drugs with safety-awareness

Key Innovation: Integrate FDA adverse event data (FAERS) to prioritize
SAFE repurposed drugs, not just efficacious ones.

Experiment Flow:
1. Download drug-gene interaction data (DGIdb)
2. Download drug-disease data (DrugBank, PharmGKB)
3. Download cancer gene expression signatures (TCGA)
4. Build heterogeneous graph
5. Train GNN (HeteroGNN / RGCN)
6. Predict drug-cancer links
7. Validate against known oncology drugs
8. Rank by safety profile (FAERS integration)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "drug_repurpose"
RESULTS_DIR = BASE_DIR / "results" / "paper1"
CKPT_DIR = BASE_DIR / "checkpoints" / "paper1"
LOG_DIR = BASE_DIR / "logs" / "paper1"

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


# ── Step 1: Download DGIdb (Drug-Gene Interactions) ─────────────────────────
def download_dgidb():
    """Download drug-gene interactions from DGIdb API."""
    output_file = DATA_DIR / "dgidb_interactions.parquet"
    if output_file.exists():
        logger.info("DGIdb data already downloaded")
        return pd.read_parquet(output_file)

    logger.info("Downloading DGIdb drug-gene interactions...")

    # DGIdb API - get all interactions
    url = "https://dgidb.org/api/v2/interactions.json"
    params = {"count": 10000, "page": 1}

    all_interactions = []
    while True:
        try:
            response = requests.get(url, params=params, timeout=60)
            data = response.json()

            if "matchedTerms" in data:
                for term in data["matchedTerms"]:
                    gene = term.get("geneName", "")
                    for interaction in term.get("interactions", []):
                        all_interactions.append({
                            "gene": gene,
                            "drug": interaction.get("drugName", ""),
                            "interaction_type": interaction.get("interactionType", ""),
                            "source": interaction.get("source", ""),
                            "score": interaction.get("score", 0),
                        })

            # Check pagination
            if len(data.get("matchedTerms", [])) < params["count"]:
                break
            params["page"] += 1

        except Exception as e:
            logger.warning(f"DGIdb API error: {e}")
            break

    # Alternative: download TSV directly
    if not all_interactions:
        logger.info("Trying DGIdb TSV download...")
        tsv_url = "https://dgidb.org/data/monthly_tsvs/2024-Feb/interactions.tsv"
        try:
            df = pd.read_csv(tsv_url, sep='\t', low_memory=False)
            df = df.rename(columns={
                'gene_name': 'gene',
                'drug_name': 'drug',
                'interaction_group_score': 'score',
            })
            df.to_parquet(output_file)
            logger.info(f"DGIdb downloaded: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"TSV download failed: {e}")

    if all_interactions:
        df = pd.DataFrame(all_interactions)
        df.to_parquet(output_file)
        logger.info(f"DGIdb saved: {df.shape}")
        return df

    return None


# ── Step 2: Download Cancer Gene Census ─────────────────────────────────────
def download_cancer_genes():
    """Get cancer-associated genes from multiple sources."""
    output_file = DATA_DIR / "cancer_genes.parquet"
    if output_file.exists():
        logger.info("Cancer genes already downloaded")
        return pd.read_parquet(output_file)

    logger.info("Downloading cancer gene lists...")

    # Use COSMIC Cancer Gene Census (public subset)
    # and OncoKB cancer gene list
    cancer_genes = []

    # Method 1: Known cancer gene lists
    # Top cancer genes from literature (oncogenes + tumor suppressors)
    known_cancer_genes = {
        # Oncogenes
        'KRAS': 'oncogene', 'BRAF': 'oncogene', 'PIK3CA': 'oncogene',
        'EGFR': 'oncogene', 'HER2': 'oncogene', 'MYC': 'oncogene',
        'ALK': 'oncogene', 'RET': 'oncogene', 'MET': 'oncogene',
        'FGFR1': 'oncogene', 'FGFR2': 'oncogene', 'FGFR3': 'oncogene',
        'CDK4': 'oncogene', 'CDK6': 'oncogene', 'CCND1': 'oncogene',
        'MDM2': 'oncogene', 'JAK2': 'oncogene', 'ABL1': 'oncogene',
        'KIT': 'oncogene', 'PDGFRA': 'oncogene', 'ROS1': 'oncogene',
        'NRAS': 'oncogene', 'HRAS': 'oncogene', 'AKT1': 'oncogene',
        'MTOR': 'oncogene', 'SMO': 'oncogene', 'CTNNB1': 'oncogene',
        'IDH1': 'oncogene', 'IDH2': 'oncogene', 'NTRK1': 'oncogene',
        'NTRK2': 'oncogene', 'NTRK3': 'oncogene',
        # Tumor suppressors
        'TP53': 'tumor_suppressor', 'RB1': 'tumor_suppressor',
        'BRCA1': 'tumor_suppressor', 'BRCA2': 'tumor_suppressor',
        'APC': 'tumor_suppressor', 'PTEN': 'tumor_suppressor',
        'VHL': 'tumor_suppressor', 'NF1': 'tumor_suppressor',
        'NF2': 'tumor_suppressor', 'WT1': 'tumor_suppressor',
        'CDKN2A': 'tumor_suppressor', 'CDKN2B': 'tumor_suppressor',
        'STK11': 'tumor_suppressor', 'SMAD4': 'tumor_suppressor',
        'ATM': 'tumor_suppressor', 'CHEK2': 'tumor_suppressor',
        'MLH1': 'tumor_suppressor', 'MSH2': 'tumor_suppressor',
        'BAP1': 'tumor_suppressor', 'ARID1A': 'tumor_suppressor',
        'FBXW7': 'tumor_suppressor', 'KEAP1': 'tumor_suppressor',
    }

    for gene, role in known_cancer_genes.items():
        cancer_genes.append({'gene': gene, 'role': role, 'source': 'curated'})

    # Method 2: Try mygene API for expanded list
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        # Query for genes annotated with cancer
        results = mg.query('cancer', fields='symbol,name,type_of_gene', species='human', size=500)
        for hit in results.get('hits', []):
            symbol = hit.get('symbol', '')
            if symbol and symbol not in known_cancer_genes:
                cancer_genes.append({
                    'gene': symbol,
                    'role': 'cancer_associated',
                    'source': 'mygene'
                })
    except Exception as e:
        logger.warning(f"mygene query failed: {e}")

    df = pd.DataFrame(cancer_genes)
    df.to_parquet(output_file)
    logger.info(f"Cancer genes saved: {df.shape}")
    return df


# ── Step 3: Build Heterogeneous Graph ───────────────────────────────────────
def build_hetero_graph(dgidb_df, cancer_genes_df, expr_signatures=None):
    """Build drug-gene-disease heterogeneous graph for PyG."""
    import torch

    logger.info("Building heterogeneous graph...")

    # Node types: drug, gene, cancer_type
    # Edge types: drug-interacts-gene, gene-associated-cancer, drug-treats-cancer

    # Get unique entities
    drugs = sorted(dgidb_df['drug'].dropna().unique().tolist())
    genes = sorted(set(
        dgidb_df['gene'].dropna().unique().tolist() +
        cancer_genes_df['gene'].unique().tolist()
    ))

    drug_to_idx = {d: i for i, d in enumerate(drugs)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    logger.info(f"Nodes: {len(drugs)} drugs, {len(genes)} genes")

    # Edge 1: drug-gene interactions
    drug_gene_edges = []
    for _, row in dgidb_df.iterrows():
        if row['drug'] in drug_to_idx and row['gene'] in gene_to_idx:
            drug_gene_edges.append([drug_to_idx[row['drug']], gene_to_idx[row['gene']]])

    logger.info(f"Drug-gene edges: {len(drug_gene_edges)}")

    # Edge 2: gene-gene co-expression (from expression correlation)
    gene_gene_edges = []
    if expr_signatures is not None:
        # Top correlated gene pairs
        common_genes = [g for g in genes if g in expr_signatures.index]
        if len(common_genes) > 100:
            expr_sub = expr_signatures.loc[common_genes[:1000]]
            corr = expr_sub.T.corr()
            # Keep top correlations
            for i in range(len(corr)):
                for j in range(i+1, len(corr)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        g1, g2 = corr.index[i], corr.columns[j]
                        if g1 in gene_to_idx and g2 in gene_to_idx:
                            gene_gene_edges.append([gene_to_idx[g1], gene_to_idx[g2]])
                            gene_gene_edges.append([gene_to_idx[g2], gene_to_idx[g1]])

    logger.info(f"Gene-gene edges: {len(gene_gene_edges)}")

    # Save graph data
    graph_data = {
        'drugs': drugs,
        'genes': genes,
        'drug_to_idx': drug_to_idx,
        'gene_to_idx': gene_to_idx,
        'drug_gene_edges': drug_gene_edges,
        'gene_gene_edges': gene_gene_edges,
    }

    with open(DATA_DIR / "hetero_graph.pkl", 'wb') as f:
        import pickle
        pickle.dump(graph_data, f)

    logger.info("Heterogeneous graph saved")
    return graph_data


# ── Step 4: GNN Model ──────────────────────────────────────────────────────
def build_gnn_model(n_drugs, n_genes, embedding_dim=128):
    """Build Graph Neural Network for drug repurposing."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DrugRepurposingGNN(nn.Module):
        """
        Heterogeneous GNN for drug-gene interaction prediction.
        Uses message passing to learn drug and gene embeddings,
        then predicts drug-cancer links through gene expression signatures.
        """
        def __init__(self, n_drugs, n_genes, emb_dim=128, hidden_dim=256):
            super().__init__()

            # Node embeddings
            self.drug_emb = nn.Embedding(n_drugs, emb_dim)
            self.gene_emb = nn.Embedding(n_genes, emb_dim)

            # Message passing layers
            self.conv1_drug = nn.Linear(emb_dim, hidden_dim)
            self.conv1_gene = nn.Linear(emb_dim, hidden_dim)

            self.conv2_drug = nn.Linear(hidden_dim, hidden_dim)
            self.conv2_gene = nn.Linear(hidden_dim, hidden_dim)

            # Attention mechanism
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

            # Prediction head
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def message_pass(self, x_src, x_dst, edge_index, conv_layer):
            """Simple message passing: aggregate neighbor features."""
            src_idx, dst_idx = edge_index
            messages = x_src[src_idx]

            # Aggregate (mean)
            agg = torch.zeros_like(x_dst)
            count = torch.zeros(x_dst.shape[0], 1, device=x_dst.device)
            agg.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(messages), messages)
            count.scatter_add_(0, dst_idx.unsqueeze(1), torch.ones_like(dst_idx.unsqueeze(1).float()))
            count = count.clamp(min=1)
            agg = agg / count

            return F.relu(conv_layer(agg + x_dst))

        def forward(self, drug_gene_edges, gene_gene_edges=None):
            """Forward pass."""
            # Initial embeddings
            h_drug = self.drug_emb.weight
            h_gene = self.gene_emb.weight

            # Layer 1: Drug -> Gene message passing
            drug_gene_edge = torch.tensor(drug_gene_edges, dtype=torch.long).T
            if drug_gene_edge.shape[1] > 0:
                h_gene = self.message_pass(
                    h_drug, h_gene,
                    drug_gene_edge, self.conv1_gene
                )
                h_drug = self.message_pass(
                    h_gene, h_drug,
                    drug_gene_edge.flip(0), self.conv1_drug
                )

            # Layer 2
            if gene_gene_edges and len(gene_gene_edges) > 0:
                gg_edge = torch.tensor(gene_gene_edges, dtype=torch.long).T
                h_gene = self.message_pass(
                    h_gene, h_gene, gg_edge, self.conv2_gene
                )

            return h_drug, h_gene

        def predict_interaction(self, h_drug, h_gene, drug_idx, gene_idx):
            """Predict drug-gene interaction probability."""
            h = torch.cat([h_drug[drug_idx], h_gene[gene_idx]], dim=-1)
            return self.predictor(h)

    return DrugRepurposingGNN(n_drugs, n_genes, embedding_dim)


# ── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Paper 1: Drug Repurposing GNN")
    parser.add_argument("--step", type=str, default="all",
                       choices=["download", "build_graph", "train", "predict", "all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--emb_dim", type=int, default=128)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Paper 1: Pan-Cancer Drug Repurposing via GNN")
    logger.info("=" * 60)

    if args.step in ["download", "all"]:
        logger.info("Step 1: Downloading data...")
        dgidb_df = download_dgidb()
        cancer_genes_df = download_cancer_genes()

        if dgidb_df is not None:
            logger.info(f"DGIdb: {len(dgidb_df)} interactions, "
                       f"{dgidb_df['drug'].nunique()} drugs, "
                       f"{dgidb_df['gene'].nunique()} genes")

    if args.step in ["build_graph", "all"]:
        logger.info("Step 2: Building heterogeneous graph...")
        dgidb_df = pd.read_parquet(DATA_DIR / "dgidb_interactions.parquet")
        cancer_genes_df = pd.read_parquet(DATA_DIR / "cancer_genes.parquet")
        graph_data = build_hetero_graph(dgidb_df, cancer_genes_df)

    if args.step in ["train", "all"]:
        logger.info("Step 3: Training GNN...")
        import pickle
        with open(DATA_DIR / "hetero_graph.pkl", 'rb') as f:
            graph_data = pickle.load(f)

        model = build_gnn_model(
            len(graph_data['drugs']),
            len(graph_data['genes']),
            args.emb_dim
        )
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training loop would go here
        # (will implement after data download completes)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
