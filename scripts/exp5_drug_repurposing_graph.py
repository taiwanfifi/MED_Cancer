#!/usr/bin/env python3
"""
Experiment 5: Drug Repurposing via Graph Neural Network
========================================================
Builds a heterogeneous drug-gene-disease knowledge graph and trains
a GNN model to predict new drug-gene interactions for cancer therapy.

Data Sources:
- DGIdb: Drug-gene interactions
- OpenTargets: Disease-drug-gene associations
- DepMap SL pairs: Gene-gene functional relationships
- TCGA DEGs: Cancer-specific gene importance

Target Paper: Paper 1 (Pan-Cancer Drug Repurposing GNN) — Nature Communications

Architecture: HeteroGNN with drug/gene/disease nodes, multiple edge types
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path("/workspace/cancer_research")
DATA_DIR = BASE_DIR / "data" / "drug_repurpose"
SL_DIR = BASE_DIR / "results" / "exp2_synthetic_lethality"
DEG_DIR = BASE_DIR / "results" / "exp3_differential_expression"
RESULTS_DIR = BASE_DIR / "results" / "exp5_drug_repurposing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_dgidb():
    """Load drug-gene interactions from DGIdb."""
    f = DATA_DIR / "dgidb_interactions.parquet"
    if not f.exists():
        logger.warning("DGIdb data not found")
        return None
    df = pd.read_parquet(f)
    logger.info(f"DGIdb: {len(df)} interactions, {df['gene'].nunique()} genes, {df['drug'].nunique()} drugs")
    return df


def load_opentargets():
    """Load disease-drug-gene data from OpenTargets."""
    f = DATA_DIR / "opentargets_cancer_drugs.parquet"
    if not f.exists():
        logger.warning("OpenTargets data not found")
        return None
    df = pd.read_parquet(f)
    logger.info(f"OpenTargets: {len(df)} entries, {df['drug_name'].nunique()} drugs, "
               f"{df['disease'].nunique()} diseases")
    return df


def load_sl_pairs():
    """Load synthetic lethality pairs from Exp 2."""
    f = SL_DIR / "synthetic_lethal_pairs.csv"
    if not f.exists():
        logger.warning("SL pairs not found")
        return None
    df = pd.read_csv(f)
    logger.info(f"SL pairs: {len(df)}, {df['driver_gene'].nunique()} drivers, "
               f"{df['target_gene'].nunique()} targets")
    return df


def build_knowledge_graph(dgidb, opentargets, sl_pairs):
    """
    Build a heterogeneous knowledge graph with:
    - Node types: drug, gene, disease
    - Edge types: drug-targets-gene, drug-treats-disease, gene-SL-gene,
                  gene-interacts-gene, gene-associated-disease
    """
    # Collect all entities
    drugs = set()
    genes = set()
    diseases = set()
    edges = []

    # 1. Drug-Gene interactions from DGIdb
    if dgidb is not None:
        for _, row in dgidb.iterrows():
            drug = str(row.get('drug', '')).strip().upper()
            gene = str(row.get('gene', '')).strip()
            if drug and gene and drug != 'NAN' and gene != 'NAN':
                drugs.add(drug)
                genes.add(gene)
                int_type = str(row.get('interaction_type', 'unknown'))
                edges.append({
                    'source': drug, 'source_type': 'drug',
                    'target': gene, 'target_type': 'gene',
                    'edge_type': f'drug_{int_type}_gene',
                    'weight': float(row.get('interaction_score', 1.0)) if pd.notna(row.get('interaction_score')) else 1.0,
                    'source_db': 'DGIdb',
                })

    # 2. Disease-Drug associations from OpenTargets
    if opentargets is not None:
        for _, row in opentargets.iterrows():
            drug = str(row.get('drug_name', '')).strip().upper()
            disease = str(row.get('disease', '')).strip()
            phase = int(row.get('phase', 0))

            if drug and disease:
                drugs.add(drug)
                diseases.add(disease)
                edges.append({
                    'source': drug, 'source_type': 'drug',
                    'target': disease, 'target_type': 'disease',
                    'edge_type': 'drug_treats_disease',
                    'weight': phase / 4.0,  # Normalize by max phase
                    'source_db': 'OpenTargets',
                })

            # Drug-Gene (via targets)
            targets_str = str(row.get('targets', ''))
            if targets_str and targets_str != 'nan':
                for gene in targets_str.split('|'):
                    gene = gene.strip()
                    if gene:
                        genes.add(gene)
                        edges.append({
                            'source': drug, 'source_type': 'drug',
                            'target': gene, 'target_type': 'gene',
                            'edge_type': 'drug_targets_gene',
                            'weight': phase / 4.0,
                            'source_db': 'OpenTargets',
                        })

                        # Gene-Disease association
                        edges.append({
                            'source': gene, 'source_type': 'gene',
                            'target': disease, 'target_type': 'disease',
                            'edge_type': 'gene_associated_disease',
                            'weight': 1.0,
                            'source_db': 'OpenTargets',
                        })

    # 3. Gene-Gene SL pairs from DepMap
    if sl_pairs is not None:
        for _, row in sl_pairs.iterrows():
            driver = str(row['driver_gene']).strip()
            target = str(row['target_gene']).strip()
            if driver and target:
                genes.add(driver)
                genes.add(target)
                edges.append({
                    'source': driver, 'source_type': 'gene',
                    'target': target, 'target_type': 'gene',
                    'edge_type': 'gene_sl_gene',
                    'weight': abs(float(row.get('delta_effect', 0.5))),
                    'source_db': 'DepMap',
                })

    logger.info(f"\nKnowledge Graph Statistics:")
    logger.info(f"  Nodes: {len(drugs)} drugs, {len(genes)} genes, {len(diseases)} diseases")
    logger.info(f"  Total nodes: {len(drugs) + len(genes) + len(diseases)}")
    logger.info(f"  Total edges: {len(edges)}")

    # Edge type breakdown
    edge_type_counts = defaultdict(int)
    for e in edges:
        edge_type_counts[e['edge_type']] += 1
    for et, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {et}: {count}")

    # Create node index mapping
    drug_list = sorted(drugs)
    gene_list = sorted(genes)
    disease_list = sorted(diseases)

    node_index = {}
    for i, d in enumerate(drug_list):
        node_index[('drug', d)] = i
    offset = len(drug_list)
    for i, g in enumerate(gene_list):
        node_index[('gene', g)] = offset + i
    offset += len(gene_list)
    for i, dis in enumerate(disease_list):
        node_index[('disease', dis)] = offset + i

    # Save graph
    graph_data = {
        'n_drugs': len(drug_list),
        'n_genes': len(gene_list),
        'n_diseases': len(disease_list),
        'n_edges': len(edges),
        'drug_list': drug_list,
        'gene_list': gene_list,
        'disease_list': disease_list,
        'edge_type_counts': dict(edge_type_counts),
    }

    edges_df = pd.DataFrame(edges)
    edges_df.to_parquet(RESULTS_DIR / "knowledge_graph_edges.parquet")

    with open(RESULTS_DIR / "knowledge_graph_stats.json", 'w') as f:
        json.dump(graph_data, f, indent=2)

    return graph_data, edges_df, node_index


def build_pytorch_geometric_graph(edges_df, node_index):
    """Convert to PyTorch Geometric HeteroData format with reverse edges."""
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        logger.warning("PyTorch Geometric not available, skipping GNN model")
        return None

    data = HeteroData()

    # Count unique nodes per type
    n_drugs = len([k for k in node_index if k[0] == 'drug'])
    n_genes = len([k for k in node_index if k[0] == 'gene'])
    n_diseases = len([k for k in node_index if k[0] == 'disease'])

    # Create node features (simple one-hot or random for now)
    feat_dim = 64
    torch.manual_seed(42)
    data['drug'].x = torch.randn(n_drugs, feat_dim)
    data['gene'].x = torch.randn(n_genes, feat_dim)
    data['disease'].x = torch.randn(n_diseases, feat_dim)

    # Create edge indices for each edge type
    edge_groups = defaultdict(lambda: {'src': [], 'dst': [], 'weight': []})

    for _, row in edges_df.iterrows():
        src_key = (row['source_type'], row['source'])
        dst_key = (row['target_type'], row['target'])

        if src_key not in node_index or dst_key not in node_index:
            continue

        src_type = row['source_type']
        dst_type = row['target_type']

        # Simplify edge types
        if 'sl' in row['edge_type']:
            edge_name = 'synthetic_lethal'
        elif 'target' in row['edge_type']:
            edge_name = 'targets'
        elif 'treat' in row['edge_type']:
            edge_name = 'treats'
        elif 'associated' in row['edge_type']:
            edge_name = 'associated_with'
        else:
            edge_name = 'interacts'

        key = (src_type, edge_name, dst_type)

        # Use type-local indices
        if src_type == 'drug':
            src_idx = node_index[src_key]
        elif src_type == 'gene':
            src_idx = node_index[src_key] - n_drugs
        else:
            src_idx = node_index[src_key] - n_drugs - n_genes

        if dst_type == 'drug':
            dst_idx = node_index[dst_key]
        elif dst_type == 'gene':
            dst_idx = node_index[dst_key] - n_drugs
        else:
            dst_idx = node_index[dst_key] - n_drugs - n_genes

        edge_groups[key]['src'].append(src_idx)
        edge_groups[key]['dst'].append(dst_idx)
        edge_groups[key]['weight'].append(row['weight'])

    for (src_type, edge_name, dst_type), group in edge_groups.items():
        edge_index = torch.tensor([group['src'], group['dst']], dtype=torch.long)
        data[src_type, edge_name, dst_type].edge_index = edge_index
        data[src_type, edge_name, dst_type].edge_attr = torch.tensor(
            group['weight'], dtype=torch.float
        ).unsqueeze(1)
        logger.info(f"  Edge ({src_type})-[{edge_name}]->({dst_type}): {edge_index.shape[1]}")

    # Add reverse edges so ALL node types receive messages
    # (drug nodes are only sources without this; disease nodes only targets)
    forward_keys = list(edge_groups.keys())
    for (src_type, edge_name, dst_type) in forward_keys:
        if src_type == dst_type:
            continue  # self-loops (gene-gene) already bidirectional enough
        rev_key = (dst_type, f'rev_{edge_name}', src_type)
        if rev_key not in edge_groups:
            fwd = data[src_type, edge_name, dst_type]
            rev_edge_index = torch.stack([fwd.edge_index[1], fwd.edge_index[0]])
            data[dst_type, f'rev_{edge_name}', src_type].edge_index = rev_edge_index
            data[dst_type, f'rev_{edge_name}', src_type].edge_attr = fwd.edge_attr.clone()
            logger.info(f"  Edge ({dst_type})-[rev_{edge_name}]->({src_type}): {rev_edge_index.shape[1]} (reverse)")

    logger.info(f"\nHeteroData: {data}")
    return data


def train_gnn_model(data, epochs=100):
    """Train a HeteroGNN for link prediction (drug-gene interaction)."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv, to_hetero
    except ImportError:
        logger.warning("PyTorch Geometric not available")
        return None

    class GNN(torch.nn.Module):
        def __init__(self, hidden_dim=64, out_dim=32):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_dim)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=0.3)
            self.conv2 = SAGEConv((-1, -1), out_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return x

    # Convert to hetero model (reverse edges ensure all node types get messages)
    model = GNN()
    model = to_hetero(model, data.metadata(), aggr='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Simple training loop with edge prediction
    # For drug-gene link prediction, we create positive and negative samples
    if ('drug', 'targets', 'gene') not in data.edge_types and \
       ('drug', 'interacts', 'gene') not in data.edge_types:
        logger.warning("No drug-gene edges found for training")
        return None

    # Find drug-gene edge type
    dg_edge_type = None
    for et in data.edge_types:
        if et[0] == 'drug' and et[2] == 'gene':
            dg_edge_type = et
            break

    if dg_edge_type is None:
        logger.warning("No drug-gene edge type found")
        return None

    edge_index = data[dg_edge_type].edge_index
    n_edges = edge_index.shape[1]
    n_drugs = data['drug'].x.shape[0]
    n_genes = data['gene'].x.shape[0]

    logger.info(f"\nTraining GNN for drug-gene link prediction")
    logger.info(f"  Edge type: {dg_edge_type}")
    logger.info(f"  Positive edges: {n_edges}")

    # Train/test split
    import torch
    perm = torch.randperm(n_edges)
    n_train = int(0.8 * n_edges)
    train_mask = perm[:n_train]
    test_mask = perm[n_train:]

    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)

        # Positive samples
        drug_emb = out['drug']
        gene_emb = out['gene']

        pos_src = edge_index[0, train_mask]
        pos_dst = edge_index[1, train_mask]
        pos_score = (drug_emb[pos_src] * gene_emb[pos_dst]).sum(dim=-1)

        # Negative samples (random)
        neg_src = torch.randint(0, n_drugs, (len(train_mask),))
        neg_dst = torch.randint(0, n_genes, (len(train_mask),))
        neg_score = (drug_emb[neg_src] * gene_emb[neg_dst]).sum(dim=-1)

        # Binary cross entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            # Evaluate on test
            model.eval()
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict)
                drug_emb = out['drug']
                gene_emb = out['gene']

                test_pos_score = (drug_emb[edge_index[0, test_mask]] *
                                 gene_emb[edge_index[1, test_mask]]).sum(dim=-1)
                test_neg_src = torch.randint(0, n_drugs, (len(test_mask),))
                test_neg_dst = torch.randint(0, n_genes, (len(test_mask),))
                test_neg_score = (drug_emb[test_neg_src] * gene_emb[test_neg_dst]).sum(dim=-1)

                # AUC
                labels = torch.cat([torch.ones(len(test_mask)), torch.zeros(len(test_mask))])
                scores = torch.cat([test_pos_score, test_neg_score])

                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels.numpy(), scores.sigmoid().numpy())

            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Test AUC={auc:.4f}")

    return model, losses, out


def predict_novel_interactions(model, data, graph_data, top_k=100):
    """Predict novel drug-gene interactions for drug repurposing."""
    try:
        import torch
    except ImportError:
        return None

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        drug_emb = out['drug']  # (n_drugs, dim)
        gene_emb = out['gene']  # (n_genes, dim)

    # Score all drug-gene pairs
    scores = torch.mm(drug_emb, gene_emb.t()).sigmoid()  # (n_drugs, n_genes)

    # Get existing edges to exclude
    existing = set()
    for et in data.edge_types:
        if et[0] == 'drug' and et[2] == 'gene':
            ei = data[et].edge_index
            for i in range(ei.shape[1]):
                existing.add((ei[0, i].item(), ei[1, i].item()))

    # Find top novel predictions
    predictions = []
    score_flat = scores.numpy().flatten()
    top_indices = np.argsort(score_flat)[::-1]

    n_drugs = len(graph_data['drug_list'])
    n_genes = len(graph_data['gene_list'])

    for idx in top_indices:
        drug_idx = idx // n_genes
        gene_idx = idx % n_genes

        if (drug_idx, gene_idx) in existing:
            continue

        predictions.append({
            'drug': graph_data['drug_list'][drug_idx],
            'gene': graph_data['gene_list'][gene_idx],
            'score': float(score_flat[idx]),
            'drug_idx': int(drug_idx),
            'gene_idx': int(gene_idx),
        })

        if len(predictions) >= top_k:
            break

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(RESULTS_DIR / "novel_drug_gene_predictions.csv", index=False)
    logger.info(f"\nTop {top_k} novel drug-gene predictions saved")
    logger.info(f"Top 20 predictions:")
    for _, row in pred_df.head(20).iterrows():
        logger.info(f"  {row['drug']} → {row['gene']} (score={row['score']:.4f})")

    return pred_df


def plot_graph_statistics(graph_data, edges_df):
    """Visualize knowledge graph statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Node type distribution
    ax = axes[0, 0]
    node_types = ['Drugs', 'Genes', 'Diseases']
    node_counts = [graph_data['n_drugs'], graph_data['n_genes'], graph_data['n_diseases']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(node_types, node_counts, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, node_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', fontweight='bold')
    ax.set_title('A) Node Type Distribution', fontweight='bold')
    ax.set_ylabel('Count')

    # Panel B: Edge type distribution
    ax = axes[0, 1]
    edge_counts = graph_data['edge_type_counts']
    edge_names = list(edge_counts.keys())
    edge_vals = list(edge_counts.values())
    # Shorten names
    short_names = [n.replace('drug_', 'D-').replace('gene_', 'G-').replace('_gene', '-G')
                     .replace('_disease', '-Dis')[:25] for n in edge_names]
    ax.barh(short_names, edge_vals, color=plt.cm.Set2(np.linspace(0, 1, len(edge_names))))
    ax.set_xlabel('Number of Edges')
    ax.set_title('B) Edge Type Distribution', fontweight='bold')

    # Panel C: Drug degree distribution
    ax = axes[1, 0]
    drug_degree = edges_df[edges_df['source_type'] == 'drug'].groupby('source').size()
    ax.hist(drug_degree.values, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Degree (number of connections)')
    ax.set_ylabel('Number of Drugs')
    ax.set_title('C) Drug Degree Distribution', fontweight='bold')
    ax.set_yscale('log')

    # Panel D: Gene degree distribution
    ax = axes[1, 1]
    gene_degree_src = edges_df[edges_df['source_type'] == 'gene'].groupby('source').size()
    gene_degree_dst = edges_df[edges_df['target_type'] == 'gene'].groupby('target').size()
    gene_degree = gene_degree_src.add(gene_degree_dst, fill_value=0)
    ax.hist(gene_degree.values, bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Degree (number of connections)')
    ax.set_ylabel('Number of Genes')
    ax.set_title('D) Gene Degree Distribution', fontweight='bold')
    ax.set_yscale('log')

    plt.suptitle('Drug Repurposing Knowledge Graph', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "knowledge_graph_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: knowledge_graph_overview.png")


def plot_training_curve(losses):
    """Plot GNN training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, color='#e74c3c', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GNN Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gnn_training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: gnn_training_loss.png")


def main():
    logger.info("=" * 60)
    logger.info("Experiment 5: Drug Repurposing via GNN")
    logger.info("=" * 60)

    # Load data sources
    dgidb = load_dgidb()
    opentargets = load_opentargets()
    sl_pairs = load_sl_pairs()

    if dgidb is None and opentargets is None:
        logger.error("No drug interaction data available!")
        return

    # Build knowledge graph
    graph_data, edges_df, node_index = build_knowledge_graph(dgidb, opentargets, sl_pairs)

    # Visualize graph
    plot_graph_statistics(graph_data, edges_df)

    # Build PyG graph and train GNN
    pyg_data = build_pytorch_geometric_graph(edges_df, node_index)

    if pyg_data is not None:
        model, losses, embeddings = train_gnn_model(pyg_data, epochs=100)

        if model is not None:
            plot_training_curve(losses)
            predictions = predict_novel_interactions(model, pyg_data, graph_data, top_k=100)

            # Cross-reference with SL targets
            if predictions is not None and sl_pairs is not None:
                sl_genes = set(sl_pairs['target_gene'].unique())
                sl_predictions = predictions[predictions['gene'].isin(sl_genes)]
                if len(sl_predictions) > 0:
                    logger.info(f"\n*** REPURPOSING HITS targeting SL genes: ***")
                    for _, row in sl_predictions.head(20).iterrows():
                        logger.info(f"  {row['drug']} → {row['gene']} (SL target, score={row['score']:.4f})")
                    sl_predictions.to_csv(RESULTS_DIR / "sl_drug_repurposing_hits.csv", index=False)

    # Save summary
    summary = {
        'experiment': 'Exp 5: Drug Repurposing GNN',
        'graph_stats': {
            'n_drugs': graph_data['n_drugs'],
            'n_genes': graph_data['n_genes'],
            'n_diseases': graph_data['n_diseases'],
            'n_edges': graph_data['n_edges'],
            'edge_types': graph_data['edge_type_counts'],
        },
    }

    with open(RESULTS_DIR / "exp5_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 5 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
