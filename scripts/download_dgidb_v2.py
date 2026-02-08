#!/usr/bin/env python3
"""Download DGIdb interactions via GraphQL API (v5.0 schema)."""
import requests
import json
import pandas as pd
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/workspace/cancer_research/data/drug_repurpose")

url = "https://dgidb.org/api/graphql"

cancer_genes = [
    "TP53", "KRAS", "BRAF", "EGFR", "PIK3CA", "PTEN", "RB1",
    "BRCA1", "BRCA2", "MYC", "ERBB2", "ALK", "RET", "MET",
    "FGFR1", "FGFR2", "FGFR3", "CDK4", "CDK6", "CCND1",
    "MDM2", "JAK2", "ABL1", "KIT", "PDGFRA", "ROS1",
    "NRAS", "AKT1", "MTOR", "IDH1", "IDH2",
    "BCL2", "VEGFA", "PARP1", "PARP2", "ATR", "WEE1",
    "HDAC1", "HDAC2", "HDAC6", "BRD4", "EZH2",
    "FLT3", "NPM1", "DNMT3A", "PDGFRB", "SRC",
    "PTCH1", "SMO", "NOTCH1", "HIF1A", "TERT",
    "AR", "ESR1", "CD274", "PDCD1", "CTLA4",
]

all_interactions = []
batch_size = 10

for i in range(0, len(cancer_genes), batch_size):
    batch = cancer_genes[i:i+batch_size]

    query = """
    query($genes: [String!]!) {
        genes(names: $genes) {
            nodes {
                name
                interactions {
                    drug {
                        name
                        conceptId
                        approved
                        antiNeoplastic
                    }
                    interactionScore
                    interactionTypes {
                        type
                        directionality
                    }
                }
            }
        }
    }
    """

    try:
        r = requests.post(
            url,
            json={"query": query, "variables": {"genes": batch}},
            timeout=60,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code == 200:
            data = r.json()
            if "errors" in data:
                logger.warning("GraphQL errors: %s", data["errors"][:1])
                continue
            genes = data.get("data", {}).get("genes", {}).get("nodes", [])
            for gene in genes:
                gene_name = gene.get("name", "")
                for ix in gene.get("interactions", []):
                    drug_info = ix.get("drug", {})
                    int_types = ix.get("interactionTypes", [])
                    all_interactions.append({
                        "gene": gene_name,
                        "drug": drug_info.get("name", ""),
                        "drug_concept_id": drug_info.get("conceptId", ""),
                        "approved": drug_info.get("approved", False),
                        "anti_neoplastic": drug_info.get("antiNeoplastic", False),
                        "interaction_score": ix.get("interactionScore", 0),
                        "interaction_type": int_types[0].get("type", "") if int_types else "",
                        "directionality": int_types[0].get("directionality", "") if int_types else "",
                    })
            logger.info("Batch %d/%d: %d genes, total: %d",
                       i // batch_size + 1, len(cancer_genes) // batch_size + 1,
                       len(genes), len(all_interactions))
        else:
            logger.warning("HTTP %d", r.status_code)
    except Exception as e:
        logger.warning("Error: %s", e)
    time.sleep(0.5)

if all_interactions:
    df = pd.DataFrame(all_interactions)
    df.to_parquet(DATA_DIR / "dgidb_interactions.parquet")
    logger.info("DGIdb saved: %s", df.shape)
    logger.info("  Unique genes: %d", df["gene"].nunique())
    logger.info("  Unique drugs: %d", df["drug"].nunique())
    logger.info("  Approved: %d", df["approved"].sum())
    logger.info("  Anti-neoplastic: %d", df["anti_neoplastic"].sum())
else:
    logger.error("No interactions downloaded!")
