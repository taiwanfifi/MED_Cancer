#!/usr/bin/env python3
"""
Download Drug-Gene Interaction Data from Multiple Sources
==========================================================
Sources: DGIdb API, DrugBank (open data), OpenTargets, PharmGKB
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path("/workspace/cancer_research/data/drug_repurpose")
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def download_dgidb_via_api():
    """Download from DGIdb GraphQL API."""
    output_file = DATA_DIR / "dgidb_interactions.parquet"
    if output_file.exists():
        logger.info("DGIdb already downloaded")
        return pd.read_parquet(output_file)

    logger.info("Downloading from DGIdb GraphQL API...")

    # DGIdb GraphQL endpoint
    url = "https://dgidb.org/api/graphql"

    # Get cancer-relevant genes first
    cancer_genes = [
        "TP53", "KRAS", "BRAF", "EGFR", "PIK3CA", "PTEN", "RB1",
        "BRCA1", "BRCA2", "MYC", "ERBB2", "ALK", "RET", "MET",
        "FGFR1", "FGFR2", "FGFR3", "CDK4", "CDK6", "CCND1",
        "MDM2", "JAK2", "ABL1", "KIT", "PDGFRA", "ROS1",
        "NRAS", "AKT1", "MTOR", "IDH1", "IDH2",
        "NTRK1", "NTRK2", "NTRK3", "VHL", "NF1", "NF2",
        "CDKN2A", "STK11", "SMAD4", "ATM", "CHEK2",
        "MLH1", "MSH2", "ARID1A", "FBXW7", "KEAP1",
        "APC", "CTNNB1", "SMO", "PTCH1", "WNT",
        "BCL2", "BAX", "VEGFA", "VEGFR2", "PDGFRB",
        "SRC", "RAF1", "MEK1", "ERK1", "ERK2",
        "NOTCH1", "NOTCH2", "HIF1A", "TERT",
        "AR", "ESR1", "PGR", "ERBB3", "ERBB4",
        "FLT3", "NPM1", "DNMT3A", "TET2", "ASXL1",
        "EZH2", "SUZ12", "SETD2", "KDM6A", "KMT2A",
        "CREBBP", "EP300", "BRD4", "DOT1L",
        "PD1", "PDL1", "CTLA4", "LAG3", "TIM3",
        "TIGIT", "CD47", "SIRPA", "CD19", "CD20",
        "HDAC1", "HDAC2", "HDAC3", "HDAC6",
        "PARP1", "PARP2", "ATR", "WEE1", "CHK1",
    ]

    all_interactions = []

    # Query in batches
    batch_size = 20
    for i in range(0, len(cancer_genes), batch_size):
        batch = cancer_genes[i:i+batch_size]

        query = """
        query($genes: [String!]!) {
            genes(names: $genes) {
                nodes {
                    name
                    longName
                    interactions {
                        nodes {
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
                            publications {
                                pmid
                            }
                            sources {
                                fullName
                            }
                        }
                    }
                }
            }
        }
        """

        try:
            response = requests.post(
                url,
                json={"query": query, "variables": {"genes": batch}},
                timeout=60,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                genes = data.get("data", {}).get("genes", {}).get("nodes", [])

                for gene in genes:
                    gene_name = gene.get("name", "")
                    for interaction in gene.get("interactions", {}).get("nodes", []):
                        drug_info = interaction.get("drug", {})
                        int_types = interaction.get("interactionTypes", [])

                        all_interactions.append({
                            "gene": gene_name,
                            "drug": drug_info.get("name", ""),
                            "drug_concept_id": drug_info.get("conceptId", ""),
                            "approved": drug_info.get("approved", False),
                            "anti_neoplastic": drug_info.get("antiNeoplastic", False),
                            "interaction_score": interaction.get("interactionScore", 0),
                            "interaction_type": int_types[0].get("type", "") if int_types else "",
                            "directionality": int_types[0].get("directionality", "") if int_types else "",
                            "n_publications": len(interaction.get("publications", [])),
                            "sources": ", ".join([s.get("fullName", "") for s in interaction.get("sources", [])]),
                        })

                logger.info(f"Batch {i//batch_size + 1}: {len(genes)} genes, running total: {len(all_interactions)} interactions")
            else:
                logger.warning(f"API error: {response.status_code}")
                # Try REST API fallback
                break

        except Exception as e:
            logger.warning(f"GraphQL error: {e}")
            break

    if all_interactions:
        df = pd.DataFrame(all_interactions)
        df.to_parquet(output_file)
        logger.info(f"DGIdb saved: {df.shape}")
        logger.info(f"  Unique drugs: {df['drug'].nunique()}")
        logger.info(f"  Unique genes: {df['gene'].nunique()}")
        logger.info(f"  Approved drugs: {df['approved'].sum()}")
        logger.info(f"  Anti-neoplastic: {df['anti_neoplastic'].sum()}")
        return df

    logger.warning("DGIdb download failed - trying REST API...")
    return download_dgidb_rest_fallback()


def download_dgidb_rest_fallback():
    """Fallback: use DGIdb REST API."""
    output_file = DATA_DIR / "dgidb_interactions.parquet"

    cancer_genes = ["EGFR", "BRAF", "KRAS", "TP53", "PIK3CA", "PTEN",
                    "BRCA1", "BRCA2", "ALK", "RET", "MET", "KIT",
                    "ERBB2", "FGFR1", "FGFR2", "CDK4", "CDK6",
                    "BCL2", "VEGFA", "MTOR", "JAK2", "ABL1",
                    "PDGFRA", "FLT3", "IDH1", "IDH2"]

    all_interactions = []

    for gene in tqdm(cancer_genes, desc="DGIdb REST"):
        try:
            url = f"https://dgidb.org/api/v2/interactions.json?genes={gene}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for term in data.get("matchedTerms", []):
                    gene_name = term.get("geneName", gene)
                    for ix in term.get("interactions", []):
                        all_interactions.append({
                            "gene": gene_name,
                            "drug": ix.get("drugName", ""),
                            "interaction_type": ix.get("interactionTypes", ""),
                            "interaction_score": ix.get("score", 0),
                            "sources": ", ".join(ix.get("sources", [])),
                            "n_publications": ix.get("pmids", 0) if isinstance(ix.get("pmids"), int) else len(ix.get("pmids", [])),
                        })
        except Exception as e:
            logger.warning(f"REST API error for {gene}: {e}")

    if all_interactions:
        df = pd.DataFrame(all_interactions)
        df.to_parquet(output_file)
        logger.info(f"DGIdb REST saved: {df.shape}")
        return df

    return None


def download_opentargets_cancer_drugs():
    """Download cancer drug-target data from Open Targets."""
    output_file = DATA_DIR / "opentargets_cancer_drugs.parquet"
    if output_file.exists():
        logger.info("OpenTargets data already downloaded")
        return pd.read_parquet(output_file)

    logger.info("Downloading OpenTargets cancer drug data...")

    # Open Targets Platform GraphQL API
    url = "https://api.platform.opentargets.org/api/v4/graphql"

    # Cancer disease IDs (EFO)
    cancer_efo_ids = [
        "EFO_0000311",  # cancer
        "EFO_0000305",  # breast carcinoma
        "EFO_0001071",  # lung cancer
        "EFO_0000182",  # hepatocellular carcinoma
        "EFO_0000756",  # melanoma
        "EFO_0000564",  # lymphoma
        "EFO_0000574",  # neuroblastoma
        "EFO_0000616",  # ovarian cancer
    ]

    all_drugs = []

    for disease_id in cancer_efo_ids:
        query = """
        query($diseaseId: String!) {
            disease(efoId: $diseaseId) {
                name
                knownDrugs(size: 500) {
                    rows {
                        drug {
                            name
                            id
                            mechanismsOfAction {
                                rows {
                                    actionType
                                    targets {
                                        approvedSymbol
                                    }
                                }
                            }
                        }
                        phase
                        status
                        ctIds
                    }
                }
            }
        }
        """

        try:
            response = requests.post(
                url,
                json={"query": query, "variables": {"diseaseId": disease_id}},
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                disease = data.get("data", {}).get("disease", {})
                disease_name = disease.get("name", "")
                drugs = disease.get("knownDrugs", {}).get("rows", [])

                for drug_entry in drugs:
                    drug = drug_entry.get("drug", {})
                    moas = drug.get("mechanismsOfAction", {}).get("rows", [])

                    targets = []
                    action_types = []
                    for moa in moas:
                        action_types.append(moa.get("actionType", ""))
                        for t in moa.get("targets", []):
                            targets.append(t.get("approvedSymbol", ""))

                    all_drugs.append({
                        "disease": disease_name,
                        "disease_id": disease_id,
                        "drug_name": drug.get("name", ""),
                        "drug_id": drug.get("id", ""),
                        "phase": drug_entry.get("phase", 0),
                        "status": drug_entry.get("status", ""),
                        "targets": "|".join(set(targets)),
                        "action_types": "|".join(set(action_types)),
                        "n_trials": len(drug_entry.get("ctIds", [])),
                    })

                logger.info(f"  {disease_name}: {len(drugs)} drug entries")

        except Exception as e:
            logger.warning(f"OpenTargets error for {disease_id}: {e}")

    if all_drugs:
        df = pd.DataFrame(all_drugs)
        df.to_parquet(output_file)
        logger.info(f"OpenTargets saved: {df.shape}")
        logger.info(f"  Unique drugs: {df['drug_name'].nunique()}")
        logger.info(f"  Phase 4 (approved): {(df['phase'] == 4).sum()}")
        logger.info(f"  Phase 3: {(df['phase'] == 3).sum()}")
        return df

    return None


def download_depmap_gene_effect():
    """Download DepMap CRISPR gene effect scores."""
    output_file = DATA_DIR / "depmap_gene_effect.parquet"
    if output_file.exists():
        logger.info("DepMap gene effect already downloaded")
        return

    logger.info("Downloading DepMap CRISPR gene effect data...")

    # DepMap Public 23Q4 - Gene Effect (CERES)
    # This is the key dataset for synthetic lethality analysis
    urls_to_try = [
        # DepMap 24Q2
        "https://ndownloader.figshare.com/files/46489425",
        # DepMap 23Q4
        "https://ndownloader.figshare.com/files/34990036",
    ]

    for url in urls_to_try:
        try:
            logger.info(f"Trying DepMap download: {url}")
            response = requests.get(url, timeout=600, stream=True)
            if response.status_code == 200:
                # Save raw file first
                raw_file = DATA_DIR / "depmap_gene_effect_raw.csv"
                total_size = int(response.headers.get('content-length', 0))

                with open(raw_file, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            pct = downloaded / total_size * 100
                            if downloaded % (50*1024*1024) < 1024*1024:
                                logger.info(f"  Downloaded {downloaded/(1024*1024):.0f}MB / {total_size/(1024*1024):.0f}MB ({pct:.0f}%)")

                # Convert to parquet
                logger.info("Converting to parquet...")
                df = pd.read_csv(raw_file, index_col=0)
                df.to_parquet(output_file)
                raw_file.unlink()  # Remove CSV to save space
                logger.info(f"DepMap gene effect: {df.shape}")
                return
        except Exception as e:
            logger.warning(f"DepMap download failed: {e}")

    logger.error("Could not download DepMap data")


def main():
    logger.info("=" * 60)
    logger.info("Drug & Gene Interaction Data Download")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)

    # 1. DGIdb
    dgidb_df = download_dgidb_via_api()

    # 2. OpenTargets
    ot_df = download_opentargets_cancer_drugs()

    # 3. DepMap
    download_depmap_gene_effect()

    logger.info("All downloads complete!")


if __name__ == "__main__":
    main()
