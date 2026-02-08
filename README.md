# MED_Cancer: Pan-Cancer Synthetic Lethality & Precision Oncology Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Experiments](https://img.shields.io/badge/Experiments-24%20completed-green.svg)](#experiment-overview)
[![Papers](https://img.shields.io/badge/Papers-15%20planned-orange.svg)](#publication-plan)

**A computational framework for discovering synthetic lethal targets, drug repurposing opportunities, and precision oncology strategies across 33 cancer types, integrating TCGA, DepMap, DGIdb, and OpenTargets data.**

---

## Abstract

Cancer treatment faces a fundamental challenge: tumor heterogeneity limits single-target therapies. **Synthetic lethality (SL)** — where simultaneous loss of two genes is lethal while individual loss is viable — offers a path to exploit tumor-specific vulnerabilities without harming normal cells. The clinical success of PARP inhibitors in BRCA-mutant cancers (Bryant et al., *Nature*, 2005) proved this concept, but systematic pan-cancer SL mapping remains incomplete.

This project conducts **24 computational experiments** spanning SL discovery, drug repurposing, immune microenvironment analysis, ferroptosis pathway mapping, epigenetic vulnerability profiling, network medicine, and prognostic signature development. We integrate multi-omics data from **11,428 patients** (TCGA), **1,086 cell lines** (DepMap CRISPR screens), **5,826 drug-gene interactions** (DGIdb), and independent validation cohorts (GEO) to build a comprehensive precision oncology framework.

Key outputs include **126 novel SL targets**, a **6-gene prognostic signature validated in LUAD** (C-index = 0.734, p = 6.8×10⁻⁴), identification of **GPX4 as a pan-cancer ferroptosis vulnerability**, and actionable drug recommendations for **61% of TCGA patients** through mutation-SL-drug mapping.

---

## Key Breakthroughs

### 1. FGFR1 — Top Multi-Evidence Therapeutic Target
FGFR1 ranks #1 across all evidence dimensions: 35 approved drugs, dependency prediction AUC = 0.835, strong differential expression and SL evidence. FGFR inhibitors (Infigratinib, Nintedanib) emerge as strong pan-cancer candidates.
> **Experiments**: 2, 5, 7, 10, 12, 13

### 2. GPX4 — Novel Pan-Cancer SL Target via Ferroptosis
GPX4 (glutathione peroxidase 4) is synthetically lethal with **5 major oncogenic drivers** (BRAF, CTNNB1, EGFR, ERBB2, VHL) with **no approved drugs** targeting it. GPX4 inhibition induces ferroptosis — iron-dependent cell death. PTEN-mutant cancers show extreme vulnerability to glutathione pathway disruption (Cohen's d = −2.00 for GSS, p < 10⁻¹⁹).
> **Experiments**: 2, 7, 15 | **Significance**: Novel druggable vulnerability across multiple cancer types

### 3. SWI/SNF → PRC2 Epigenetic Rewiring
25 significant SWI/SNF–PRC2 rewiring pairs confirmed in DepMap data. EZH2 shows 5 rewiring partners from the SWI/SNF complex, supporting Tazemetostat (FDA-approved EZH2 inhibitor) use in SWI/SNF-mutant cancers. BRD4 is essential in 93% of cell lines — a near-universal vulnerability.
> **Experiments**: 20 | **Reference**: Kadoch & Crabtree, *Nature Genetics* (2013)

### 4. 6-Gene SL-Derived Prognostic Signature
LASSO Cox regression yields a 6-gene signature (BRAF, KIF14, CDK6, ORC6, CHAF1B, MAD2L1) with cross-validated C-index = 0.733. **Independent LUAD validation: C-index = 0.734, log-rank p = 6.76×10⁻⁴.** All genes are SL targets or pan-cancer differentially expressed genes from our analyses — the first SL-derived prognostic signature with clinical validation.
> **Experiments**: 22, 17

### 5. Precision Oncology at Scale — 61% of Patients Targetable
Mutation → SL target → approved drug mapping across 1,577 TCGA patients shows **965 patients (61%)** have SL-targetable mutations, and **210 (13%)** have druggable SL targets with existing approved therapies. BCL2 upregulation in immune-cold breast cancer (FDR = 0.028) suggests Venetoclax + checkpoint inhibitor combinations for cold tumors.
> **Experiments**: 14, 19

---

## Theoretical Foundation & Key References

This work builds on established principles in cancer genomics and synthetic lethality:

| Concept | Key Reference | Relevance |
|---------|--------------|-----------|
| Synthetic lethality in cancer | Hartwell et al., *Science* (1997) | Foundational concept for our SL discovery |
| BRCA-PARP synthetic lethality | Bryant et al., *Nature* (2005) | Clinical proof-of-concept we extend pan-cancer |
| SL in precision medicine | Lord & Ashworth, *Science* (2017) | Comprehensive SL review informing our strategy |
| DepMap CRISPR screens | Tsherniak et al., *Cell* (2017) | Primary data source for gene dependency |
| Ferroptosis in cancer | Dixon et al., *Cell* (2012) | Basis for GPX4 and PTEN vulnerability findings |
| SWI/SNF–PRC2 antagonism | Kadoch & Crabtree, *Nat. Genet.* (2013) | Foundation for epigenetic rewiring (Exp 20) |
| Immune checkpoint therapy | Ribas & Wolchok, *Science* (2018) | Context for immune-SL combination strategies |
| TCGA pan-cancer atlas | Hoadley et al., *Cell* (2018) | Multi-omics data framework |
| SL computational methods | De Kegel et al., *Nat. Commun.* (2024) | ML benchmarks for SL prediction |
| SL transcriptional buffering | SYLVER, *Nat. Genet.* (2025) | Complementary computational SL approach |

---

## Hypotheses Tested

| ID | Hypothesis | Status | Key Evidence |
|----|-----------|--------|-------------|
| H1 | SL targets from DepMap predict patient survival | Partially supported | 6-gene signature C=0.734 in LUAD; KIRC not significant |
| H2 | GPX4 is a pan-cancer SL vulnerability | **Supported** | SL with 5 drivers, no drugs, ferroptosis pathway |
| H3 | SL pairs transfer across cancer lineages | **Supported** | 671 universal pairs, 10.3% mean transfer rate |
| H4 | PTEN mutations create ferroptosis vulnerability | **Strongly supported** | GSH pathway genes d = −1.5 to −2.0, p < 10⁻¹⁹ |
| H5 | Immune TME composition predicts SL sensitivity | Partially supported | BCL2 in immune-cold BRCA (FDR=0.028) |
| H6 | SWI/SNF loss creates PRC2 dependency | **Supported** | 25 rewiring pairs, EZH2 with 5 SWI/SNF partners |
| H7 | Drug synergy can be predicted from SL co-dependency | Supported | 146/159 pairs show negative co-dependency |
| H8 | GNN can predict novel drug-gene interactions | Supported | AUC = 0.99 on held-out test set |

---

## Repository Structure

```
MED_Cancer/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── PROGRESS_LOG.md                # Detailed experiment log with all results
├── RESEARCH_REPORT.md             # Comprehensive research report (Chinese)
├── docs/
│   ├── IDEAS_AND_HYPOTHESES.md    # Research hypotheses and next steps
│   ├── RESEARCH_ROADMAP.md        # Long-term research plan
│   ├── HONEST_ASSESSMENT.md       # Candid assessment of strengths/limitations
│   └── DATASETS_GUIDE.md          # Data sources and download instructions
├── scripts/                       # All experiment scripts (Python)
│   ├── exp0_data_exploration.py   # TCGA/DepMap/DGIdb data survey
│   ├── exp1_pan_cancer_survival.py
│   ├── ...                        # exp2 through exp23
│   ├── download_dgidb_v2.py       # DGIdb v5.0 GraphQL downloader
│   ├── setup_server.sh            # Environment setup
│   └── run_all_pending.sh         # Batch experiment runner
└── results/                       # All experiment outputs
    ├── exp0_data_exploration/     # 62 figures (PNG), 70 data files (CSV/JSON)
    ├── exp1_survival/
    ├── ...                        # exp2 through exp23
    └── exp23_pan_cancer_atlas/
```

---

## Experiment Overview

| Exp | Title | Method | Key Finding |
|-----|-------|--------|-------------|
| 0 | Data Exploration | Descriptive statistics | TCGA: 33 projects, 11,428 cases |
| 1 | Pan-Cancer Survival | Cox regression, KM curves | 18 significant survival associations, THCA age HR=11.6 |
| 2 | Synthetic Lethality Discovery | DepMap co-dependency | **126 SL targets**, KRAS–mitochondria link |
| 3 | Differential Expression | DESeq2-style analysis | BRCA: 3,570 DEGs; LUAD: 3,162 DEGs |
| 4 | SL Transferability | Cross-lineage validation | **671 universal SL pairs**, 10.3% transfer rate |
| 5 | Drug Repurposing GNN | HeteroSAGE link prediction | AUC=0.99, 100 novel drug predictions |
| 6 | Immune Microenvironment | ssGSEA deconvolution + Cox | T_reg protective in BRCA (HR=0.35) |
| 7 | Multi-Omics Integration | Composite scoring | **GPX4 = top SL hub**, 10 actionable targets |
| 8 | Deep Learning Survival | DeepSurv, RSF, Cox-PH | Clinical features dominate at TCGA scale |
| 9 | Biomarker Panel | LASSO stability selection | MammaPrint best for KIRC (C=0.71) |
| 10 | Drug Sensitivity | Gene dependency prediction | MDM2 AUC=0.88, PTEN 0.87, FGFR1 0.84 |
| 11 | Outlier Analysis | Survival outlier profiling | Immune profiles distinguish exceptional survivors |
| 12 | Cross-Experiment Integration | Multi-evidence scoring | **FGFR1 = #1 target** (4,750 genes scored) |
| 13 | Literature Validation | PubMed mining | Spearman ρ=0.50 (comp. vs lit.), 45 novel targets |
| 14 | Mutation→SL→Drug Mapping | Precision oncology pipeline | **61% patients SL-targetable**, 13% druggable |
| 15 | Ferroptosis Deep Dive | Pathway-focused SL | **PTEN→GSH vulnerability** (d=−2.00) |
| 16 | Patient Clustering | Consensus clustering | LUAD k=3 subtypes (survival p=0.070) |
| 17 | External Validation | GEO cohort validation | GSE72094: 5/12 genes validated (42%) |
| 18 | Drug Synergy | Co-dependency prediction | BRCA1-MYC top synergy (score=0.746) |
| 19 | Immune-SL Combination | Immune stratification + SL | **BCL2↑ in immune-cold BRCA** (FDR=0.028) |
| 20 | Epigenetic Vulnerability | Chromatin rewiring | **25 SWI/SNF→PRC2 pairs**, BRD4 93% essential |
| 21 | Network Medicine | Pathway-level SL network | RAS-MAPK = central SL hub (score=2280) |
| 22 | Prognostic Signature | LASSO Cox signature | **6-gene sig, LUAD C=0.734, p=6.8×10⁻⁴** |
| 23 | Pan-Cancer Atlas | Cross-cancer integration | Pan-cancer SL landscape visualization |

---

## How to Reproduce

### Prerequisites

```bash
# Python 3.11+ required
pip install numpy pandas scipy scikit-learn matplotlib seaborn lifelines
pip install torch torch_geometric  # For GNN experiments (Exp 5)
pip install gseapy mygene          # For pathway/gene annotation
```

### Data Download

All raw data must be downloaded separately (not included due to size):

```bash
# TCGA expression and clinical data (via GDC API)
python scripts/download_tcga_expression.py

# DGIdb drug-gene interactions (via GraphQL API)
python scripts/download_dgidb_v2.py

# DepMap gene effect data
# Download from https://depmap.org/portal/download/
# Place CRISPRGeneEffect.csv in data/depmap/
```

See [`docs/DATASETS_GUIDE.md`](docs/DATASETS_GUIDE.md) for complete download instructions.

### Running Experiments

```bash
# Run individual experiments
python scripts/exp2_synthetic_lethality.py
python scripts/exp15_ferroptosis_deep_dive.py

# Run all pending experiments
bash scripts/run_all_pending.sh
```

Results are saved to `results/exp{N}_{name}/` with CSV data files, JSON summaries, and PNG figures.

---

## Data Sources & Citations

| Dataset | Source | Access | Citation |
|---------|--------|--------|----------|
| TCGA | [GDC Data Portal](https://portal.gdc.cancer.gov/) | Open | TCGA Research Network (2013–2018) |
| DepMap | [Broad Institute](https://depmap.org/portal/) | Open | Tsherniak et al., *Cell* (2017) |
| DGIdb v5.0 | [dgidb.org](https://www.dgidb.org/) | Open | Freshour et al., *NAR* (2021) |
| OpenTargets | [opentargets.org](https://www.opentargets.org/) | Open | Ochoa et al., *NAR* (2023) |
| GEO GSE20685 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) | Open | Kao et al., *BMC Cancer* (2011) |
| GEO GSE72094 | [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) | Open | Schabath et al., *JNCI* (2016) |

---

## Publication Plan

| # | Title | Experiments | Target Journal | Status |
|---|-------|------------|----------------|--------|
| 1 | Pan-Cancer Drug Repurposing via GNN | 5, 7, 10, 12 | Nature Communications | Data ready |
| 2 | Pan-Cancer Synthetic Lethality Map | 2, 4 | Cell | Data ready |
| 3 | SL Map + Pathway Enrichment | 2, 3, 4 | Nature Genetics | Data ready |
| 4 | Multi-Omics Precision Oncology | 3, 6, 7, 12 | Nature Medicine | Data ready |
| 5 | AI Biomarker Panel + Prognosis | 1, 8, 9 | Lancet Digital Health | Data ready |
| 6 | GPX4 Ferroptosis as Pan-Cancer Vulnerability | 2, 7 | Cancer Cell | Data ready |
| 7 | TME + Survival Outliers | 6, 11 | Nature Immunology | Data ready |
| 8 | PTEN–Ferroptosis Vulnerability | 2, 15, 17 | Cancer Cell | Data ready |
| 9 | Drug Combination & Synergy Prediction | 2, 5, 10, 12, 18 | Cancer Discovery | Data ready |
| 10 | Pan-Cancer Therapeutic Target Atlas | 12, 14 | Cancer Cell | Data ready |
| 11 | Immune-SL Combination Strategy | 6, 11, 19 | Nature Immunology | Data ready |
| 12 | Epigenetic Vulnerability Mapping | 20 | Molecular Cell | Data ready |
| 13 | Precision Oncology Pipeline | 14, 16, 17 | Nature Medicine | Data ready |
| 14 | Network Medicine / Pathway SL | 21, 2, 4 | Cell Systems | Data ready |
| 15 | SL-Based Prognostic Signature | 22, 17 | Clinical Cancer Research | Data ready, **validated** |

---

## Team & Acknowledgments

**Taipei Medical University — Medical Data Engineering Lab**
- **Principal Investigator**: Prof. Yang (Pharmacovigilance, Drug Safety, CDSS)

Computational experiments performed on NVIDIA RTX 5880 Ada GPU (48GB VRAM). All analyses use publicly available datasets (TCGA, DepMap, DGIdb, OpenTargets, GEO).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions about the code, data, or collaboration opportunities, please open an [issue](https://github.com/taiwanfifi/MED_Cancer/issues) on this repository.
