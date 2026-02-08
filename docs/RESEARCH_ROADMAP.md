# Cancer Research Roadmap — TMU MDE Lab
**PI**: Prof. Yang | **Started**: 2026-02-08
**Goal**: 10+ top-tier publications toward understanding and overcoming cancer
**Compute**: NVIDIA RTX 5880 Ada (48GB VRAM)

---

## Overall Strategy

We leverage Prof. Yang's pharmacovigilance/drug safety expertise + computational genomics + deep learning to attack cancer from multiple angles. Each paper builds on shared infrastructure (TCGA data, models, pipelines). The approach is: **data-driven discovery → hypothesis generation → validation → publication**.

---

## Paper Portfolio (10 Papers)

### TRACK A: Drug Discovery & Safety (connects to Prof. Yang's expertise)

#### Paper 1: Pan-Cancer Drug Repurposing via Graph Neural Networks
- **Target**: Nature Communications / npj Precision Oncology
- **Hypothesis**: GNNs integrating gene expression profiles + drug-gene interaction networks can identify repurposable drugs for cancers where no targeted therapy exists
- **Data**: TCGA (33 cancer types), DGIdb, DrugBank, GDSC
- **Method**: Build heterogeneous graph (gene-drug-disease), train GNN to predict drug-cancer sensitivity
- **Novel angle**: Integrate FDA adverse event data (FAERS) to prioritize SAFE repurposed drugs
- **Status**: 🔴 Not started

#### Paper 2: Drug Combination Synergy Prediction with Safety Constraints
- **Target**: Briefings in Bioinformatics / Cancer Research
- **Hypothesis**: Deep learning can predict synergistic anti-cancer drug combinations while accounting for cumulative toxicity
- **Data**: DrugComb, ALMANAC, NCI-60, FAERS
- **Method**: Multi-modal fusion (drug structure + gene expression + clinical AE profiles)
- **Novel angle**: Safety-constrained optimization (not just efficacy, but safety-efficacy Pareto frontier)
- **Status**: 🔴 Not started

#### Paper 3: Adverse Event Prediction for Cancer Immunotherapy
- **Target**: JAMA Oncology / Annals of Oncology
- **Hypothesis**: Patient genomic profiles can predict severe immune-related adverse events (irAEs) before treatment
- **Data**: TCGA + immunotherapy clinical data, FAERS, cBioPortal
- **Method**: Multi-task learning (predict both response AND irAE risk)
- **Novel angle**: Pharmacovigilance signal detection + genomics = precision immunotherapy
- **Status**: 🔴 Not started

### TRACK B: Genomic Discovery

#### Paper 4: Pan-Cancer Synthetic Lethality Map via Foundation Models
- **Target**: Nature Medicine / Cancer Cell
- **Hypothesis**: Large-scale CRISPR screen data + genomic foundation models can predict novel synthetic lethal gene pairs across cancer types
- **Data**: DepMap/Achilles (CRISPR), TCGA mutations, Cancer Cell Line Encyclopedia
- **Method**: Fine-tune genomic foundation model (Geneformer) for SL prediction
- **Novel angle**: Cross-cancer transfer learning of SL relationships
- **Status**: 🔴 Not started

#### Paper 5: Multi-Omics Cancer Subtype Discovery
- **Target**: Cell Reports Medicine / Nature Cancer
- **Hypothesis**: Integrating mutation + expression + methylation + CNV reveals clinically actionable subtypes missed by single-omics
- **Data**: TCGA multi-omics (33 cancer types), METABRIC
- **Method**: Multi-modal variational autoencoder (MVAE) + survival analysis
- **Novel angle**: Pharmacogenomic stratification (subtypes predict drug response)
- **Status**: 🔴 Not started

#### Paper 6: Cancer Gene Regulatory Network Master Regulators
- **Target**: Genome Biology / Cell Systems
- **Hypothesis**: Network analysis of cancer gene regulation can identify master regulators as therapeutic targets
- **Data**: TCGA, GTEx (normal), GRNdb, ENCODE
- **Method**: GENIE3/GRNBoost2 + differential network analysis (cancer vs normal)
- **Novel angle**: Druggability scoring of master regulators
- **Status**: 🔴 Not started

### TRACK C: AI/ML for Cancer (connects to LLM expertise)

#### Paper 7: Cancer Treatment LLM with Safety Guardrails
- **Target**: npj Digital Medicine / Lancet Digital Health
- **Hypothesis**: LLMs fine-tuned on cancer guidelines + safety guardrails outperform general medical LLMs in treatment recommendation
- **Data**: NCCN guidelines, OncoKB, cancer clinical trial data
- **Method**: RAG + fine-tuning + AESOP guardrail (from our existing work)
- **Novel angle**: Apply our AESOP framework to oncology-specific CDSS
- **Status**: 🔴 Not started

#### Paper 8: Pathology Foundation Model for Cancer Grading
- **Target**: Nature Medicine / The Lancet Oncology
- **Hypothesis**: Self-supervised vision transformer trained on histopathology images can achieve expert-level cancer grading
- **Data**: TCGA diagnostic slides, CAMELYON, PathDT
- **Method**: ViT/DINO pre-training → cancer-type-specific fine-tuning
- **Novel angle**: Uncertainty quantification for clinical deployment
- **Status**: 🔴 Not started

### TRACK D: Biomarker & Precision Medicine

#### Paper 9: Liquid Biopsy Biomarker Panel via Machine Learning
- **Target**: Clinical Cancer Research / JCO Precision Oncology
- **Hypothesis**: ML can optimize cfDNA/ctDNA biomarker panels for multi-cancer early detection
- **Data**: cfDNA datasets, GRAIL/Galleri-like public data, methylation arrays
- **Method**: Feature selection + ensemble learning for multi-cancer classification
- **Novel angle**: Cost-effectiveness optimization (minimum panel for maximum detection)
- **Status**: 🔴 Not started

#### Paper 10: Immunotherapy Response Prediction from Tumor Microenvironment
- **Target**: Nature Immunology / Cancer Immunology Research
- **Hypothesis**: Deep characterization of tumor microenvironment (TME) composition predicts immunotherapy response better than TMB/PD-L1 alone
- **Data**: TCGA, single-cell atlases, immunotherapy clinical cohorts
- **Method**: Cell deconvolution (CIBERSORTx) + deep learning integration
- **Novel angle**: TME-based patient stratification for combination immunotherapy
- **Status**: 🔴 Not started

---

## Execution Priority

| Priority | Paper | Reason |
|----------|-------|--------|
| 1st | Paper 5 (Multi-Omics Subtyping) | Foundation data + quick results, TCGA well-established |
| 2nd | Paper 1 (Drug Repurposing GNN) | Connects to Prof. Yang's drug safety expertise |
| 3rd | Paper 4 (Synthetic Lethality) | High impact, DepMap data is rich |
| 4th | Paper 10 (TME Immunotherapy) | Hot topic, strong clinical impact |
| 5th | Paper 2 (Drug Combo Synergy) | Builds on Paper 1 infrastructure |
| 6th | Paper 7 (Cancer LLM) | Builds on existing AESOP work |
| 7th | Paper 6 (Gene Reg Networks) | Builds on TCGA data from Paper 5 |
| 8th | Paper 3 (irAE Prediction) | Needs immunotherapy data |
| 9th | Paper 9 (Liquid Biopsy) | Specialized data needed |
| 10th | Paper 8 (Pathology) | Large image data, longer training |

---

## Phase 1 (Current): Infrastructure + Paper 5 Data
- [x] Server connected (RTX 5880, 48GB)
- [ ] Install PyTorch + bio packages
- [ ] Download TCGA multi-omics data
- [ ] Paper 5: Initial data exploration
- [ ] Paper 1: Download drug-gene interaction data

---

## Key Hypotheses Log

### H1: Multi-omics integration reveals hidden cancer subtypes
- **Rationale**: Single-omics misses cross-layer interactions (e.g., methylation silencing of mutant gene)
- **Testable**: Compare clustering from single vs multi-omics, validate with survival analysis
- **If true**: Each new subtype could guide different treatment strategies

### H2: Drug safety profiles can guide repurposing prioritization
- **Rationale**: Many repurposing studies ignore safety; FAERS data can pre-filter dangerous candidates
- **Testable**: Compare repurposing candidates ranked by efficacy-only vs efficacy+safety
- **If true**: Safer clinical trials, faster translation

### H3: Synthetic lethality is cancer-type transferable
- **Rationale**: SL in one cancer type might work in another with similar driver mutations
- **Testable**: Train on common cancers, test on rare cancers
- **If true**: Therapeutic options for rare cancers where trial data is minimal

### H4: TME composition is a better predictor than TMB alone
- **Rationale**: TMB is noisy; TME captures immune context directly
- **Testable**: Head-to-head comparison on immunotherapy cohorts
- **If true**: Better patient selection for immunotherapy

---

## Dependencies
- PyTorch + CUDA 12.8
- scikit-learn, pandas, scipy, matplotlib, seaborn
- Bioinformatics: scanpy, anndata, pydeseq2, lifelines
- Graph ML: torch_geometric, dgl
- Genomics: gseapy, mygene, pybiomart
- Data: GDC API (Python)

## Checkpoint & Resume Protocol
All experiments save:
1. Config/hyperparameters → `checkpoints/<paper_id>/config.json`
2. Model weights → `checkpoints/<paper_id>/model_best.pt`
3. Results → `results/<paper_id>/`
4. Logs → `logs/<paper_id>/`
5. Progress → `checkpoints/<paper_id>/progress.json`
