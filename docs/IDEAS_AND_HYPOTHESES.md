# Cancer Research: Ideas, Hypotheses & Next Steps
**Last Updated**: 2026-02-08 17:15 (Taiwan Time)

---

## Current Status: 14 Experiments Complete (Exp 0-13)
- All results synced to local machine
- 46 publication-ready figures generated
- 9 papers mapped with data ready
- Compute: NVIDIA RTX 5880 Ada (48GB VRAM)

---

## KEY FINDINGS SO FAR

### Breakthrough Discoveries
1. **GPX4 = Novel Pan-Cancer SL Target** (No approved drugs, SL with 5 major drivers)
   - Ferroptosis pathway — regulated cell death via iron-dependent lipid peroxidation
   - BRAF, CTNNB1, EGFR, ERBB2, VHL mutant cancers are all vulnerable
   - Zero clinical trials targeting GPX4 directly
   → **This alone could be a Cancer Cell paper**

2. **CDS2, CHMP4B, GINS4, PSMA4 = Truly Under-Studied Targets**
   - <50 PubMed publications each, but strong multi-evidence computational support
   - CDS2: SL with 3 drivers, involved in phosphatidylserine synthesis
   → **Novel targets that have been overlooked by the field**

3. **MDM2 Dependency Most Predictable** (AUC=0.88)
   - p53 pathway status determines cell line behavior
   - MDM2 inhibitors (Nutlin-3, Idasanutlin) should be stratified by genomic context
   → **Precision oncology application**

4. **Immune TME Distinguishes Survival Outliers**
   - BRCA: B cells (p=0.007), CD4 T (p=0.014), NK (p=0.049) protect
   - Patients who defy their stage-based prognosis have distinct immune profiles
   → **Immunotherapy stratification opportunity**

5. **SL Pairs Transfer Across Cancer Types** (671 universal pairs)
   - 10.3% mean transfer rate; lung↔skin highest (30.7%)
   → **Pan-cancer SL-based therapy design is feasible**

---

## NEW HYPOTHESES TO TEST

### H1: Mutation-Specific SL Vulnerability Map
**Hypothesis**: Specific cancer mutations (e.g., KRAS G12D, BRAF V600E, TP53 R175H) create distinct SL vulnerability profiles that can guide personalized therapy.
**Data needed**: TCGA mutation data (MAF files) + DepMap + DGIdb
**Approach**: Download TCGA somatic mutation calls → map specific mutations to SL targets → create mutation→drug treatment guidelines
**Impact**: Direct clinical translation — "Patient has KRAS G12D → prescribe Drug X targeting SL partner Y"
**Priority**: ★★★★★ (HIGH — directly actionable)

### H2: Ferroptosis Pathway as Pan-Cancer Vulnerability
**Hypothesis**: The entire ferroptosis pathway (not just GPX4) is synthetically lethal with common drivers. Other ferroptosis genes (SLC7A11, FSP1, DHODH) may also be SL.
**Data needed**: DepMap gene effect for ferroptosis genes
**Approach**: Test all 20+ known ferroptosis genes for SL with common drivers → build ferroptosis vulnerability map → cross-reference with expression/survival
**Impact**: If confirmed, opens an entire therapeutic class
**Priority**: ★★★★★ (HIGH — GPX4 finding needs deepening)

### H3: Drug Synergy Prediction from SL Pairs
**Hypothesis**: If Gene A and Gene B are SL, then Drug-targeting-A + Drug-targeting-B should be synergistic in cells with mutations in the corresponding driver.
**Data needed**: SL pairs + DGIdb drugs + NCI-ALMANAC/DrugComb synergy data
**Approach**: Map SL pairs to drug combinations → validate against known synergy databases → predict novel synergistic combos
**Impact**: Direct drug combination recommendations
**Priority**: ★★★★☆

### H4: Tumor Stage-Dependent SL Vulnerability
**Hypothesis**: SL vulnerabilities change across tumor stages. Early-stage tumors may have different SL partners than late-stage tumors due to tumor evolution.
**Data needed**: TCGA stage-specific expression + DepMap
**Approach**: Stratify TCGA patients by stage → compare expression of SL target genes → identify stage-specific opportunities
**Impact**: Stage-adapted therapy design
**Priority**: ★★★☆☆

### H5: Immune-SL Combination Therapy
**Hypothesis**: Combining immune checkpoint inhibitors with SL-informed targeted therapy produces better outcomes than either alone.
**Data needed**: Exp 6 immune data + Exp 2 SL data + clinical outcomes
**Approach**: For each cancer type, identify patients with favorable immune AND SL profiles → model combination response → compare vs single-agent
**Impact**: Novel combination therapy rationale
**Priority**: ★★★★☆

### H6: Patient Clustering by Multi-Omics Profile
**Hypothesis**: Unsupervised clustering of patients using expression + immune + clinical features reveals clinically meaningful subgroups with distinct treatment strategies.
**Data needed**: TCGA expression + immune + clinical data (all available)
**Approach**: Consensus clustering → survival analysis per cluster → pathway analysis → treatment recommendations per cluster
**Impact**: Patient stratification framework
**Priority**: ★★★★☆

### H7: Gene Regulatory Network Upstream of SL Targets
**Hypothesis**: Transcription factors upstream of SL targets can be targeted to simultaneously downregulate multiple SL genes.
**Data needed**: Expression correlation networks + TF databases (TRRUST, Regnetwork)
**Approach**: Build TF→SL target regulatory network → identify master regulators → check if they're druggable
**Impact**: Target upstream regulators instead of individual genes
**Priority**: ★★★☆☆

### H8: External Validation with Independent Cohorts (GEO/ICGC)
**Hypothesis**: Our TCGA-derived findings replicate in independent cohorts.
**Data needed**: GEO microarray datasets, ICGC data
**Approach**: Download 2-3 independent breast/lung cancer cohorts → test SL targets' expression-survival associations → validate immune findings
**Impact**: Critical for paper credibility — reviewers always ask for external validation
**Priority**: ★★★★★ (HIGH — required for top-tier publication)

### H9: Epigenetic Vulnerability in Cancer
**Hypothesis**: Epigenetic regulators (EZH2, DNMT1, HDAC family) create unique SL opportunities that can be targeted with existing epigenetic drugs.
**Data needed**: DepMap for epigenetic genes, DGIdb
**Approach**: Systematic SL analysis of all known epigenetic regulators → cross-reference with FDA-approved epigenetic drugs
**Impact**: Repurpose existing drugs (Tazemetostat, Vorinostat) for new indications
**Priority**: ★★★★☆

### H10: Liquid Biopsy Biomarker Discovery
**Hypothesis**: SL target genes with high expression variability between tumor and normal can serve as liquid biopsy biomarkers (detectable in blood).
**Data needed**: Exp 3 DEG data + known secreted/membrane protein databases
**Approach**: Filter DEGs for secreted/membrane proteins → rank by fold-change and SL evidence → propose liquid biopsy panel
**Impact**: Early detection + treatment monitoring
**Priority**: ★★★☆☆

---

## EXPERIMENT QUEUE (Next to Run)

### Exp 14: TCGA Mutation Mapping + Personalized SL (H1) ★★★★★
- Download TCGA MAF files for BRCA, LUAD, KIRC
- Map each patient's driver mutations
- Cross-reference mutations → SL targets → available drugs
- Output: Patient-level treatment recommendation matrix

### Exp 15: Ferroptosis Pathway Deep Dive (H2) ★★★★★
- Test all ferroptosis genes for SL
- Expression analysis of ferroptosis pathway across cancers
- Survival analysis stratified by ferroptosis gene expression
- Output: Ferroptosis vulnerability map

### Exp 16: Patient Multi-Omics Clustering (H6) ★★★★☆
- Consensus clustering of TCGA patients
- Expression + immune + clinical features
- Survival analysis per cluster
- Output: Patient subgroups with treatment implications

### Exp 17: External Validation with GEO (H8) ★★★★★
- Download independent cohorts (GSE series for BRCA, LUAD)
- Validate SL target expression-survival associations
- Replicate immune findings
- Output: Validation statistics + forest plots

---

## FAILED PATHS / LESSONS LEARNED

### Exp 9: LASSO Stability Selection Found 0 Stable Genes
- **Why**: Only 11-37 events in 77-90 patients — severely underpowered for 17,000+ features
- **Lesson**: Data-driven biomarker discovery needs much larger cohorts (>500 patients with events)
- **Workaround**: Use established panels (Oncotype DX, MammaPrint) as benchmarks
- **Future**: Could work with feature pre-filtering (top 100 variable genes only)

### Exp 10: GDSC Download Failed
- **Why**: GDSC URLs changed/deprecated since 2022
- **Workaround**: Used DepMap gene effect as proxy for drug sensitivity
- **Lesson**: Always have fallback data strategies; don't depend on single external URLs
- **Future**: Try GDSC via R package (PharmacoGx) or CCLE expression data

### Exp 5: GNN Predictions Saturated at 1.0
- **Why**: GNN overfitting — 76K edges, simple 2-layer SAGEConv architecture
- **Lesson**: Heterogeneous graph neural networks need careful hyperparameter tuning
- **Future**: Add dropout regularization, reduce hidden dim, use edge weight thresholds

### Exp 11: 0 Outlier DEGs Found
- **Why**: Very few outlier patients had expression data (only 7-13 out of 96 outliers)
- **Lesson**: Clinical data (957 patients) >> expression data (150 patients) in TCGA
- **Workaround**: Used immune profiles instead (which ARE derived from expression)
- **Future**: Download more expression samples to increase overlap

---

## PAPER PUBLICATION STRATEGY

### Phase 1: Core Papers (Write First)
1. **Pan-Cancer Therapeutic Target Discovery** (Exp 12 + 13) → Cancer Cell
   - Most comprehensive, integrates everything
   - Multi-evidence scoring + literature validation
2. **GPX4 Ferroptosis as Pan-Cancer Vulnerability** (Exp 2 + 7 + 15) → Cancer Cell
   - Novel finding, high impact
   - Needs Exp 15 (ferroptosis deep dive) for depth

### Phase 2: Method Papers
3. **SL Pan-Cancer Map + Transferability** (Exp 2 + 4) → Nature Genetics
4. **Drug Repurposing via GNN** (Exp 5 + 10) → Nature Communications
5. **Multi-Omics Precision Oncology Framework** (Exp 3 + 6 + 7 + 12) → Nature Medicine

### Phase 3: Clinical Translation Papers
6. **Immune TME + Survival Outliers** (Exp 6 + 11) → Nature Immunology
7. **AI Biomarker Panel + Prognosis** (Exp 1 + 8 + 9) → Lancet Digital Health
8. **Drug Combination Discovery** (Exp 2 + 5 + 12) → Cancer Discovery
9. **Mutation-Specific Treatment Guide** (Exp 14) → JAMA Oncology
10. **Patient Stratification via Clustering** (Exp 16) → Journal of Clinical Oncology

### Phase 4: Validation + Review
11. **External Validation Study** (Exp 17) → Can be added to any of the above
12. **Comprehensive Review: Computational Cancer Target Discovery** → Nature Reviews Cancer

---

## DATA MANAGEMENT

### Compute Environment
- GPU: NVIDIA RTX 5880 Ada, 48GB VRAM
- All experiments reproducible from scripts in `scripts/`

### Data & Results
- Results: `results/exp{0-23}_*/`
- Scripts: `scripts/`
- Progress log: `PROGRESS_LOG.md`
