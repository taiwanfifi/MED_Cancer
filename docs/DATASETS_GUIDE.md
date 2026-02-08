# Cancer Research Datasets & Computational Trends Guide
**Generated**: 2026-02-08

## Key Datasets We Should Add (Not Yet Used)

### Priority 1 (High Impact)
1. **GDSC** (Genomics of Drug Sensitivity in Cancer) — IC50/AUC for hundreds of drugs × ~1,000 cell lines. Python: `GDSCTools`. Critical for Paper 1 (Drug Repurposing GNN) and Paper 2 (Drug Combo Synergy).
2. **NCI-ALMANAC** — 104 FDA-approved drugs × NCI-60 pairwise synergy (290,000 measurements). Essential for drug combination prediction.
3. **cBioPortal REST API** — Mutation frequency + clinical outcomes. Better mutation data than our DepMap gene-effect proxy for SL analysis.
4. **STRING API** — Protein-protein interaction networks. Add PPI edges to our knowledge graph (Exp 5 GNN).

### Priority 2 (Next Phase)
5. **DrugComb** — 437,923 drug combination experiments. For synergy prediction models.
6. **PharmGKB** — Pharmacogenomics annotations. Links to drug safety (Prof. Yang's expertise).
7. **AlphaFold API** — Protein structures for SL target druggability assessment.
8. **CancerSEA** — Single-cell functional states for TME analysis.

### Priority 3 (Future Papers)
9. **SEER API** — Population-level outcomes for validation.
10. **BioGRID** — 2.9M protein interactions for network analysis.

## Hottest Computational Directions (2024-2026)

1. **Foundation Models for Cancer Biology** — >200 published. TITAN (Nature Medicine 2025): WSI foundation model. Our Paper 8 (Pathology FM) aligns perfectly.
2. **Spatial Transcriptomics + scRNA-seq** — Explosive growth. TME spatial mapping, immunotherapy prediction. Our Paper 10 (TME) is well-positioned.
3. **GNN Drug Discovery** — Genesis/Genentech partnership. 83% success rate for pancreatic cancer drug combos (Nature Communications 2025). Our Paper 1 (GNN) is on-trend.
4. **Digital Twins / Virtual Cell Models** — npj Digital Medicine 2025. Multi-omics + deep generative models.
5. **AI Pathology** — DNA methylation classification >97% precision. GPT-4 matching pathologists.
6. **Precision Immunomodulation** — AI + small molecules for immune modulation.

## Action Items for Next Session
- [ ] Download GDSC drug sensitivity data → integrate with Exp 5 GNN
- [ ] Download NCI-ALMANAC → new experiment for drug synergy prediction
- [ ] Use cBioPortal API for actual mutation data → improve SL analysis (fix TP53 issue)
- [ ] Add STRING PPI edges to knowledge graph
- [ ] Consider `depmap-downloader` Python package for cleaner data access
