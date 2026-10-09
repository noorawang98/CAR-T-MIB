# Clinical GSEA in perivascular niches

- pseudobulk_adjtime_gsea: The model is `~ group + tp`; response levels are SD/PD then PR, and tp levels are Pre then Post.
- The coefficient is groupPR. The final GSEA reads `pervimib_de_<tag>_adjTime.csv`, ranks by the limma log2 fold change, and uses clusterProfiler::GSEA with human Hallmark, GO:BP, KEGG_LEGACY and Reactome sets (minGSSize=10, maxGSSize=500; seed 1). Positive NES corresponds to PR-associated enrichment; negative NES corresponds to SD/PD-associated enrichment.
