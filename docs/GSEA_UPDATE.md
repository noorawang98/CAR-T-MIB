# Final clinical GSEA path confirmed by the author

Final path: pervimib_pseudobulk → pervimib_de_only → pseudobulk_adjtime_gsea. The model is `~ group + tp`; response levels are SD/PD then PR, and tp levels are Pre then Post. The coefficient is groupPR. The final GSEA reads `pervimib_de_<tag>_adjTime.csv`, ranks by the limma log2 fold change, and uses clusterProfiler::GSEA with human Hallmark, GO:BP, KEGG_LEGACY and Reactome sets (minGSSize=10, maxGSSize=500; seed 1). Positive NES corresponds to PR-associated enrichment; negative NES corresponds to SD/PD-associated enrichment.

The former selected pervimib_enrich branch reads unadjusted DE tables and is not the final sampling-time-adjusted GSEA. It is retained only as a separate enrichment branch.

## Source correction affecting result validity

The supplied pseudobulk_adjtime_gsea sorted the numeric statistics and then assigned the original unsorted gene names. This can mismatch genes and ranking statistics. The release now retains each statistic's name through filtering, duplicate removal and sorting. The console's negative-NES=PR label was also corrected to match the groupPR coefficient. The group+Pre/Post model is unchanged; the ranking column was updated from t to logFC following the author’s explicit confirmation.

Only source/static and synthetic name-pairing checks were performed. Real results were not rerun. Existing Figure 6G/S7G results cannot be certified from this correction alone; check whether their actual generating code contained the source defect.

The author additionally confirmed log2FC, rather than normalized expression such as log-transformed CP10K, as the final ranking metric. The originally archived script used d$t; the published source now explicitly reads d$logFC from the adjusted DE table. This is an author-directed metric update, not a claim that the archived source already used logFC.
