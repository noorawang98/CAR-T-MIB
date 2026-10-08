# Inputs and dependencies

The release reorganizes supplied source; it does not reconstruct missing author code or private data.

| Requirement | Used by |
|---|---|
| `src/common.py`, now supplied and bundled | Original sample metadata, paths, BH helper, `load_base`, legacy `grpr_labels` and `scissor_labels`. `AUTHOR_COMMON_DIR` is an optional override, not a required missing module. `grpr_labels()` reads def5 quartile labels; it does not return `GRPR_quad_M5`. |
| `Scissor5.R`, schard `functions.R` and `h5ad_util.R` | Scissor R workflows. Configure `SCISSOR_HELPER_R`, `SCHARD_FUNCTIONS_R`, `SCHARD_H5AD_R`. |
| Spatial `.h5ad` objects, count layers, coordinates, reference proportions; bulk expression/response metadata | Annotation, Scissor and functional-state workflows. Preserve the original subdirectories under configured data roots. |
| `grpr_gsva_v2/gsva_scores_z.csv` and historical functional-label tables | Feature/classifier and earlier ROC inputs; the corresponding original intermediate exports must be available. |
| `results/spot_full_table.csv`, `results/spot_regions.csv`, Scissor label tables and `GRPR_mem_exh_final_labels.csv` | Latest mouse region/neighbourhood analyses. `region_annot.py` now supplies the region generator; the curated `prepare_scissor_spot_table.py` supplies the integrated table. Tumour-program and perivascular-definition differences remain documented. |
| Raw Stereo-seq level-13 matrices; reclustered `.h5ad` objects; `mouse/filtered_mm.gtf` | `raw_car_counts` raw counts → `step01` spot master → existing `step02` CAR calling → `region_annot` spatial CNV and regions. |
| Existing original per-cohort Scissor `.RData` and bulk tables | Restored `scissor_coefs.R` exports `scissor_coefs_long.csv`. It is distinct from the pooled M5/BEST label workflow. |
| Clinical annotated/deconvolved sections, original reference files, patient/time/response/survival metadata | Clinical spatial, pseudobulk and survival workflows. |
| `myeloid_CAF_correlation_by_group.csv`, `myeloid_CAF_colocalisation_by_group.csv` | Correlation/OR summary script. Its upstream `step26_coloc_marker_heatmap.py` was not supplied. |
| External `CIBERSORT.R`, signature matrix and TCGA expression/clinical tables | CIBERSORT/pan-cancer analyses. These are not bundled. |
| Imaging input tables / calibration used by the author | Imaging-to-h5ad conversion. |

The raw scRNA-seq QC/reference pipeline, scVI/DestVI training, communication inference and imaging-neighbourhood statistical entry points were not supplied as clean executable workflows. The mixed historical scRNA/annotation drawing notebooks were excluded rather than advertised as complete upstream pipelines.

`pending_entrypoint/analyze.py` additionally requires `car_call.py`, `mics.py`, `mib.py`, `pseudobulk.py`, `roc.py`, `neighborhood.py`, `model_family.py` and `fourier.py` in `src`. These modules were not supplied. The available `fourier_core.py` was not renamed to pretend it implements that entire interface.

Python dependencies include NumPy, pandas, SciPy, anndata, scikit-learn, statsmodels, h5py, tifffile and joblib. R dependencies include survival, survminer (for computational `surv_cutpoint`), limma, edgeR, GSVA, jsonlite, msigdbr, fgsea, clusterProfiler, org.Hs.eg.db, infercnv, Seurat, Scissor, homologene, sva, data.table, tidyverse, Biobase, e1071 and preprocessCore. Exact requirements vary by branch; see imports and historical environment snapshots.
