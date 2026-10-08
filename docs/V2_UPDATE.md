# Second-archive update

`github_release(2).zip` adds 31 analysis Python scripts and `src/common.py`; earlier analysis source files were unchanged. Its changed README/audit/mapping documents were reviewed rather than copied over the curated guide.

The updated release retains **65 analysis scripts plus the supplied shared `common.py`**. Eight newly supplied analysis scripts and three previously excluded prerequisites were added; all drawing code was excluded. Source/release paths and hashes are in `v2_added_scripts.csv` and the main manifest.

| Included calculation | Reason |
|---|---|
| `raw_car_counts.py` | Extract raw total/CAR UMI from Stereo-seq matrices. |
| `build_spot_master.py` | Assemble original spot metadata/counts; also export the same spot order required by the Scissor coefficient loader. |
| `region_annot.py` | Spatial CNV score, tumour voting and tumour zones. |
| `src/common.py` | Supply original shared data loaders, metadata and BH helper. |
| `gsva_grpr.py`, `kmeans_grpr.py`, `marker_scores.py`, `final_labels.py` | Minimal legacy support chain needed to construct the existing def5 input read by `common.grpr_labels()`. Kept under `legacy_dependencies`; not promoted to the current grouping. |
| `scissor_coefs.R` | Supply the per-cohort coefficients consumed by `common.scissor_labels()`. Requires original cohort-fit `.RData`. |
| `prepare_scissor_spot_table.py` | Keep only original table assembly/derived metadata and `spot_full_table.csv` export; omit superseded comparison statistics. |
| `car_proportions.py` | CAR-positive fractions and tabular comparisons. Explicitly loads the original `GRPR_sample` sidecar required for its legacy split. The original fixed Vehicle background rate is preserved. |
| `niche_v2.py` | Supply the ring-niche input consumed by the already-retained `neighbour_rings` mixed-model features; drawing removed. It is a separate niche definition. |

Remaining module-13 scripts are historical variants, batch/normalisation diagnostics, unrelated subtype analyses or drawing workflows; they were not indiscriminately added. The alternate `step31_lym_downstream.py` and MAN1A analysis were excluded. The complete exclusion inventory was refreshed against the second archive.

## Input sequence

```text
raw_car_counts raw counts → step01 spot master → step02 CAR calls → region_annot CNV/regions
```

For the chromosome-window score, use the source's chosen tumour program. The code provides `TUMOR_MOD=mel` (default) and `TUMOR_MOD=lym` with `SUF=_lym`; it provides no epithelial program. Do not rename these branches to malignant epithelial without changing the biological input definition.

The optional legacy sidecar sequence is `step04 → kmeans_grpr → marker_scores → final_labels`. `step04` reads existing GSVA scores; it does not generate the full original MSigDB GSVA matrix. The current functional label chain remains `clinical_gsva → clinical_gsva_run → mem_exh_final`, with current region consumers reading `GRPR_quad_M5`.

The existing ring-feature branch needs `region_annot` with `TUMOR_MOD=lym, SUF=_lym`, followed by `niche_v2`. Its outputs do not replace the default `spot_regions.csv`.

The CNV algorithm, marker sets, thresholds, voting rules, neighbourhood tests and random seeds were preserved. Changes outside rendering are table/input wiring: spot-order export, explicit legacy label-sidecar loading, a calculation-only integrated-table export and bundled-module routing in the runner.
