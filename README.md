# CAR-T myeloid–fibroblast barrier: key analysis code

Analysis scripts curated from the author-supplied revision archive. Numbered folders follow the methodological sequence: annotation, Scissor, functional states, Fourier analysis, MIB definitions, neighbourhood statistics, functional correlations/enrichment, imaging conversion, clinical pseudobulk and survival.

The release contains calculations and tabular exports. Figure generation, layout checking, historical alternatives and duplicate scripts were removed. Scientific thresholds, model formulas, permutation counts and seeds were preserved. Original step numbers remain in filenames to preserve provenance and cross-script references.

## Start here

1. Read [the method-order guide](docs/METHOD_ORDER.md) and [input requirements](docs/INPUTS_AND_DEPENDENCIES.md).
2. Copy `configs/paths.example.env` to `.env` and supply absolute paths to the original input data and author helper modules. Load the variables into your shell before running an analysis.
3. Use `python run_analysis.py --check analysis/<folder>/<script>` to check prerequisites, then omit `--check` to run that script. This runner does not automatically execute every folder: mouse, clinical and pan-cancer workflows have separate inputs.
4. Run `python tests/test_fourier_core.py` for the data-independent Fourier checks.

Example:

```bash
set -a
source .env
set +a
python run_analysis.py --check analysis/06_spatial_neighbourhood/perivascular_permutation_test.py
python run_analysis.py analysis/06_spatial_neighbourhood/perivascular_permutation_test.py
```

Project data are not redistributed. The second author archive supplied spatial CNV/region generation and `common.py`; these are now included. Other upstream entry points and external R helpers remain required for a complete rerun. See the explicit dependency list rather than treating this as a self-contained raw-data pipeline.

The supplied environment snapshots are historical records, not a tested lockfile. In particular, the listed Python and pandas versions should not be interpreted as a compatible environment specification. GSVA scripts use the legacy `gsva(matrix, gene_sets, ...)` interface and require a compatible GSVA version.

## Provenance and scope

- [Script manifest](docs/script_manifest.csv): original and release paths and SHA-256 hashes.
- [Excluded scripts](docs/excluded_scripts.csv): files removed from the release and their reasons.
- [Methods/code alignment](docs/METHOD_ALIGNMENT.md): differences that cannot be resolved by reorganizing code.
- [Validation](docs/VALIDATION.md): checks performed and their limits.
- [Final clinical GSEA update](docs/GSEA_UPDATE.md): author-confirmed adjusted path and a source sorting correction.
- [Second-archive update](docs/V2_UPDATE.md): added calculations, restored dependencies and remaining definition differences.
- `pending_entrypoint/analyze.py`: an additional author-supplied CSV interface; its imported source modules were not supplied, so it is not the executable entry point for this release.

No software licence was assigned on the author's behalf; see `LICENSE_TBD.md`.

## Script filenames

Scripts use descriptive filenames without step-number prefixes. See [the rename mapping](docs/script_rename_map.csv) and [rename notes](docs/SCRIPT_RENAMING.md).
