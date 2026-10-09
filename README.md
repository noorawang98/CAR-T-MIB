# CAR-T myeloid–fibroblast barrier: key analysis code

Analysis scripts curated from the [CART_ST2025](https://github.com/noorawang98/CART_ST2025) revision archive, following the methodological sequence: annotation, Scissor, functional states, Fourier analysis, MIB definitions, neighbourhood statistics, functional correlations/enrichment, imaging conversion, clinical pseudobulk and survival.

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

Project data are not redistributed. The second archive supplied spatial CNV/region generation and `common.py`;.

## Provenance and scope

- [Script manifest](docs/script_manifest.csv): original and release paths and SHA-256 hashes.
- [Excluded scripts](docs/excluded_scripts.csv): files removed from the release and their reasons.
- [Methods/code alignment](docs/METHOD_ALIGNMENT.md): differences that cannot be resolved by reorganizing code.
- [Validation](docs/VALIDATION.md): checks performed and their limits.
- [Final clinical GSEA update](docs/GSEA_UPDATE.md): user-confirmed adjusted path and a source sorting correction.
- [Second-archive update](docs/V2_UPDATE.md): added calculations, restored dependencies and remaining definition differences.
- `pending_entrypoint/analyze.py`: an additional user-supplied CSV interface; its imported source modules were not supplied, so it is not the executable entry point for this release.

No software licence was assigned on the author's behalf; see `LICENSE_TBD.md`.

## Script filenames

Scripts use descriptive filenames without step-number prefixes. See [the rename mapping](docs/script_rename_map.csv) and [rename notes](docs/SCRIPT_RENAMING.md).
