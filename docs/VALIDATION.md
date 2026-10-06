# Validation performed

- All 52 Python files passed syntax compilation, including the additional pending interface and retained Fourier example/test.
- All 19 R scripts passed string/comment-aware delimiter checks. An R runtime was unavailable; R parsing, package compatibility and execution were not tested.
- No matplotlib/seaborn drawing calls, figure-save calls, ggplot/ggsave or ggsurvplot calls remain in the retained analysis scripts.
- 89 non-rendering computational/helper functions matched the original Python AST exactly. The only other non-rendering helper difference was a path repair in the generated GSVA R script (`HOME_ROOT + ...` changed to R string concatenation). Model parameters were not changed.
- Scientific-call comparisons found only intended removals: one repeated Pearson calculation used for a plot title, and obsolete statistics after the clinical spot-table preparation. The CIBERSORT component-scoring script excludes its later historical stage/race Cox block; the retained pan-cancer Cox entry uses age/gender adjustment.
- The original Fourier synthetic suite passed all checks: rasterisation, band selection, reconstruction, toroidal shifts, plus-one P values and BH adjustment.
- The runner failed clearly when data roots were not configured, as intended.

There was no full-data rerun, no validation against manuscript numerical results, and no claim that absent author helpers or upstream inputs have been reconstructed. See `validation_checks.json` for the static-check summary.

## Second-archive checks

The full spatial-CNV script and common module match the supplied Python AST exactly. 16 additional computational/helper functions are unchanged. CNV edge-padding/window-length checks passed. Original table exports were preserved, apart from the explicitly excluded historical step07 comparison outputs. The runner resolves bundled common.py without an override. All release Python files were recompiled; no drawing calls remain. There was no real-data rerun.
