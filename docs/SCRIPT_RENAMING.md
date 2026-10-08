# Descriptive script filenames

All bundled scripts previously named with a step-number prefix now use descriptive filenames. Python imports, script-path arguments, usage examples, documentation and release manifests were updated. The complete old-to-new mapping is in `script_rename_map.csv`. Historical source paths in provenance tables are deliberately preserved.

Input/output data filenames, model parameters and statistical procedures are unchanged. Use the current paths from `METHOD_ORDER.md` when invoking `run_analysis.py`; old script paths are no longer accepted. No plotting scripts were added.

Validation includes Python syntax, local import targets, runner prerequisite checks, R delimiter checks, ZIP integrity and the retained Fourier synthetic tests. This is a filename/refactoring update, not a full-data rerun.
