# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(msigdbr); library(dplyr)})
OUT <- paste0(PROJECT_ROOT, "/mouse/cart_region/grpr_gsva_v2")
M <- msigdbr(species = "Mus musculus") %>%
  dplyr::select(gs_name, gene_symbol, gs_collection) %>% distinct() %>% filter(!is.na(gs_name), !is.na(gene_symbol))
keep <- grepl("_UP$", M$gs_name) | grepl("^(GOBP|GOCC|GOMF|KEGG|REACTOME|WP|BIOCARTA|PID)_", M$gs_name)
M <- M[keep, ]
pat <- "EXHAUST|EFFECTOR|CYTOTOX|GZMB|PRF1|MEMORY|STEMNESS|STEM_CELL|HYPOXIA|INFLAMMAT|CYTOKINE|CHEMOKINE"
M <- M[grepl(pat, M$gs_name, ignore.case = TRUE), ]
cat("kept rows:", nrow(M), "| sets:", length(unique(M$gs_name)), "\n")
write.csv(M, gzfile(file.path(OUT, "msigdb_sets.csv.gz")), row.names = FALSE)
