# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(msigdbr); library(dplyr)})
OUT <- paste0(PROJECT_ROOT, "/mouse/cart_region/grpr_pred_v3/clinical_gsva"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
M <- msigdbr(species = "Mus musculus", category = "C7", subcategory = "IMMUNESIGDB") %>%
  dplyr::select(gs_name, gene_symbol) %>% distinct() %>% filter(!is.na(gs_name), !is.na(gene_symbol))
write.csv(M, gzfile(file.path(OUT, "msigdb_c7_mouse.csv.gz")), row.names = FALSE)
cat("C7 mouse rows:", nrow(M), "| sets:", length(unique(M$gs_name)), "\n")
