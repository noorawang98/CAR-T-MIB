# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
setwd(paste0(PROJECT_ROOT, "/mouse/cart_region/ccr1mib_tcga"))
source(paste0(HOME_ROOT, "/TCGA/CIBERSORT.R"))
res <- CIBERSORT(paste0(HOME_ROOT, "/LJX/st/CIBERSORTx_Job22_ST_NR_mm2hsa.txt"),
                 paste0(HOME_ROOT, "/TCGA/tcga_sel.txt"), perm = 0, QN = TRUE)
write.csv(res, "CIBERSORT_STNR_TCGA.csv")
cat("done", dim(res), "\n")
