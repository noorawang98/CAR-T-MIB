# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step06: export the continuous Scissor weak label (per-spot coefficients) for every
# bulk cohort of scissor_revision.R.
#
# Scissor5() returns $Coefs = the per-column coefficients of the network-penalised
# logistic model fitted on X = cor(bulk, spot).  colnames(X) = c(bulk sample ids,
# spot barcodes), therefore the last n_spot entries are the per-spot weak labels:
#   Coefs < 0  -> Scissor-  -> 'DR'  -> responder      (R)
#   Coefs > 0  -> Scissor+  -> 'NDR' -> non-responder  (NR)
# (identical to add_scissorlab() in scissor_revision.R)
suppressPackageStartupMessages(library(data.table))

RDATA <- paste0(LEGACY_DATA_ROOT, "/ST_20241110/spatial_input/h5ad/recluster_obj/Scissor_rdata/RData/Scissor_5bulk_all_samples.RData")
BULKDIR <- paste0(LEGACY_DATA_ROOT, "/ST_20241110/spatial_input/h5ad/scissor_no_relapse_exprllist/bulk")
OUT <- paste0(PROJECT_ROOT, "/mouse/cart_region/results")
SAMPLES <- c("Vehicle", "NR1", "NR2", "NR3", "R1", "R2", "R3")

load(RDATA)   # infos2_hm2mm_list

SO <- read.csv(file.path(OUT, "spot_order.csv"), stringsAsFactors = FALSE)
spot_order <- split(SO$barcode, SO$sample)

res <- list()
for (co in names(infos2_hm2mm_list)) {
  bf <- file.path(BULKDIR, paste0("bulk_", co, ".txt"))
  n_bulk <- ncol(fread(bf, nrows = 1)) - 1
  for (s in SAMPLES) {
    x <- infos2_hm2mm_list[[co]][[s]]
    if (is.null(x)) next
    cf <- as.numeric(x$Coefs)
    so <- spot_order[[s]]
    ns <- length(so)
    # cor(x, y) -> ncol(x) x ncol(y): X columns are the SPOTS only, so Coefs is
    # already the per-spot vector, in the order of the Seurat/h5ad obs_names.
    if (length(cf) != ns) {
      cat("WARN length mismatch", co, s, length(cf), ns, "\n")
      next
    }
    spot_cf <- cf
    names(spot_cf) <- so
    okp <- if (length(x$Scissor_pos)) mean(spot_cf[x$Scissor_pos] > 0, na.rm = TRUE) else NA
    okn <- if (length(x$Scissor_neg)) mean(spot_cf[x$Scissor_neg] < 0, na.rm = TRUE) else NA
    cat(sprintf("check %-10s %-8s alpha=%-7s nbulk=%d pos=%d(%.2f) neg=%d(%.2f)\n",
                co, s, x$para$alpha, n_bulk, length(x$Scissor_pos), okp,
                length(x$Scissor_neg), okn))
    res[[paste(co, s, sep = "|")]] <- data.frame(
      cohort = co, sample = s, barcode = so,
      scissor_coef = as.numeric(spot_cf), stringsAsFactors = FALSE)
  }
}
D <- do.call(rbind, res)
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
write.csv(D, file.path(OUT, "scissor_coefs_long.csv"), row.names = FALSE)
cat("wrote scissor_coefs_long.csv", nrow(D), "rows\n")
print(table(D$cohort))
