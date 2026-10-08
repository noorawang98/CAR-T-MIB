# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step52: pooled (all-cohort, no-relapse) Scissor labelling of every mouse spatial sample
# + Scissor's internal reliability test.  Fixes the broken run in
# 07_scissor/scissor_validation.R (that helper never called reliability.test, so all AUC
# values were NA and every rds was NULL).
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(Scissor); library(tidyverse); library(homologene)
  library(sva); library(data.table); library(pROC); library(progress)
})
source(Sys.getenv("SCISSOR_HELPER_R"))
source(Sys.getenv("SCHARD_FUNCTIONS_R"))
source(Sys.getenv("SCHARD_H5AD_R"))

BULKDIR <- Sys.getenv("BULKDIR",
             paste0(LEGACY_DATA_ROOT, "/ST_20241110/spatial_input/h5ad/scissor_no_relapse_exprllist"))
RCL     <- paste0(LEGACY_DATA_ROOT, "/ST_20241110/spatial_input/h5ad/recluster_obj")
OUT     <- Sys.getenv("OUT", paste0(PROJECT_ROOT, "/mouse/cart_region/scissor_pooled_noRelapse"))
CFG     <- Sys.getenv("CONFIG", "M5")                       # M5 (all 5) | M4 | M2
TAG     <- Sys.getenv("TAG", CFG)                          # output suffix (e.g. M5a001)
SAMPLES <- strsplit(Sys.getenv("SAMPLES", "Vehicle,NR1,NR2,NR3,R1,R2,R3"), ",")[[1]]
NPERM   <- as.integer(Sys.getenv("NPERM", "10"))
NFOLD   <- as.integer(Sys.getenv("NFOLD", "5"))
ALPHA   <- as.numeric(Sys.getenv("ALPHA", "0.005"))
CUTOFF  <- as.numeric(Sys.getenv("CUTOFF", "0.2"))
for (d in c("labels", "logs", "auc")) dir.create(file.path(OUT, d), showWarnings = FALSE, recursive = TRUE)

CONFIGS <- list(M5 = c("CC2025", "GSE153437", "GSE153438", "GSE197977", "GSE248835"),
                M4 = c("CC2025", "GSE153437", "GSE153438", "GSE248835"),
                M2 = c("CC2025", "GSE153437"))

## ---------------- merged, batch-corrected bulk ----------------
build_bulk <- function(cohorts) {
  mats <- list()
  for (co in cohorts) {
    d <- fread(file.path(BULKDIR, "bulk", paste0("bulk_", co, ".txt")), data.table = FALSE)
    g <- toupper(as.character(d[[1]])); d <- d[, -1, drop = FALSE]
    m <- as.matrix(d); storage.mode(m) <- "double"; rownames(m) <- g
    m <- m[!duplicated(g) & !is.na(g) & g != "", , drop = FALSE]
    int_like <- mean(m == round(m), na.rm = TRUE)
    if (int_like > 0.9 && max(m, na.rm = TRUE) > 100) {        # raw counts -> CPM
      cpm <- sweep(m, 2, pmax(colSums(m, na.rm = TRUE), 1), "/") * 1e6
      m <- log2(cpm + 1)
    } else {
      m <- log2(m + 1)                                        # already linear-normalised
    }
    mats[[co]] <- m
    cat(sprintf("  %-11s %d genes x %d samples (log2 scale, median %.2f)\n",
                co, nrow(m), ncol(m), median(m)))
  }
  common <- Reduce(intersect, lapply(mats, rownames))
  E <- do.call(cbind, lapply(mats, function(m) m[common, , drop = FALSE]))
  ph <- read.csv(file.path(BULKDIR, "pheno.csv"), stringsAsFactors = FALSE)
  ph <- ph[ph$sample_id %in% colnames(E), ]
  ph <- ph[match(colnames(E), ph$sample_id), ]
  ok <- !is.na(ph$label)
  E <- E[, ok, drop = FALSE]; ph <- ph[ok, ]
  cat(sprintf("merged %s: %d genes x %d samples | DR %d / NDR %d\n", CFG, nrow(E), ncol(E),
              sum(ph$label == "DR"), sum(ph$label == "NDR")))
  mod <- model.matrix(~ label, data = data.frame(label = factor(ph$label, levels = c("DR", "NDR"))))
  Eb <- ComBat(dat = E, batch = factor(ph$GSE), mod = mod, par.prior = TRUE)
  Eb[is.na(Eb)] <- 0
  Eb <- Eb[rowSums(Eb) > 0, , drop = FALSE]
  list(expr = Eb, pheno = ph)
}

## ---------------- mouse spatial -> human symbols ----------------
mm2hm <- function(obj, keep = NULL) {
  genes <- rownames(obj)                                     # mouse symbols
  map <- homologene::homologene(genes, inTax = 10090, outTax = 9606)
  map <- map[!duplicated(map$`10090`), , drop = FALSE]
  m <- map$`9606`[match(genes, map$`10090`)]
  ok <- !is.na(m) & !duplicated(m)
  if (!is.null(keep)) ok <- ok & (m %in% keep)
  count <- t(as.matrix(FetchData(obj, vars = genes[ok])))     # genes x cells
  colnames(count) <- colnames(obj)
  rownames(count) <- m[ok]
  cat(sprintf("  spatial: %d genes x %d spots (human symbols)\n", nrow(count), ncol(count)))
  Scissor::Seurat_preprocessing(count, verbose = FALSE, resolution = 1)
}

main <- function() {
  B <- build_bulk(CONFIGS[[CFG]])
  saveRDS(B, file.path(OUT, paste0("bulk_merged_", CFG, ".rds")))
  phenotype <- as.numeric(factor(B$pheno$label, levels = c("DR", "NDR"))) - 1
  names(phenotype) <- B$pheno$sample_id
  auc_rows <- list()
  for (s in SAMPLES) {
    lab_file <- file.path(OUT, "labels", sprintf("%s.%s.scissor_labels.csv", s, TAG))
    if (file.exists(lab_file)) { cat("skip (exists):", s, "\n"); next }
    t0 <- Sys.time()
    cat(sprintf("\n===== %s (%s) =====\n", s, TAG)); flush.console()
    obj <- h5ad2seurat(file.path(RCL, paste0(s, "_sc2st_DestVI_destvi_recluster.h5ad")))
    obj@meta.data$barcode <- paste0(rownames(obj@meta.data), "-", s)
    st <- mm2hm(obj)   # network from all homolog genes; Scissor uses the bulk intersection
    rdata <- file.path(OUT, sprintf("%s.%s.Scissor_inputs.RData", s, TAG))
    infos <- Scissor5(bulk_dataset = B$expr, sc_dataset = st, phenotype = phenotype,
                      tag = c("DR", "NDR"), family = "binomial", alpha = ALPHA,
                      cutoff = CUTOFF, Save_file = rdata)
    co <- infos$Coefs; names(co) <- colnames(as.matrix(st@assays$RNA$data))
    lab <- ifelse(co > 0, "NDR", ifelse(co < 0, "DR", "0"))
    write.csv(data.frame(spot = names(co), coef = as.numeric(co), label = lab),
              lab_file, row.names = FALSE)
    cat(sprintf("  Scissor+ %d, Scissor- %d, selected %.1f%% (alpha=%s, lambda=%.3g)\n",
                length(infos$Scissor_pos), length(infos$Scissor_neg),
                100 * (length(infos$Scissor_pos) + length(infos$Scissor_neg)) / length(co),
                infos$para$alpha, infos$para$lambda))
    auc_rows[[s]] <- data.frame(sample = s, config = TAG, n_spots = length(co),
                                n_NDR = length(infos$Scissor_pos), n_DR = length(infos$Scissor_neg),
                                alpha = infos$para$alpha, lambda = infos$para$lambda)
    ## ---- Scissor internal reliability test ----
    e <- new.env(); load(rdata, envir = e)
    cn <- length(infos$Scissor_pos) + length(infos$Scissor_neg)
    res <- tryCatch(
      Scissor::reliability.test(X = e$X, Y = e$Y, network = e$network, alpha = infos$para$alpha,
                                family = "binomial", cell_num = cn, n = NPERM, nfold = NFOLD),
      error = function(err) { cat("reliability error:", conditionMessage(err), "\n"); NULL })
    saveRDS(res, file.path(OUT, "auc", sprintf("%s.%s.reliability.rds", s, TAG)))
    if (!is.null(res)) {
      ar <- as.numeric(res$AUC_test_real)
      auc_rows[[s]]$auc_status <- "success"
      auc_rows[[s]]$auc_mean <- res$statistic
      auc_rows[[s]]$reliability_p <- res$p
      auc_rows[[s]]$auc_sd <- sd(ar)
      auc_rows[[s]]$auc_n <- length(ar)
      auc_rows[[s]]$auc_min <- min(ar); auc_rows[[s]]$auc_max <- max(ar)
      cat(sprintf("  reliability: AUC = %.3f (p = %.3f, n = %d, folds = %d, %.1f min)\n",
                  res$statistic, res$p, length(ar), NFOLD,
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    } else {
      auc_rows[[s]]$auc_status <- "error"
    }
    flush.console()
  }
  if (length(auc_rows)) {
    f <- file.path(OUT, "auc", sprintf("reliability_%s.csv", TAG))
    old <- if (file.exists(f)) read.csv(f) else NULL
    A <- bind_rows(old, bind_rows(auc_rows))
    A <- A[!duplicated(A[c("sample", "config")]), ]
    write.csv(A, f, row.names = FALSE)
    print(A)
  }
  cat("\ndone:", CFG, "\n")
}
main()
