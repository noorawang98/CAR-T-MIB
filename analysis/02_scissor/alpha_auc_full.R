# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step91: complete alpha -> reliability-AUC sweep for every slide (parallel over pairs).
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(Seurat); library(Scissor); library(pROC); library(parallel)})
source(Sys.getenv("SCISSOR_HELPER_R"))
OUT <- paste0(PROJECT_ROOT, "/mouse/cart_region/scissor_pooled_noRelapse")
RES <- paste0(PROJECT_ROOT, "/mouse/cart_region/figs_final")
SAMPLES <- c("Vehicle","NR1","NR2","NR3","R1","R2","R3")
ALPHAS  <- c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
NPERM <- as.integer(Sys.getenv("NPERM", "5")); NFOLD <- as.integer(Sys.getenv("NFOLD", "5"))
CORES <- as.integer(Sys.getenv("CORES", "12")); NLAM <- as.integer(Sys.getenv("NLAM", "30"))
set.seed(123)
grid <- expand.grid(sample = SAMPLES, alpha = ALPHAS, stringsAsFactors = FALSE)
cat("fits:", nrow(grid), "on", CORES, "cores\n")
res <- mclapply(seq_len(nrow(grid)), function(i) {
  s <- grid$sample[i]; a <- grid$alpha[i]
  f <- file.path(OUT, sprintf("%s.M5.Scissor_inputs.RData", s))
  if (!file.exists(f)) return(NULL)
  e <- new.env(); load(f, envir = e); X <- e$X; Y <- e$Y; network <- e$network
  fu <- tryCatch({
    set.seed(123)
    f0 <- APML1(X, Y, family = "binomial", penalty = "Net", alpha = a, Omega = network,
                nlambda = NLAM, nfolds = min(5, nrow(X)))
    f1 <- APML1(X, Y, family = "binomial", penalty = "Net", alpha = a, Omega = network, lambda = f0$lambda.min)
    as.numeric(f1$Beta[2:(ncol(X) + 1)])
  }, error = function(err) NULL)
  if (is.null(fu)) return(data.frame(sample = s, alpha = a, n_sel = NA, sel_pct = NA, auc = NA, p = NA))
  nsel <- sum(fu != 0)
  r <- tryCatch(Scissor::reliability.test(X, Y, network, alpha = a, family = "binomial",
                                          cell_num = max(nsel, 10), n = NPERM, nfold = NFOLD),
                error = function(err) NULL)
  data.frame(sample = s, alpha = a, n_sel = nsel, sel_pct = 100 * nsel / ncol(X),
             auc = if (is.null(r)) NA else r$statistic, p = if (is.null(r)) NA else r$p)
}, mc.cores = CORES)
R <- do.call(rbind, res[!vapply(res, is.null, logical(1))])
write.csv(R, file.path(RES, "alpha_auc_full.csv"), row.names = FALSE)
cat("done:", nrow(R), "rows\n"); print(R)
