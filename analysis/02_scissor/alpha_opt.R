# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step57: optimise Scissor's elastic-net alpha (sparsity) by its own internal reliability
# test, then regenerate the labels with the best alpha and validate all samples.
# Reuses the cached X / Y / network from the alpha=0.005 run -> no h5ad re-reading.
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(Seurat); library(Scissor); library(pROC); library(progress)})
source(Sys.getenv("SCISSOR_HELPER_R"))
OUT <- paste0(PROJECT_ROOT, "/mouse/cart_region/scissor_pooled_noRelapse")
SAMPLES <- strsplit(Sys.getenv("SAMPLES", "Vehicle,NR1,NR2,NR3,R1,R2,R3"), ",")[[1]]
SCAN <- as.numeric(strsplit(Sys.getenv("ALPHAS", "0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5"), ",")[[1]])
REL  <- as.numeric(strsplit(Sys.getenv("REL_ALPHAS", "0.01,0.05,0.1,0.2,0.3"), ",")[[1]])   # alphas tested for AUC
RELS <- strsplit(Sys.getenv("REL_SAMPLES", "NR1,NR2,R1,R2"), ",")[[1]]
NPERM <- as.integer(Sys.getenv("NPERM", "5")); NFOLD <- as.integer(Sys.getenv("NFOLD", "5"))
NLAM <- as.integer(Sys.getenv("NLAM", "30")); CVFOLDS <- as.integer(Sys.getenv("CVFOLDS", "5"))
TAGOUT <- Sys.getenv("TAGOUT", "BEST")
dir.create(file.path(OUT, "labels"), showWarnings = FALSE)
fits <- list()
scan <- list()
for (s in SAMPLES) {
  f <- file.path(OUT, sprintf("%s.M5.Scissor_inputs.RData", s))
  if (!file.exists(f)) { cat("missing", f, "\n"); next }
  e <- new.env(); load(f, envir = e)
  X <- e$X; Y <- e$Y; network <- e$network
  for (a in SCAN) {
    r <- tryCatch({
      set.seed(123)
      f0 <- APML1(X, Y, family = "binomial", penalty = "Net", alpha = a, Omega = network, nlambda = NLAM, nfolds = min(CVFOLDS, nrow(X)))
      f1 <- APML1(X, Y, family = "binomial", penalty = "Net", alpha = a, Omega = network, lambda = f0$lambda.min)
      co <- as.numeric(f1$Beta[2:(ncol(X) + 1)])
      list(co = co, lambda = f0$lambda.min)
    }, error = function(err) { cat("  alpha", a, "error:", conditionMessage(err), "\n"); NULL })
    if (is.null(r)) next
    np <- sum(r$co > 0); nn <- sum(r$co < 0)
    scan[[length(scan) + 1]] <- data.frame(sample = s, alpha = a, lambda = r$lambda,
                                           n_pos = np, n_neg = nn,
                                           sel_pct = 100 * (np + nn) / length(r$co),
                                           min_side = min(np, nn))
    fits[[paste(s, a)]] <- r$co
  }
  cat("scanned", s, "\n"); flush.console()
}
S <- do.call(rbind, scan)
write.csv(S, file.path(OUT, "alpha_scan.csv"), row.names = FALSE)
print(aggregate(cbind(sel_pct, min_side) ~ alpha, data = S, FUN = mean))

rel <- list()
for (s in RELS) {
  e <- new.env(); load(file.path(OUT, sprintf("%s.M5.Scissor_inputs.RData", s)), envir = e)
  X <- e$X; Y <- e$Y; network <- e$network
  for (a in REL) {
    co <- fits[[paste(s, a)]]
    if (is.null(co)) next
    cn <- sum(co != 0)
    if (cn < 10) { cat("skip", s, a, "too few\n"); next }
    set.seed(123)
    res <- tryCatch(Scissor::reliability.test(X, Y, network, alpha = a, family = "binomial",
                                              cell_num = cn, n = NPERM, nfold = NFOLD),
                    error = function(err) { cat("rel error", s, a, conditionMessage(err), "\n"); NULL })
    if (is.null(res)) next
    rel[[length(rel) + 1]] <- data.frame(sample = s, alpha = a, n_sel = cn,
                                         sel_pct = 100 * cn / ncol(X), auc = res$statistic, p = res$p)
    cat(sprintf("  rel %s alpha=%s AUC=%.3f p=%.3f\n", s, a, res$statistic, res$p)); flush.console()
  }
}
R <- do.call(rbind, rel)
write.csv(R, file.path(OUT, "alpha_reliability.csv"), row.names = FALSE)
A <- aggregate(auc ~ alpha, data = R, FUN = mean)
A$sel_pct <- aggregate(sel_pct ~ alpha, data = R, FUN = mean)$sel_pct
A$min_side <- aggregate(n_sel ~ alpha, data = R, FUN = min)$n_sel
print(A)
ok <- A[A$sel_pct >= 3 & A$sel_pct <= 60, ]
best <- if (nrow(ok)) ok$alpha[which.max(ok$auc)] else A$alpha[which.max(A$auc)]
cat(sprintf("\nbest alpha = %s (mean reliability AUC %.3f, selected %.1f%%)\n",
            best, A$auc[A$alpha == best], A$sel_pct[A$alpha == best]))
write.csv(data.frame(best_alpha = best, mean_auc = A$auc[A$alpha == best],
                     mean_sel_pct = A$sel_pct[A$alpha == best],
                     criterion = "max mean reliability AUC with 3-60% of spots selected"),
          file.path(OUT, "alpha_best.csv"), row.names = FALSE)

## ---- regenerate labels with the best alpha + full reliability test ----
final <- list()
for (s in SAMPLES) {
  e <- new.env(); load(file.path(OUT, sprintf("%s.M5.Scissor_inputs.RData", s)), envir = e)
  X <- e$X; Y <- e$Y; network <- e$network
  co <- fits[[paste(s, best)]]
  if (is.null(co)) {
    set.seed(123)
    f0 <- APML1(X, Y, family = "binomial", penalty = "Net", alpha = best, Omega = network, nlambda = NLAM, nfolds = min(CVFOLDS, nrow(X)))
    f1 <- APML1(X, Y, family = "binomial", penalty = "Net", alpha = best, Omega = network, lambda = f0$lambda.min)
    co <- as.numeric(f1$Beta[2:(ncol(X) + 1)])
  }
  lab <- ifelse(co > 0, "NDR", ifelse(co < 0, "DR", "0"))
  write.csv(data.frame(spot = colnames(X), coef = co, label = lab),
            file.path(OUT, "labels", sprintf("%s.%s.scissor_labels.csv", s, TAGOUT)), row.names = FALSE)
  cn <- sum(co != 0)
  res <- tryCatch(Scissor::reliability.test(X, Y, network, alpha = best, family = "binomial",
                                            cell_num = cn, n = 10, nfold = NFOLD),
                  error = function(err) NULL)
  final[[length(final) + 1]] <- data.frame(sample = s, config = TAGOUT, n_spots = ncol(X),
                                           n_NDR = sum(co > 0), n_DR = sum(co < 0), alpha = best,
                                           lambda = NA, auc_status = ifelse(is.null(res), "error", "success"),
                                           auc_mean = ifelse(is.null(res), NA, res$statistic),
                                           reliability_p = ifelse(is.null(res), NA, res$p),
                                           auc_sd = ifelse(is.null(res), NA, sd(as.numeric(res$AUC_test_real))),
                                           auc_n = ifelse(is.null(res), NA, length(as.numeric(res$AUC_test_real))))
  if (!is.null(res)) saveRDS(res, file.path(OUT, "auc", sprintf("%s.%s.reliability.rds", s, TAGOUT)))
  cat(sprintf("  %s: NDR %d / DR %d (%.1f%% selected), AUC %.3f p %.3f\n", s, sum(co > 0), sum(co < 0),
              100 * cn / ncol(X), ifelse(is.null(res), NA, res$statistic), ifelse(is.null(res), NA, res$p)))
  flush.console()
}
F <- do.call(rbind, final)
write.csv(F, file.path(OUT, sprintf("auc/reliability_%s.csv", TAGOUT)), row.names = FALSE)
print(F)
cat("done\n")
