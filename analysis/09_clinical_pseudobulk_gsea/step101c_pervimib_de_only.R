# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step101c: 仅做 limma-voom DE (两种 perviMIB 口径), 输出 CSV (富集另跑 step101b)
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(limma); library(edgeR)})
WD <- Sys.getenv("ST_WD", paste0(PROJECT_ROOT, "/destvi_260911")); PROG <- file.path(WD, "prognosis")
PB <- read.csv(file.path(PROG, "pervimib_pseudobulk_counts.csv.gz"), check.names = FALSE)
PB$key <- paste(PB$sample, PB$mib_def, sep = "|")
M <- read.csv(file.path(PROG, "pervimib_pseudobulk_meta.csv")); M$key <- paste(M$sample, M$mib_def, sep = "|")
for (defn in c("perviMIB_mouse", "perviMIB_DestVI")) {
  tag <- ifelse(defn == "perviMIB_mouse", "mouse", "destvi")
  m <- M[M$mib_def == defn & M$n_pervimib >= 5, ]
  m$group <- factor(m$resp, levels = c("SD/PD", "PR")); m$tp <- factor(m$time, levels = c("Pre", "Post"))
  sub <- PB[match(m$key, PB$key), ]
  x <- t(as.matrix(sub[, setdiff(colnames(PB), c("key", "sample", "mib_def"))]))
  x[is.na(x)] <- 0; storage.mode(x) <- "double"
  keep <- rowSums(x >= 10) >= 3; x <- x[keep, , drop = FALSE]
  cat(sprintf("\n[%s] sections=%d (PR %d / SD-PD %d), genes=%d\n", defn, nrow(m), sum(m$group=="PR"), sum(m$group=="SD/PD"), nrow(x)))
  dge <- calcNormFactors(DGEList(counts = x))
  des <- model.matrix(~ group, data = m); fit <- eBayes(lmFit(voom(dge, des), des))
  tt <- topTable(fit, coef = "groupPR", number = Inf, sort.by = "P"); tt$gene <- rownames(tt)
  write.csv(tt, file.path(PROG, paste0("pervimib_de_", tag, ".csv")), row.names = FALSE)
  des2 <- model.matrix(~ group + tp, data = m); fit2 <- eBayes(lmFit(voom(dge, des2), des2))
  tt2 <- topTable(fit2, coef = "groupPR", number = Inf, sort.by = "P"); tt2$gene <- rownames(tt2)
  write.csv(tt2, file.path(PROG, paste0("pervimib_de_", tag, "_adjTime.csv")), row.names = FALSE)
  cat(sprintf("  FDR<0.1: %d;  p<0.01: %d (%d up in PR, %d up in SD/PD)\n", sum(tt$adj.P.Val < .1), sum(tt$P.Value < .01),
              sum(tt$P.Value < .01 & tt$logFC > 0), sum(tt$P.Value < .01 & tt$logFC < 0)))
  cat(sprintf("  top up in PR: %s\n", paste(head(tt$gene[tt$logFC > 0], 10), collapse = ", ")))
  cat(sprintf("  top up in SD/PD: %s\n", paste(head(tt$gene[tt$logFC < 0], 10), collapse = ", ")))
  cat(sprintf("  (校正取样时点后) FDR<0.1: %d, p<0.01: %d\n", sum(tt2$adj.P.Val < .1), sum(tt2$P.Value < .01)))
}
cat("\nDONE\n")
