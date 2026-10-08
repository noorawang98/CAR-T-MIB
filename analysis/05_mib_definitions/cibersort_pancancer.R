# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step105b: 用 ST 签名矩阵 (CIBERSORTx Job22_ST_NR_mm2hsa) 对 MIB_DestVI 伪bulk 做 CIBERSORT 打分,
#           选出 Pre SD/PD MIB_DestVI 富集的组件, 再用 TCGA 的同一 CIBERSORT 输出重复泛癌 Cox 分析。
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(parallel); library(e1071); library(preprocessCore); library(survival);   library(dplyr)})
source(paste0(HOME_ROOT, "/TCGA/CIBERSORT.R"))
WD <- Sys.getenv("ST_WD", paste0(PROJECT_ROOT, "/destvi_260911")); PROG <- file.path(WD, "prognosis")
OUT <- file.path(WD, "pancancer_mibdestvi"); dir.create(OUT, showWarnings = FALSE)
SIG <- paste0(HOME_ROOT, "/LJX/st/CIBERSORTx_Job22_ST_NR_mm2hsa.txt"); TCGA_CIB <- paste0(PROJECT_ROOT, "/mouse/cart_region/ccr1mib_tcga/CIBERSORT_STNR_TCGA.csv")
# ---- 1) ST 伪bulk 打分 (若已有缓存直接复用) ----
CACHEF <- file.path(OUT, "ST_mibdestvi_cibersort_fractions.csv")
PB <- read.csv(file.path(PROG, "mibdestvi_pseudobulk_counts.csv.gz"), check.names = FALSE, row.names = 1)
M <- read.csv(file.path(PROG, "mibdestvi_pseudobulk_meta.csv"))
X <- read.table(SIG, header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)   # signature matrix
Y <- data.matrix(PB); Y[is.na(Y)] <- 0   # 切片基因空间不同, 缺失计 0
gm <- intersect(rownames(X), rownames(Y)); cat("shared genes:", length(gm), "\n")
X <- data.matrix(X[gm, , drop = FALSE]); Y <- Y[gm, , drop = FALSE]
cn <- colnames(Y); Y <- normalize.quantiles(Y); rownames(Y) <- gm; colnames(Y) <- cn   # normalize.quantiles 会丢 dimnames
X <- (X - mean(X)) / sd(as.vector(X))
frac <- t(sapply(seq_len(ncol(Y)), function(i) {
  r <- CoreAlg(X, Y[, i, drop = FALSE])
  v <- r$w; names(v) <- colnames(X); v }))
frac <- as.data.frame(frac)
comp <- setdiff(colnames(frac), "key")
frac$key <- colnames(Y)
frac <- cbind(M[match(frac$key, M$key), , drop = FALSE], frac[, comp, drop = FALSE])   # 不用 merge, 避免列名冲突
write.csv(frac, file.path(OUT, "ST_mibdestvi_cibersort_fractions.csv"), row.names = FALSE)
cat("\nST 打分完成:", nrow(frac), "个伪bulk x", length(comp), "组件\n")
# Pre SD/PD MIB_DestVI vs Pre PR MIB_DestVI (以及 nonMIB 对照) 的组件差异
res <- do.call(rbind, lapply(comp, function(cn) {
  d <- frac[frac$set == "MIB_DestVI", ]
  a <- d[[cn]][d$time == "Pre" & d$resp == "SD/PD"]; b <- d[[cn]][d$time == "Pre" & d$resp == "PR"]
  data.frame(component = cn, mean_Pre_SDPD = mean(a), mean_Pre_PR = mean(b),
             log2_ratio = log2((mean(a) + 1e-6) / (mean(b) + 1e-6))) }))
res <- res[order(-res$log2_ratio), ]
write.csv(res, file.path(OUT, "ST_mibdestvi_component_ranking.csv"), row.names = FALSE)
print(head(res, 6))
sel <- head(res$component[res$log2_ratio > log2(1.5)], 3)
if (!length(sel)) sel <- head(res$component, 2)
cat("选定 MIB_DestVI 关联组件:", paste(sel, collapse = ", "), "\n")
