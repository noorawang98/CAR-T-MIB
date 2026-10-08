# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
## Step4c (v2): inferCNV 后处理 —— i6 HMM 分组状态 + 每细胞 CNV score -> 二分 cnv_verdict, 回填全部细胞
## 输入: cnv/infercnv_out_group/{14_invert_log_transformHMMi6.infercnv_obj,
##       HMM_CNV_predictions.HMMi6.hmm_mode-samples.Pnorm_0.5.pred_cnv_genes.dat}, cnv/cells_sub_meta.csv
## 输出: tables/cnv_*.csv
.libPaths(c("~/R_4.3", .libPaths()))
suppressMessages(library(infercnv))

WD <- paste0(PROJECT_ROOT, "/destvi_260911"); OUT <- file.path(WD, "cnv", "infercnv_out_group")
TAB <- file.path(WD, "tables"); dir.create(TAB, showWarnings = FALSE)
EPI <- c("Malignant epithelial CEACAM5+", "Malignant epithelial proliferating", "Malignant epithelial gastric")

## ---- 1. 每细胞 CNV score (STEP14: 平滑+参考扣除后的 log2FC->FC, 正常≈1) ----
o14 <- readRDS(file.path(OUT, "14_invert_log_transformHMMi6.infercnv_obj"))
E <- o14@expr.data                                     # genes x cells
score <- colMeans(abs(E - 1))
meta <- read.csv(file.path(WD, "cnv", "cells_sub_meta.csv"), row.names = 1)[names(score), ]

## ---- 2. i6 HMM 分组状态 (Bayes 过滤后) ----
p <- read.delim(file.path(OUT, "HMM_CNV_predictions.HMMi6.hmm_mode-samples.Pnorm_0.5.pred_cnv_genes.dat"))
grp <- sub("\\..*$", "", p$cell_group_name)   # 组名 = "<cell_type>.<cell_type>_s1" -> cell_type
hmm <- data.frame(cell_type = grp, state = p$state, stringsAsFactors = FALSE)
tab <- table(hmm$cell_type, hmm$state)
hmm_sum <- data.frame(cell_type = rownames(tab), n_gene_state = rowSums(tab),
                      frac_non_neutral = 1 - (tab[, "3"] / rowSums(tab)),
                      stringsAsFactors = FALSE)
hmm_sum$gain_frac <- rowSums(tab[, intersect(colnames(tab), c("4", "5", "6")), drop = FALSE]) / rowSums(tab)
hmm_sum$loss_frac <- rowSums(tab[, intersect(colnames(tab), c("1", "2")), drop = FALSE]) / rowSums(tab)
## 补齐没有 HMM 区段(=完全中性)的细胞类型
missing_ct <- setdiff(unique(meta$cell_type), hmm_sum$cell_type)
if (length(missing_ct)) {
  hmm_sum <- rbind(hmm_sum, data.frame(cell_type = missing_ct, n_gene_state = 0,
                                       frac_non_neutral = 0, gain_frac = 0, loss_frac = 0))
}
hmm_sum$cnv_verdict_HMM <- ifelse(hmm_sum$frac_non_neutral > 0.30, "CNV+", "CNV-")  # 恶性组 0.44-0.64 vs 参考组 <=0.19
hmm_sum <- hmm_sum[order(-hmm_sum$frac_non_neutral), ]
write.csv(hmm_sum, file.path(TAB, "cnv_hmm_group_summary.csv"), row.names = FALSE)
cat("=== i6 HMM 分组非中性基因组比例 (前 8) ===\n"); print(head(hmm_sum, 8), row.names = FALSE, digits = 3)

## ---- 3. 每细胞二分 verdict ----
ref_types <- setdiff(unique(meta$cell_type), EPI)
thr <- as.numeric(quantile(score[meta$cell_type %in% ref_types], 0.95))
verdict_hmm <- hmm_sum$cnv_verdict_HMM[match(meta$cell_type, hmm_sum$cell_type)]
df <- data.frame(barcode = names(score), cnv_score = as.numeric(score), cell_type = meta$cell_type,
                 leiden = meta$leiden, sample = meta$sample, sample_group = meta$sample_group,
                 cnv_verdict_HMM = verdict_hmm, stringsAsFactors = FALSE)
df$cnv_verdict_cell <- ifelse(df$cnv_score > thr, "CNV+", "CNV-")   # 阈值 = 参考细胞 score 的 95 分位
df$cnv_verdict_HMM[is.na(df$cnv_verdict_HMM)] <- "CNV-"
df$cnv_verdict <- ifelse(df$cnv_verdict_HMM == "CNV+" & df$cnv_verdict_cell == "CNV+", "CNV+", "CNV-")
write.csv(df, file.path(TAB, "cnv_score_subset.csv"), row.names = FALSE)
cat(sprintf("\n阈值(参考细胞 95%% 分位)=%.4f\n", thr))
print(aggregate(cbind(score = df$cnv_score, hmm_pos = as.numeric(df$cnv_verdict_HMM == "CNV+"),
                      cell_pos = as.numeric(df$cnv_verdict_cell == "CNV+"),
                      pos = as.numeric(df$cnv_verdict == "CNV+")) ~ cell_type, df, mean),
      row.names = FALSE, digits = 3)

## ---- 4. 回填全部 148884 细胞 (用更新命名后的 h5ad obs) ----
suppressMessages(library(rhdf5))
h5 <- file.path(WD, "scrna_reference_celltype.h5ad")
obs <- h5read(h5, "obs")
dec <- function(x) if (is.list(x) && !is.null(x$categories)) x$categories[as.integer(x$codes) + 1] else as.character(x)
allc <- data.frame(barcode = as.character(obs[["_index"]]),
                   leiden = dec(obs[["leiden"]]),
                   cell_type_detail = dec(obs[["cell_type_detail"]]), stringsAsFactors = FALSE)
hmm_pos_types <- hmm_sum$cell_type[hmm_sum$cnv_verdict_HMM == "CNV+"]
allc$cnv_verdict <- ifelse(allc$cell_type_detail %in% EPI & allc$cell_type_detail %in% hmm_pos_types,
                           "CNV+", "CNV-")
m <- match(allc$barcode, df$barcode)
has <- !is.na(m)
allc$cnv_verdict[has] <- df$cnv_verdict[m[has]]
allc$cnv_score <- df$cnv_score[m]
allc$cnv_source <- ifelse(has, "per-cell (inferCNV subset)",
                          ifelse(allc$cell_type_detail %in% EPI, "group HMM (inferCNV subset)",
                                 "reference group (non-epithelial)"))
write.csv(allc, file.path(TAB, "cnv_verdict_per_cell.csv"), row.names = FALSE)
cat("\n=== 全部细胞 cnv_verdict ===\n"); print(table(allc$cell_type_detail, allc$cnv_verdict))

## ---- 5. 分箱 CNV 谱 (每细胞类型沿染色体 20 箱/染色体平均) ----
go <- read.table(file.path(WD, "cnv", "gene_order.txt"), stringsAsFactors = FALSE)
go <- go[go$V1 %in% rownames(E), ]
chrlen <- tapply(go$V4, go$V2, max)
go$bin <- paste0(go$V2, "_", sprintf("%02d", floor(go$V3 / (chrlen[go$V2] / 20) * 20)))
go <- go[!is.na(go$bin), ]
Eb <- E[go$V1, , drop = FALSE]
binmean <- t(sapply(split(seq_len(nrow(go)), go$bin), function(i) colMeans(Eb[i, , drop = FALSE])))
ord <- order(as.numeric(sub("chr(\\d+).*", "\\1", rownames(binmean))), as.numeric(sub(".*_", "", rownames(binmean))))
binmean <- binmean[ord, ]
colnames(binmean) <- names(score)
write.csv(data.frame(bin = rownames(binmean), binmean, check.names = FALSE),
          file.path(TAB, "cnv_binned_profile_by_cell.csv"), row.names = FALSE)
cat("\nDONE\n")
