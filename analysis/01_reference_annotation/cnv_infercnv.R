# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
## inferCNV (infercnv 1.16.0, ~/R_4.3) — 确认上皮细胞恶性程度
## 输入: cnv/counts_sub.mtx, genes_sub.tsv, barcodes_sub.tsv, annot_sub.tsv, gene_order.txt
## 输出: cnv/infercnv_out/  (run.final.infercnv_obj.rds, HMM 预测, 日志)
.libPaths(c("~/R_4.3", .libPaths()))
suppressMessages({library(infercnv); library(Matrix)})

WD   <- paste0(PROJECT_ROOT, "/destvi_260911/cnv")
OUT  <- file.path(WD, "infercnv_out_group")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

## ---- 读入 ----
expr  <- t(as(readMM(file.path(WD, "counts_sub.mtx")), "dgCMatrix"))  # -> genes x cells
genes <- readLines(file.path(WD, "genes_sub.tsv"))
cells <- readLines(file.path(WD, "barcodes_sub.tsv"))
rownames(expr) <- genes; colnames(expr) <- cells
annot <- read.table(file.path(WD, "annot_sub.tsv"), sep = "\t", header = FALSE,
                    row.names = 1, stringsAsFactors = FALSE)
colnames(annot) <- "cell_type"
annot <- annot[colnames(expr), , drop = FALSE]
gorder <- read.table(file.path(WD, "gene_order.txt"), header = FALSE, stringsAsFactors = FALSE)
write.table(annot, file.path(WD, "annot_infercnv.txt"), sep = "\t", quote = FALSE, col.names = FALSE)
refs   <- readLines(file.path(WD, "reference_groups.txt"))
cat("matrix:", dim(expr), "| groups:", length(unique(annot$cell_type)),
    "| ref groups:", length(refs), "\n")

## ---- 核心调用 (与原函数 infercnv::run 默认值比较见 cnv/README_cnv.md) ----
infercnv_obj <- infercnv::CreateInfercnvObject(
  raw_counts_matrix = expr,
  annotations_file  = file.path(WD, "annot_infercnv.txt"),
  gene_order_file   = file.path(WD, "gene_order.txt"),
  ref_group_names   = refs,
  chr_exclude       = c("chrM", "chrX", "chrY", "chrUn", "random", "chrEBV")
)

infercnv_obj <- infercnv::run(
  infercnv_obj,
  cutoff              = 1,              # 默认
  out_dir             = OUT,
  analysis_mode       = "samples",      # 分组模式; 默认 subclusters 在 Seurat5 下 .single_tumor_leiden_subclustering 报错
  cluster_by_groups   = TRUE,           # 默认
  denoise             = FALSE,          # 默认
  HMM                 = TRUE,           # 默认 FALSE -> 本处开启
  HMM_type            = "i6",           # 默认 i6 (i6/i3 中第一项)
  BayesMaxPNormal     = 0.5,            # 默认
  num_threads         = 16,             # 默认 4 -> 本处 16
  no_plot             = TRUE,           # 默认 FALSE -> 本处 TRUE (按要求不绘图)
  save_rds            = TRUE,
  save_final_rds      = TRUE,
  plot_steps          = FALSE,
  resume_mode         = TRUE
)
cat("inferCNV done ->", OUT, "\n")
