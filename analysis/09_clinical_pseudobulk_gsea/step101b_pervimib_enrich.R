# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step101b: perviMIB DE 结果的通路富集 (clusterProfiler: GO-BP + KEGG + Hallmark GSEA) 与火山图
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(clusterProfiler); library(org.Hs.eg.db);  library(msigdbr); library(fgsea); library(data.table); })
WD <- Sys.getenv("ST_WD", paste0(PROJECT_ROOT, "/destvi_260911")); PROG <- file.path(WD, "prognosis"); FIG <- file.path(WD, "figs")
pl <- function(v) v/25.4
for (tag in c("mouse", "destvi")) {
  f <- file.path(PROG, paste0("pervimib_de_", tag, ".csv"))
  if (!file.exists(f)) { cat("missing", f, "\n"); next }
  tt <- read.csv(f, check.names = FALSE)
  gl <- unique(tt$gene[tt$P.Value < 0.01 & abs(tt$logFC) > 0.3])
  cat(sprintf("\n[%s] DE genes (p<0.01 & |logFC|>0.3): %d\n", tag, length(gl)))
  # ---- GO-BP ----
  ego <- tryCatch(enrichGO(gl, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP",
                           pAdjustMethod = "BH", pvalueCutoff = 0.25, qvalueCutoff = 0.25, readable = TRUE),
                  error = function(e) NULL)
  if (!is.null(ego) && nrow(as.data.frame(ego))) {
    E <- as.data.frame(ego); write.csv(E, file.path(PROG, paste0("pervimib_enrichGO_", tag, ".csv")), row.names = FALSE)
    cat("  GO-BP 显著项:", nrow(E), "; top:", paste(head(E$Description, 5), collapse = " | "), "\n")
  } else cat("  GO-BP: 无\n")
  # ---- KEGG ----
  eg <- tryCatch(bitr(gl, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db), error = function(e) NULL)
  ekg <- NULL
  if (!is.null(eg) && nrow(eg) > 5) {
    ekg <- tryCatch(enrichKEGG(eg$ENTREZID, organism = "hsa", pvalueCutoff = 0.25, qvalueCutoff = 0.25), error = function(e) NULL)
    if (!is.null(ekg) && nrow(as.data.frame(ekg))) {
      K <- as.data.frame(setReadable(ekg, org.Hs.eg.db, "ENTREZID"))
      write.csv(K, file.path(PROG, paste0("pervimib_enrichKEGG_", tag, ".csv")), row.names = FALSE)
      cat("  KEGG 显著项:", nrow(K), "; top:", paste(head(K$Description, 5), collapse = " | "), "\n")
    } else cat("  KEGG: 无\n")
  }
  # ---- Hallmark GSEA (fgsea, 保护式) ----
  hm <- suppressWarnings(msigdbr(species = "Homo sapiens", collection = "H"))
  gs <- split(hm$gene_symbol[!is.na(hm$gene_symbol)], hm$gs_name)
  gs <- lapply(gs, function(x) unique(x[!is.na(x)])); gs <- gs[sapply(gs, length) >= 10]
  rk <- tt$t; names(rk) <- tt$gene; rk <- rk[is.finite(rk)]; rk <- rk[!duplicated(names(rk))]; rk <- sort(rk, decreasing = TRUE)
  fg <- tryCatch(as.data.frame(fgsea(pathways = gs, stats = rk, minSize = 10, maxSize = 500, nPermSimple = 5000)),
                 error = function(e) { cat("  fgsea 报错:", conditionMessage(e), "\n"); NULL })
  if (!is.null(fg)) {
    fg <- fg[order(fg$padj), c("pathway","size","NES","pval","padj")]
    write.csv(fg, file.path(PROG, paste0("pervimib_enrichHallmark_", tag, ".csv")), row.names = FALSE)
    cat("  Hallmark padj<0.25:", sum(fg$padj < .25, na.rm = TRUE), "; top:", paste(head(fg$pathway, 3), collapse = " | "), "\n")
  }
}
cat("DONE\n")
