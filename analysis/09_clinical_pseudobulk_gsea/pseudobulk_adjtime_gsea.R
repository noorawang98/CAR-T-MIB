# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step108c: 仅对伪bulk 时点校正模型 (~ group + tp) 的 DE 结果做 GSEA, 并汇总关键通路 NES
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(clusterProfiler); library(msigdbr); library(data.table)})
WD <- Sys.getenv("ST_WD", paste0(PROJECT_ROOT, "/destvi_260911")); PROG <- file.path(WD, "prognosis")
set.seed(1); T2G <- list()
for (cs in list(list("H", NULL), list("C5","GO:BP"), list("C2","CP:KEGG_LEGACY"), list("C2","CP:REACTOME"))) {
  g <- tryCatch(if (is.null(cs[[2]])) suppressWarnings(msigdbr(species="Homo sapiens", collection=cs[[1]]))
                else suppressWarnings(msigdbr(species="Homo sapiens", collection=cs[[1]], subcollection=cs[[2]])), error=function(e) NULL)
  if (is.null(g) || !nrow(g)) next
  T2G[[if (is.null(cs[[2]])) cs[[1]] else cs[[2]]]] <- as.data.frame(g[, c("gs_name","gene_symbol")])
}
runf <- function(stats, label) {
  nmv <- names(stats); keep <- is.finite(stats) & !is.na(nmv) & nzchar(nmv) & grepl("^[A-Za-z][A-Za-z0-9.\\-]*$", nmv)
  st <- stats[keep]
  st <- st[!duplicated(names(st))]
  st <- sort(st, decreasing = TRUE)
  out <- list()
  for (nm in names(T2G)) {
    r <- tryCatch(as.data.frame(clusterProfiler::GSEA(geneList=st, TERM2GENE=T2G[[nm]], minGSSize=10, maxGSSize=500,
                                                     pvalueCutoff=1, verbose=FALSE, seed=TRUE)), error=function(e) NULL)
    if (is.null(r) || !nrow(r)) next
    r$collection <- nm; out[[nm]] <- r[, c("ID","setSize","NES","pvalue","p.adjust","collection")]
  }
  R <- rbindlist(out); setnames(R, "ID", "pathway"); R <- R[order(p.adjust)]; R$label <- label
  write.csv(as.data.frame(R), file.path(PROG, paste0("gsea_strat_", label, ".csv")), row.names = FALSE)
  cat(sprintf("[%s] 基因 %d; padj<0.25 %d\n", label, length(st), sum(R$p.adjust<.25)))
  R
}
ALL <- list()
for (tag in c("mouse","destvi")) {
  f <- file.path(PROG, paste0("pervimib_de_", tag, "_adjTime.csv"))
  if (!file.exists(f)) { cat("缺", f, "\n"); next }
  d <- read.csv(f)
  gcol <- if ("gene" %in% names(d)) d$gene else rownames(d)
  ALL[[paste0("pseudobulk_", tag, "_adjTime")]] <- runf(setNames(d$logFC, gcol), paste0("pseudobulk_", tag, "_adjTime"))
}
KEY <- c("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION","REACTOME_COLLAGEN_CHAIN_TRIMERIZATION","REACTOME_ECM_PROTEOGLYCANS",
         "HALLMARK_TGF_BETA_SIGNALING","REACTOME_SIGNALING_BY_TGFB_FAMILY_MEMBERS","GOBP_FIBROBLAST_PROLIFERATION")
K <- rbindlist(lapply(names(ALL), function(k) { R <- ALL[[k]]; if (is.null(R)) return(NULL)
  R2 <- R[pathway %in% KEY]; if (!nrow(R2)) return(NULL); data.frame(label=k, R2[, .(pathway, NES, p.adjust)]) }), fill=TRUE)
write.csv(K, file.path(PROG, "gsea_strat_keypathways_adjTime.csv"), row.names = FALSE)
cat("\n=== 时点校正模型的关键通路 NES (正=PR 侧富集；负=SD/PD 侧富集) ===\n"); print(K)
cat("\nDONE\n")
