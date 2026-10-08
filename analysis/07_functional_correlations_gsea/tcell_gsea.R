# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")
#!/usr/bin/env Rscript
# =============================================================================
# Step02 (nr1r1_analysis_M5/gsea): GSEA of the NR1 + R1 CAR+ spots ranked by log2FC(PR/GR)
# -----------------------------------------------------------------------------
# Grouping : GRPR_quad_M5 (current M5-based T-cell-function grouping; 66 GR / 50 PR
#            CAR+ spots, 103 GR / 89 PR) -> ranking from nr1r1_rank.py
# Ranking  : log2( (CPM_PR + 1) / (CPM_GR + 1) ) of the NR1 pseudobulk CPM, descending
#            => positive metric / positive NES = PR-high (exhausted / poor-response side)
#               negative NES                    = GR-high (memory-functional side)
# Gene sets: MSigDB mouse (msigdbr 24.1.0)
#            primary universe = T-cell-function terms of H + C2:CP:REACTOME +
#            C2:CP:KEGG_LEGACY + C5:GO:BP, each annotated with the direction expected
#            from T-cell biology; supplementary = C7:IMMUNESIGDB T-cell x state sets
#            (flagged: the quad_M5 panels were derived from C7 -> partly circular).
# Outputs  : tables/*.csv, rds/NR1_Tcell_GSEA_results.rds (+ combined gseaResult)
# =============================================================================
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({
  library(clusterProfiler); library(msigdbr); library(data.table); })

WD   <- paste0(PROJECT_ROOT, "/mouse/cart_region/nr1r1_analysis_M5/gsea")
TAB  <- file.path(WD, "tables"); RDS <- file.path(WD, "rds")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)
dir.create(RDS, showWarnings = FALSE, recursive = TRUE)
RANK_CSV <- file.path(TAB, "NR1R1_rank_log2FC_PR_GR.csv")
NPERM <- 10000

## ---------------------------------------------------------------- 1. ranking
R <- fread(RANK_CSV)
## fgsea needs a tie-free ordering: rank_metric = log2FC(PR/GR) with exact ties spread inside a
## band narrower than the smallest gap between distinct values (see step01)
gl <- if ("rank_metric" %in% names(R)) R$rank_metric else R$log2FC_PR_GR
names(gl) <- R$gene
gl <- gl[is.finite(gl) & !is.na(names(gl)) & nzchar(names(gl))]
gl <- sort(gl[!duplicated(names(gl))], decreasing = TRUE)
cat(sprintf("ranking: %d genes | %+.2f .. %+.2f (descending log2FC(PR/GR))\n",
            length(gl), min(gl), max(gl)))

## ------------------------------------------------- 2. MSigDB mouse collections
get_coll <- function(coll, sub = NULL, tag) {
  g <- suppressWarnings(if (is.null(sub)) msigdbr(species = "Mus musculus", collection = coll)
                        else msigdbr(species = "Mus musculus", collection = coll, subcollection = sub))
  d <- unique(as.data.frame(g[, c("gs_name", "gene_symbol")]))
  d <- d[!is.na(d$gs_name) & !is.na(d$gene_symbol) & nzchar(d$gene_symbol), ]
  d$collection <- tag
  setDT(d)          # data.table syntax is used on the result below (C7 block)
  d[]
}
COLL <- rbindlist(list(
  get_coll("H",  NULL,             "H"),
  get_coll("C2", "CP:REACTOME",    "C2:CP:REACTOME"),
  get_coll("C2", "CP:KEGG_LEGACY", "C2:CP:KEGG_LEGACY"),
  get_coll("C5", "GO:BP",          "C5:GO:BP")
))
COLL <- unique(COLL)
cat(sprintf("MSigDB mouse: %d sets / %d rows\n", length(unique(COLL$gs_name)), nrow(COLL)))

## ------------------------------------------- 3. T-cell-function gene-set filter
## word-boundary anchored so that FAT_CELL / FIBROBLAST_CELL do not leak in
TC_PAT <- paste0(
  "(^|_)T_CELL", "(^|_)TCELL", "(^|_)T_LYMPHOCYTE", "(^|_)CD4(_|$)", "(^|_)CD8(_|$)",
  "|(^|_)CD4_POSITIVE", "|(^|_)CD8_POSITIVE", "|(^|_)TH1(_|$)", "|(^|_)TH2(_|$)",
  "|(^|_)TH17(_|$)", "|(^|_)TREG", "|REGULATORY_T_CELL", "|REGULATORY_T_LYMPHOCYTE",
  "|(^|_)TCR(_|$)", "|T_CELL_RECEPTOR", "|IMMUNOLOGIC_MEMORY", "|ADAPTIVE_IMMUNE_MEMORY",
  "|MEMORY_T_CELL", "|LYMPHOCYTE_ACTIVATION", "|LYMPHOCYTE_COSTIMULATION", "|LYMPHOCYTE_ANERGY",
  "|LYMPHOCYTE_MEDIATED_IMMUNITY", "|LYMPHOCYTE_DIFFERENTIATION", "|LYMPHOCYTE_PROLIFERATION",
  "|T_CELL_MEDIATED_CYTOTOXICITY", "|LEUKOCYTE_MEDIATED_CYTOTOXICITY", "|NK_CELL",
  "|NATURAL_KILLER", "|CYTOLYSIS", "|INTERFERON_GAMMA", "|IL2_STAT5", "|INTERLEUKIN_2",
  "|RESPONSE_TO_INTERLEUKIN_2", "|ALLOGRAFT", "|GRAFT_VERSUS_HOST", "|ANTIGEN_PROCESSING",
  "|ANTIGEN_PRESENTATION", "|MHC_CLASS_I", "|MHC_CLASS_II", "|T_CELL_TOLERANCE",
  "|T_CELL_ANERGY", "|T_CELL_ACTIVATION", "|T_CELL_COSTIMULATION", "|T_CELL_EXHAUSTION",
  "|T_CELL_MEDIATED_IMMUNITY", "|GRANZYME", "|PERFORIN")
SETS <- unique(COLL$gs_name)
TC_SETS <- SETS[grepl(TC_PAT, SETS)]
cat(sprintf("T-cell-function sets in H+C2+C5: %d\n", length(TC_SETS)))
T2G_TC <- unique(COLL[gs_name %in% TC_SETS, .(gs_name, gene_symbol)])

## ----------------------------------- 4. expected direction from T-cell biology
## PR-high (positive NES) = the dysfunctional / suppressed / exhausted side
PR_EXP <- paste0("EXHAUST|ANERG|TOLERAN|REGULATORY_T|(^|_)TREG|SUPPRESS|SENESCEN|",
                 "NEGATIVE_REGULATION|APOPTOT|INHIBIT|CHECKPOINT|PD_1|PD_L1|CTLA4|LAG3|",
                 "HAVCR2|TIGIT|DYSFUNCTION|DYSREGULAT|ATTENUAT|TERMINAL_DIFFERENTIATION")
## GR-high (negative NES) = the functional / memory / effector side
GR_EXP <- paste0("ACTIVATION|ACTIVAT|EFFECTOR|CYTOTOX|CYTOLYSIS|MEMORY|COSTIMUL|PROLIFERAT|",
                 "DIFFERENTIAT|MATURATION|SELECTION|HOMEOSTASIS|EXPANSION|SIGNALING|RESPONSE|",
                 "IL2|INTERLEUKIN_2|STAT5|INTERFERON_GAMMA|ANTIGEN|MHC_CLASS|LYMPHOCYTE_MEDIATED|",
                 "GRANZYME|PERFORIN|NK_CELL|NATURAL_KILLER|CYTOTOXICITY|SURVIVAL|MIGRATION|ADHESION")
classify <- function(nm) {
  if (grepl(PR_EXP, nm)) return(c("PR-high (dysfunction / suppression)", 1))
  if (grepl(GR_EXP, nm)) return(c("GR-high (T-cell function / effector / memory)", -1))
  c("context (not assessed)", NA)
}
EXP <- as.data.table(t(vapply(TC_SETS, classify, character(2))))
setnames(EXP, c("expect", "dir"))
EXP[, `:=`(ID = TC_SETS, dir = as.numeric(dir))]
cat("expected-direction breakdown:\n"); print(EXP[, .N, by = expect])

## ------------------------------------------------------------- 5. GSEA runs
run_gsea <- function(t2g, tag) {
  set.seed(1)
  t0 <- Sys.time()
  o <- tryCatch(clusterProfiler::GSEA(geneList = gl, TERM2GENE = as.data.frame(t2g),
                                      minGSSize = 10, maxGSSize = 500, pvalueCutoff = 1,
                                      pAdjustMethod = "BH", eps = 0, nPermSimple = NPERM,
                                      verbose = FALSE, seed = TRUE),
                error = function(e) {
                  cat("  !! ", tag, ":", conditionMessage(e), "\n")
                  cl <- sys.calls()
                  cat("     calls:", paste(utils::tail(vapply(cl, function(z)
                      paste(deparse(z)[1], collapse = ""), character(1)), 6), collapse = " <- "), "\n")
                  NULL })
  if (!is.null(o)) cat(sprintf("  %-22s sets tested %5d  (%.1fs)\n", tag, nrow(as.data.frame(o)),
                               as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  o
}
cat("\nGSEA (T-cell-function universe):\n")
g_tc  <- run_gsea(T2G_TC, "T-cell combined")
g_by  <- list()
for (cl in unique(COLL$collection)) {
  s <- intersect(TC_SETS, unique(COLL[collection == cl, gs_name]))
  if (!length(s)) next
  g_by[[cl]] <- run_gsea(unique(COLL[gs_name %in% s, .(gs_name, gene_symbol)]), cl)
}

## global context: the same ranking against each *unfiltered* collection
cat("\nGSEA (unfiltered collections, for global q context):\n")
g_full <- list()
for (cl in unique(COLL$collection)) {
  g_full[[cl]] <- run_gsea(unique(COLL[collection == cl, .(gs_name, gene_symbol)]), paste0(cl, " (all)"))
}

## supplementary: C7 T-cell x state sets (direction parsed from the set name)
C7 <- get_coll("C7", "IMMUNESIGDB", "C7:IMMUNESIGDB")
c7sets <- unique(C7$gs_name)
c7sets <- c7sets[grepl("T_CELL|TCELL|T_LYMPHOCYTE|_CD4|_CD8|CD4_|CD8_|TH1|TH2|TH17|TREG", c7sets) &
                 grepl("EFF|EFFECTOR|ACTIV|MEMOR|EXHAUST|NAIVE|ANERG|TOLERAN|SENESC", c7sets)]
cat_stat <- function(s) {
  s <- paste(s, collapse = "_")
  if (grepl("EXHAUST|ANERG|TOLERAN|DYSFUNCT|SENESC", s)) "dys"
  else if (grepl("TREG", s)) "treg"
  else if (grepl("NAIVE", s)) "naive"
  else if (grepl("MEMORY|EFF|EFFECTOR|ACTIV|STEM_CELL", s)) "fun"
  else NA_character_
}
parse_c7 <- function(nm) {
  end <- if (grepl("_UP$", nm)) "UP" else if (grepl("_DN$", nm)) "DN" else NA_character_
  core <- sub("_(UP|DN)$", "", nm); tk <- strsplit(core, "_")[[1]]
  vi <- which(tk == "VS")
  if (!length(vi) || is.na(end)) return(list(state_hi = NA, state_lo = NA, expect = "context (not assessed)", dir = NA))
  a <- tk[seq_len(vi[1] - 1)]; b <- tk[(vi[1] + 1):length(tk)]
  sa <- paste(utils::tail(a, 2), collapse = "_"); sb <- paste(utils::head(b, 3), collapse = "_")
  ca <- cat_stat(sa); cb <- cat_stat(sb)
  ## state A reads "<study>_<STATE>" (tail), state B reads "<STATE>_<cell>" (head); _UP = higher in A
  chi <- if (end == "UP") ca else cb
  clo <- if (end == "UP") cb else ca
  ex <- if (!is.na(chi) && chi == "fun" && !is.na(clo) && clo %in% c("dys", "treg", "naive"))
          "GR-high (T-cell function / effector / memory)"
        else if (!is.na(chi) && chi %in% c("dys", "treg") && !is.na(clo) && clo == "fun")
          "PR-high (dysfunction / suppression)"
        else "context (not assessed)"
  dr <- if (grepl("^GR-high", ex)) -1 else if (grepl("^PR-high", ex)) 1 else NA
  list(state_hi = if (end == "UP") sa else sb, state_lo = if (end == "UP") sb else sa, expect = ex, dir = dr)
}
cat(sprintf("\nC7 T-cell x state sets: %d\n", length(c7sets)))
g_c7 <- if (length(c7sets)) run_gsea(unique(C7[gs_name %in% c7sets, .(gs_name, gene_symbol)]), "C7 T-cell x state") else NULL

## ------------------------------------------------------- 6. tables + RDS
fmt_res <- function(o, tag) {
  if (is.null(o)) return(NULL)
  d <- as.data.table(as.data.frame(o))
  if ("enrichmentScore" %in% names(d)) setnames(d, "enrichmentScore", "ES")
  d[, `:=`(collection = tag, enriched_in = ifelse(NES < 0, "GR", "PR"))]
  d[]
}
A <- fmt_res(g_tc, "T-cell (H+C2+C5, combined)")
A <- merge(A, EXP[, .(ID, expect, dir)], by = "ID", all.x = TRUE)
A[, `:=`(dir_match = ifelse(is.na(dir), NA, sign(NES) == dir),
         call = ifelse(is.na(expect), "not assessed",
                       ifelse(sign(NES) == dir, "direction matches expectation",
                              "direction opposite to expectation")))]
A <- A[order(-NES)]
fwrite(A[, .(ID, setSize, ES, NES, pvalue, p.adjust, collection, enriched_in, expect, dir_match, call,
             core_enrichment)],
       file.path(TAB, "NR1R1_Tcell_GSEA_universe.csv"))

SEL <- A[!is.na(dir_match) & dir_match == TRUE & pvalue < 0.05][order(-abs(NES))]
SEL[, tier := "match, p<0.05"]
TR <- A[!is.na(dir_match) & dir_match == TRUE & pvalue >= 0.05 & pvalue < 0.25][order(-abs(NES))]
TR[, tier := "match, p<0.25"]
OPP <- A[!is.na(dir_match) & dir_match == FALSE & pvalue < 0.05][order(-abs(NES))]
OPP[, tier := "opposite, p<0.05"]
fwrite(rbind(SEL, TR, OPP, fill = TRUE)[, .(tier, ID, setSize, NES, pvalue, p.adjust, enriched_in,
                                           expect, core_enrichment)],
       file.path(TAB, "NR1R1_Tcell_GSEA_selected.csv"))

BY <- rbindlist(lapply(names(g_by), function(n) fmt_res(g_by[[n]], n)), fill = TRUE)
BY <- merge(BY, EXP[, .(ID, expect, dir)], by = "ID", all.x = TRUE)
BY[, call := ifelse(is.na(dir), "not assessed",
                    ifelse(sign(NES) == dir, "direction matches expectation",
                           "direction opposite to expectation"))]
fwrite(BY[order(collection, -NES), .(collection, ID, setSize, NES, pvalue, p.adjust, expect, call,
                                     core_enrichment)],
       file.path(TAB, "NR1R1_Tcell_GSEA_by_collection.csv"))

FULL <- rbindlist(lapply(names(g_full), function(n) fmt_res(g_full[[n]], n)), fill = TRUE)
fwrite(FULL[order(collection, pvalue), .(collection, ID, setSize, NES, pvalue, p.adjust, enriched_in)],
       file.path(TAB, "NR1R1_Tcell_GSEA_global_context.csv"))

## global q of every tested T-cell set (same ranking, unfiltered H / C2 / C5 background)
FULL[, key := ID]
GQ <- FULL[, .(ID, global_collection = collection, global_p = pvalue, global_q = p.adjust, global_NES = NES)]
A2 <- merge(A, GQ[!duplicated(GQ, by = c("ID", "global_collection"))], by = "ID", all.x = TRUE)
fwrite(A2[order(pvalue), .(ID, setSize, NES, pvalue, p.adjust, enriched_in, expect, call,
                           global_collection, global_p, global_q, global_NES)],
       file.path(TAB, "NR1R1_Tcell_GSEA_universe_with_global_q.csv"))

C7T <- if (!is.null(g_c7)) {
  d <- fmt_res(g_c7, "C7:IMMUNESIGDB")
  p <- rbindlist(lapply(d$ID, function(n) as.data.table(parse_c7(n))))
  d <- cbind(d, p)
  d[, call := ifelse(is.na(dir), "not assessed",
                     ifelse(sign(NES) == dir, "direction matches expectation",
                            "direction opposite to expectation"))]
  # overlap with the clinical panels used to build the quad_M5 grouping (circularity flag)
  PJ <- paste0(PROJECT_ROOT, "/mouse/cart_region/grpr_pred_v3/clinical_gsva/panels_final.json")
  if (file.exists(PJ) && requireNamespace("jsonlite", quietly = TRUE)) {
    pan <- unique(unlist(jsonlite::fromJSON(PJ)))
    ov <- function(nm) { g <- unique(d[ID == nm, strsplit(core_enrichment, "/")[[1]]])
      length(intersect(g, pan)) / max(length(g), 1) }
    d[, panel_overlap := vapply(ID, ov, numeric(1))]
  }
  d[order(-NES)]
} else NULL
if (!is.null(C7T))
  fwrite(C7T[, .(ID, setSize, NES, pvalue, p.adjust, enriched_in, state_hi, state_lo, expect, call,
                 panel_overlap, core_enrichment)],
         file.path(TAB, "NR1R1_Tcell_GSEA_C7_supplementary.csv"))

## ------------------------------------------------------------- plot set list
PLOT <- SEL[!duplicated(ID)]
if (nrow(PLOT) == 0) {
  PLOT <- TR[!duplicated(ID)]
  cat("\nNOTE: no direction-matching set at p<0.05 -> plotting the p<0.25 matches\n")
}
PLOT <- head(PLOT[order(-abs(NES))], 6L)
if (nrow(PLOT) == 0) {   # still nothing: strongest sets overall, flagged
  PLOT <- head(A[order(-abs(NES))], 6L); PLOT[, tier := "no direction match - strongest |NES|"]
  cat("NOTE: no direction-matching set at p<0.25 either -> plotting the strongest |NES| sets\n")
}
PLOT[, direction_label := ifelse(NES < 0, "GR", "PR")]
PLOT[, source := "primary (H+C2+C5 T-cell universe)"]

## one supplementary C7 panel per direction (C7 = the collection the panels were derived from,
## so these are flagged separately) - this is where the only PR-side exhaustion set sits
if (!is.null(C7T)) {
  C7S <- C7T[!is.na(dir) & sign(NES) == dir & pvalue < 0.05]
  if (nrow(C7S)) {
    pk <- rbind(C7S[enriched_in == "PR"][order(-abs(NES))][1], C7S[enriched_in == "GR"][order(-abs(NES))][1])
    pk <- pk[!is.na(ID)]
    if (nrow(pk)) {
      pk[, `:=`(tier = "C7 supplementary, p<0.05", source = "C7:IMMUNESIGDB (supplementary)",
                direction_label = ifelse(NES < 0, "GR", "PR"))]
      COLS <- c("ID", "setSize", "NES", "pvalue", "p.adjust", "enriched_in", "expect", "tier",
                "direction_label", "source")
      pk <- pk[!ID %in% PLOT$ID]
      if (nrow(pk)) PLOT <- rbind(PLOT[, ..COLS], pk[, ..COLS], fill = TRUE)
    }
  }
}
fwrite(PLOT[, .(ID, setSize, NES, pvalue, p.adjust, enriched_in, expect, tier, direction_label, source)],
       file.path(TAB, "NR1R1_Tcell_GSEA_plot_sets.csv"))

saveRDS(list(ranking = gl, rank_table = R, params = list(metric = "log2FC(PR/GR) pseudobulk CPM NR1",
          grouping = "GRPR_quad_M5", samples = "NR1+R1 (Control/Vehicle excluded)", minGSSize = 10, maxGSSize = 500, nPermSimple = NPERM,
          collections = unique(COLL$collection), n_genes = length(gl)),
          tcell_universe = T2G_TC, expectation = EXP, universe_annotated = A,
          selected = rbind(SEL, TR, OPP, fill = TRUE), plot_sets = PLOT,
          gsea = list(tcell_combined = g_tc, by_collection = g_by, unfiltered = g_full, c7 = g_c7),
          global_context = FULL, c7_table = C7T, session = sessionInfo()),
          file.path(RDS, "NR1R1_Tcell_GSEA_results.rds"))
if (!is.null(g_tc)) saveRDS(g_tc, file.path(RDS, "NR1R1_Tcell_GSEA_gseaResult_combined.rds"))

## ------------------------------------------------------------- console report
cat("\n================ direction-matching T-cell-function GSEA (p<0.05) ================\n")
if (nrow(SEL)) print(SEL[, .(ID, setSize, NES = round(NES, 2), p = signif(pvalue, 2),
                             q = signif(p.adjust, 2), enriched_in)]) else cat("(none)\n")
cat(sprintf("\nmatching p<0.25: %d | matching p<0.05: %d | opposite p<0.05: %d | tested: %d\n",
            nrow(TR) + nrow(SEL), nrow(SEL), nrow(OPP), nrow(A)))
if (nrow(OPP)) { cat("\nopposite-direction sets (p<0.05), listed for transparency:\n")
  print(OPP[, .(ID, setSize, NES = round(NES, 2), p = signif(pvalue, 2), expect)]) }
cat("\nplot sets:\n"); print(PLOT[, .(ID, NES = round(NES, 2), p = signif(pvalue, 2), tier)])
cat("\nwrote tables/ (6 csv) + rds/NR1_Tcell_GSEA_results.rds\n")
