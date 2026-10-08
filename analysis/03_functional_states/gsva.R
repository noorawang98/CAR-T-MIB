# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(GSVA); library(jsonlite)})
OUT <- paste0(PROJECT_ROOT, "/mouse/cart_region/grpr_gsva_v2")
E <- t(as.matrix(read.csv(file.path(OUT, "expr_for_gsva.csv"), row.names = 1, check.names = FALSE)))  # genes x spots
P <- fromJSON(file.path(OUT, "panels_for_gsva.json"))
P <- lapply(P, function(g) intersect(g, rownames(E)))
P <- P[sapply(P, length) >= 5]
cat("expr:", dim(E), "| panels:", paste(names(P), sapply(P, length), sep="=", collapse=" "), "\n")
set.seed(1)
sc <- GSVA::gsva(E, P, kcdf = "Gaussian", mx.diff = TRUE, verbose = FALSE)
S <- as.data.frame(t(sc))
write.csv(S, file.path(OUT, "gsva_scores.csv"))
cat("GSVA scores:", dim(S), "\n"); print(round(sapply(S, mean), 3))
