# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# =============================================================================
# survival_tests_generic.R — log-rank + Cox for a list of grouping/exposure
#                           columns, using the `survival` package.
#
#   R_LIBS_USER=/home/wr/R_4.3 /usr/bin/Rscript scripts/survival_tests_generic.R \
#       <patient_table.csv> <outdir> <time_col> <event_col> <col1,col2,...> [km_col]
#
# A column with exactly 2 distinct values is treated as a binary grouping
# (survdiff + coxph on an explicit 0/1 numeric, so the hazard ratio is
# unambiguously "1 vs 0"); anything else is treated as a continuous exposure
# (coxph per 1 SD, sample SD as in scale()).
#
# Age and medication/treatment columns are asserted absent.
# The optional 6th argument names one binary column to draw a KM curve for.
# =============================================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) stop("usage: survival_tests_generic.R <table> <outdir> <time> <event> <cols> [km_col]")
in_csv <- args[1]; outdir <- args[2]; time_col <- args[3]; event_col <- args[4]
cols <- strsplit(args[5], ",")[[1]]
km_col <- if (length(args) >= 6) args[6] else NA

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages({library(survival);  })
cat("R:", R.version.string, "| survival:", as.character(packageVersion("survival")), "\n")

d <- read.csv(in_csv, stringsAsFactors = FALSE, check.names = FALSE)
forbidden <- grep("(^|[_ .])age([_ .]|$)|drug|medicat|treatment|therapy|chemo|regimen",
                  names(d), ignore.case = TRUE, value = TRUE)
if (length(forbidden)) stop("forbidden covariates: ", paste(forbidden, collapse = ", "))
cat("asserted: no Age / medication / treatment column\n")

d$.t <- as.numeric(d[[time_col]]); d$.e <- as.integer(d[[event_col]])
d <- d[!is.na(d$.t) & !is.na(d$.e) & d$.t > 0, , drop = FALSE]
cat(sprintf("cohort: %d patients, %d events\n", nrow(d), sum(d$.e)))

# every row must carry the SAME columns or rbind() fails; binary-group and
# continuous rows populate different subsets of this schema
mkrow <- function(v, kind, ...) {
  r <- data.frame(variable = v, kind = kind,
                  level1 = NA_character_, level0 = NA_character_,
                  n = NA_integer_, n_events = NA_integer_,
                  n_level1 = NA_integer_, n_level0 = NA_integer_,
                  median_surv_level1 = NA_real_, median_surv_level0 = NA_real_,
                  logrank_chisq = NA_real_, logrank_df = NA_integer_, logrank_p = NA_real_,
                  HR_level1_vs_level0 = NA_real_, HR_per_SD = NA_real_,
                  CI_low = NA_real_, CI_high = NA_real_, z_stat = NA_real_,
                  cox_p = NA_real_, note = NA_character_, stringsAsFactors = FALSE)
  vals <- list(...)
  for (nm in names(vals)) r[[nm]] <- vals[[nm]]
  r
}

rows <- list()
for (v in cols) {
  if (!v %in% names(d)) next
  x <- d[[v]]
  u <- unique(na.omit(x))
  km <- FALSE
  if (length(u) == 2) {
    bin <- as.integer(as.character(x) == as.character(sort(u)[2]))
    keep <- !is.na(x)
    dd <- d[keep, , drop = FALSE]; dd$g <- bin[keep]
    fit <- survfit(Surv(.t, .e) ~ g, data = dd)
    sd_ <- survdiff(Surv(.t, .e) ~ g, data = dd)
    # explicit numeric 0/1 so the HR is "1 vs 0" and cannot be silently inverted
    cx <- coxph(Surv(.t, .e) ~ g, data = dd)
    scx <- summary(cx)
    chisq <- as.numeric(sd_$chisq); dfree <- length(sd_$n) - 1
    p_lr <- pchisq(chisq, dfree, lower.tail = FALSE)
    tb <- summary(fit)$table
    if (is.null(dim(tb))) tb <- matrix(tb, nrow = 1, dimnames = list("g=1", names(tb)))
    med <- function(pat) { i <- grep(pat, rownames(tb)); if (!length(i)) NA_real_ else as.numeric(tb[i[1], "median"]) }
    rows[[v]] <- mkrow(v, "binary_group",
      level1 = sprintf("g=1 (%s)", as.character(sort(u)[2])),
      level0 = sprintf("g=0 (%s)", as.character(sort(u)[1])),
      n = nrow(dd), n_events = sum(dd$.e), n_level1 = sum(dd$g == 1), n_level0 = sum(dd$g == 0),
      median_surv_level1 = med("g=1"), median_surv_level0 = med("g=0"),
      logrank_chisq = chisq, logrank_df = as.integer(dfree), logrank_p = p_lr,
      HR_level1_vs_level0 = as.numeric(scx$conf.int[1, "exp(coef)"]),
      CI_low = as.numeric(scx$conf.int[1, "lower .95"]),
      CI_high = as.numeric(scx$conf.int[1, "upper .95"]),
      cox_p = as.numeric(scx$coefficients[1, "Pr(>|z|)"]),
      note = "survival::survdiff + coxph, unadjusted; Age excluded")
    if (!is.na(km_col) && identical(v, km_col)) km <- TRUE
    kmobj <- if (km) list(fit = fit, dd = dd, v = v) else NULL
  } else {
    z <- as.numeric(scale(as.numeric(x)))   # sample SD, matching scale()
    keep <- !is.na(z)
    if (sum(keep) < 5 || length(u) < 3) {
      rows[[v]] <- mkrow(v, "continuous", n = sum(keep), n_events = sum(d$.e[keep]),
                          note = "not estimable")
      next
    }
    cx <- coxph(Surv(d$.t[keep], d$.e[keep]) ~ z[keep])
    scx <- summary(cx)
    rows[[v]] <- mkrow(v, "continuous", n = sum(keep), n_events = sum(d$.e[keep]),
      HR_per_SD = as.numeric(scx$conf.int[1, "exp(coef)"]),
      CI_low = as.numeric(scx$conf.int[1, "lower .95"]),
      CI_high = as.numeric(scx$conf.int[1, "upper .95"]),
      z_stat = as.numeric(scx$coefficients[1, "z"]),
      cox_p = as.numeric(scx$coefficients[1, "Pr(>|z|)"]),
      note = "survival::coxph, single covariate z-scored to 1 SD (sample SD), unadjusted; Age excluded")
  }
}
res <- do.call(rbind, rows)
res$p_BH <- p.adjust(res$cox_p, method = "BH")
res$significant_BH <- res$p_BH < 0.05
if ("logrank_p" %in% names(res)) res$logrank_p_BH <- p.adjust(res$logrank_p, method = "BH")
write.csv(res, file.path(outdir, "composition_survival_tests.csv"), row.names = FALSE)

cat("\n=== survival tests ===\n")
print(res, row.names = FALSE)

if (!is.na(km_col) && exists("kmobj") && !is.null(kmobj)) {
  for (ext in c("pdf", "png")) {
    f <- file.path(outdir, paste0("km_composition_subtype.", ext))
  }
  cat("\nKM written for", km_col, "\n")
}
cat("\nwritten:", file.path(outdir, "composition_survival_tests.csv"), "\n")
