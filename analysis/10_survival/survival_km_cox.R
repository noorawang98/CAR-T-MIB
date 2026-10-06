# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# =============================================================================
# survival_km_cox.R — Kaplan-Meier, log-rank and Cox for the CCR1+ M2-dominated
#                     pvMIB-like grouping, using the `survival` package.
#
# Environment (R 4.3.2 with the user library ~/R_4.3):
#   R_LIBS_USER=/home/wr/R_4.3 /usr/bin/Rscript scripts/survival_km_cox.R \
#       <survival_patient_table.csv> <outdir>
#
# Age is deliberately absent from the input table (dropped upstream), and
# clinical.tsv contains no medication/treatment field, so no such covariate can
# enter any model. This is asserted below and recorded in the output.
#
# Outputs (in <outdir>):
#   survival_logrank_cox.csv     survdiff + coxph per grouping
#   survival_km_curves.csv       KM step coordinates with 95% CI
#   survival_cox_continuous.csv  coxph on continuous exposure (per 1 SD)
#   survival_followup.csv        median follow-up (reverse KM)
#   km_curve.pdf / km_curve.png  two-panel KM with risk tables
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: survival_km_cox.R <patient_table.csv> <outdir>")
in_csv <- args[1]
outdir <- args[2]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(survival)
  })

cat("R:", R.version.string, "\n")
cat("survival:", as.character(packageVersion("survival")), "\n")
cat("survminer:", as.character(packageVersion("survminer")), "\n")

d <- read.csv(in_csv, stringsAsFactors = FALSE, check.names = FALSE)

# ---- assert the forbidden covariates are absent ---------------------------- #
forbidden <- grep("(^|[_ .])age([_ .]|$)|drug|medicat|treatment|therapy|chemo|regimen",
                  names(d), ignore.case = TRUE, value = TRUE)
if (length(forbidden)) {
  stop("forbidden covariates present in the survival table: ",
       paste(forbidden, collapse = ", "))
}
cat("asserted: no Age / medication / treatment column in the input table\n")

d$os_days <- as.numeric(d$os_days)
d$os_event <- as.integer(d$os_event)
d <- d[!is.na(d$os_days) & !is.na(d$os_event) & d$os_days > 0, , drop = FALSE]
cat(sprintf("survival cohort: %d patients, %d events\n", nrow(d), sum(d$os_event)))

# ---- median follow-up by reverse KM ---------------------------------------- #
fu <- survfit(Surv(os_days, 1 - os_event) ~ 1, data = d)
med_fu <- summary(fu)$table[["median"]]
followup <- data.frame(n = nrow(d), events = sum(d$os_event),
                       median_followup_days = as.numeric(med_fu),
                       method = "reverse Kaplan-Meier")
write.csv(followup, file.path(outdir, "survival_followup.csv"), row.names = FALSE)

groupings <- list(
  list(name = "pvMIB_positive_vs_negative", var = "pvMIB_positive",
       pos = "pvMIB-like positive", neg = "pvMIB-like negative", primary = TRUE),
  list(name = "M1_region_positive_vs_negative", var = "M1_region_positive",
       pos = "M1 MIB region positive", neg = "M1 MIB region negative", primary = FALSE),
  list(name = "activated_pvMIB_vs_rest", var = "activated_pvMIB_present_any",
       pos = "activated pvMIB positive", neg = "no activated pvMIB", primary = FALSE)
)

as01 <- function(x) as.integer(as.logical(x))

stat_rows <- list()
km_rows <- list()
plots <- list()

for (gp in groupings) {
  if (!gp$var %in% names(d)) {
    cat("skip (column missing):", gp$var, "\n"); next
  }
  dd <- d
  dd$g <- as01(dd[[gp$var]])
  if (length(unique(dd$g)) < 2) {
    cat("skip (single group):", gp$name, "\n"); next
  }
  # `g` (factor, positive first) is only for survfit/survdiff plotting. The Cox
  # model uses an explicit numeric 0/1 so the HR is unambiguously
  # "positive vs negative": fitting coxph() directly on the factor makes the
  # FIRST level the reference and silently inverts the reported HR, which the
  # independent cross-check in validate_myeloid_survival.py is there to catch.
  dd$g <- factor(dd$g, levels = c(1, 0), labels = c("positive", "negative"))
  dd$g_num <- as01(dd[[gp$var]])   # 1 = positive = exposed

  fit <- survfit(Surv(os_days, os_event) ~ g, data = dd)
  sd  <- survdiff(Surv(os_days, os_event) ~ g, data = dd)
  cx  <- coxph(Surv(os_days, os_event) ~ g_num, data = dd)
  scx <- summary(cx)

  tb <- summary(fit)$table
  if (is.null(dim(tb))) tb <- matrix(tb, nrow = 1,
                                     dimnames = list("g=positive", names(tb)))
  getrow <- function(patt) {
    i <- grep(patt, rownames(tb))
    if (!length(i)) return(c(NA, NA, NA))
    c(tb[i[1], "median"], tb[i[1], "0.95LCL"], tb[i[1], "0.95UCL"])
  }
  posm <- getrow("positive"); negm <- getrow("negative")

  # proportional-hazards diagnostic: curves can cross, in which case HR is not
  # a valid summary of the effect
  zph <- tryCatch(cox.zph(cx), error = function(e) NULL)
  zph_p <- if (!is.null(zph)) as.numeric(zph$table["GLOBAL", "p"]) else NA_real_

  # PH-free alternative: restricted mean survival time up to tau, group difference
  # with a bootstrap CI (n is small, so this is cheap)
  tau <- as.numeric(quantile(dd$os_days, 0.75))
  rmst_at <- function(t, e, g, tau) {
    f <- survfit(Surv(t, e) ~ g)
    sm <- summary(f, times = sort(unique(c(0, t[t <= tau], tau))), extend = TRUE)
    st <- as.character(sm$strata)
    out <- c()
    for (lv in levels(g)) {
      idx <- grepl(paste0("g=", lv), st, fixed = TRUE)
      tt <- c(0, sm$time[idx]); ss <- c(1, sm$surv[idx])
      o <- order(tt); tt <- tt[o]; ss <- ss[o]
      keep <- tt <= tau
      tt <- c(tt[keep], tau); ss <- c(ss[keep], ss[length(ss[keep])])
      out[lv] <- sum(diff(tt) * head(ss, -1))
    }
    out
  }
  set.seed(20260918)
  r_obs <- rmst_at(dd$os_days, dd$os_event, dd$g, tau)
  diff_obs <- as.numeric(r_obs["positive"] - r_obs["negative"])
  boot <- replicate(500, {
    i <- sample(nrow(dd), replace = TRUE)
    tryCatch({
      rr <- rmst_at(dd$os_days[i], dd$os_event[i], dd$g[i], tau)
      as.numeric(rr["positive"] - rr["negative"])
    }, error = function(e) NA_real_)
  })
  diff_ci <- quantile(boot, c(0.025, 0.975), na.rm = TRUE)

  chisq <- as.numeric(sd$chisq)
  df_lr <- length(sd$n) - 1
  p_lr <- pchisq(chisq, df_lr, lower.tail = FALSE)
  hr  <- scx$conf.int[1, "exp(coef)"]
  lo  <- scx$conf.int[1, "lower .95"]
  hi  <- scx$conf.int[1, "upper .95"]
  p_cx <- scx$coefficients[1, "Pr(>|z|)"]

  n_pos <- sum(dd$g == "positive"); n_neg <- sum(dd$g == "negative")
  stat_rows[[gp$name]] <- data.frame(
    analysis = gp$name,
    grouping = sprintf("%s vs %s", gp$pos, gp$neg),
    is_primary = gp$primary,
    n_total = nrow(dd), n_events = sum(dd$os_event),
    n_positive = n_pos, n_negative = n_neg,
    median_survival_positive = posm[1], median_survival_positive_CI_low = posm[2],
    median_survival_positive_CI_high = posm[3],
    median_survival_negative = negm[1], median_survival_negative_CI_low = negm[2],
    median_survival_negative_CI_high = negm[3],
    logrank_chisq = chisq, logrank_df = df_lr, logrank_p = p_lr,
    logrank_O_positive = as.numeric(sd$obs[1]), logrank_E_positive = as.numeric(sd$exp[1]),
    cox_HR = as.numeric(hr), cox_HR_CI_low = as.numeric(lo), cox_HR_CI_high = as.numeric(hi),
    cox_z = as.numeric(scx$coefficients[1, "z"]), cox_p = as.numeric(p_cx),
    cox_n = as.numeric(scx$n), cox_nevent = as.numeric(scx$nevent),
    ph_global_p = zph_p,
    ph_assumption_ok = isTRUE(zph_p > 0.05),
    rmst_tau_days = tau,
    rmst_positive = as.numeric(r_obs["positive"]),
    rmst_negative = as.numeric(r_obs["negative"]),
    rmst_difference = diff_obs,
    rmst_difference_CI_low = as.numeric(diff_ci[1]),
    rmst_difference_CI_high = as.numeric(diff_ci[2]),
    cox_contrast = "positive vs negative (g_num: 1 = positive)",
    note = "survival::survdiff + coxph, unadjusted; Age excluded; no treatment covariate",
    stringsAsFactors = FALSE)

  sm <- summary(fit)
  strata_chr <- if (is.null(sm$strata)) rep("positive", length(sm$time)) else as.character(sm$strata)
  km_rows[[gp$name]] <- data.frame(
    analysis = gp$name,
    strata = strata_chr,
    group = ifelse(grepl("positive", strata_chr) & !grepl("negative", strata_chr),
                   "positive", "negative"),
    time = sm$time, n_risk = sm$n.risk, n_event = sm$n.event,
    n_censor = sm$n.censor, survival = sm$surv,
    lower = sm$lower, upper = sm$upper, stringsAsFactors = FALSE)

}

stats <- do.call(rbind, stat_rows)
stats$logrank_p_BH <- p.adjust(stats$logrank_p, method = "BH")
stats$significant_BH <- stats$logrank_p_BH < 0.05
write.csv(stats, file.path(outdir, "survival_logrank_cox.csv"), row.names = FALSE)

kmdf <- do.call(rbind, km_rows)
write.csv(kmdf, file.path(outdir, "survival_km_curves.csv"), row.names = FALSE)

# ---- continuous Cox sensitivity (per 1 SD), still unadjusted ---------------- #
cont_vars <- list(
  c("pvMIB_area_fraction", "pvMIB-like area fraction"),
  c("M1_MIB_core_spot_fraction", "M1 MIB spot fraction"),
  c("M2_MIB_immune_excluding_spot_count", "M2 spot count"),
  c("pvMIB_region_count", "pvMIB region count"))
crow <- list()
for (vv in cont_vars) {
  v <- vv[1]
  if (!v %in% names(d)) next
  x <- suppressWarnings(as.numeric(d[[v]]))
  x[is.na(x)] <- 0
  if (sd(x) == 0 || length(unique(x)) < 2) {
    crow[[v]] <- data.frame(term = vv[2], n = length(x), n_events = sum(d$os_event),
                            HR_per_SD = NA, CI_low = NA, CI_high = NA, p_value = NA,
                            note = "not estimable (no variation)", stringsAsFactors = FALSE)
    next
  }
  z <- as.numeric(scale(x))
  cx <- coxph(Surv(d$os_days, d$os_event) ~ z)
  s <- summary(cx)
  crow[[v]] <- data.frame(
    term = vv[2], n = as.numeric(s$n), n_events = as.numeric(s$nevent),
    HR_per_SD = as.numeric(s$conf.int[1, "exp(coef)"]),
    CI_low = as.numeric(s$conf.int[1, "lower .95"]),
    CI_high = as.numeric(s$conf.int[1, "upper .95"]),
    p_value = as.numeric(s$coefficients[1, "Pr(>|z|)"]),
    note = "survival::coxph, continuous exposure z-scored to 1 SD, unadjusted",
    stringsAsFactors = FALSE)
}
cont <- do.call(rbind, crow)
if (!is.null(cont)) {
  cont$p_BH <- p.adjust(cont$p_value, method = "BH")
  write.csv(cont, file.path(outdir, "survival_cox_continuous.csv"), row.names = FALSE)
}

cat("\n=== survival::survdiff / coxph ===\n")
print(stats[, c("analysis", "n_positive", "n_negative", "logrank_p", "logrank_p_BH",
                "cox_HR", "cox_HR_CI_low", "cox_HR_CI_high", "cox_p")], row.names = FALSE)
if (!is.null(cont)) { cat("\n=== continuous Cox (per 1 SD) ===\n"); print(cont, row.names = FALSE) }
cat("\nwritten to:", outdir, "\n")
