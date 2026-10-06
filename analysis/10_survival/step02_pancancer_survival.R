# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

# Pan-cancer clinical association of the CCR1+ MIB CIBERSORTx components NR_2 / NR_9 / NR_10
# (signature: CIBERSORTx_Job22_ST_NR_mm2hsa.txt, 13 components)
# best cutoff per cancer type -> high vs low; Cox PH adjusted for age and gender;
# forest plot (png + ai) and KM curves (high = red, low = blue) for p<0.05 & HR>1.
.libPaths(c(paste0(HOME_ROOT, "/R_4.3"), .libPaths()))
suppressPackageStartupMessages({library(survival); library(survminer); library(dplyr); })
setwd(paste0(PROJECT_ROOT, "/mouse/cart_region/ccr1mib_tcga"))
dir.create("figs", showWarnings = FALSE)
COMPONENTS <- c("NR_2", "NR_9", "NR_10")
MIN_N <- 20; MIN_GROUP <- 5; MIN_EVENTS <- 5

ci <- read.csv("CIBERSORT_STNR_TCGA.csv", row.names = 1, check.names = FALSE)
ci[] <- lapply(ci, function(x) suppressWarnings(as.numeric(x)))
ci <- ci[stats::complete.cases(ci[, COMPONENTS]), ]   # drop the 439 random SVR failures
cat("CIBERSORT ok samples:", nrow(ci), "| median fit correlation:",
    round(median(ci$Correlation, na.rm = TRUE), 3), "\n")
clin <- read.delim(paste0(HOME_ROOT, "/TCGA/Survival_SupplementalTable_S1_20171025_xena_sp"),
                   check.names = FALSE, stringsAsFactors = FALSE)
clin <- clin[, c("sample", "cancer type abbreviation", "age_at_initial_pathologic_diagnosis",
                 "gender", "OS", "OS.time")]
names(clin) <- c("sample", "cancer", "age", "gender", "OS", "OS.time")
D <- merge(data.frame(sample = rownames(ci), ci[, COMPONENTS, drop = FALSE]),
           clin, by = "sample")
D <- D[!is.na(D$OS.time) & !is.na(D$OS) & D$OS.time > 0, ]
D$age <- suppressWarnings(as.numeric(D$age))
D$gender <- factor(D$gender)
cat("merged samples:", nrow(D), "| cancers:", length(unique(D$cancer)), "\n")

best_cut <- function(x, time, event) {
  cp <- NA_real_
  r <- try(survminer::surv_cutpoint(data.frame(x = x, time = time, event = event),
                                    time = "time", event = "event", variables = "x"), silent = TRUE)
  if (!inherits(r, "try-error") && !is.null(r$cutpoint))
    cp <- suppressWarnings(as.numeric(r$cutpoint$cutpoint[1]))
  qs <- unique(quantile(x, probs = seq(.15, .85, by = .01), na.rm = TRUE))
  sc <- function(c) {
    g <- x > c
    if (sum(g) < MIN_GROUP || sum(!g) < MIN_GROUP) return(NA_real_)
    s <- try(survdiff(Surv(time, event) ~ g), silent = TRUE)
    if (inherits(s, "try-error")) NA_real_ else s$chisq
  }
  ch <- vapply(qs, sc, numeric(1))
  cpb <- if (all(is.na(ch))) NA_real_ else qs[which.max(ch)]
  bad <- is.na(cp) || sum(x > cp) < MIN_GROUP || sum(x <= cp) < MIN_GROUP
  if (bad) cp <- cpb
  if (is.na(cp)) return(list(cut = NA_real_, high = rep(NA, length(x))))
  list(cut = cp, high = x > cp)
}

rows <- list(); km <- list()
for (comp in COMPONENTS) {
  for (ca in sort(unique(D$cancer))) {
    d <- D[D$cancer == ca & !is.na(D[[comp]]), ]
    if (nrow(d) < MIN_N) next
    bc <- best_cut(d[[comp]], d$OS.time, d$OS)
    if (is.na(bc$cut)) next
    d$group <- factor(ifelse(bc$high, "High", "Low"), levels = c("Low", "High"))
    if (any(table(d$group) < MIN_GROUP)) next
    ev <- sum(d$OS == 1)
    if (ev < MIN_EVENTS) next
    fit <- try(coxph(Surv(OS.time, OS) ~ group + age + gender, data = d), silent = TRUE)
    if (inherits(fit, "try-error")) next
    s <- summary(fit)$coefficients
    sconf <- summary(fit)$conf.int
    if (!"groupHigh" %in% rownames(s)) next
    hr <- sconf["groupHigh", "exp(coef)"]; lo <- sconf["groupHigh", "lower .95"]
    hi <- sconf["groupHigh", "upper .95"]; p <- s["groupHigh", "Pr(>|z|)"]
    sig <- ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", "ns")))
    rows[[paste(comp, ca)]] <- data.frame(component = comp, cancer = ca, n = nrow(d),
      n_high = sum(d$group == "High"), n_low = sum(d$group == "Low"), events = ev,
      cutoff = bc$cut, HR = hr, lower = lo, upper = hi, pvalue = p, significance = sig)
    if (p < .05 && hr > 1) km[[paste(comp, ca)]] <- d[, c("sample", "OS.time", "OS", "group", comp)]
  }
  cat(comp, "done\n")
}
T <- do.call(rbind, rows); rownames(T) <- NULL
write.csv(T, "pancancer_CCR1MIB_NR2_NR9_NR10_cox.csv", row.names = FALSE)

# ---------------- forest plots (png + ai) ----------------
suppressPackageStartupMessages(library(forestplot))
for (comp in COMPONENTS) {
  t <- T[T$component == comp, ]
  if (!nrow(t)) next
  o <- order(t$HR); t <- t[o, ]
  lab <- paste0(t$cancer, "  (n=", t$n, ")")
  txt <- rbind(c("Cancer type (n)", "HR (95% CI)", "p", ""),
               cbind(lab,
                     sprintf("%.2f (%.2f-%.2f)", t$HR, t$lower, t$upper),
                     formatC(t$pvalue, format = "g", digits = 2),
                     t$significance))
}

# ---------------- KM curves for p<0.05 & HR>1 ----------------
for (nm in names(km)) {
  d <- km[[nm]]; comp <- strsplit(nm, " ")[[1]][1]; ca <- sub(paste0(comp, " "), "", nm)
  d$OS.time <- d$OS.time / 365
  f <- survfit(Surv(OS.time, OS) ~ group, data = d)
}
cat("KM plots:", length(km), "\nDONE\n")
