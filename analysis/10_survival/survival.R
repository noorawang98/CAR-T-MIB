# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

#!/usr/bin/env Rscript
# Step83b: bin50_v3 的 perviMIB / MIB 丰度 与 OS / PFS 的关系 (Cox + KM)
# 输入 prognosis/survival_exposure_patient.csv (step83a 生成)
suppressPackageStartupMessages({library(survival);  })
WD <- Sys.getenv("ST_WD", paste0(PROJECT_ROOT, "/destvi_260911")); OUT <- file.path(WD, "prognosis")
T <- read.csv(file.path(OUT, "survival_exposure_patient.csv"), check.names = FALSE)
T$pfs_event <- as.numeric(T$pfs_event); T$os_event <- as.numeric(T$os_event)
T$pfs_days <- as.numeric(T$pfs_days);   T$os_days <- as.numeric(T$os_days)
EXPOS <- c("perviMIB_mouse", "perviMIB_DestVI", "CCR1_MIB_mouse", "MIB_DestVI")
TPS <- c("Post", "Pre", "All")
cox_rows <- list(); km_rows <- list(); i <- 1; j <- 1
for (tp in TPS) for (ex in EXPOS) {
  col <- paste0(ex, "_", tp)
  if (!col %in% names(T)) next
  d <- T[!is.na(T[[col]]), ]
  x <- d[[col]]
  if (length(unique(x)) < 2) next
  for (ep in c("os", "pfs")) {
    ev <- d[[paste0(ep, "_event")]]; tm <- d[[paste0(ep, "_days")]]
    if (sum(ev) < 2) next
    sx <- (x - mean(x)) / sd(x)                 # 每 +1 SD
    fit <- tryCatch(coxph(Surv(tm, ev) ~ sx), error = function(e) NULL)
    if (!is.null(fit)) {
      s <- summary(fit)
      cox_rows[[i]] <- data.frame(timepoint = tp, exposure = ex, endpoint = toupper(ep),
        model = "continuous (per +1 SD)", n = nrow(d), events = sum(ev),
        HR = s$conf.int[1, "exp(coef)"], lo = s$conf.int[1, "lower .95"], hi = s$conf.int[1, "upper .95"],
        p = s$coefficients[1, "Pr(>|z|)"], C = s$concordance[1]); i <- i + 1
    }
    fit2 <- tryCatch(coxph(Surv(tm, ev) ~ I(x * 100)), error = function(e) NULL)   # 每 +1 个百分点
    if (!is.null(fit2)) {
      s2 <- summary(fit2)
      cox_rows[[i]] <- data.frame(timepoint = tp, exposure = ex, endpoint = toupper(ep),
        model = "continuous (per +1% abundance)", n = nrow(d), events = sum(ev),
        HR = s2$conf.int[1, "exp(coef)"], lo = s2$conf.int[1, "lower .95"], hi = s2$conf.int[1, "upper .95"],
        p = s2$coefficients[1, "Pr(>|z|)"], C = s2$concordance[1]); i <- i + 1
    }
    # 中位数分组 KM
    grp <- ifelse(x > median(x), "High", "Low")
    if (length(unique(grp)) == 2) {
      km <- survfit(Surv(tm, ev) ~ grp, data = data.frame(tm, ev, grp))
      lr <- survdiff(Surv(tm, ev) ~ grp, data = data.frame(tm, ev, grp))
      p_lr <- 1 - pchisq(lr$chisq, length(lr$n) - 1)
      med <- summary(km)$table[, "median"]
      km_rows[[j]] <- data.frame(timepoint = tp, exposure = ex, endpoint = toupper(ep), split = "median",
        n_high = sum(grp == "High"), n_low = sum(grp == "Low"),
        median_high = ifelse(is.na(med[2]), Inf, med[2]), median_low = ifelse(is.na(med[1]), Inf, med[1]),
        logrank_p = p_lr); j <- j + 1
      f <- survfit(Surv(tm, ev) ~ grp, data = data.frame(tm, ev, grp))

    }
    # 存在/缺失 分组 (应对大量 0 值)
    if (any(x == 0) && any(x > 0)) {
      grp2 <- ifelse(x > 0, "present", "absent")
      lr2 <- survdiff(Surv(tm, ev) ~ grp2, data = data.frame(tm, ev, grp2))
      p2 <- 1 - pchisq(lr2$chisq, length(lr2$n) - 1)
      km2 <- survfit(Surv(tm, ev) ~ grp2, data = data.frame(tm, ev, grp2))
      med2 <- summary(km2)$table[, "median"]
      km_rows[[j]] <- data.frame(timepoint = tp, exposure = ex, endpoint = toupper(ep), split = "present_vs_absent",
        n_high = sum(grp2 == "present"), n_low = sum(grp2 == "absent"),
        median_high = ifelse(is.na(med2[2]), Inf, med2[2]), median_low = ifelse(is.na(med2[1]), Inf, med2[1]),
        logrank_p = p2); j <- j + 1
      fit3 <- tryCatch(coxph(Surv(tm, ev) ~ I(x > 0)), error = function(e) NULL)
      if (!is.null(fit3)) {
        s3 <- summary(fit3)
        cox_rows[[i]] <- data.frame(timepoint = tp, exposure = ex, endpoint = toupper(ep),
          model = "binary (present vs absent)", n = nrow(d), events = sum(ev),
          HR = s3$conf.int[1, "exp(coef)"], lo = s3$conf.int[1, "lower .95"], hi = s3$conf.int[1, "upper .95"],
          p = s3$coefficients[1, "Pr(>|z|)"], C = s3$concordance[1]); i <- i + 1
      }
    }
  }
}
COX <- do.call(rbind, cox_rows); KM <- do.call(rbind, km_rows)
write.csv(COX, file.path(OUT, "survival_cox_bin50.csv"), row.names = FALSE)
write.csv(KM, file.path(OUT, "survival_km_bin50.csv"), row.names = FALSE)
options(width = 220)
cat("=== Cox 单变量 (bin50_v3; HR 与 95%CI) ===\n")
print(COX[order(COX$endpoint, COX$timepoint, COX$exposure, COX$model),
          c("timepoint","exposure","endpoint","model","n","events","HR","lo","hi","p","C")], row.names = FALSE, digits = 3)
cat("\n=== KM / log-rank ===\n")
print(KM[order(KM$endpoint, KM$timepoint, KM$exposure), ], row.names = FALSE, digits = 3)
