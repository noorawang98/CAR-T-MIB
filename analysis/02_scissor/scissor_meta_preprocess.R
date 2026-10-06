# --- released copy: personal absolute paths were replaced mechanically (see CHANGELOG.md) ---
PROJECT_ROOT <- Sys.getenv("PROJECT_ROOT", unset = "")
LEGACY_DATA_ROOT <- Sys.getenv("LEGACY_DATA_ROOT", unset = "")
HOME_ROOT <- Sys.getenv("HOME_ROOT", unset = "")

library(tidyverse)
library(Biobase)
library(limma)
library(sva)
setwd(Sys.getenv("BULK_METADATA_ROOT"))
outdir <- "Scissor_GEO"
dir.create(outdir, showWarnings = FALSE)

clean <- function(x){
  x <- as.character(x)
  x <- str_squish(x)
  x[x %in% c("", "NA", "na", "N/A", "Missing", "missing")] <- NA
  x
}

key <- function(x){
  x %>% str_to_lower() %>% str_replace_all("[^a-z0-9]+", "_") %>% str_replace_all("^_|_$", "")
}

get_char <- function(pd){
  cc <- grep("^characteristics_ch1", colnames(pd), value = TRUE)
  if(length(cc) == 0) return(pd)
  x <- pd %>%
    rownames_to_column("sample_id") %>%
    select(sample_id, all_of(cc)) %>%
    pivot_longer(-sample_id, values_to = "kv") %>%
    mutate(kv = clean(kv)) %>%
    filter(!is.na(kv), str_detect(kv, ":")) %>%
    mutate(k = key(str_replace(kv, ":.*$", "")),
           v = str_trim(str_replace(kv, "^[^:]+:", ""))) %>%
    select(sample_id, k, v) %>%
    distinct() %>%
    pivot_wider(names_from = k, values_from = v, values_fn = ~paste(unique(.x), collapse="; "))
  pd %>% rownames_to_column("sample_id") %>% left_join(x, by="sample_id") %>% column_to_rownames("sample_id")
}

pick <- function(pd, nm){
  cn <- key(colnames(pd)); names(cn) <- colnames(pd)
  hit <- names(cn)[cn %in% key(nm)]
  if(length(hit) == 0) rep(NA, nrow(pd)) else clean(pd[[hit[1]]])
}

pickn <- function(pd, nm) suppressWarnings(as.numeric(pick(pd, nm)))

## 1. metadata
# lapply(dataset,function(i){
#   eset =i[[1]]
#   pd = phenoData(eset)
#   colnames(pd)
# })
# 
# 

meta <- map2_dfr(dataset, GEOID, function(ds, gse){
  eset <- ds[[1]]
  pd <- phenoData(eset)
  tibble(
    sample_id = rownames(pd),
    GSE = gse,
    platform = annotation(eset),
    title = clean(pd$title),
    response = pick(pd, c("response:ch1")),
    bestresponse = pick(pd, c("bestresponse:ch1")),
    visit = pick(pd, c("visit:ch1")),
    treatment_arm = pick(pd, c("treatment arm:ch1", "treatment_arm:ch1")),
    ongoing_response = pick(pd, c("ongoing.response:ch1", "ongoing_response:ch1")),
    ongoing_2grps = pick(pd, c("ongoing_2grps:ch1")),
    DOR_m = pickn(pd, c("duration.of.response.months:ch1")),
    DOR_e = pickn(pd, c("duration.of.response.event:ch1")),
    EFS_m = pickn(pd, c("event.free.survival.months:ch1")),
    EFS_e = pickn(pd, c("event.free.survival.event:ch1")),
    tumor_burden = pickn(pd, c("baseline tumour burden (mm2):ch1", "baseline tumor burden (spd):ch1")),
    coo = pick(pd, c("cell of origin:ch1", "molecular subgroup:ch1")),
    sex = pick(pd, c("sex:ch1", "gender:ch1"))
  )
})

## 2. label
meta_clean <- meta %>%
  mutate(
    response_u = str_to_upper(response),
    best_u = str_to_upper(bestresponse),
    ongoing_l = str_to_lower(ongoing_response),
    ongoing2_l = str_to_lower(ongoing_2grps),
    arm_l = str_to_lower(treatment_arm),
    rel_prog = str_detect(paste(title, visit, ongoing_response, sep=" "),
                          regex("relapse|progress|PROGFCB", ignore_case=TRUE)),
    label = case_when(
      GSE %in% c("GSE153437","GSE153438") & response_u == "DR" ~ "DR",
      GSE %in% c("GSE153437","GSE153438") & response_u == "NDR" ~ "NDR",
      GSE == "GSE197977" & rel_prog ~ "NDR",
      GSE == "GSE197977" & best_u %in% c("CR","PR") ~ "DR",
      GSE == "GSE197977" & best_u %in% c("SD","PD") ~ "NDR",
      GSE == "GSE248835" & str_detect(ongoing_l, "ongoing") ~ "DR",
      GSE == "GSE248835" & str_detect(ongoing_l, "relapse|progress|non|no") ~ "NDR",
      GSE == "GSE248835" & ongoing2_l == "ongoing" ~ "DR",
      GSE == "GSE248835" & ongoing2_l %in% c("others","other") ~ "NDR",
      # GSE == "GSE248835" & DOR_m >= 6 & DOR_e == 0 ~ "DR",
      # GSE == "GSE248835" & DOR_m < 6 & DOR_e == 1 ~ "NDR",
      # GSE == "GSE248835" & EFS_m >= 6 & EFS_e == 0 ~ "DR",
      # GSE == "GSE248835" & EFS_m < 6 & EFS_e == 1 ~ "NDR",
      TRUE ~ NA_character_
    ),
    keep = case_when(
      GSE %in% c("GSE153437","GSE153438","GSE197977") ~ TRUE,
      GSE == "GSE248835" & str_detect(arm_l, "axicabtagene|ciloleucel|axi") ~ TRUE,
      TRUE ~ FALSE
    )
  ) %>%
  filter(keep, !is.na(label))

meta_raw = meta
meta = meta_clean

table(meta$GSE, meta$label)


## 3. expression merge
collapse_gene <- function(x){
  sample_cols <- intersect(meta$sample_id, colnames(x))
  x <- x[, c(sample_cols, "genename"), drop=FALSE]
  x[sample_cols] <- lapply(x[sample_cols], as.numeric)
  x <- x %>%
    mutate(genename = clean(genename)) %>%
    filter(!is.na(genename)) %>%
    group_by(genename) %>%
    summarise(across(all_of(sample_cols), ~mean(.x, na.rm=TRUE)), .groups="drop")
  m <- as.matrix(x[,-1])
  rownames(m) <- x$genename
  m
}

meta= bulk_meta

expr_mats <- map(exprlist[rna_gse], collapse_gene)

rna_gse = c('GSE153437','CC2025')

phenotype = factor(meta$label,levels=c('DR','NDR'))

phenotype = as.numeric(phenotype)-1

dir.create('Scissor_rdata_rna')
infos2_hm2mm_rna_list = lapply(1:length(filelist),function(i){
  file = filelist[i]
  st_obj = h5ad2seurat(file)
  st_obj@meta.data$barcode =paste0(rownames(st_obj@meta.data),"-",st_obj$sample)
  st_obj@meta.data$barcode = paste0(rownames(st_obj@meta.data),"-",gsub("_sc2st_DestVI","",st_obj$sample))
  st_obj=mm2hm(obj=st_obj)
  common_gene <- intersect(rownames(expr_cb), rownames(st_obj))
  bulk_use <- expr_cb[common_gene, , drop = FALSE]
  st_use <- st_obj[common_gene, , drop = FALSE]
  # qc$Response =factor(qc$Response,levels=c('Response','Nonresponse'))
  # qc$response =factor(qc$response,levels=c('NDR','DR'))
  # sc_dataset=Seurat_preprocessing(st@assays$Spatial$counts)
  # phenotype= as.numeric(qc$Response)-1
  # phenotype = as.numeric(qc$response)-1
  phenotype=phenotype
  # tag = c('Response','Nonresponse')
  tag = c('DR','NDR')
  infos2 <- Scissor5(
    # bulk_dataset = as.matrix(qc@assays$RNA$counts_removebatch),
    # bulk_dataset = as.matrix(qc@assays$RNA$counts),
    bulk_dataset =bulk_use,
    sc_dataset = st_obj, 
    phenotype=phenotype, 
    tag = tag, 
    family = "binomial", 
    #alpha = 0.4,
    # alpha = 0.5,
    # cutoff = 0.5,
    Save_file = paste0('./Scissor_rdata_rna/',file, ".Scissor_260514.RData"))
  return(infos2)
  
})


# exprlist[rna_gserna_gse
merge_expr_function = function(exprlist,GEOID,meta0){
  # expr_mats <- map(exprlist, collapse_gene)
  meta = meta0
  expr_mats <-purrr::map(exprlist[GEOID], collapse_gene)
  # expr_mats <- map(exprlist[rna_gse], collapse_gene)
  names(expr_mats) <- GEOID
  # names(expr_mats) <- rna_gse
  # names(expr_mats.rna) <-rna_gse
  common_gene <- Reduce(intersect, map(expr_mats, rownames))
  length(common_gene)
  
  expr <- do.call(cbind, map(expr_mats, ~.x[common_gene,,drop=FALSE]))
  # meta=meta0[meta0$GSE%in%GEOID,]
  # meta <- meta %>% filter(sample_id %in% colnames(expr)) %>% arrange(match(sample_id, colnames(expr)))
  meta = meta[meta$sample_id%in%colnames(expr),]
  # expr <- expr[, meta0$sample_id]
  all.equal(meta$sample_id,colnames(expr))
  ## 4. transform + z-score
  # if(quantile(expr, 0.99, na.rm=TRUE) > 50) expr <- log2(expr + 1)
  
  # for(g in unique(meta$GSE)){
  #   idx <- meta$GSE == g
  #   expr[,idx] <- t(scale(t(expr[,idx])))
  # }
  expr[is.na(expr)] <- 0
  # meta$tumor_burden[is.na(meta$tumor_burden)]<-0
  
  # 5. ComBat: remove GSE/platform, keep label
  meta <- meta %>%
    mutate(
      label = factor(label, levels=c("NDR","DR")),
      # batch = interaction(GSE, platform, drop=TRUE),
      batch = factor(clean(GSE)),
      visit = factor(clean(visit)),
      coo = factor(clean(coo)),
      # sex = factor(clean(sex)),
      # tumor_burden_log = log2(tumor_burden + 1)
    )
  
  # design <- model.matrix(~ label, meta)
  
  expr_cb <- ComBat(
    dat = as.matrix(expr),
    batch = meta$batch,
    # mod = design,
    par.prior = TRUE
    
  )
  
  # ## 6. optional covariate removal: no sex if unavailable
  # covars <- c()
  # if(nlevels(droplevels(meta$coo)) > 1) covars <- c(covars, "coo")
  # # if(nlevels(droplevels(meta$sex)) > 1) covars <- c(covars, "sex")
  # # if(sum(!is.na(meta$tumor_burden_log)) >= 3) covars <- c(covars, "tumor_burden_log")
  # 
  # if(length(covars) > 0){
  #   mm <- model.matrix(as.formula(paste("~", paste(covars, collapse="+"))), meta)
  #   mm <- mm[, colnames(mm)!="(Intercept)", drop=FALSE]
  #   fit <- lmFit(expr_cb, cbind(design, mm))
  #   expr_final <- expr_cb - fit$coefficients[, colnames(mm), drop=FALSE] %*% t(mm)
  # } else {
  #   expr_final <- expr_cb
  # }
  
  return(expr_cb)
}






# expr_mat = merge_expr_function(exprlist = exprlist,GEOID = rna_gse,meta0 = bulk_meta)

# common_genes = intersect(exprlist[[rna_gse[1]]]$genename,exprlist[[rna_gse[2]]]$genename)

# sampleid = intersect(bulk_meta$sample_id,c(colnames(exprlist[[rna_gse[1]]]),colnames(exprlist[[rna_gse[2]]])))

merge_expr_function <- function(exprlist, GEOID, meta0) {
  expr_mats <- purrr::map(exprlist[GEOID], collapse_gene)
  names(expr_mats) <- GEOID
  
  # meta_list <- list()
  # 
  # for (g in GEOID) {
  #   expr_g <- expr_mats[[g]]
  #   
  #   meta_g <- meta0 %>%
  #     dplyr::filter(GSE == g, sample_id %in% colnames(expr_g)) %>%
  #     dplyr::arrange(match(sample_id, colnames(expr_g)))
  #   
  #   expr_g <- expr_g[, meta_g$sample_id, drop = FALSE]
  #   
  #   ## 给合并后的列名加 GSE 前缀，避免不同 GSE 样本名重复
  #   uid <- paste(g, meta_g$sample_id, sep = "__")
  #   colnames(expr_g) <- uid
  #   meta_g$uid <- uid
  #   
  #   expr_mats[[g]] <- expr_g
  #   meta_list[[g]] <- meta_g
  # }
  # 
  # meta <- dplyr::bind_rows(meta_list)
  
  message("Matched samples:")
  print(table(meta$GSE, meta$label, useNA = "ifany"))
  
  common_gene <- Reduce(intersect, purrr::map(expr_mats, rownames))
  message("Common genes: ", length(common_gene))
  
  expr <- do.call(
    cbind,
    purrr::map(expr_mats, ~ .x[common_gene, , drop = FALSE])
  )
  
  expr <- expr[, meta$uid, drop = FALSE]
  
  ## transform + z-score
  if (quantile(expr, 0.99, na.rm = TRUE) > 50) {
    expr <- log2(expr + 1)
  }
  
  for (g in unique(meta$GSE)) {
    idx <- meta$GSE == g
    expr[, idx] <- t(scale(t(expr[, idx, drop = FALSE])))
  }
  expr[is.na(expr)] <- 0
  
  meta <- meta %>%
    dplyr::mutate(
      label = factor(label, levels = c("NDR", "DR")),
      batch = factor(clean(GSE)),
      visit = factor(clean(visit)),
      coo = factor(clean(coo))
    )
  
  ## ComBat: remove GSE batch
  if (nlevels(droplevels(meta$batch)) < 2) {
    stop("ComBat requires at least two batches after sample matching.")
  }
  
  expr_cb <- sva::ComBat(
    dat = as.matrix(expr),
    batch = droplevels(meta$batch),
    par.prior = TRUE
  )
  
  return(list(
    expr = expr_cb,
    meta = meta,
    common_gene = common_gene
  ))
}

rna_merge <- merge_expr_function(
  exprlist = exprlist,
  GEOID = rna_gse,
  meta0 = bulk_meta
)

expr_mat <- rna_merge$expr
meta_merge.sub <- rna_merge$meta

phenotype <- ifelse(meta_merge.sub$label == "NDR", 1, 0)
names(phenotype) <- meta_merge.sub$uid

table(meta_merge.sub$GSE, meta_merge.sub$label)
table(phenotype)

## 7. save Scissor input
phenotype <- ifelse(meta$label == "NDR", 1, 0)
phenotype = as.numeric(meta$label)-1
names(phenotype) <- meta$sample_id

stopifnot(identical(colnames(expr_final), names(phenotype)))

saveRDS(expr_final, file.path(outdir, "bulk_expr_scissor.rds"))
saveRDS(phenotype, file.path(outdir, "phenotype_DR1_NDR0.rds"))
write.csv(meta, file.path(outdir, "meta_scissor.csv"), row.names=FALSE)

table(meta$GSE, meta$label)
dim(expr_final)

#-------CC2025-----
library(dplyr)
library(tibble)

meta_CC2025 <- bulk_clinical %>%
  as.data.frame() %>%
  rownames_to_column("sample_id") %>%
  mutate(
    GSE = "CC2025",
    platform = "RNAseq",
    title = sample_id,
    
    # 原始最佳疗效
    bestresponse = Best.Overall.response,
    
    # 和 meta_clean 保持一致
    response = NA_character_,
    response_u = NA_character_,
    best_u = bestresponse,
    
    # 定义二分类 label
    # CR / PR = DR
    # SD / PD = NDR
    label = case_when(
      Best.Overall.response %in% c("CR", "PR") ~ "DR",
      Best.Overall.response %in% c("SD", "PD") ~ "NDR",
      TRUE ~ NA_character_
    ),
    
    keep = !is.na(label),
    
    # 其他临床信息
    age = Age..years.,
    sex = Sex,
    disease_type = Disease.type,
    crs_grade = Max..CRS..grade.,
    icans_grade = Max..ICANS..grade.,
    car_product = CAR.T.cell.product,
    survival = Survival,
    bulk_rna_seq = Bulk.RNA.seq.,
    single_cell_rna_seq = Single.cell.RNA.seq.,
    ffpe = FFPE,
    extra_X = X
  )

meta_CC2025 <- meta_CC2025 %>%
  mutate(
    visit = NA_character_,
    treatment_arm = car_product,
    ongoing_response = NA_character_,
    ongoing_2grps = NA_character_,
    DOR_m = NA_real_,
    DOR_e = NA_real_,
    EFS_m = NA_real_,
    EFS_e = NA_real_,
    tumor_burden = NA_real_,
    coo = NA_character_,
    rel_prog = NA
  )

meta_4 = data.frame('sample_id'= meta_clean$sample_id,
                    'GSE'=meta_clean$GSE,
                    'visit'=meta_clean$visit,
                    'label'=meta_clean$label,
                    'coo'=meta_clean$coo)

meta_CC2025_sel = data.frame('sample_id' = meta_CC2025$sample_id,
                             'GSE'=rep('CC2025',nrow(meta_CC2025)),
                             'visit'=rep('Screening',nrow(meta_CC2025)),
                             'label'=meta_CC2025$Response,
                             'coo'=meta_CC2025$Disease.type )

bulk_meta=rbind(meta_4,meta_CC2025_sel)

#---- RNA-seq -----




#---- nanostring --------


