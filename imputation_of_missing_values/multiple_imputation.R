# MULTIPLE IMPUTATION OF MISSING VALUES


# The predictorsMatrix is manually defined considering: 1) patterns of missingness; 2) vars with higher % of missingness
# 3) correlation between demographics and itemid values;

# Defined parameters:
# max_corr_missingness_patterns = 0.15
# min_corr_itemids = 0.01
# min_corr_tslots = 1 (tslots not included as predictor for any variable)
# for MI:
# m = 10
# maxit = 20

## FUNCTIONS:
load_packages <- function() {
  library(dplyr)
  library(reshape2)
  library(mice)
  library(VIM)
  library(lattice)
  library(ggplot2)
}

# To combine desired demographic variables:
combine_adm_data <- function(data, adm_file, static_vars) {
  require(dplyr)
  adm = read.csv(adm_file, header=TRUE, sep="\t", stringsAsFactors = FALSE)
  extra_info = adm %>%
    select(AE, all_of(static_vars)) %>%
    as.data.frame()
  
  df <- merge(data, extra_info, by = "AE", all = FALSE)
  df = df %>% filter(GENDER_DESC != "Unknown") %>% as.data.frame()
  select <- static_vars
  for (s in select){
    df[[s]] <- as.factor(df[[s]])
  }
  return(df)
}


## To create the missingness matrix (where 0 == NA and 1 == nonNA)
missingnes_matrix <- function(data, itemid_list, row_names = TRUE){
  misspatterns <- data
  itemid_list <- itemid_list
  if (row_names == TRUE){
    rownames(misspatterns) <- misspatterns$AE
  } 
  misspatterns = misspatterns %>% select(all_of(itemid_list)) %>% as.data.frame
  misspatterns[!(is.na(misspatterns))] <- 1
  misspatterns[(is.na(misspatterns))] <- 0
  return(misspatterns)
}

# To create the predictors matrix
manual_pred_matrix <- function(df, max_corr_missingness_patterns = FALSE, min_corr_itemids = 0.00, min_corr_tslots = 0.1){
  mask <- missingnes_matrix(df, itemid_list, matrix)
  obs2 <- lapply(names(mask), function(c){
    p <- c(sum(mask[[c]]))
    p <- as.data.frame(p, stringsAsFactors = FALSE)
    p[,2] <- c
    return(p)
  })
  names(obs2) <- names(mask)
  obs <- bind_rows(obs2)
  names(obs) <- c("nobs", "ITEMID")
  obs = obs %>% select(ITEMID, nobs) %>% as.data.frame()
  zero_variance = obs$ITEMID[which(obs$nobs == nrow(mask))]
  mod_itemid_list <- itemid_list[!itemid_list %in% zero_variance]
  corr <- cor(mask[,c(mod_itemid_list)])
  corr2 <- abs(corr)
  # If max_corr_missingness_patterns is set to a value, it has preference over min_corr_itemids in the definition of predictors later on.
  # To avoid this, I can set it to FALSE In that case, it's as it had the value NULL
  if (max_corr_missingness_patterns != FALSE){
    max_corr_missingness_patterns = as.numeric(max_corr_missingness_patterns)
    corr2[corr2 >= max_corr_missingness_patterns] <- NA
    corr2[corr2 < max_corr_missingness_patterns] <- 1
    corr2[is.na(corr2)] <- 0
  } else if (max_corr_missingness_patterns == FALSE){
    max_corr_missingness_patterns = 0
    corr2[corr2 >= max_corr_missingness_patterns] <- NA
    corr2[corr2 < max_corr_missingness_patterns] <- 1
    corr2[is.na(corr2)] <- 0
  }
  
  
  # 3. minimum correlation between itemids:
  corr3 <- quickpred(df[,c(itemid_list)], mincor = min_corr_itemids)
  
  # 4. To create the combined matrix
  if (max_corr_missingness_patterns != FALSE){
    # To create a matrix with both corr2 and corr3
    my.matrices <- list(corr2, corr3)
    names(my.matrices) <- c("corr2", "corr3")
    matrix.df.list <- lapply(my.matrices, function(l){
      lcorr <- reshape2::melt(l, varnames=c("row", "col"), stringsAsFactors = FALSE)
      lcorr$row <- as.character(lcorr$row)
      lcorr$col <- as.character(lcorr$col)
      return(lcorr)
    })
    names(matrix.df.list) <- names(my.matrices)
    matrix.df <- bind_rows(matrix.df.list, .id = ".id")
    head(matrix.df)
    corr4 <- acast(matrix.df, row ~ col, sum)
    # Rearrange the column and rows
    corr4[itemid_list,itemid_list]
    corr4[corr4 == 1] <- 0 # if the value = 1, means that it is 0 in one of the matrices
    corr4[corr4 == 2] <- 1 # if the value = 2, means it's 1 in both matrices
  } else if(max_corr_missingness_patterns == FALSE){
    # To create a matrix with both corr2 and corr3
    my.matrices <- list(corr2, corr3)
    names(my.matrices) <- c("corr2", "corr3")
    matrix.df.list <- lapply(my.matrices, function(l){
      lcorr <- reshape2::melt(l, varnames=c("row", "col"), stringsAsFactors = FALSE)
      lcorr$row <- as.character(lcorr$row)
      lcorr$col <- as.character(lcorr$col)
      return(lcorr)
    })
    names(matrix.df.list) <- names(my.matrices)
    matrix.df <- bind_rows(matrix.df.list, .id = ".id")
    head(matrix.df)
    corr4 <- acast(matrix.df, row ~ col, sum)
    # Rearrange the column and rows
    corr4[itemid_list,itemid_list]
    corr4[corr4 == 1] <- 1 # we keep the 1 in corr3
    corr4[corr4 == 2] <- 1 # if the value = 2, means it's 1 in both matrices
  }
  
  corr4 <- as.data.frame(corr4, stringsAsFactors = F)
  corr4$ITEMID = row.names(corr4)
  corr4$sum <- rowSums(corr4[,c(itemid_list)])
  table(corr4$sum)
  
  # 5. Correlation between tslot and ITEMID value (cutoff value = 0.1):
  df2 <- na.omit(df)
  corr <- cor(df2[,c("tslot",itemid_list)])
  tslotdf = as.data.frame(corr, stringsAsFactors = F)
  tslotdf = tslotdf %>% select(tslot) %>% mutate(ITEMID = row.names(tslotdf)) %>% 
    mutate(tslot = abs(tslot)) %>% as.data.frame()
  tslotdf$cat <- ifelse(tslotdf$tslot <= min_corr_tslots, 0 , 1)
  #tslot[1,3] <- 0
  tslotdf = tslotdf %>% select(cat, ITEMID) %>% mutate(tslot = cat) %>% select(tslot) %>% as.data.frame()
  tslotdf[1,] <- 0
  
  row <- rep(0, length(corr4))
  corr5 <- rbind(row, corr4)
  corr5[1,"ITEMID"] <- "tslot"
  corr5 <- cbind(tslotdf, corr5)
  
  ## 6. To include demographics as descriptors:
  demog2 <- c("GENDER_DESC", "AGE_RANGE_AT_ADM", "HOSPITAL_DISCH_SERVICE","PRIMARY_ADM_DIAG")
  m <- as.data.frame(matrix(1, ncol = length(demog2), nrow = nrow(corr5)))
  names(m) <- demog2
  corr6 <- cbind(corr5, m)
  
  ## 7. ITEMIDs with very large % missingness should not be used as predictors (> 10%):
  large_missing_itemids <- c("glucose", "alp", "alt", "bilirubin", "urea")
  for (l in large_missing_itemids){
    corr6[,l] <- 0
  }
  corr6[1,"ITEMID"] <- "tslot"
  
  # To calculate the final number of predictors for each itemid.
  row.names(corr6) <- c(unique(corr6$ITEMID))
  corr6 = corr6 %>% select(-c(ITEMID, sum)) %>% as.data.frame()
  
  # To add the last rows for demographics:
  demog2 <- c("GENDER_DESC", "AGE_RANGE_AT_ADM", "HOSPITAL_DISCH_SERVICE","PRIMARY_ADM_DIAG")
  m2 <- as.data.frame(matrix(0, ncol = length(corr6), nrow = length(demog2)))
  row.names(m2) <- demog2
  names(m2) <- names(corr6)
  corr6 <- rbind(corr6, m2)
  corr6$ITEMID <- row.names(corr6)
  #row.names(corr6) <- c(unique(corr6$ITEMID))
  corr6$ITEMID <- NULL
  # Tslot should not be predicted
  corr6[1,] <- 0
  diag(corr6) <- 0
  corr6$sum = rowSums(corr6)
  table(corr6$sum)
  corr6$sum <- NULL
  pred <- as.matrix(corr6)
  # Rearrange again the columns and rows in right order:
  r_order <- c(itemid_list, demog2)
  pred[r_order,r_order]
  return(pred)
}


############################################
############################################
############################################

load_packages()


files_path <- "~/files_path/"
generated_data_path <- paste0(files_path, "generated_data_path/")
results_path <- "~/results_path/"

adm_file = paste0(files_path, "adm.txt")
# Define the demographic variables to be used as predictors
demog2 <- c("GENDER_DESC", "AGE_RANGE_AT_ADM", "HOSPITAL_DISCH_SERVICE","PRIMARY_ADM_DIAG")
vars <- demog2


# 1. To read the data:
mdf <- read.csv(paste0(files_path, "data.txt"), header=TRUE, sep="\t", stringsAsFactors = FALSE)
itemid_list <- unique(mdf$ITEMID)
df <- reshape2::dcast(mdf, AE + tslot ~ ITEMID, value.var = c("norm"))

# 2. To compute the predictors matrix with predefined criteria
pred <- manual_pred_matrix(df, max_corr_missingness_patterns = 0.15, min_corr_itemids = 0.01, min_corr_tslots = 1) ## Correlation with day of admission (tslot) is not included as a descriptor by setting min_corr_tslots = 1
print(rowSums(pred)) 
# To save the matrix
write.csv(pred, file = paste0(results_path, "mi_predMatrix.csv"), row.names= TRUE, quote=F)


# 3. To prepare data for imputation:
df <- combine_adm_data(df, adm_file, vars)
# To inspect and format the variables:
# 2.a. To format (and level) appropriatley the descriptors
df$AGE_RANGE_AT_ADM <- relevel(df$AGE_RANGE_AT_ADM, ref = '85-89') 
df$HOSPITAL_DISCH_SERVICE <- relevel(df$HOSPITAL_DISCH_SERVICE, ref = "GM")
# To transform the levels of PRIMARY_ADM_DIAG
df$PRIMARY_ADM_DIAG <- as.character(df$PRIMARY_ADM_DIAG)
df$PRIMARY_ADM_DIAG <- ifelse(df$PRIMARY_ADM_DIAG == "A" | df$PRIMARY_ADM_DIAG == "B", "AB", df$PRIMARY_ADM_DIAG) # A & B can be the same category
df$PRIMARY_ADM_DIAG <- ifelse(df$PRIMARY_ADM_DIAG == "Q" | df$PRIMARY_ADM_DIAG == "Z" | df$PRIMARY_ADM_DIAG == "H", "QZH", df$PRIMARY_ADM_DIAG) # Q, Z and H have a very small number. I group them
df$PRIMARY_ADM_DIAG <- as.factor(as.character(df$PRIMARY_ADM_DIAG))
df$PRIMARY_ADM_DIAG <- relevel(df$PRIMARY_ADM_DIAG, ref = "R")


# 4. Imputation:
toimp = df

start_time <- Sys.time()
m = 10
maxit = 20
imp = mice(toimp, m = m, maxit = maxit, predictorMatrix = pred, printFlag = TRUE)
end_time <- Sys.time()  
print(paste0("time to impute with pmm ", end_time - start_time))

# To save the loggedevents
write.table(imp$loggedEvents, file = paste0(results_path, "loggedEvents.txt"), sep="\t", row.names= TRUE, quote=F)
print("loggedEvents saved")

# To save the mids object:
name <- "mids_object"
mids_path <- paste0(generated_data_path, "pmm_imputation/",name,".rds")
saveRDS(imp, file = mids_path)
