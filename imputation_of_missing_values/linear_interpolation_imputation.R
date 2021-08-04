# LINEAR INTERPOLATION IMPUTATION #


## FUNCTIONS:

load_packages <- function() {
  library(lubridate)
  library(dplyr)
  library(reshape2)
  library(imputeTS)
  library(mice)
}

sanityc <- function(data){
  for (c in names(data)){
    print(paste0(c, " ", sum(is.na(data[[c]]))))
  }
}

# To identify the lab tests closest to ref date
select_closest_lab_tests <- function(reference_dates, exc_vars){
  require(dplyr)
  wd <- paste0(files_path, "lab_tests/")
  setwd(wd)
  file_list <- list.files() # The files in the specified wd
  # To exclude the desired variables:
  to_exclude <- paste0(exc_vars, ".txt")
  file_list2 <- file_list[!file_list %in% to_exclude]
  
  for (file in file_list2){
    
    if (!exists("dataset")){
      dataset <- data.frame(stringsAsFactors = FALSE)
    }
    if (exists("dataset")){
      data <- read.csv(file, header=TRUE, sep="\t", stringsAsFactors = FALSE)
      data$ITEMID <- data$NewTestName
      df2 = data %>% select(AE, ITEMID, coll_date, nobs, order_obs, original,norm, MainTestGroup) %>% as.data.frame()
      sum(is.na(df2$coll_date)) # 0
      df2 = df2 %>% filter(!is.na(coll_date)) %>% as.data.frame()  # in case there are NA
      df2$DATE_t <- as.POSIXct(df2$coll_date, tz = "GMT", format = "%Y-%m-%d %H:%M:%S")
      timestamp = "day"
      df2$DATE <- floor_date(df2$DATE_t, timestamp)
      df2$original <- as.numeric(df2$original)
      df2$norm <- round(as.numeric(df2$norm), digits = 4)
      print(length(unique(df2$AE)))
      sanityc(df2)
      # To group by DATE:
      df2$DATE <- as.character(df2$DATE)
      # To merge with the ref date for each tslot and AE
      df3 <- merge(reference_dates, df2,by = c("AE", "DATE"))
      # To format DATE_t and ref_date as Posixct
      df3$DATE_t <- as.POSIXct(df3$DATE_t,  format="%Y-%m-%d %H:%M:%S", tz = "GMT")
      df3$ref_date <- as.POSIXct(df3$ref_date,  format="%Y-%m-%d %H:%M:%S", tz = "GMT")
      # To calculate the diff between DATE_t and ref_date
      df3$dif <- df3$DATE_t - df3$ref_date
      # df4 = df3 %>% select(AE, DATE, tslot, ref_date, DATE_t, dif) %>% as.data.frame()
      df3$dif <- abs(df3$dif)
      df3 = df3 %>% dplyr::group_by(AE, DATE) %>% 
        dplyr::arrange(AE, DATE, dif) %>% 
        dplyr::mutate(count = row_number()) %>% as.data.frame()
      print(file)
      print(table(df3$count))
      df3 = df3 %>% filter(count == 1) %>% 
        select(AE, tslot, DATE, ITEMID, original, norm, MainTestGroup, ref_date, rd) %>% as.data.frame()     
    }
    
    dataset<- rbind(dataset, df3)
    rm(data, df3) 
  }
  return(dataset)
}

## Select the desired observation number and merge the inidividual files (for vital signs)
select_closest_vital_signs <- function(reference_dates, exc_vars){
  # "closest" --> closest recorded observation to the ref_date for lab tests
  require(dplyr)
  wd <- paste0(files_path, "vital_signs/")
  setwd(wd)
  file_list <- list.files() # The files in the specified wd
  # To exclude the desired variables:
  to_exclude <- paste0("cleaned_", exc_vars, "_data.txt")
  file_list2 <- file_list[!file_list %in% to_exclude]
  
  for (file in file_list2){
    
    if (!exists("dataset")){
      dataset <- data.frame(stringsAsFactors = FALSE)
    }
    if (exists("dataset")){
      data <- read.csv(file, header=TRUE, sep="\t", stringsAsFactors = FALSE)
      df2 = data %>% select(AE, ITEMID, rec_date, nobs, order_obs, original,norm) %>% as.data.frame()
      sum(is.na(df2$rec_date)) # 0
      df2 = df2 %>% filter(!is.na(rec_date)) %>% as.data.frame()  # in case there are NAs
      df2$DATE_t <- as.POSIXct(df2$rec_date, tz = "GMT", format = "%Y-%m-%d %H:%M:%S")
      timestamp = "day"
      df2$DATE <- floor_date(df2$DATE_t, timestamp)
      df2$original <- as.numeric(df2$original)
      df2$norm <- round(as.numeric(df2$norm), digits = 4)
      df2$MainTestGroup <- "vitals"
      print(length(unique(df2$AE)))
      sanityc(df2)
      # To group by DATE:
      df2$DATE <- as.character(df2$DATE)
      # To merge with the ref date for each tslot and AE
      df3 <- merge(reference_dates, df2,by = c("AE", "DATE"))
      # To format DATE_t and ref_date as Posixct
      df3$DATE_t <- as.POSIXct(df3$DATE_t,  format="%Y-%m-%d %H:%M:%S", tz = "GMT")
      df3$ref_date <- as.POSIXct(df3$ref_date,  format="%Y-%m-%d %H:%M:%S", tz = "GMT")
      # To calculate the diff between DATE_t and ref_date
      df3$dif <- df3$DATE_t - df3$ref_date
      df3$dif <- abs(df3$dif)
      df3 = df3 %>% dplyr::group_by(AE, DATE) %>% 
        dplyr::arrange(AE, DATE, dif) %>% 
        dplyr::mutate(count = row_number()) %>% as.data.frame()
      print(file)
      print(table(df3$count))
      df3 = df3 %>% filter(count == 1) %>% 
        select(AE, tslot, DATE, ITEMID, original, norm, MainTestGroup, ref_date, rd) %>% as.data.frame()     
    }
    
    dataset<- rbind(dataset, df3)
    rm(data, df3) 
  }
  return(dataset)
}


## To combine desired demographic variables:
combine_adm_data <- function(data, adm_file, static_vars) {
  require(dplyr)
  adm = read.csv(adm_file, header=TRUE, sep="\t", stringsAsFactors = FALSE)
  extra_info = adm %>%
    select(AE, all_of(static_vars)) %>% as.data.frame()
  df <- merge(data, extra_info, by = "AE", all = FALSE)
  df = df %>% filter(GENDER_DESC != "Unknown") %>% as.data.frame()
  select <- static_vars
  for (s in select){
    df[[s]] <- as.factor(df[[s]])
  }
  return(df)
}


#################################
#################################
#################################

load_packages()


files_path <- "~/files_path/"
generated_data_path <- paste0(files_path, "generated_data_path/")
results_path <- "~/results_path/"

# With several datasets: To read the mids_object from MI 
name <- "mids_object"
mids_path <- paste0(generated_data_path,"pmm_imputation/",name,".rds")
imp <- readRDS(mids_path)
m = 10 # The number of datasets that we imputed:
imp_datasets <- seq(1:m)

# To read the data before MI imputation:
matrix <- read.csv(paste0(files_path, "data.txt"), header=TRUE, sep="\t", stringsAsFactors = FALSE)
itemid_list <- unique(matrix$ITEMID)
# And format it to be used in pmm_imp_list:
m2b <- matrix
m2b$value <- as.numeric(m2b$norm)
m2b$norm <- NULL
sanityc(m2b)
m2b$impMI <- ifelse(is.na(m2b$value), 1, 0) # All 0
m2b = m2b %>% arrange(AE, ITEMID, tslot) %>% as.data.frame()
m2b$value <- NULL
changed.tslots <- unique(m2b$AE[which(m2b$changed == 1)])
# length(changed.tslots) # 625

# To extract and format all the m imputed datasets (with mi pmm) and prepare them for "isolated" observation inclusion and linear interpolation
pmm_imp_list <- lapply(imp_datasets, function(n){
  
  matrix_imputed <- complete(imp,n)
  print(length(unique(matrix_imputed$AE))) # 11158
  
  matrix_imputed = matrix_imputed %>% select(AE, tslot, all_of(itemid_list)) %>% as.data.frame()
  matrix_imputed <- reshape2::melt(matrix_imputed, id.vars = c("AE", "tslot"), variable.name = "ITEMID", value.name = "value")
  # Is this tslot the original (starting with 2 for some AEs) or is it tslotN?
  matrix_imputed = matrix_imputed %>% group_by(AE, ITEMID) %>% mutate(mints = min(tslot)) %>% as.data.frame()
  table(matrix_imputed$mints) # I have 1s and 2s, therefore it's tlot (not tslotN)
  
  
  # to merge with m2b:
  matrix_imputed1 <- merge(matrix_imputed, m2b, by = c("AE", "tslot", "ITEMID"), all.x = TRUE)
  print(length(unique(matrix_imputed1$AE))) # 11158
  sanityc(matrix_imputed1)
  # impMI should have missing values for those that were imputed with MI
  matrix_imputed1$impMI <- ifelse(is.na(matrix_imputed1$impMI), 1, matrix_imputed1$impMI)
  table(matrix_imputed1$impMI)
  # changed and tslotN also have missing values:
  matrix_imputed1$changed <- ifelse(matrix_imputed1$AE %in% changed.tslots, 1, 0)
  matrix_imputed1$tslotN <- ifelse(matrix_imputed1$changed == 1, matrix_imputed1$tslot-1, matrix_imputed1$tslot)
  matrix_imputed1 = matrix_imputed1 %>% arrange(AE, ITEMID, tslot) %>% as.data.frame()
  matrix_imputed1 = matrix_imputed1 %>% select(AE, tslot, tslotN, changed, ITEMID, value, impMI) %>% as.data.frame()
  
  return(matrix_imputed1)
})



### IN BATCH JOBS (for each imputed dataset)
#job_index = as.numeric(as.character(Sys.getenv("LSB_JOBINDEX")))
job_index = 1 # Example of imputation on MI imputed dataset 1.

# To read pmm-imputed matrix
mdf <- pmm_imp_list[[job_index]]
itemid_list <- unique(as.character(mdf$ITEMID))


## TO INSERT "ISOLATED" OBSERVATIONS FOR NON-IMPUTABLE DAYS BEFORE LI ##
# Not all non-imputable days are missing for all ITEMIDs. I include the actual recorded value for those
# which were not missing (before the previously established max tslot. This is because I want to avoid considering the tail of the time series with a small number of observations)

# I use the refDates computed earlier on:
# 1.1. To idenfity the maxtslot per AEs (to avoid including observations after the last imputable day)
# Important to work here with original tslot (because we'll merge other datasets)
mdf = mdf %>% dplyr::group_by(AE) %>% dplyr::mutate(maxt = max(tslot)) %>% as.data.frame()
# To save the maxtslot for each AE
maxtslots  = mdf %>% select(AE, maxt) %>% unique() %>% as.data.frame()
mdf$maxt <- NULL


# 1.2. To read the refDates:
file_name <- paste0(files_path, "/dsic_with_ref_dates2.txt")
refDates <- read.csv(file_name, header=TRUE, sep="\t", stringsAsFactors = FALSE)
rd = refDates %>% select(AE, tslot, ref_date, rd) %>% as.data.frame() 
rd2 = merge(rd, mdf, by = c("AE", "tslot"), all = TRUE)
# To identify obs not included in matrix:
isolated = rd2 %>% filter(is.na(impMI)) %>% select(AE, tslot, ref_date, rd) %>% unique() %>% as.data.frame()
isolated$DATE = format(as.POSIXct(strptime(isolated$ref_date,"%Y-%m-%d %H:%M:%S",tz="")) ,format = "%Y-%m-%d")
aes <- unique(isolated$AE)
length(aes) # 7964
# To read labtest data for these AEs:
labtests <- select_closest_lab_tests(isolated, exc_vars = c("albumin", "aptt", "crp", "inr"))
# 1.3. To select closest obs for vital signs for isolated days:
vitals <- select_closest_vital_signs(isolated, exc_vars = c("gcs", "oxig"))
test = vitals %>% group_by(AE, tslot, ITEMID) %>% mutate(count = row_number()) %>% as.data.frame()
table(test$count) # all 1s
rm(test)
# To bind them:
isobs <- rbind(labtests, vitals)
nrow(isobs)
# Any tail observation?
isobs <- merge(isobs, maxtslots, by = "AE")
nrow(isobs)
length(isobs$AE[which(isobs$tslot <= isobs$maxts)]) # 0
sanityc(isobs)

# To format isobs as mdf and merge them
changed.tslots <- unique(mdf$AE[which(mdf$changed == 1)])
isobs$changed = ifelse(isobs$AE %in% changed.tslots, 1, 0)
isobs$tslotN <- ifelse(isobs$changed == 1, isobs$tslot - 1, isobs$tslot)
isobs$value <- as.numeric(round(isobs$norm, digits = 4))
isobs$impMI <- 0
isobs = isobs %>% select(AE, tslot, tslotN, changed, ITEMID, value, impMI, rd) %>% as.data.frame()
mdf$rd <- "LT"

merged <- rbind(mdf, isobs)
merged = merged %>% arrange(AE, ITEMID, tslot) %>% as.data.frame()
mwide <- reshape2::dcast(merged, AE + tslot + tslotN ~ ITEMID, value.var = "value")
sanityc(mwide)
mlong <- reshape2::melt(mwide, id.vars = c("AE", "tslot", "tslotN"))


# 2. LINEAR INTERPOLATION FOR EVERY AE AND ITEMID INDEPENDENTLY
long <- lapply(itemid_list, function(i){
 
  df = merged %>% filter(ITEMID == i) %>% as.data.frame()
  admissions <- unique(df$AE)
  
  imp.df <- lapply(admissions, function(ae){
    d <- df %>% filter(AE == ae) %>% as.data.frame()
    v <- as.numeric(d$tslotN)
    tslotN <- seq(min(v),max(v), by = 1)
    dd <- as.data.frame(tslotN)
    finald <- merge(d, dd, by = "tslotN", all = TRUE)
    finald$AE <- ae
    finald$ITEMID <- i
    finald$impMI <- ifelse(is.na(finald$impMI), 0, finald$impMI)
    finald$impLI <- ifelse(is.na(finald$value), 1, 0)
    
    # Linear interpolation:
    miss <- as.numeric(c(finald$value))
    pred <- round(na_interpolation(miss, option = "linear"), digits = 3)
    pred2 <- as.data.frame(cbind(pred, miss))
    pred2 = pred2 %>% mutate(ITEMID = i) %>% mutate(AE = ae) %>% mutate(tslotN = as.numeric(row.names(pred2))) %>% as.data.frame()
    pred2 = pred2 %>% select(AE, tslotN, ITEMID, pred) %>% as.data.frame()
    pred3 <- merge(pred2, finald, by = c("AE", "tslotN", "ITEMID"))
    pred3 = pred3 %>% select(-c(value)) %>% mutate(value = pred) %>% select(AE, tslotN, ITEMID, value, impMI, impLI) %>%
      arrange(tslotN) %>% as.data.frame()
    return(pred3)
    
  })
  imp <- bind_rows(imp.df)
 
  return(imp)
  print(paste0("all done for ", i))
})

# TO bind all the dataframes for the different ITEMIDs
final <- bind_rows(long)
table(final$impLI)*100/nrow(final) # % overall missingness
# To add tslot and changed variables
final$changed <- ifelse(final$AE %in% changed.tslots, 1, 0)
final$tslot <- ifelse(final$changed == 1, final$tslotN+1, final$tslotN)
final = final %>% select(AE, tslot, tslotN, changed, ITEMID,value, impMI, impLI) %>% as.data.frame()
# To add the type of reference date:
rdtype = merged %>% select(AE, tslotN, rd) %>% unique() %>% as.data.frame()
final2 = merge(final, rdtype, by = c("AE", "tslotN"), all.x = TRUE)
sanityc(final2)
final2$rd <- ifelse(is.na(final2$rd), "None", final2$rd)

write.table(final2, file = paste0(generated_data_path, "linear_interpolation_",job_index,".txt"), sep="\t", row.names= TRUE, quote=F)