### RF TRAINING AND EVALUATION FOR MULTILABEL OUTCOMES ###
### DIAGNOSIS AT DISCHARGE ###

# The script is divided in 4 parts:
  # 1. Get the data ready and save some parameters to check later that the files used are the correct (assessment)
  # 2. Hyperparameter Tuning (HT_results)
  # 3. Model training and cross validation (CV_results)
  # 4. Model training and predictions (TEST_results)

# The script is divided in 4 parts:
  # 1. Data formatting
  # 2. Hyperparameter Tuning (HT_results)
  # 3. Model training and cross validation (CV_results)
  # 4. Model training and predictions (TEST_results)

## Intended to be run with arguments:
# bsub "python RF_training_DD.py Day1 D1_raw_data D1_no_pda D1_no_pda_PDA"


###############################################
##### 1. DATA PREPARATION AND ASSESSMENT  #####
###############################################

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, getopt



# TO PASS THE ARGUMENTS:
day = sys.argv[1]
data_type = sys.argv[2]
demog = sys.argv[3]
outcome = sys.argv[4]
# Example:
# day = 'Day1'
# data_type = 'D1_raw_data'
# demog = 'D1_pda'
# outcome = 'D1_no_pda_DD'


# RESULTS PATHS:
# results_root = results_root_path
# assessment_path = results_root+'assessment/'
# ht_path = results_root+'HT_results/'
# cv_path = results_root+'CV_results/'
# test_path = results_root+'TEST_results/'
# TEST_features_importance = results_root+'/TEST_features_importance/'
# TEST_trained_models = results_root+'/TEST_trained_models/'
# TEST_predictions = test_path+'/TEST_predictions/'
# TEST_probabilities = results_root+'/TEST_conf_matrix/'
# TEST_conf_matrix = results_root+'/TEST_probabilities/'
# TEST_roc_parameters = results_root+'/TEST_roc_parameters/'


# TO READ THE INPUT DATA (The datasets have been previously created to include only the relevant variables)
# root_path = root_path
file_name = 'file_name.txt'
TTdata = root_path + file_name
df = pd.read_table(TTdata)
df = df.set_index('AE') 

# Some ICD-10 are very rare and this can have a bad impact on the training of the models:
target_names = ['DD_A', 'DD_B', 'DD_C', 'DD_D', 'DD_E', 'DD_F', 'DD_G', 'DD_H','DD_I', 'DD_J', 'DD_K', 'DD_L', 'DD_M', 'DD_N', 'DD_O','DD_P','DD_Q','DD_R', 'DD_S', 'DD_T','DD_U','DD_V','DD_W','DD_X','DD_Y','DD_Z']
df_outc = pd.DataFrame(df[target_names])
outcome_numbers = df_outc.sum(axis = 0, skipna = True)
# I drop those with < 10 occurrences (O,P,U,X)
df = df[df.DD_O == 0] ### And we drop the AEs that had any of those ICD-10 codes
df = df.drop('DD_O', axis = 1)
df = df[df.DD_P == 0]
df = df.drop('DD_P', axis = 1)
df = df[df.DD_U == 0]
df = df.drop('DD_U', axis = 1)
df = df[df.DD_X == 0]
df = df.drop('DD_X', axis = 1)


input_variables = list(df.columns)
with open(assessment_path+'input_data_column_names.txt', "w") as output:
    output.write(str(input_variables))

# DATA PROCESSING: Features and Targets and Convert Data to Arrays
target_names = ['DD_A', 'DD_B', 'DD_C', 'DD_D', 'DD_E', 'DD_F', 'DD_G', 'DD_H','DD_I', 'DD_J', 'DD_K', 'DD_L', 'DD_M', 'DD_N','DD_Q','DD_R', 'DD_S', 'DD_T','DD_V','DD_W','DD_Y','DD_Z']
outcome = pd.DataFrame(df[target_names])
descriptors = df.drop(target_names, axis = 1)
# Saving feature names for later use
descriptors_list = list(descriptors.columns)
with open(assessment_path+'input_data_features.txt', "w") as output:
    output.write(str(descriptors_list))

# TRAINING/VALIDATION (TV, for hyperparameter tuning) and TEST (Tt, for model evaluation) Sets:
# Split the data into training and testing sets:
TV_features_df, Tt_features_df, TV_outcome_df, Tt_outcome_df = train_test_split(descriptors, outcome, 
                           test_size = 0.30, random_state = 11)

# To transform to numpy arrays without index:
TV_features = np.array(TV_features_df)
Tt_features = np.array(Tt_features_df)
TV_outcome = np.array(TV_outcome_df[target_names])
Tt_outcome = np.array(Tt_outcome_df[target_names])

# Percentage of indviduals in each class:
TV_class_frac = TV_outcome_df.sum(axis = 0, skipna = True)*100/len(TV_outcome)
Tt_class_frac = Tt_outcome_df.sum(axis = 0, skipna = True)*100/len(Tt_outcome)
# Save it:
fractions = pd.DataFrame(columns=target_names)
fractions = fractions.append(TV_class_frac,ignore_index=True)
fractions = fractions.append(Tt_class_frac,ignore_index=True)
fractions = fractions.set_index([pd.Index(['TV', 'Test'])])
fractions.to_csv(assessment_path+'perc_class_split.csv', index=True)

target_names2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'Q','R', 'S', 'T','V','W','Y','Z']
n_classes = len(target_names)

# To prepare real Test dataset to save results in STEP 4.
Tt_outcome_real = Tt_outcome_df
Tt_outcome_real['run'] = 'real'
Tt_outcome_real = Tt_outcome_real.reset_index()

print('All done for 1. DATA PREPARATION AND ASSESSMENT')


###############################################
#####    2.HYPERPARAMETER TUNING    ###########
###############################################

from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, make_scorer

# 4. Hyperparameter tunning: Stratified CV with Random Grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'log2'] # auto = sqrt
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
               'estimator__n_estimators': n_estimators,
               'estimator__max_features': max_features,
               'estimator__max_depth': max_depth,
               'estimator__min_samples_split': min_samples_split,
               'estimator__min_samples_leaf': min_samples_leaf,
               'estimator__bootstrap': bootstrap
               }
# print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
model_to_set = OneVsRestClassifier(RandomForestClassifier(class_weight='balanced'))
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
# The score chosen to select the best combination of parameters is weghted roc_auc with a one vs rest approach
scoring = make_scorer(roc_auc_score, multi_class="ovr",average="weighted")

rf_random = RandomizedSearchCV(estimator = model_to_set, param_distributions = random_grid, 
                                # scoring = 'roc_auc', # With very unbalanced datasets, using accuracy as scoring metric to choose the best combination of parameters will not be the best strategy. Use ROC Area instead
                                scoring = scoring,
                                return_train_score=True,
                                n_iter = 100, cv = 5,
                                # n_iter = 2, cv = 2, 
                                verbose=2, 
                                random_state=42, 
                                n_jobs = -1)
# Note: For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

# Fit the random search model
rf_random.fit(TV_features, TV_outcome)
## To see the best parameters and results:
# rf_random.best_params_
# rf_random.cv_results_
# rf_random.best_score_
# rf_random.best_index_
# rf_random.scorer_

# To save the csv with the following details:
# Hyperparameter tuning results:
results = pd.DataFrame.from_dict(rf_random.cv_results_)
results.to_csv(os.path.join(ht_path,'HT_results.csv'))
# Hyperparameter tuning best parameters and best score (roc auc)
best_parameters = pd.DataFrame(rf_random.best_params_, index=[0])
best_parameters['best_score'] = rf_random.best_score_
best_parameters.to_csv(os.path.join(ht_path,'HT_best_parameters_and_score.csv'),index=False)

print('All done for 2.HYPERPARAMETER TUNING')


###############################################
##### 3. MODEL TRAINING AND VALIDATION ########
###############################################

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Define the model with the selected parameters during HT
model = rf_random.best_estimator_
# define evaluation procedure
# cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

# #scoring = make_scorer(roc_auc_score, multi_class="ovr",average="weighted")
# scoring = {'weighted_roc_auc': make_scorer(roc_auc_score, multi_class="ovr",average="weighted"),
#            'micro_roc_auc': make_scorer(roc_auc_score, multi_class="ovr",average="micro"),
#            'macro_roc_auc': make_scorer(roc_auc_score, multi_class="ovr",average="macro"),
#            'weighted_precision': make_scorer(precision_score, average='weighted'),
#            'weighted_recall': make_scorer(recall_score, average='weighted'),
#            'weighted_f1': make_scorer(f1_score, average='weighted'),
#            'micro_precision': make_scorer(precision_score, average='micro'),
#            'micro_recall': make_scorer(recall_score, average='micro'),
#            'micro_f1': make_scorer(f1_score, average='micro'),
#            'macro_precision': make_scorer(precision_score, average='macro'),
#            'macro_recall': make_scorer(recall_score, average='macro'),
#            'macro_f1': make_scorer(f1_score, average='macro')}

# # Evaluate the model
# scores = cross_validate(model, TV_features, TV_outcome, scoring=scoring, cv=cv, n_jobs=-1)
# # To store as pandas dataframe and SAVE the results:
# results_cv = pd.DataFrame.from_dict(scores)
# results_cv.to_csv(os.path.join(cv_path,'CV_results.csv'), index=False)
# print('All done for 3. MODEL TRAINING AND VALIDATION')
print('SKIP STEP 3. MODEL TRAINING AND VALIDATION')


###############################################
#### 4. MODEL TRAINING AND PREDICTION #########
###############################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import pickle
from matplotlib import pyplot
from itertools import cycle
from scipy import interp


# The predicted outcome cannot be saved as a unique column, so I'll save an individual file with the predictions for each run
# To create an empty dataframe to store the performance results:
performance = pd.DataFrame(columns=['run', 'weighted_precision', 'weighted_recall','weighted_f1',
                                    'micro_precision', 'micro_recall','micro_f1',
                                    'macro_precision', 'macro_recall','macro_f1'])
# To create an empty dataframe to store roc_auc results:
auc_results = pd.DataFrame(columns=['run','weighted_roc_auc', 'micro_roc_auc', 'macro_roc_auc'])
report = pd.DataFrame(columns=['precision', 'recall', 'f1-score','support','run'])


# We train the model and obtain the predicitons on the test set. This is repeated N times
N = 10
for i in range(N):
  # fit the model on the whole dataset and obtain the probability for each class:
  run_name = 'run_'
  run_name +=str(i)
  model.fit(TV_features, TV_outcome)
  probs = model.predict_proba(Tt_features)
  # Probabilities to dataframe:
  probabilities = pd.DataFrame(probs, columns = target_names)
  probabilities['run'] = run_name
  # And save it:
  probabilities.to_csv(os.path.join(results_root,'TEST_probabilities','prob_'+run_name+'.csv'))
  # to get the PREDICTIONS on the test set:
  predictions = model.predict(Tt_features) # 1. get the predictions
  predf = pd.DataFrame(predictions, columns = target_names) # 2.transform into a df with column names = labels
  predf = predf.set_index(np.array(Tt_outcome_real["AE"])) #3. To assign AE ids as index
  predf["run"] = run_name # 4. To create a column indacating the run number
  predf = predf.reset_index() #5. To transform the index into a column AE (in a way that its equivalent to the real outcomes df (Tt_outcome_real))
  predf = predf.rename(columns={"index": "AE"})
  predfinal = predf.append(Tt_outcome_real, ignore_index = True)
  predfinal.to_csv(os.path.join(TEST_predictions,'preds_'+run_name+'.csv'))
  # To save the CONFUSION MATRIX for each run:
  cm = multilabel_confusion_matrix(Tt_outcome, predictions)
  # To save it (as a dataframe?)
  cmdf = pd.DataFrame() # To create an empty dataframe
  # To iterate through each individual CM and save all together
  for n in range(n_classes):
    partial = pd.DataFrame(cm[n])
    partial['class'] = target_names2[n]
    cmdf = cmdf.append(partial)
  # To save a unique dataframe for each run:
  cmdf.to_csv(os.path.join(results_root,'TEST_conf_matrix','cm_'+run_name+'.csv'))
  # Compute ROC AUC and other parameters:
  weighted_roc_auc = roc_auc_score(Tt_outcome, probs, multi_class="ovr",average="weighted")
  weighted_precision = precision_score(Tt_outcome, predictions, average="weighted")
  weighted_recall = recall_score(Tt_outcome, predictions, average="weighted")
  weighted_f1 = f1_score(Tt_outcome, predictions, average="weighted")
  micro_roc_auc = roc_auc_score(Tt_outcome, probs, multi_class="ovr",average="micro")
  micro_precision = precision_score(Tt_outcome, predictions, average="micro")
  micro_recall = recall_score(Tt_outcome, predictions, average="micro")
  micro_f1 = f1_score(Tt_outcome, predictions, average="micro")
  macro_roc_auc = roc_auc_score(Tt_outcome, probs, multi_class="ovr",average="macro")
  macro_precision = precision_score(Tt_outcome, predictions, average="macro")
  macro_recall = recall_score(Tt_outcome, predictions, average="macro")
  macro_f1 = f1_score(Tt_outcome, predictions, average="macro")
  auc_results.loc[i] = [run_name, weighted_roc_auc, micro_roc_auc, macro_roc_auc]
  performance.loc[i] = [run_name, weighted_precision, weighted_recall,weighted_f1,
                        micro_precision, micro_recall,micro_f1,
                        macro_precision, macro_recall,macro_f1]
  # Performance by class: classification_report:
  cr = classification_report(Tt_outcome, predictions, target_names=target_names, output_dict=True)
  crdf = pd.DataFrame(cr).transpose()
  crdf['run'] = run_name
  report = report.append(crdf)
  ## FEATURE IMPORTANCE:
  # Get numerical feature importances
  feature_importances_df = pd.DataFrame()
  for n in range(n_classes):
    cname = target_names[n]
    importances = list(model.estimators_[n].feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(descriptors_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    new_feature_importances_df = pd.DataFrame(feature_importances, columns=['feature_class_'+cname, 'importance_class_'+cname])
    feature_importances_df = pd.concat([feature_importances_df, new_feature_importances_df], axis=1)
  # To save it individually for each run:
  feature_importances_df.to_csv(os.path.join(results_root,'TEST_features_importance','feat_imp_'+run_name+'.csv'))
  # To save the ESTIMATORS or models:
  # save the model to disk
  filename = results_root+'/TEST_trained_models/RF_'+run_name+'.sav'
  pickle.dump(model, open(filename, 'wb'))

# To save predictions:
# predicted_outcome.to_csv(os.path.join(test_path,'TEST_predictions.csv'))
# To save ROC AUC:
auc_results.to_csv(os.path.join(test_path,'TEST_auc_results.csv'))
# To save performance (precision, recall, f1)
performance.to_csv(os.path.join(test_path,'TEST_precision_recall_f1.csv'))
report.to_csv(os.path.join(test_path,'TEST_report.csv'))



print('All done for 4. MODEL TRAINING AND PREDICTION')
print('END OF SCRIPT')