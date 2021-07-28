### RF TRAINING AND EVALUATION FOR BINARY OUTCOMES ###

# The script is divided in 4 parts:
  # 1. Data rformatting
  # 2. Hyperparameter Tuning (HT_results)
  # 3. Model training and cross validation (CV_results)
  # 4. Model training and predictions (TEST_results)


## Intended to be run with arguments:
# bsub "python RF_training_binary.py Day1 D1_raw_data D1_no_pda D1_no_pda_IM"

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
# outcome = 'D1_pda_IM'

# RESULTS PATHS:
# results_root = results_root_path
# assessment_path = results_root+'assessment/'
# ht_path = results_root+'HT_results/'
# cv_path = results_root+'CV_results/'
# test_path = results_root+'TEST_results/'


# TO READ THE INPUT DATA (The datasets have been previously created to include only the relevant variables)
# root_path = root_path
file_name = 'file_name.txt'
TTdata = root_path + file_name
df = pd.read_table(TTdata)
df = df.set_index('AE') # To establish AE id as index
input_variables = list(df.columns)
with open(assessment_path+'input_data_column_names.txt', "w") as output:
    output.write(str(input_variables))

# DATA PROCESSING: Features and Targets and Convert Data to Arrays
# Outcome (or labels) are the values we want to predict
outcome = pd.DataFrame(df['IM'])
descriptors = df.drop('IM', axis = 1)
descriptors_list = list(descriptors.columns)
with open(assessment_path+'input_data_features.txt', "w") as output:
    output.write(str(descriptors_list))

# TRAINING/VALIDATION (TV, for hyperparameter tuning) and TEST (Tt, for model evaluation) Sets:
# Split the data into training and testing sets:
TV_features_df, Tt_features_df, TV_outcome_df, Tt_outcome_df = train_test_split(descriptors, outcome, 
                           test_size = 0.30, random_state = 11, 
                           stratify=outcome)  # Important to keep the % of classes similar in TV and Tt

# To transform to numpy arrays without index:
TV_features = np.array(TV_features_df)
Tt_features = np.array(Tt_features_df)
TV_outcome = np.array(TV_outcome_df['IM'])
Tt_outcome = np.array(Tt_outcome_df['IM'])

# Percentage of indviduals in each class:
TV_class_frac = np.bincount(TV_outcome)*100/len(TV_outcome)
Tt_class_frac = np.bincount(Tt_outcome)*100/len(Tt_outcome)
# Save it:
fractions = pd.DataFrame(index=['TV', 'Test'],columns=['0','1'])
fractions.loc['TV'] = TV_class_frac.reshape(-1, len(TV_class_frac))
fractions.loc['Test'] = Tt_class_frac.reshape(-1, len(Tt_class_frac))
fractions.to_csv(assessment_path+'perc_class_split.csv', index=True)

print('All done for 1. DATA PREPARATION AND ASSESSMENT')


###############################################
#####    2.HYPERPARAMETER TUNING    ###########
###############################################

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(class_weight='balanced')
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                scoring = 'roc_auc', # With very unbalanced datasets, using accuracy as scoring metric to choose the best combination of parameters will not be the best strategy. Use ROC Area instead
                                return_train_score=True,
                                n_iter = 100, cv = 5, 
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

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate



# Define the model with the selected parameters during HT
model = rf_random.best_estimator_
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
scoring = ['roc_auc', 'accuracy', 'precision', 'recall'] # precision and recall call the respective functions from sklearn.metrics and the default pos_label is 1
# But we can change it (or make sure we are getting the performance for the class of interest) by creating our custom_scorer
scoring = {'AUC': 'roc_auc', 
                'Accuracy': 'balanced_accuracy', # The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
                                                 # The best value is 1 and the worst value is 0 when adjusted=False.
                'Recall': 'recall',
                'Precision': 'precision',
                'F1': 'f1'}
# Evaluate the model
scores = cross_validate(model, TV_features, TV_outcome, scoring=scoring, cv=cv, n_jobs=-1)
results_cv = pd.DataFrame.from_dict(scores)
results_cv.to_csv(os.path.join(cv_path,'CV_results.csv'), index=False)

print('All done for 3. MODEL TRAINING AND VALIDATION')

###############################################
#### 4. MODEL TRAINING AND PREDICTION #########
###############################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import pickle
from matplotlib import pyplot


# To use the Tt_outcome_df with the observed outcome and AE id to save the predictions:
predicted_outcome = Tt_outcome_df
# To create an empty dataframe to store the performance results:
performance = pd.DataFrame(columns=['0','1', 'run'])
# To create an empty dataframe to store accuracy results:
accuracy = pd.DataFrame(columns=['run','balanced_accuracy'])
# To create an empty dataframe to store roc_auc results:
auc_results = pd.DataFrame(columns=['run','roc_auc', 'prc_auc'])


# We train the model and obtain the predicitons on the test set. This is repeated N times
N = 10
for i in range(N):
  # fit the model on the whole dataset:
  model.fit(TV_features, TV_outcome)
  probs = model.predict_proba(Tt_features)[:, 1]
  # to get the PREDICTIONS on the test set:
  predictions = model.predict(Tt_features)
  run_name = 'run_'
  run_name +=str(i)
  predicted_outcome[run_name] = predictions # In this way, I can identify the AEs which are always well predicted (and see if, for example, they share a shorter LOS)
  # To save the CONFUSION MATRIX for each run:
  cm = pd.DataFrame(confusion_matrix(Tt_outcome, predictions))
  cm.to_csv(os.path.join(results_root,'TEST_conf_matrix','cm_'+run_name+'.csv'))
  # To calculate ACCURACY on the Test dataset:
  acc = metrics.balanced_accuracy_score(Tt_outcome, predictions)
  accdf = pd.DataFrame([i, acc])
  accuracy.loc[i] = [i, acc]
  # To caculate precision, recall, F1
  perf = precision_recall_fscore_support(Tt_outcome, predictions)
  perfdf = pd.DataFrame(list(perf), columns=['0','1'], index=['precision','recall','F1','support'])
  perfdf['run'] = run_name
  performance = performance.append(perfdf)
  # To calculate AUC OF THE PRECISION-RECALL CURVE:
  precision, recall, thresholdsPR = precision_recall_curve(Tt_outcome, probs)
  # calculate precision-recall AUC
  prc_auc = auc(recall, precision)
  # To calculate ROC AUC score:
  roc_auc = roc_auc_score(Tt_outcome, probs)
  # roc_results.loc[i] = [i, roc_auc]
  auc_results.loc[i] = [i, roc_auc, prc_auc]
  ## FEATURE IMPORTANCE:
  # Get numerical feature importances
  importances = list(model.feature_importances_)
  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(descriptors_list, importances)]
  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
  feature_importances_df = pd.DataFrame(feature_importances, columns=['feature', 'importance'])
  # To save it individually for each run:
  feature_importances_df.to_csv(os.path.join(results_root,'TEST_features_importance','feat_imp_'+run_name+'.csv'))
  # To save the ESTIMATORS or models:
  # save the model to disk
  filename = results_root+'TEST_trained_models/RF_'+run_name+'.sav'
  pickle.dump(model, open(filename, 'wb'))
  # To save the parameters to plot the ROC and Precision-Recall curve (only for the first run):
  if i == 0:
    fpr, tpr, thresholds = roc_curve(Tt_outcome, probs) # for binary classification only
    # To plot the ROC:
    # pyplot.plot(fpr, tpr, marker='.', lw=2, label='RF AUC = '+str(round(roc_auc,2)))
    # To save fpr, tpr and thresholds. These can be read later on (as numpy.ndarray and shape (215,) in this case) to plot ROC curves for various models in one plot
    with open(results_root+'TEST_roc_parameters/'+'ROC_fpr.npy', 'wb') as f1:
      np.save(f1, fpr)
    with open(results_root+'TEST_roc_parameters/'+'ROC_tpr.npy', 'wb') as f2:
      np.save(f2, tpr)
    with open(results_root+'TEST_roc_parameters/'+'ROC_thresholds.npy', 'wb') as f3:
      np.save(f3, thresholds)
    with open(results_root+'TEST_prc_parameters/'+'PRC_recall.npy', 'wb') as f4:
      np.save(f4, recall)
    with open(results_root+'TEST_prc_parameters/'+'PRC_precision.npy', 'wb') as f5:
      np.save(f5, precision)
    with open(results_root+'TEST_prc_parameters/'+'PRC_thresholds.npy', 'wb') as f6:
      np.save(f6, thresholdsPR)

# To save predictions:
predicted_outcome.to_csv(os.path.join(test_path,'TEST_predictions.csv'))
# To save accuracy results:
accuracy.to_csv(os.path.join(test_path,'TEST_accuracy.csv'))
# To save ROC AUC:
auc_results.to_csv(os.path.join(test_path,'TEST_auc_results.csv'))
# To save performance (precision, recall, f1)
performance.to_csv(os.path.join(test_path,'TEST_precision_recall_f1.csv'))

print('All done for 4. MODEL TRAINING AND PREDICTION')
print('END OF SCRIPT')