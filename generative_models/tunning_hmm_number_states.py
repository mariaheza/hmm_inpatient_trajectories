# 2-fold cross validation process where the ‘training and test’ set was randomly divided into two subsets, which were separately used to train a model and then fit the entire dataset. The output of both models was compared on a concordance matrix of co-occurrences for patient and day with the aim to find an association between states in both output datasets
# The outputs are for the two TT_cv datasets:
	# 1. The original training datasets with the predicted states
	# 2. The logprob for each training dataset
	# 3. The transitions probability matrix for each training dataset

## To run the python script with arguments as the number of states:
#bsub python hmm_2cv.py 5"


import os
import numpy as np
from hmmlearn import hmm
import pandas as pd
import sys, getopt


def run_cv_hmm(dataset_train, dataset_fit,length_train, length_fit, n_coms=20):
	X = dataset_train.to_numpy()
	Xl = length_train
	Y = dataset_fit.to_numpy()
	Yl = length_fit
	model = hmm.GaussianHMM(n_components=n_coms, covariance_type="full", n_iter=100)
	model.fit(X, Xl)
	logprob = model.score(X, Xl)
	transmat = model.transmat_
	Z = model.predict(Y, Yl)
	zdf = pd.DataFrame(data=Z, columns=["states"])
	Ydf = dataset_fit.reset_index(level=['AE','tslotN'])
	mergedDf = Ydf.merge(zdf, left_index=True, right_index=True)
	return([mergedDf, logprob, transmat])


########################
########################

# To run as ijob with different number of states as arguments:
n_states = int(sys.argv[1])

TTdata = "~/file_name.txt"
df = pd.read_table(TTdata)
df.shape
list(df.columns)
df = df.drop(['tslot','changed','impMI','impLI'], axis=1)
df = df[['AE','tslotN','ITEMID','value']]
df[:10]

# Dataset formatting:
# We need to know the length of each series (ie., the number of days by AE)
ldf = df.groupby('AE').max()[['tslotN']]
ldf = ldf['tslotN'].values 
TTdf = df.pivot_table(
        	values='value', 
        	index=['AE', 'tslotN'], 
        	columns='ITEMID')


# To split the dataset in 2 for 2-CV:
Naes = len(np.unique(df['AE'].values))
aes = np.unique(df['AE'].values)
type(aes)
sample_size = int(Naes/2)
index = np.random.choice(aes, sample_size, replace=False)

S1 = df[df.AE.isin(index)] # subset1
S2 = df[~df.AE.isin(index)] # subset2

# Length of each series (ie., the number of days by AE)
lS1 = S1.groupby('AE').max()[['tslotN']]
lS1 = lS1['tslotN'].values

lS2 = S2.groupby('AE').max()[['tslotN']]
lS2 = lS2['tslotN'].values

S1df = S1.pivot_table(
        	values='value', 
        	index=['AE', 'tslotN'], 
        	columns='ITEMID')

S2df = S2.pivot_table(
        	values='value', 
        	index=['AE', 'tslotN'], 
        	columns='ITEMID')


### TRAIN THE HMM ###

# States trained with S1
states_S1, logprobS1, transmatS1 = run_cv_hmm(S1df, TTdf, lS1, ldf, n_coms = n_states)
# States trained with S2
states_S2, logprobS2, transmatS2 = run_cv_hmm(S2df, TTdf, lS2, ldf, n_coms = n_states)


## To format the outputs:
logprobS1df = ["tt_cv1","all vars", n_states, logprobS1]
logprobS1df = pd.DataFrame(logprobS1df).T
logprobS1df.columns = ['training_dataset','hmm_vars','n_states','logLikelihood']
transmatS1df = pd.DataFrame(transmatS1)


logprobS2df = ["tt_cv2","all vars", n_states, logprobS2]
logprobS2df = pd.DataFrame(logprobS2df).T
logprobS2df.columns = ['training_dataset','hmm_vars','n_states','logLikelihood']
transmatS2df = pd.DataFrame(transmatS2)