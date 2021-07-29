import os
import numpy as np
from hmmlearn import hmm
import pandas as pd
import sys, getopt # To run as ijobs:
from datetime import datetime
import time

def run_simple_hmm(dataset_train, length_train, n_coms=20):
	X = dataset_train.to_numpy() # pandas df to numpy array
	Xl = length_train
	model = hmm.GaussianHMM(n_components=n_coms, covariance_type="full", n_iter=100)
	model.fit(X, Xl)
	logprob = model.score(X, Xl) # Returns logprob on training dataset
	transmat = model.transmat_ # Returns the transmisions probability matrix in training dataset
	Z = model.predict(X, Xl)
	zdf = pd.DataFrame(data=Z, columns=["states"]) # Arrray to pd dataframe
	# Reset index in the test pandas dataframe:
	Ydf = dataset_train.reset_index(level=['AE','tslotN'])
	mergedDf = Ydf.merge(zdf, left_index=True, right_index=True)
	return([mergedDf, logprob, transmat])

########

n_states = 17


TTdata = "file_name.txt"
df = pd.read_table(TTdata)
df.shape
list(df.columns)
df = df.drop(['tslot','changed','impMI','impLI'], axis=1)
df = df[['AE','tslotN','ITEMID','value']]

# To extract the length of each series (ie., the number of days by AE)
ldf = df.groupby('AE').max()[['tslotN']]
ldf = ldf['tslotN'].values # Transform it into a numpy array
TTdf = df.pivot_table(
        	values='value', 
        	index=['AE', 'tslotN'], 
        	columns='ITEMID')


### TRAINING THE HMM ###

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Starting Time =", current_time)

# States trained with TT_dataset
states_TT, logprobTT, transmatTT = run_simple_hmm(TTdf, ldf, n_coms = n_states)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Ending Time =", current_time)


## To format the outputs:
logprobTTdf = ["TT_dataset", "all vars", n_states, logprobTT]
logprobTTdf = pd.DataFrame(logprobTTdf).T
logprobTTdf.columns = ['training_dataset','hmm_vars','n_states','logLikelihood']
transmatTTdf = pd.DataFrame(transmatTT)
