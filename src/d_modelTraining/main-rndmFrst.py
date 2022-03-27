# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""

import numpy as np
import multiprocessing as mp
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
from os import mkdir
import dill as pickle

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import   accuracy_score
from sklearn.ensemble import  RandomForestClassifier

from xgboost import XGBClassifier

import localPkg.datmgmt as DataManager

#%% PATHS 
# Path to file
cfpath = dirname(__file__) 
# Path to images to be processed
folderName = abspath(join(cfpath,"..","a_dataGeneration","rawData"))
# Path to save bin : saves basic information
saveBin = join(cfpath,"saveBin")
# Path to training files
trainDatDir = abspath(join(cfpath,"..","b_dataAggregation","processedData","EL-11122021"))
# Path to model
modelDir = abspath(join(saveBin,"saveRF"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
# Make directory for saves
try:
    mkdir(abspath(join(modelDir)))
except FileExistsError:
  print('Save folder for model already exists!')
#endtry

#%% DEFINITIONS & PARAMS
def robust_save(fname):
    plt.savefig(join(saveBin,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
#enddef

#%% PARAMS
dTime = '12242021' #date.today().strftime('%d%m%Y')

#%% Load k-split Data (k=10)
tmpSaveDir = join(cvDatDir, ('CVjoined_data_'+dTime+'.pkl'))
tmpSave = DataManager.load_obj(tmpSaveDir)
X_train = tmpSave[0]
X_test = tmpSave[1]
y_train = tmpSave[2]
y_train = y_train.reshape(len(y_train),1)
y_test = tmpSave[3]
y_test = y_test.reshape(len(y_test),1)
X = np.vstack((X_train,X_test))
y = np.ravel(np.vstack((y_train,y_test)))
print("y_train: " + str(np.unique(y_train)))
print("y_test: " + str(np.unique(y_test)))


#%% RANDOM FOREST ALGORITHM 
print('Random Forest:')

#%% CREATE RANDOMFOREST PIPELINE
print("starting modeling career...")
coef = [671,10,68,3,650,87,462]
RFmodel = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
                                       max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
                                       n_estimators = coef[4], max_samples = coef[5],
                                       max_features = coef[6])

#%% MODEL FITTING
print('fitting...')
model = RFmodel.fit(X_train,y_train)
y_score = model.decision_function(X_test)
print(model.score(X_test,y_test))
filename = join(modelDir,('fittedRF_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done')

y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
print('RF Train accuracy',accuracy_score(y_train, y_train_predict))
print('RF Test accuracy',accuracy_score(y_test,y_predict))

#%% Cross Validate
scores = cross_val_score(estimator = model,
                          X = X,
                          y = y,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('RF CV accuracy scores: %s' % scores)
print('RF CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

#Best coefficients so far:
    #coef = [671,10,68,3,650,87,462]

"""
#%% SAMPLE CODE FOR OPTIMIZING PARAMETERS
score = 0.75       
coef = [671,10,68,3,650,192,462]
for ii in range(2,500,10): 
        model = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
                                       max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
                                       n_estimators = coef[4], max_samples = coef[5],
                                       max_features = ii)
        model.fit(X_train,y_train) 
        y_predict = model.predict(X_test) 
        y_train_predict = model.predict(X_train)
        newscore = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        print(ii,newscore, end="")
        if newscore > score:
            print(' best so far')
            score = newscore
        else:
            print()
"""