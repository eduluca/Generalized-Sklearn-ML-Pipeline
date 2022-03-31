# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.20
"""
#%% IMPORTS
import time
import numpy as np
import multiprocessing as mp
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from os.path import dirname, join, abspath
from os import mkdir
from datetime import date
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

from localPkg.datmgmt import DataManager
import dill as pickle

#%% PATHS 
print("Number of processors: ", mp.cpu_count())
# Path to file
cfpath = dirname(__file__) 
# Path to images to be processed
folderName = abspath(join(cfpath,"..","a_dataGeneration","rawData"))
# Path to save bin : saves basic information
saveBin = join(cfpath,"saveBin")
# Path to training files
trainDatDir = abspath(join(cfpath,"..","b_dataAggregation","processedData","EL-11122021"))
# Path to model
modelDir = abspath(join(saveBin,"saveSVM"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
# Make directory for saves
try:
  mkdir(abspath(join(modelDir)))
except FileExistsError:
  print('Save folder for model already exists!')
#endtry
#%% Script Params
# PARMS
dTime = '28032022' #date.today().strftime('%d%m%Y')
#%% GRIDSEARCH PARAMS

# param_range = [0.0001,0.001,0.01,0.1,1,10,100,1000]
# param_range= np.arange(0.01,1,0.001)
param_range2_C = [40,50,60,70,80,90,100,110,120,130,140]
param_range2_ga = [0.0005,0.0006,0.0007,0.001,0.002,0.003,0.004]
deg_range = [2,3,4,5,6,7]
deg_range2 = [2,3,4,5,6,10]
poly_range = np.arange(2,10,1)
poly_range_C = np.arange(1e-15,1e-7,6e-10)
poly_range_ga = np.arange(1e8,1e15,6e12)
# param_grid = [{'svc__C':param_range,
#                 'svc__kernel':['linear']},
#               {'svc__C': param_range,
#                 'svc__gamma':param_range,
#                 'svc__kernel':['rbf']},
#               {'svc__C': param_range,
#                 'svc__gamma':param_range,
#                 'svc__kernel':['poly'],
#                 'svc__degree':deg_range}]
param_grid2 = [{'svc__C': param_range2_C,
                'svc__gamma':param_range2_ga,
                'svc__decision_function_shape':['ovo','ovr'],
                'svc__kernel':['rbf']},
                {'svc__C': param_range2_C,
                  'svc__gamma':param_range2_ga,
                  'svc__kernel':['poly'],
                  'svc__decision_function_shape':['ovo','ovr'],
                  'svc__degree':deg_range2}]
# param_grid2 = [{'svc__C': param_range2_C,
#                 'svc__gamma':param_range2_ga,
#                 'svc__decision_function_shape':['ovo','ovr'],
#                 'svc__kernel':['rbf']}]
# param_grid3 = [{'svc__C': poly_range_C,
#                 'svc__gamma':poly_range_ga,
#                 'svc__kernel':['poly'],
#                 'svc__degree':poly_range}]

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

#%% Create SVM Pipeline
pipe_svc = make_pipeline(RobustScaler(),SVC(),verbose=True)

#%% SVM MODEL FITTING
# Create an instance of SVM and fit out data.
print("starting modeling career...")

#%% GRIDSEARCH (IF NECESSARY)
"""
gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid2,
                  scoring = 'f1',
                  cv = 5,
                  n_jobs = -1,
                  verbose = 10)


print("Fitting...")
gs = gs.fit(X_train,y_train)
print('best score: ' + str(gs.best_score_))
print(gs.best_params_)
pipe_svc = gs.best_estimator_
"""
### END Gridsearch ####

#%% PARAMETER SETTING (IF AVAILABLE)
### Setting Parameters ###
print('fitting...')
#{'svc__C': 100, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'} (~0.72% f1_score)
#{svc__C=130, svc__decision_function_shape=ovr, svc__gamma=0.0005, svc__kernel=rbf}

pipe_svc.set_params(svc__C =  130, 
                    svc__gamma = 0.0005, 
                    svc__kernel =  'rbf',
                    svc__probability = True,
                    svc__shrinking = False,
                    svc__decision_function_shape = 'ovr')

#%% MODEL FITTING
model = pipe_svc.fit(X_train,y_train) # Train Model
print('done fitting.')
#%% MODEL EVALUATION
print('evaluating...')
y_score = model.decision_function(X_test) # get scores and predictions for test set
print(model.score(X_test,y_test)) # print roc-auc of model fit
print('done evaluating.')
#%% MODEL SAVE
print('saving...')
filename = join(modelDir,('fittedSVM_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done.')

y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
print('RF Train accuracy',accuracy_score(y_train, y_train_predict))
print('RF Test accuracy',accuracy_score(y_test,y_predict))
#%% CROSS VALIDATION
scores = cross_val_score(estimator = model,
                          X = X,
                          y = y,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = 5,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 