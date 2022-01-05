# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.20
"""
#%% IMPORTS
import time
from datetime import date
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

import numpy as np
from os.path import dirname, join, abspath

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import localModules.DataManager as DataManager
import dill as pickle

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
modelDir = abspath(join(cfpath,"saveSVM"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
#%% Script Params
# PARMS
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42
dTime = '12242021' #date.today().strftime('%d%m%Y')
#%%
tmpSaveDir = join(cvDatDir, ('CVjoined_data_'+dTime+'.pkl'))
tmpSave = DataManager.load_obj(tmpSaveDir)
X_train = tmpSave[0]
X_test = tmpSave[1]
y_train = tmpSave[2]
y_test = tmpSave[3]
#%% Create SVM Pipeline
pipe_svc = make_pipeline(RobustScaler(),SVC())

#%% SVM MODEL FITTING
# Create an instance of SVM and fit out data.
print("starting modeling career...")

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

#%% GRIDSEARCH (IF NECESSARY)

# gs = GridSearchCV(estimator = pipe_svc,
#                   param_grid = param_grid2,
#                   scoring = 'roc_auc',
#                   cv = 5,
#                   n_jobs = -1,
#                   verbose = 10)


# print("Fitting...")
# gs = gs.fit(X_train,y_train)
# print('best score: ' + str(gs.best_score_))
# print(gs.best_params_)
# pipe_svc = gs.best_estimator_
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
model = pipe_svc.fit(X_train,y_train)
y_score = model.decision_function(X_test)
print(pipe_svc.score(X_test,y_test))
filename = join(modelDir,('fitted_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done')
#%% CROSS VALIDATION
scores = cross_val_score(estimator = model,
                          X = X_test,
                          y = y_test,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = 5,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 