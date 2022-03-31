# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""

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

import src.localModules.DataManager as DataManager

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
modelDir = abspath(join(saveBin,"saveDT"))
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
dTime = '03022022' #date.today().strftime('%d%m%Y')

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

#%% XGBOOST ALGORITHM 
print('XGBoost:')

##XGBoost with Optimal HyperParameters
print("starting modeling career...")
coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]
XGBmodel = XGBClassifier(max_depth = coef[0],subsample = coef[1],n_estimators = coef[2],
                      colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate=coef[5], 
                      min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
                      reg_lambda = coef[9])


#%% MODEL FITTING
print('fitting...')
model = XGBmodel.fit(X_train,y_train)
y_score = evals_result = model.evals_result()
print(model.score(X_test,y_test))
filename = join(modelDir,('fittedXGB_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done')

y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
print('XGB Train accuracy',accuracy_score(y_train, y_train_predict))
print('XGB Test accuracy',accuracy_score(y_test,y_predict))

#%% CROSS VALIDATE k-fold (k=10)
scores = cross_val_score(estimator = model,
                          X = X_train,
                          y = y_train,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('XGB CV accuracy scores: %s' % scores)
print('XGB CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 
        
#Best coefficients so far:
    #coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]

"""
#%% SAMPLE CODE FOR OPTIMIZING PARAMETERS
score = 0.7159090909090909       
coef = [2,0.4,150,.8,1,.1,1,0,1,0.5]
for ii in range(2,31): 
    model = XGBClassifier(max_depth = ii,subsample = coef[1],n_estimators = coef[2],
                        colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate=coef[5], 
                        min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
                        reg_lambda = coef[9])
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