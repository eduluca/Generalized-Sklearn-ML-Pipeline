# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:19:38 2021

@author: jsalm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""
from os import mkdir
from os.path import join, abspath, dirname

import numpy as np
import matplotlib.pyplot as plt
import cv2
import dill as pickle


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import localPkg.preproc as ProcessPipe
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
modelDir = abspath(join(saveBin,"saveKNN"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
# Make directory for saves
mkdir(abspath(join(modelDir)))

#%% DEFINITIONS & PARAMS
def robust_save(fname):
    plt.savefig(join(saveBin,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
#enddef

#%% PARAMS
param_grid = {'kneighborsclassifier__n_neighbors':[5,7,10,13,15,18,20]}
dTime = '12242021' #date.today().strftime('%d%m%Y')

#%% Load k-split Data (k=10)
tmpSaveDir = join(cvDatDir, ('CVjoined_data_'+dTime+'.pkl'))
tmpSave = DataManager.load_obj(tmpSaveDir)
X_train = tmpSave[0]
X_test = tmpSave[1]
y_train = tmpSave[2]
y_test = tmpSave[3]
X = np.vstack((X_train,X_test))
y = np.vstack((y_train,y_test))
print("y_train: " + str(np.unique(y_train)))
print("y_test: " + str(np.unique(y_test)))

#%% Create KNN pipeline
pipe_knn = make_pipeline(RobustScaler(),KNeighborsClassifier())

#%% SVM MODEL FITTING
#we create an instance of SVM and fit out data.
print("starting modeling career...")

#%% GRIDSEARCH (IF NECESSARY)
# gs = GridSearchCV(estimator = pipe_knn,
#                   param_grid = param_grid,
#                   scoring = 'roc_auc',
#                   cv = 5,
#                   n_jobs = -1,
#                   verbose = 10)


# print("Fitting...")
# gs = gs.fit(X_train,y_train)
# print('best score: ' + str(gs.best_score_))
# print(gs.best_params_)
# pipe_knn = gs.best_estimator_
### END Gridsearch ####

#%% SETTING PARAMETERS
#{'kneighborsclassifier__n_neighbors': 7}
print('fitting...')

pipe_knn.set_params(kneighborsclassifier__n_neighbors = 7)

#%% MODEL FITTING
model = pipe_knn.fit(X_train,y_train)
y_score = model.decision_function(X_test)
print(model.score(X_test,y_test))
filename = join(modelDir,('fittedKNN_'+dTime+'.sav'))
pickle.dump(model, open(filename, 'wb'))
print('done')

y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
print('RF Train accuracy',accuracy_score(y_train, y_train_predict))
print('RF Test accuracy',accuracy_score(y_test,y_predict))
#%% CROSS VALIDATE
scores = cross_val_score(estimator = pipe_knn,
                          X = X,
                          y = y,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 