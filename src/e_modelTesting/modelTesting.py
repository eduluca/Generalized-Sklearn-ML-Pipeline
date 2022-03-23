# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.0
"""
#%% IMPORTS
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sklearn.pipeline import Pipeline
import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt

from os.path import dirname, join, abspath
from datetime import date
from sklearn.metrics import confusion_matrix, auc, roc_curve
from localPkg.preproc import ProcessPipe
from localPkg.disp import LabelMaker
from localPkg.datmgmt import DataManager

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
modelDir = abspath(join(cfpath,"..", "d_modelTraining","saveBin","saveSVM"))
# Path to cross-validated files
cvDatDir = abspath(join(cfpath,"..","c_dataValidation","saveBin"))
#%% DEFINITIONS

#%% Script Params
# PARMS
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42
dTime = '03022022' #'12242021' #date.today().strftime('%d%m%Y')
#%% MODEL TESTING
#load model
modelPath = join(modelDir,('fittedSVM_'+dTime+'.sav')) # join(modelDir,'fittedSVM_10012022.sav')
model = DataManager.load_obj(modelPath)
#load CV data
tmpSaveDir = join(cvDatDir, ('CVjoined_data_'+dTime+'.pkl'))
tmpDat = DataManager.load_obj(tmpSaveDir)
X_train = tmpDat[0]
X_test = tmpDat[1]
y_train = tmpDat[2]
y_test = tmpDat[3]
#%% SVM MODEL PREDICTION 
predictions = model.predict(X_test)
#%% FIGURES
### Confusion Matrix: Save fig if interesting ###
confmat = confusion_matrix(y_true = y_test, y_pred=predictions)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    #endfor
#endfor

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig(join(saveBin,'confussion-matrix.png'),dpi=200,bbox_inches='tight')
plt.show()

#%% FIGURES
### ROC Curve ###
y_score = model.predict_proba(X_test)
fpr, tpr,_ = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr, tpr)
ProcessPipe.write_auc(fpr,tpr,join(saveBin,f"datAUCROC_{dTime}.csv"))
#fpr,tpr,roc_auc = SVM.read_auc()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig(join(saveBin,'roc_auc_curve.png'),dpi=200,bbox_inches='tight')
#%% PLOT IMAGES
# predict_im = ProcessPipe.data_to_img(boolim2_2,predictions)
# ProcessPipe.overlay_predictions(image, train_bool, predictions, y_test, ind_test, domains)

#%% TEST RANDOM IMAGE AND VALIDATE VISUALLY
#imports
#pick a test image
im_list_test = [0]
reduceFactor = 2
#image directory
im_dir = DataManager.DataMang(folderName)
for i in im_list_test:
    xTest, image, domains = ProcessPipe.mainLoopTest(i)
#endfor
#%% VISUALIZE & VALIDATE
predictions = model.predict(xTest)
# predict_im = data_to_img(boolim2_2,predictions)
ProcessPipe.overlayValidate(image, predictions, domains)

#%% EDUARDO'S ROC IMPLEMENTATION
"""
# ==== RANDOM FOREST MODEL ==== #
#ROC/AUC score
RFmodel = 
print('RF Train ROC/AUC Score', roc_auc_score(y_train, RFmodel.predict_proba(X_train)[:,1]))
print('RF Test ROC/AUC Score', roc_auc_score(y_test, RFmodel.predict_proba(X_test)[:,1]))

#ROC/AUC plotting
plt.figure(1)

fpr, tpr, thresholds = roc_curve(y_test, RFmodel.predict_proba(X_test)[:,1],drop_intermediate=False)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Serotonin Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot(fpr,tpr,label = 'RForest(area = %0.2f)' 
         % roc_auc_score(y_test, RFmodel.predict_proba(X_test)[:,1]), color='blue',lw=3)
"""
"""
# ==== XGBOOST MODEL ==== #
#ROC/AUC score
print('XGB Train ROC/AUC Score', roc_auc_score(y_train, XGBmodel.predict_proba(X_train)[:,1]))
print('XGB Test ROC/AUC Score', roc_auc_score(y_test, XGBmodel.predict_proba(X_test)[:,1]))

#ROC/AUC plotting
plt.figure(1)

fpr, tpr, thresholds = roc_curve(y_test, XGBmodel.predict_proba(X_test)[:,1],drop_intermediate=False)
plt.plot(fpr,tpr,label = 'XGBoost(area = %0.2f)' 
         % roc_auc_score(y_test, XGBmodel.predict_proba(X_test)[:,1]), color='red',lw=3)
"""
"""
# ==== KNN MODEL ==== #
### ROC Curve ###
fpr, tpr,_ = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr, tpr)
ProcessPipe.write_auc(fpr,tpr)
#fpr,tpr,roc_auc = kNN.read_auc()
plt.figure(1)
plt.plot(fpr, tpr, color='lightblue', lw=3, label='KNN (area = %0.2f)' % roc_auc)
"""