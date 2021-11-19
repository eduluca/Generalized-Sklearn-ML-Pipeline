# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.20
"""
#%% IMPORTS
import time
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy.ndimage import convolve
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score

import ProcessPipe
import DataManager

#%% PATHS 
# Path to file
cfpath = os.path.dirname(__file__) 
# Path to images to be processed
folderName = os.path.join(cfpath,"images-5HT")
# Path to save bin : saves basic information
saveBin = os.path.join(cfpath,"save_bin")
# Path to training files
trainDatDir = os.path.join(cfpath,'archive-image-bin\\trained-bin-EL-11122021\\')
#%% Load Data
tmpLoadDir = os.path.join(trainDatDir, 'joined_data.pkl')
tmpDat = DataManager.load_obj(tmpLoadDir)
X = tmpDat[0]
y = tmpDat[1]
#Typing for memory constraints
X = np.float32(X)
y = np.float16(y)
#%% BASIC PADDING
lenX = []
padedX = []
count = 0
fill_val = 0
for im in X:
    for seg in im:
        lenX.append(len(seg))
    'end for'
'end for'
uVals = np.unique(lenX)
uMax = np.max(uVals)
for i in range(len(X)):
    for j in range(len(X[i])):
        nPad = uMax-len(X[i][j])
        padedX.append(np.append(X[i][j],np.zeros(nPad)))
    'end for'
'end for'
#%% Train-Test Split
#stack X and y
X = np.vstack(padedX)
y = np.vstack(y)
#adding in some refence numbers for later
idx = np.array([[i for i in range(0,len(y))]]).T
y = np.hstack((y,idx))
#split dataset
print('Splitting dataset...')
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        random_state=count)
ind_train = y_train[:,1]
ind_test = y_test[:,1]
y_train = y_train[:,0]
y_test = y_test[:,0]
# Print train-test characteristics
print('   '+"Training Data (N): " + str(len(y_train)))
print('     '+"Testing Data (N): " + str(len(y_test)))
print('     '+"y_train: " + str(np.unique(y_train)))
print('     '+"y_test: " + str(np.unique(y_test)))

#%% Create SVM Pipeline
pipe_svc = make_pipeline(RobustScaler(),SVC())

#%% SVM MODEL FITTING
# Create an instance of SVM and fit out data.
print("starting modeling career...")

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


#%% CROSS VALIDATION
scores = cross_val_score(estimator = pipe_svc,
                          X = X_train,
                          y = y_train,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = 5,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

#%% MODEL FITTING
fitted = pipe_svc.fit(X_train,y_train)
y_score = pipe_svc.fit(X_train,y_train).decision_function(X_test)
print(pipe_svc.score(X_test,y_test))

#%% MODEL TESTING
#pick a test image
# os.chdir(r'C:\Users\jsalm\Documents\Python Scripts\SVM_7232020')
for gen in im_dir.open_dir(im_list_test,'test'):
    image,nW,nH,chan,name = gen
    Test_im = image

#extract features
# im_segs_test, _, domains_test, paded_im_seg_test, _, hog_features_test = SVM.feature_extract(Test_im, ff_width, wiener_size, med_size,False)
# X_test = SVM.create_data(im_segs_test,False)

### SVM MODEL PREDICTION ###
predictions = fitted.predict(X_test)   

# predict_im = data_to_img(boolim2_2,predictfions)
ProcessPipe.overlay_predictions(image, train_bool, predictions, y_test, ind_test,domains)

### Confusion Matrix: Save fig if interesting ###
confmat = confusion_matrix(y_true = y_test, y_pred=predictions)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.savefig(os.path.join(saveBin,'confussion-matrix.png'),dpi=200,bbox_inches='tight')
plt.show()

### ROC Curve ###
fpr, tpr,_ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
ProcessPipe.write_auc(fpr,tpr)
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
plt.savefig(os.path.join(saveBin,'roc_auc_curve.png'),dpi=200,bbox_inches='tight')
# #PLOT IMAGES
# # Filters.imshow_overlay(Test_im,predict_im,'predictions2',True)

# name_list = ["image","denoised_im","median_im","thresh_im","dir_im","gau_im","di_im","t_im"]
# for i in range(0,len(image_tuple)):
#     plt.figure(name_list[i]);plt.imshow(image_tuple[i])