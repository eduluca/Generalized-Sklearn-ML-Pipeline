# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy.ndimage import convolve
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

import SVM
import Filters
import ML_interface_SVM_V3
import DataManager

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")

def robust_save(fname):
    plt.savefig(os.path.join(save_bin,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
####PARAMS####
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
###END###

### DATA PROCESSING IMAGE 1 ###

#initialize test data set
dirname = os.path.dirname(__file__)
foldername = os.path.join(dirname,"images_5HT")
im_dir = DataManager.DataMang(foldername)

### PARAMS ###
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42
###

#load image folder for training data
dirname = os.path.dirname(__file__)
foldername = os.path.join(dirname,"images_5HT")
im_dir = DataManager.DataMang(foldername)
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [i for i in range(start,im_dir.dir_len)]
hog_features = []
for gen in im_dir.open_dir(im_list):
    #load image and its information
    image,nW,nH,chan,name = gen
    print('procesing image : {}'.format(name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))
    image = image[:,:,channel]
    #Import train data (if training your model)
    train_bool = ML_interface_SVM_V3.import_train_data(name,(nW,nH),'train_71420')
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, domains, paded_im_seg, paded_bool_seg, hog_features = SVM.feature_extract(image, ff_width, wiener_size, med_size,True,train_bool)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    X,y = SVM.create_data(hog_features,True,bool_segs)
    break

print('done')
#adding in some refence numbers for later
y = np.vstack([y,np.arange(0,len(y),1)]).T
#split dataset

print('Splitting dataset...')
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=count)
ind_train = y_train[:,1]
ind_test = y_test[:,1]

y_train = y_train[:,0]
y_test = y_test[:,0]



print("y_train: " + str(np.unique(y_train)))
print("y_test: " + str(np.unique(y_test)))



#create SVM pipline
#try using a GBC
pipe_svc = make_pipeline(RobustScaler(),SVC())



##### RANDOM FOREST ALGORITHM #####

print('Random Forest:')


##Random Forest with Optimal HyperParameters
print("starting modeling career...")
coef = [671,10,68,3,650,87,462]
RFmodel = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
                                       max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
                                       n_estimators = coef[4], max_samples = coef[5],
                                       max_features = coef[6])

##Cross Validate
scores = cross_val_score(estimator = RFmodel,
                          X = X_train,
                          y = y_train,
                          cv = 5,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('RF CV accuracy scores: %s' % scores)
print('RF CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 


##Fitting Model
print('fitting...')
RFmodel.fit(X_train,y_train)
y_predict = RFmodel.predict(X_test)
y_train_predict = RFmodel.predict(X_train)
print('RF Train accuracy',accuracy_score(y_train, y_train_predict))
print('RF Test accuracy',accuracy_score(y_test,y_predict))

#ROC/AUC score
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


#Best coefficients so far:
    #coef = [671,10,68,3,650,87,462]
    
#Sample code for testing hyperparameters
# score = 0.75       
# coef = [671,10,68,3,650,192,462]
# for ii in range(2,500,10): 
#         model = RandomForestClassifier(max_depth = coef[0], min_samples_split = coef[1], 
#                                        max_leaf_nodes = coef[2], min_samples_leaf = coef[3],
#                                        n_estimators = coef[4], max_samples = coef[5],
#                                        max_features = ii)
#         model.fit(X_train,y_train) 
#         y_predict = model.predict(X_test) 
#         y_train_predict = model.predict(X_train)
#         newscore = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
#         print(ii,newscore, end="")
#         if newscore > score:
#             print(' best so far')
#             score = newscore
#         else:
#             print()





##### XGBOOST ALGORITHM #####

print('XGBoost:')

##XGBoost with Optimal HyperParameters
print("starting modeling career...")
coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]
XGBmodel = XGBClassifier(max_depth = coef[0],subsample = coef[1],n_estimators = coef[2],
                      colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate=coef[5], 
                      min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
                      reg_lambda = coef[9])

##Cross Validate (Takes a long time to run, may want to be removed for efficiency)
scores = cross_val_score(estimator = XGBmodel,
                          X = X_train,
                          y = y_train,
                          cv = 5,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('XGB CV accuracy scores: %s' % scores)
print('XGB CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 


##Fitting Model
print('fitting...')
XGBmodel.fit(X_train,y_train)
y_predict = XGBmodel.predict(X_test)
y_train_predict = XGBmodel.predict(X_train)
print('XGB Train accuracy',accuracy_score(y_train, y_train_predict))
print('XGB Test accuracy',accuracy_score(y_test,y_predict))



#ROC/AUC score
print('XGB Train ROC/AUC Score', roc_auc_score(y_train, XGBmodel.predict_proba(X_train)[:,1]))
print('XGB Test ROC/AUC Score', roc_auc_score(y_test, XGBmodel.predict_proba(X_test)[:,1]))

#ROC/AUC plotting
plt.figure(1)

fpr, tpr, thresholds = roc_curve(y_test, XGBmodel.predict_proba(X_test)[:,1],drop_intermediate=False)
plt.plot(fpr,tpr,label = 'XGBoost(area = %0.2f)' 
         % roc_auc_score(y_test, XGBmodel.predict_proba(X_test)[:,1]), color='red',lw=3)


        
#Best coefficients so far:
    #coef = [2,0.28,150,0.57,0.36,0.1,1,0,0.75,0.42]
    
#Sample code for testing hyperparameters
    # score = 0.7159090909090909       
    # coef = [2,0.4,150,.8,1,.1,1,0,1,0.5]
    # for ii in range(2,31): 
    #     model = XGBClassifier(max_depth = ii,subsample = coef[1],n_estimators = coef[2],
    #                       colsample_bylevel = coef[3], colsample_bytree = coef[4],learning_rate=coef[5], 
    #                       min_child_weight = coef[6], random_state = coef[7],reg_alpha = coef[8],
    #                       reg_lambda = coef[9])
    #     model.fit(X_train,y_train) 
    #     y_predict = model.predict(X_test) 
    #     y_train_predict = model.predict(X_train)
    #     newscore = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    #     print(ii,newscore, end="")
    #     if newscore > score:
    #         print(' best so far')
    #         score = newscore
    #     else:
    #         print()
    
### XGBoost MODEL PREDICTION ###
### Fitting Model ###
fitted = XGBmodel.fit(X_train,y_train)
y_score = XGBmodel.fit(X_train,y_train).predict_proba(X_test)

predictions = fitted.predict(X_test)   

# predict_im = data_to_img(boolim2_2,predictfions)
SVM.overlay_predictions(image, train_bool, predictions, y_test, ind_test,domains)


##### KNN ALGORITHM #####


print('KNN:')
pipe_knn = make_pipeline(RobustScaler(),KNeighborsClassifier())

#SVM MODEL FITTING
#we create an instance of SVM and fit out data.
print("starting modeling career...")

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

### Setting Parameters ###
#{'kneighborsclassifier__n_neighbors': 7}
print('fitting...')

pipe_knn.set_params(kneighborsclassifier__n_neighbors = 7)

### Cross Validate ###
scores = cross_val_score(estimator = pipe_knn,
                          X = X_train,
                          y = y_train,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

### Fitting Model ###
fitted = pipe_knn.fit(X_train,y_train)
y_score = pipe_knn.fit(X_train,y_train).predict_proba(X_test)
print(pipe_knn.score(X_test,y_test))

### DATA PROCESSING IMAGE 2 ###
#pick a test image
# os.chdir(r'C:\Users\jsalm\Documents\Python Scripts\SVM_7232020')
Test_im = np.array(cv2.imread("images_5HT/injured 60s_sectioned_CH2.tif")[:,:,2]/255).astype(np.float32)

#extract features
# im_segs_test, _, domains_test, paded_im_seg_test, _, hog_features_test = SVM.feature_extract(Test_im, ff_width, wiener_size, med_size,False)
# X_test = SVM.create_data(im_segs_test,False)

# ### SVM MODEL PREDICTION ###
# predictions = fitted.predict(X_test)   

# # predict_im = data_to_img(boolim2_2,predictfions)
# SVM.overlay_predictions(image, train_bool, predictions, y_test, ind_test,domains)

### Confusion Matrix: Save fig if interesting ###
# confmat = confusion_matrix(y_true = y_test, y_pred=predictions)
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

# plt.xlabel('Predicted label')
# plt.ylabel('True label')

# plt.tight_layout()
# plt.savefig(os.path.join(save_bin,'confussion_matrix.png'),dpi=200,bbox_inches='tight')
# plt.show()

### ROC Curve ###
fpr, tpr,_ = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr, tpr)
SVM.write_auc(fpr,tpr)
#fpr,tpr,roc_auc = SVM.read_auc()


plt.figure(1)

plt.plot(fpr, tpr, color='lightblue', lw=3, label='KNN (area = %0.2f)' % roc_auc)


##### SVM ALGORITHM #####


#we create an instance of SVM and fit out data.
print('SVM:')
print("starting modeling career...")

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

### Cross Validate ###
scores = cross_val_score(estimator = pipe_svc,
                          X = X_train,
                          y = y_train,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

### Fitting Model ###
fitted = pipe_svc.fit(X_train,y_train)
y_score = pipe_svc.fit(X_train,y_train).decision_function(X_test)
print(pipe_svc.score(X_test,y_test))

### DATA PROCESSING IMAGE 2 ###
#pick a test image
# os.chdir(r'C:\Users\jsalm\Documents\Python Scripts\SVM_7232020')
Test_im = np.array(cv2.imread("images_5HT/injured 60s_sectioned_CH2.tif")[:,:,2]/255).astype(np.float32)

#extract features
# im_segs_test, _, domains_test, paded_im_seg_test, _, hog_features_test = SVM.feature_extract(Test_im, ff_width, wiener_size, med_size,False)
# X_test = SVM.create_data(im_segs_test,False)

# ### SVM MODEL PREDICTION ###
# predictions = fitted.predict(X_test)   

# # predict_im = data_to_img(boolim2_2,predictfions)
# SVM.overlay_predictions(image, train_bool, predictions, y_test, ind_test,domains)

### Confusion Matrix: Save fig if interesting ###
# confmat = confusion_matrix(y_true = y_test, y_pred=predictions)
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

# plt.xlabel('Predicted label')
# plt.ylabel('True label')

# plt.tight_layout()
# plt.savefig(os.path.join(save_bin,'confussion_matrix.png'),dpi=200,bbox_inches='tight')
# plt.show()

### ROC Curve ###
fpr, tpr,_ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
SVM.write_auc(fpr,tpr)
#fpr,tpr,roc_auc = SVM.read_auc()

plt.figure(1)

plt.plot(fpr, tpr, color='darkorange',lw=3, label='SVM (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.legend(loc="lower right")
plt.show()
plt.savefig(os.path.join(save_bin,'roc_auc_curve.png'),dpi=200,bbox_inches='tight')
# #PLOT IMAGES
# # Filters.imshow_overlay(Test_im,predict_im,'predictions2',True)

# name_list = ["image","denoised_im","median_im","thresh_im","dir_im","gau_im","di_im","t_im"]
# for i in range(0,len(image_tuple)):
#     plt.figure(name_list[i]);plt.imshow(image_tuple[i])