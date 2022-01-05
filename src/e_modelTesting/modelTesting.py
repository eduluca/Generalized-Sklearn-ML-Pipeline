# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.0
"""
#%% IMPORTS
import time
from datetime import date
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join, abspath

from sklearn.metrics import confusion_matrix, auc, roc_curve

import localModules.ProcessPipe as ProcessPipe
import localModules.DataManager as DataManager

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
#%% MODEL TESTING
#load model
modelPath = join(modelDir,('SVMfitted_'+dTime+'.sav'))
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
plt.savefig(join(saveBin,'roc_auc_curve.png'),dpi=200,bbox_inches='tight')
#%% PLOT IMAGES
# predict_im = ProcessPipe.data_to_img(boolim2_2,predictions)
# ProcessPipe.overlay_predictions(image, train_bool, predictions, y_test, ind_test, domains)

#%% TEST RANDOM IMAGE AND VALIDATE VISUALLY
#imports
import localModules.ProcessPipe as ProcessPipe
#pick a test image
im_list_test = [0]
reduceFactor = 2
#image directory
im_dir = DataManager.DataMang(folderName)
X = [] 
for gen in im_dir.open_dir(im_list_test,'test'):
    t_start = time.time()
    image,nW,nH,chan,name,count = gen
    Test_im = image
    print('   '+'{}.) Procesing Image : {}'.format(count,name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    image = image[:,:,channel]
    #extract features from image using method(ProcessPipe.feature_extract) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, domains, paded_im_seg, paded_bool_seg, hog_features = ProcessPipe.feature_extract(image, ff_width, wiener_size, med_size,reduceFactor,False)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    tmp_X = ProcessPipe.create_data(hog_features,False,bool_segs)
    X.append(tmp_X)
    t_end = time.time()
    print('     '+'Number of Segments : %i'%(len(im_segs)))
    print('     '+"Processing Time for %s : %0.2f"%(name,(t_end-t_start)))
#stack X
X = np.vstack(X)
#Typing for memory constraints
X = np.float32(X)
print('done')
#%% VISUALIZE & VALIDATE
predictions = model.predict(X)
# predict_im = data_to_img(boolim2_2,predictions)
ProcessPipe.overlayValidate(image, predictions, domains)
