# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.20
"""
#%% IMPORTS
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import time
import multiprocessing as mp
import numpy as np

from os.path import dirname, join, abspath
from datetime import date
from localPkg.datmgmt import DataManager
from localPkg.preproc.ProcessPipe import overlayValidate
from localPkg.disp import LabelMaker

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
# Path to Aggregate data
aggDatDir = abspath(join(cfpath,"..", "b_dataAggregation","aggregateData"))
#%% PARAMS
dTime = date.today().strftime('%d%m%Y')
count = 42
#%% Load Data
print('Loading Data...')
tmpLoadDir = join(aggDatDir, 'train-data-ALL.pkl') #join(aggDatDir, ('joined_data_'+dTime+'.pkl'))
tmpDat = DataManager.load_obj(tmpLoadDir)
X = tmpDat[0]
y = tmpDat[1]
doms = tmpDat[2]
# del tmpDat
#%% Train-Test Split
# print('Splitting Data...')
# #stack X and y
# X = np.vstack(X)
# y = np.vstack(y)
# #Typing for memory constraints
# X = np.float64(X)
# y = np.int16(y)
# #adding in some refence numbers for later
# idx = np.array([[i for i in range(0,len(y))]]).T
# y = np.hstack((y,idx))
# #split dataset
# X_train, X_test, y_train, y_test = train_test_split(X,y,
#                                                         test_size=0.3,
#                                                         shuffle=True,
#                                                         random_state=count)
# ind_train = y_train[:,1]
# ind_test = y_test[:,1]
# y_train = y_train[:,0]
# y_test = y_test[:,0]
# # Print train-test characteristics
# print('   '+"Training Data (N): " + str(len(y_train)))
# print('     '+"Testing Data (N): " + str(len(y_test)))
# print('     '+"y_train: " + str(np.unique(y_train)))
# print('     '+"y_test: " + str(np.unique(y_test)))

#%%
# im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35]
# channel = 2
# ##
# for fileNum in range(0,len(y)):
#     rotateSlice = np.arange(0,len(y[fileNum]),4)
#     imDir = DataManager.DataMang(folderName)
#     predictsI = y[fileNum][rotateSlice,0]
#     domainI = np.array(doms[fileNum])
#     imageOut,nW,nH,_,imName,imNum = imDir.openFileI(y[fileNum][0,1],'train')
#     print('   '+'{},{}.) Observing Image : {}'.format(fileNum,imNum,imName))
#     #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
#     #the notation RGB (0,1,2 respectively))=
#     # imageIn = imageOut[:,:,channel]
#     overlayValidate(imageOut,predictsI,domainI,saveBin)
# #endfor

#%%
dirName = os.path.dirname(__file__)

# rawFName = join(dirName,"rawData")
# im_dir = DataManager.DataMang(rawFName)
# print('INFO: Directory contains %i files'%(im_dir.dir_len))
# #Find file similar between the bin and tif files
# comparedDir = im_dir.compareDir(permSaveF)

im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35]
channel = 2
permSaveF = abspath(join(dirName,"..","b_dataAggregation","processedData","EL-11122021"))
##
for fileNum in range(0,len(y)):
    rotateSlice = np.arange(0,len(y[fileNum]),4)
    imDir = DataManager.DataMang(folderName)
    predictsI = y[fileNum][rotateSlice,0]
    domainI = np.array(doms[fileNum])
    imageOut,nW,nH,_,imName,imNum = imDir.openFileI(y[fileNum][0,1],'train')
    print('   '+'{},{}.) Observing Image : {}'.format(fileNum,imNum,imName))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    # imageIn = imageOut[:,:,channel]
    train_bool = LabelMaker.import_train_data(imName,(nH,nW),permSaveF)
    overlayValidate(imageOut,predictsI,domainI,saveBin,boolIm = train_bool)
#endfor
