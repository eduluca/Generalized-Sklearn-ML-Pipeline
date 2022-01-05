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

from sklearn.model_selection import train_test_split

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
# Path to Aggregate data
aggDatDir = abspath(join(cfpath,"..", "b_dataAggregation","aggregateData"))
#%% Script Params
# PARMS
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42
dTime = '12242021' #date.today().strftime('%d%m%Y')
#%% Load Data
print('Loading Data...')
tmpLoadDir = join(aggDatDir, ('joined_data_'+dTime+'.pkl'))
tmpDat = DataManager.load_obj(tmpLoadDir)
X = tmpDat[0]
y = tmpDat[1]
del tmpDat
#%% BASIC PADDING
# print('Padding Data...')
# X = ProcessPipe.padPreProcessed(X)
#%% Train-Test Split
print('Splitting Data...')
#stack X and y
X = np.vstack(X)
y = np.vstack(y)
#Typing for memory constraints
X = np.float32(X)
y = np.float16(y)
#adding in some refence numbers for later
idx = np.array([[i for i in range(0,len(y))]]).T
y = np.hstack((y,idx))
#split dataset
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

tmpDat = [X_train,X_test,y_train,y_test]
tmpSaveDir = join(saveBin, ('CVjoined_data_'+dTime+'.pkl'))
DataManager.save_obj(tmpSaveDir,tmpDat)
