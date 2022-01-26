"""
Created on Tues Jan 25 19:06:00 2022

@author: Jacob Salminen
@version: 1.0
"""
#%% IMPORTS
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from os.path import join, abspath, dirname
from localPkg.preproc import ProcessPipe
from localPkg.datmgmt import DataManager
from localPkg.disp import LabelMaker
#%% Globals
dTime = date.today().strftime('%d%m%Y')
results = []
#%% PATHS 
# Path to file
cfpath = dirname(__file__) 
# Path to images to be processed
folderName = abspath(join(cfpath,"..","a_dataGeneration","rawData"))
# Path to training files
trainDatDir = join(cfpath,"processedData","EL-11122021")
# Path to save bin : saves basic information
saveBin = join(cfpath,"saveBin")
# Path to aggregate data files
aggDatDir = join(cfpath,"aggregateData")
#%% 
dir = DataManager.DataMang(aggDatDir)
files = []

for r,d,f in os.walk(aggDatDir):
    for name in f:
        if name == '_folderLog.txt' or name == 'train-data-ALL':
            continue
        #endif
        print(join(r,name))
        files.append(DataManager.load_obj(join(r,name)))
    #endfor
#endfor

#%%
saveName = join(aggDatDir,"train-data-ALL")
X = []
y = []
for i in range(0,len(files)):
    X.append(files[i][0])
    y.append(files[i][1])
#endfor
DataManager.save_obj(saveName,(X,y))
