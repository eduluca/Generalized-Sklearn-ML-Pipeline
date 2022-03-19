# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0
"""
#%% IMPORTS
from genericpath import exists
import time
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, abspath, dirname, exists
from datetime import date
from localPkg.preproc import ProcessPipe
from localPkg.datmgmt import DataManager
from localPkg.disp import LabelMaker
#%% Globals
dTime = date.today().strftime('%d%m%Y')

#%% Initialize Image Parsing/Pre-Processing 
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35] #[i for i in range(start,im_dir.dir_len)]

#%% DEFINITIONS
def robust_save(fname):
    plt.savefig(join(fname,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
#enddef

# Callback function to collec the output from parallel processing in 'result'
def collect_result(result):
    global results
    results.append(result)

def directoryHandler(dirDir):
    curDir = DataManager.DataMang(folderName)
    fs = curDir.files
#enddef

def mainLoop(fileNum):
    #%% Globals
    global dTime, cfpath, folderName, trainDatDir, aggDatDir, savePath
    
    #%% PATHS 
    # Path to file
    cfpath = dirname(__file__) 
    # Path to images to be processed
    folderName = abspath(join(cfpath,"..","a_dataGeneration","rawData"))
    # Path to training files
    trainDatDir = join(cfpath,"processedData","EL-11122021")
    # Path to save bin : saves basic information
    # saveBin = join(cfpath,"saveBin")
    # Path to aggregate data files
    aggDatDir = join(cfpath,"aggregateData")
    savePath = join(aggDatDir,dTime)
    if ~exists(savePath):
        os.mkdir(savePath)
    #endif

    #%% Initialize Image Parsing/Pre-Processing 
    #load image folder for training data
    imDir = DataManager.DataMang(folderName)

    #%% PARAMS
    channel = 2
    fftWidth = 121
    wienerWindowSize = (5,5)
    medWindowSize = 10
    # seedN = 42
    reduceFactor = 2

    #%% MAIN PROCESS
    ProcessPipe.dispTS()
    #opend filfe
    imageOut,nW,nH,_,imName,imNum = imDir.openFileI(fileNum,'train')
    #load image and its information
    print('   '+'{}.) Procesing Image : {}'.format(imNum,imName))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    imageIn = imageOut[:,:,channel]
    #Import train data (if training your model)
    trainBool = LabelMaker.import_train_data(imName,(nH,nW),trainDatDir)
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    imSegs, boolSegs, _, _, _, hogFeats = ProcessPipe.feature_extract(imageIn, fftWidth, wienerWindowSize, medWindowSize, reduceFactor, True, trainBool)
    chosenFeats = hogFeats
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    result = ProcessPipe.create_data(chosenFeats,boolSegs,fileNum,True)
    
    #%% WRAP-UP MAIN
    ProcessPipe.dispTS(False)
    print('     '+'Number of Segments : %i'%(len(chosenFeats)))
    tmpSaveDir = join(savePath, (f'trained_data_{dTime}_{fileNum}.pkl'))
    DataManager.save_obj(tmpSaveDir,result)
    return result
    #endfor
#enddef

#%% LOOP: Image Parsing/Pre-Processing 
if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())
    #%% Loop Start - Basic Loop
    # print('Starting PreProcessing Pipeline...')
    # for i in im_list:
    #     result = mainLoop(i)
    #     break
    # #endfor

    #%% Loop Start - multiprocessing documentation ex
    #! see. https://docs.python.org/3/library/multiprocessing.html !#
    # mp.set_start_method('spawn')
    # # q = mp.Queue()
    # p = mp.Process(target = mainLoop, args = (im_list,)) 
    # p.start()
    # p.join()

    #%% Loop Start - parallel processing
    # import concurrent.futures
    # with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.map(mainLoop, im_list)
    #endwith                        

    #%% Loop Start - async-multi processing num. 1
    from joblib import Parallel, delayed
    threadN = mp.cpu_count()
    results = Parallel(n_jobs=threadN)(delayed(ProcessPipe.mainLoop)(i) for i in im_list) # only one that works? (03/04/2022)

    #%% Loop Start - async-multi processing num. 2
    # pool = mp.Pool(mp.cpu_count())
    # #! extract valid data into a neater structure !#
    # tmpDat = results[0]
    # X = []
    # y = []
    # for i in range(0,len(tmpDat)):
    #     X.append(tmpDat[i][0])
    #     y.append(tmpDat[i][1])
    # #endfor
    # print('done')
    # # Save Data
    # tmpDat = (X,y)
    # tmpSaveDir = join(aggDatDir, ('joined_data_'+dTime+'.pkl'))
    # DataManager.save_obj(tmpSaveDir,tmpDat)
#endif