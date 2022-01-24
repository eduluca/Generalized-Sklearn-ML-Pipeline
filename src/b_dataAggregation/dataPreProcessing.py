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
# from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as pltc
from os.path import join, abspath, dirname

import src.localModules.ProcessPipe as ProcessPipe
from src.a_dataGeneration import TrainGUI
import src.localModules.DataManager as DataManager

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
#%% Initialize Image Parsing/Pre-Processing 
#load image folder for training data
im_dir = DataManager.DataMang(folderName)
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35] #[i for i in range(start,im_dir.dir_len)]

#%% DEFINITIONS
# Callback function to collec the output from parallel processing in 'result'
def collect_result(result):
    global results
    results.append(result)

def mainLoop(fileNum):
    global dTime, cfpath, trainDatDir, saveBin, aggDatDir, im_dir, im_list
    #%% PARAMS ###
    channel = 2
    ff_width = 121
    wiener_size = (5,5)
    med_size = 10
    count = 42
    reduceFactor = 2
    
    #define variables for loops
    hog_features = [np.array([],dtype='float64')]
    im_segs = [np.array([],dtype='float64')]
    bool_segs = [np.array([],dtype='float64')]

    t_start = time.time()
    #opend filfe
    image,nW,nH,_,name,count = im_dir.openFileI(fileNum)
    #load image and its information
    print('   '+'{}.) Procesing Image : {}'.format(count,name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    image = image[:,:,channel]
    #Import train data (if training your model)
    train_bool = TrainGUI.import_train_data(name,(nH,nW),trainDatDir)
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, _, _, _, hog_features = ProcessPipe.feature_extract(image, ff_width, wiener_size, med_size,reduceFactor,True,train_bool)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    tmp_X,tmp_y = ProcessPipe.create_data(hog_features,bool_segs,fileNum,True)
    t_end = time.time()
    print('     '+'Number of Segments : %i'%(len(im_segs)))
    print('     '+"Processing Time for %s : %0.2f"%(name,(t_end-t_start)))
    return (tmp_X, tmp_y)
    #endfor
#enddef
#%% LOOP: Image Parsing/Pre-Processing 
if __name__ == '__main__':
    for i in im_list:
        result = mainLoop(i)
        break
    #endfor

    '''
    print("Number of processors: ", mp.cpu_count())
    #%% Loop Start
    print('Starting PreProcessing Pipeline...')
    reduceFactor = 2
    ## change what channel is being imported on the main image.
    im_dir = DataManager.DataMang(folderName)
    pool = mp.Pool(mp.cpu_count())
    # result = Parallel(n_jobs=mp.cpu_count())(delayed(mainLoop)(i) for i in range(0,len(im_list))) # old - del 01/05/2022
    pool.map_async(mainLoop, im_list, callback=collect_result)
    # close pool and let all the processes complete
    pool.close()
    pool.join()
    # extract valid data into a neater structure
    tmpDat = results[0]
    X = []
    y = []
    for i in range(0,len(tmpDat)):
        X.append(tmpDat[i][0])
        y.append(tmpDat[i][1])
    #endfor
    print('done')
    # Save Data
    tmpDat = (X,y)
    tmpSaveDir = join(aggDatDir, ('joined_data_'+dTime+'.pkl'))
    DataManager.save_obj(tmpSaveDir,tmpDat)
    '''
#endif