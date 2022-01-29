# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0
"""
#%% IMPORTS
import time
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, abspath, dirname
from datetime import date
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
#%% Initialize Image Parsing/Pre-Processing 
#load image folder for training data
im_dir = DataManager.DataMang(folderName)
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

def mainLoop(fileNum):
    global dTime, cfpath, folderName, trainDatDir, saveBin, aggDatDir, im_dir, im_list
    #%% PARAMS ###
    channel = 2
    ff_width = 121
    wiener_size = (5,5)
    med_size = 10
    count = 42
    reduceFactor = 2
    # im_list NOTES: removed 3 (temporary),
    #define variables for loops
    hog_features = [np.array([],dtype='float64')]
    im_segs = [np.array([],dtype='float64')]
    bool_segs = [np.array([],dtype='float64')]

    t_start = time.time()
    #opend filfe
    image,nW,nH,_,name,count = im_dir.openFileI(fileNum,'train')
    #load image and its information
    print('   '+'{}.) Procesing Image : {}'.format(count,name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    image = image[:,:,channel]
    #Import train data (if training your model)
    train_bool = LabelMaker.import_train_data(name,(nH,nW),trainDatDir)
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, _, _, _, hog_features = ProcessPipe.feature_extract(image, ff_width, wiener_size, med_size,reduceFactor,True,train_bool)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    result = ProcessPipe.create_data(hog_features,bool_segs,fileNum,True)
    # result = (tmp_X, tmp_y)
    t_end = time.time()
    print('     '+'Number of Segments : %i'%(len(im_segs)))
    print('     '+"Processing Time for %s : %0.2f"%(name,(t_end-t_start)))
    tmpSaveDir = join(aggDatDir, (f'trained_data_{dTime}_{fileNum}.pkl'))
    DataManager.save_obj(tmpSaveDir,result)
    return result
    #endfor
#enddef
#%% LOOP: Image Parsing/Pre-Processing 
if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())
    #%% Loop Start - Basic Loop
    print('Starting PreProcessing Pipeline...')
    for i in im_list:
        result = mainLoop(i)
        break
    #endfor

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
    # from joblib import Parallel, delayed
    # threadN = mp.cpu_count()-2
    # results = Parallel(n_jobs=threadN)(delayed(mainLoop)(i) for i in im_list) # old - del 01/05/2022

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