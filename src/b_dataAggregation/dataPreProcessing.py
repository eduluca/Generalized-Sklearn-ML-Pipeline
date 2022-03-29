# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0
"""
#%% IMPORTS
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import multiprocessing as mp
from datetime import date
from localPkg.preproc import ProcessPipe
from os.path import dirname, join, abspath, exists
#%% Globals
dTime = date.today().strftime('%d%m%Y')

#%% Initialize Image Parsing/Pre-Processing 
# change the 'start' in PARAMS to choose which file you want to start with.

#%% DEFINITIONS
def mainInput(im_list,q):
    # for i in im_list:
        q.put(ProcessPipe.mainLoop(im_list,rawDatDir,trainDatDir,savePath))
    #endfor
#enddef
    

#%% LOOP: Image Parsing/Pre-Processing 
if __name__ == '__main__':
    #%% PATHS 
    # Path to file
    cfpath = dirname(__file__) 
    aDatGenDir = abspath(join(cfpath,"..","a_dataGeneration"))
    bDatAggDir = abspath(join(cfpath,"..","b_dataAggregation"))
    # Path to images to be processed
    rawDatDir = join(aDatGenDir,"rawData")
    # Path to training files
    trainDatDir = join(bDatAggDir,"processedData","EL-11122021")
    # Path to aggregate data files
    aggDatDir = join(bDatAggDir,"aggregateData")
    savePath = join(aggDatDir,dTime)

    im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35] #[i for i in range(start,im_dir.dir_len)]
    print("Number of processors: ", mp.cpu_count())
    #%% Loop Start - Basic Loop
    print('Starting PreProcessing Pipeline...')
    # for i in im_list:
    #     result = ProcessPipe.mainLoop(i,rawDatDir,trainDatDir,savePath)
    #     break
    # #endfor 

    #%% Loop Start - multiprocessing documentation ex
    #! see. https://docs.python.org/3/library/multiprocessing.html !#
    # ***(03/27/2022): make an implmentation in mpProcessPipe and mpDataManager.
    # mp.set_start_method('spawn')
    # q = mp.Queue()
    # p = mp.Process(target = mainInput, args = (im_list,q)) 
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
    results = Parallel(n_jobs=threadN)(delayed(ProcessPipe.mainLoop)(i,rawDatDir,trainDatDir,savePath) for i in im_list) # only one that works? (03/04/2022)

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