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
#%% Globals
dTime = date.today().strftime('%d%m%Y')

#%% Initialize Image Parsing/Pre-Processing 
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35] #[i for i in range(start,im_dir.dir_len)]

#%% DEFINITIONS


#%% LOOP: Image Parsing/Pre-Processing 
if __name__ == '__main__':
    print("Number of processors: ", mp.cpu_count())
    #%% Loop Start - Basic Loop
    print('Starting PreProcessing Pipeline...')
    for i in im_list:
        result = ProcessPipe.mainLoop(i)
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
    # threadN = mp.cpu_count()
    # results = Parallel(n_jobs=threadN)(delayed(ProcessPipe.mainLoop)(i) for i in im_list) # only one that works? (03/04/2022)

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