import time
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
from os.path import join, abspath
from localPkg.disp import LabelMaker
from localPkg.datmgmt import DataManager

#%% Check Train Data
if __name__ == "__main__":
    #%% cell 1
    print('Running ''Check Train Data''...')

    ### PARAMS ###
    channel = 2
    ff_width = 121
    wiener_size = (5,5)
    med_size = 10
    start = 0
    count = 42
    ###

    dirName = os.path.dirname(__file__)
    permSaveF = abspath(join(dirName,"..","b_dataAggregation","processedData","EL-11122021"))
    rawFName = join(dirName,"rawData")
    im_dir = DataManager.DataMang(rawFName)
    im_list = [i for i in range(start,im_dir.dir_len)]
    print('INFO: Directory contains %i files'%(im_dir.dir_len))
    #Find file similar between the bin and tif files
    comparedDir = im_dir.compareDir(permSaveF)
    for name in comparedDir:
        #load image and its information
        t_start = time.time()
        # name = DataManager.yasin_DataHandler(gen)
        print('   '+'Procesing Image : {}'.format(name))
        #Import train data (if training your model)
        nH = 1440
        nW = 1920
        train_bool = LabelMaker.import_train_data(name,(nH,nW),permSaveF)
        # plt.figure('Image'); plt.imshow(image); plt.show()
        plt.figure('Training Image');plt.imshow(train_bool); plt.show()
        plt.waitforbuttonpress()
        t_end = time.time()
    print('done')
'end if'