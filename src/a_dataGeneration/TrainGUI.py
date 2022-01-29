# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:04:48 2020

@author: jsalm
"""

import time
import os
import matplotlib.pyplot as plt
from os.path import join, abspath
from localPkg.disp import LabelMaker
from localPkg.datmgmt import DataManager

#%% Main GUI
"""
if __name__ == "__main__":
    from localModules import DataManager
    import matplotlib.pyplot as plt
    from os.path import join, abspath
    dirName = os.path.dirname(__file__)
    permSaveF = abspath(join(dirName,"..","b_dataAggregation","processedData"))
    tmpSaveF = join(dirName,"tmpSaves")
    foldername = join(dirName,"rawData")
    # dealing with the Channel situation: display RGB but edit gray scale
    im_dir = DataManager.DataMang(foldername)
    count = 0
    channel = 2
    ansload = input("Would you like to load a previous image? [Y/N] ")
    if ansload == 'y':
        filename = input('load: [Input Name] ')
        loadf = join(tmpSaveF,filename+'.pkl')
        wobj = DataManager.load_obj(loadf)
        wobj.init_wind()
        num = wobj.IMG_NUM
        name = wobj.IMG_NAME
        print("loading %s...."%(name))
        wind = main(wobj,tmpSaveF,permSaveF)
    else:
        ansload = input("What image number would you like to start from?: [int = 0 to %i] "%im_dir.dir_len)
        im_list = [i for i in range(im_dir.dir_len-1,-1,-1)] #[i for i in range(0,im_dir.dir_len)] #[i for i in range(im_dir.dir_len-1,-1,-1)] #[i for i in range(0,im_dir.dir_len)]
        for gen in im_dir.open_dir(im_list,'train'):
            image,nW,nH,chan,name,fN = gen
            wobj = PanZoomWindow(channel,image,name,im_list[count],windowName = name)
            print("loading %s..."%(name))
            wind = main(wobj,tmpSaveF,permSaveF)
        # tmp_image = Filters.normalize_img(image)
        bool_im = import_train_data(name,(nH,nW),permSaveF)
        wind.write2csv(tmpSaveF)
        plt.imshow(bool_im)
        plt.show()
        count += 1
        'end for'
    "end if"
'end if'
"""
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

# VERSION HISTORY
#6/29/2020: fillPoly() works, storage works, reconstruction works, just could use some user friendliness
#10/11/2021: WE BACK BB, fixed some menu functionality and added comments. Specifically added infinite redaction of lines and improved menu responsiveness.
