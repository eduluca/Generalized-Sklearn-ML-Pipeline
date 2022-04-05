# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:04:48 2020

@author: jsalm
"""
#%%
import time
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
from os.path import join, abspath
from localPkg.disp import LabelMaker
from localPkg.datmgmt import DataManager

#%% Main GUI
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
    chsnDir = join(permSaveF,'EL-11122021')

    comparedDir = im_dir.compareDir(chsnDir)
    ansload = input("Would you like to load a previous image? [Y/N] ")
    if ansload == 'Y':
        print('Available files to load:')
        cnt = 0
        for ss in comparedDir:
            print(f'{cnt}) {ss}')
            cnt += 1
        #endfor
        fNum = input('load: [Input Number:int] ')        
        loadf = join(foldername,comparedDir[int(fNum)]+'.tif')
        im = im_dir._load_image_train(loadf)
        nH,nW,chan = im.shape
        wobj = LabelMaker.PanZoomWindow(channel,im,comparedDir[int(fNum)],int(fNum),windowName = comparedDir[int(fNum)])
        wobj.overlayOldData(comparedDir[int(fNum)],(nH,nW),chsnDir)
        num = wobj.IMG_NUM
        name = wobj.IMG_NAME
        print("loading %s...."%(name))
        print(f"start with old data: {wobj.OOD}")
        wind = wobj.main(tmpSaveF,permSaveF)
    else:
        print('Available files to load:')
        cnt = 0
        for ss in im_dir.files:
            print(f'{cnt}) {ss}')
            cnt += 1
        #endfor
        ansload = input("What image number would you like to start from?: [int = 0 to %i] "%im_dir.dir_len)
        im_list = [i for i in range(im_dir.dir_len-1,-1,-1)] #[i for i in range(0,im_dir.dir_len)] #[i for i in range(im_dir.dir_len-1,-1,-1)] #[i for i in range(0,im_dir.dir_len)]
        for gen in im_dir.open_dir(im_list,'train'):
            image,nW,nH,chan,name,fN = gen
            wobj =LabelMaker.PanZoomWindow(channel,image,name,im_list[count],windowName = name)
            print("loading %s..."%(name))
            wind = wobj.main(wobj,tmpSaveF,permSaveF)
        # tmp_image = Filters.normalize_img(image)
        bool_im = LabelMaker.import_train_data(name,(nH,nW),permSaveF)
        wind.write2csv(tmpSaveF)
        plt.imshow(bool_im)
        plt.show()
        count += 1
        'end for'
    "end if"
'end if'
# VERSION HISTORY
#6/29/2020: fillPoly() works, storage works, reconstruction works, just could use some user friendliness
#10/11/2021: WE BACK BB, fixed some menu functionality and added comments. Specifically added infinite redaction of lines and improved menu responsiveness.

# %%
