# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:31:36 2020

@author: jsalm
"""

import os
import numpy as np
import cv2
import dill as pickle
import multiprocessing

class DataMang:
    def __init__(self,directory):
        # dir_n = os.path.join(os.path.dirname(__file__),dirname)
        self.directory = directory
        self.dir_len = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory,name))])
        self.files = []
        self.root = ""
        self._get_DirInf()
    'end def'
    
    def _get_ImageRootDir(im_name):
        pass
        # get image directory based on image name
        return 0

    def _load_image(self,rootdir):
        im = np.array(cv2.imread(rootdir)[:,:,:]/255).astype(np.float32)
        return im

    def _load_image_train(self,rootdir):
        im = cv2.imread(rootdir)[:,:,:]
        return im
    
    def _get_DirInf(self):
        tmp_root = str()
        tmp_files = []
        for root,_,files in os.walk(self.directory):
            for f in files:
                tmp_files.append(f)
            'end'
            tmp_root = root
        'end'
        self.files = sorted(tmp_files)
        self.root = tmp_root
        return 0

    def compareDir(self,otherDir):
        def yasinSplit(file):
            tmp = file.split('.')
            name = '.'.join([tmp[0],tmp[1]])
            return name
        #enddef
        tmpHandle1 = []
        outNames = []
        otherF = [name for name in os.listdir(otherDir) if os.path.isfile(os.path.join(otherDir,name))]
        for f in otherF:
            tmpHandle1.append(yasinSplit(f))
        #endfor
        for f in self.files:
            if any([yasinSplit(f) in tmpHandle1]):
                outNames.append(yasinSplit(f))
            #endif
        #endfor
        return outNames

    def open_dir(self,im_list,step):
        """
        This is a chunky directory manager. 

        Parameters
        ----------
        *args : int or list of int
            

        Yields
        ------
        im : image in directory
            DESCRIPTION.
        nW : TYPE
            DESCRIPTION.
        nH : TYPE
            DESCRIPTION.

        """
        for count in im_list:
            f = self.files[count]
            if step == 'train':
                im = self._load_image_train(os.path.join(self.root,f))
            elif step == 'test':
                im = self._load_image(os.path.join(self.root,f))
            'end if'
            name = [x for x in map(str.strip, f.split('.')) if x]
            # addition to handle naming convention provided by Yasin (Note: no spaces in names prefered -_-)
            if len(name)>2:
                n_out = '.'.join(name[:2])
            else:
                n_out = '.'.join(name[:1])
            nH,nW,chan = im.shape
            yield (im,nW,nH,chan,n_out,count)
        'end for'
    'end def'

    def openFileI(self,i,step):
        f = self.files[i]
        if step == 'train':
            im = self._load_image_train(os.path.join(self.root,f))
        elif step == 'test':
            im = self._load_image(os.path.join(self.root,f))
        #endif
        name = [x for x in map(str.strip, f.split('.')) if x]
        # addition to handle naming convention provided by Yasin (Note: no spaces in names prefered -_-)
        if len(name)>2:
            n_out = '.'.join(name[:2])
        else:
            n_out = '.'.join(name[:1])
        #endif
        nH,nW,chan = im.shape
        return (im,nW,nH,chan,n_out,i)
'end class'



def yasin_DataHandler(imageName):
    tmpS1 = imageName.split('.')
    # tmpS2 = tmpS1[1].split('_')
    # tmpJ1 = '_'.join(tmpS2[:-1])
    name = '.'.join([tmpS1[0],tmpS1[1]])
    return name


def save_obj(rootPath, obj):
    # include .pkl for rootPath
    with open(rootPath, 'wb') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
    'end with'
    print('object saved.')
'end def'

def load_obj(rootPath):
    with open(rootPath, 'rb') as infile:
        result = pickle.load(infile)
    'end with'
    print('object loaded.')
    return result
'end def'

if __name__ == "__main__":
    pass
#endif