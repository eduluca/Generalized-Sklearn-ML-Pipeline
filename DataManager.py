# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:31:36 2020

@author: jsalm
"""

import os
import numpy as np
import cv2
import pickle
import sys

class DataMang:
    def __init__(self,directory):
        # dir_n = os.path.join(os.path.dirname(__file__),dirname)
        self.directory = directory
        self.dir_len = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory,name))])
    'end def'
    
    def _load_image(self,rootdir):
        im = np.array(cv2.imread(rootdir)[:,:,:]/255).astype(np.float32)
        return im

    def _load_image_train(self,rootdir):
        im = cv2.imread(rootdir)[:,:,:]
        return im

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
        tmp_root = str()
        tmp_files = []
        for root,_,files in os.walk(self.directory):
            for f in files:
                tmp_files.append(f)
            'end'
            tmp_root = root
        'end'
        tmp_files = sorted(tmp_files)
        for count in im_list:
            f = tmp_files[count]
            if step == 'train':
                im = self._load_image_train(os.path.join(tmp_root,f))
            elif step == 'test':
                im = self._load_image(os.path.join(tmp_root,f))
            'end if'
            name = [x for x in map(str.strip, f.split('.')) if x]
            # addition to handle naming convention provided by Yasin (Note: no spaces in names prefered -_-)
            if len(name)>2:
                n_out = '.'.join(name[:2])
            else:
                n_out = '.'.join(name[:1])
            nH,nW,chan = im.shape
            yield (im,nW,nH,chan,n_out)
        'end for'
    'end def'
'end class'

def save_obj(rootPath, obj):
    # include .pkl for rootPath
    with open(rootPath, 'wb') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
    'end with'
'end def'

def load_obj(rootPath):
    with open(rootPath, 'rb') as infile:
        result = pickle.load(infile)
    'end with'
    return result
'end def'
