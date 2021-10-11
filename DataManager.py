# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:31:36 2020

@author: jsalm
"""

import os
import numpy as np
import cv2
import pickle

class DataMang:
    def __init__(self,directory):
        # dir_n = os.path.join(os.path.dirname(__file__),dirname)
        self.directory = directory
        self.dir_len = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory,name))])
    'end def'
    
    def save_obj(obj):
        with open('mat.pkl', 'wb') as outfile:
            pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
        'end with'
    'end def'
    
    def load_obj(obj):
        with open('mat.pkl', 'rb') as infile:
            result = pickle.load(infile)
        'end with'
        return result
    'end def'
    
    def _load_image(self,rootdir):
        im = np.array(cv2.imread(rootdir)[:,:,:]/255).astype(np.float32)
        # im[im==0] = "nan"
        # im[im==1] = np.nanmin(im)
        # im[np.isnan(im)] = np.nanmin(im)
        return im
    
    def open_dir(self,im_list):
        """
        This is a chunky directory manager. 

        Parameters
        ----------
        *args : int or list of int
            

        Yields
        ------
        im : TYPE
            DESCRIPTION.
        nW : TYPE
            DESCRIPTION.
        nH : TYPE
            DESCRIPTION.

        """
        count = 0
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                if isinstance(im_list,list):
                    if count in im_list:
                        impath = os.path.join(root,f)
                        im = self._load_image(impath)
                        name = [x for x in map(str.strip, f.split('.')) if x]
                        nW,nH,chan = im.shape
                        yield (im,nW,nH,chan,name[0])
                    'end if'
                count += 1
                'end if'
            'end for'
        'end for'
    'end def'
'end class'