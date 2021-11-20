# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: Jacob Salminen
@version: 1.0.20
"""
#%% IMPORTS
import time
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy.ndimage import convolve
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.ensemble import GradientBoostingClassifier

import ProcessPipe
import Filters
import TrainGUI
import DataManager

#%% DEFINITIONS
def robust_save(fname):
    plt.savefig(os.path.join(fname,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
'end def'

#%% GRIDSEARCH PARAMS

# param_range = [0.0001,0.001,0.01,0.1,1,10,100,1000]
# param_range= np.arange(0.01,1,0.001)
param_range2_C = [40,50,60,70,80,90,100,110,120,130,140]
param_range2_ga = [0.0005,0.0006,0.0007,0.001,0.002,0.003,0.004]
deg_range = [2,3,4,5,6,7]
deg_range2 = [2,3,4,5,6,10]
poly_range = np.arange(2,10,1)
poly_range_C = np.arange(1e-15,1e-7,6e-10)
poly_range_ga = np.arange(1e8,1e15,6e12)
# param_grid = [{'svc__C':param_range,
#                 'svc__kernel':['linear']},
#               {'svc__C': param_range,
#                 'svc__gamma':param_range,
#                 'svc__kernel':['rbf']},
#               {'svc__C': param_range,
#                 'svc__gamma':param_range,
#                 'svc__kernel':['poly'],
#                 'svc__degree':deg_range}]
param_grid2 = [{'svc__C': param_range2_C,
                'svc__gamma':param_range2_ga,
                'svc__decision_function_shape':['ovo','ovr'],
                'svc__kernel':['rbf']},
                {'svc__C': param_range2_C,
                  'svc__gamma':param_range2_ga,
                  'svc__kernel':['poly'],
                  'svc__decision_function_shape':['ovo','ovr'],
                  'svc__degree':deg_range2}]
# param_grid2 = [{'svc__C': param_range2_C,
#                 'svc__gamma':param_range2_ga,
#                 'svc__decision_function_shape':['ovo','ovr'],
#                 'svc__kernel':['rbf']}]
# param_grid3 = [{'svc__C': poly_range_C,
#                 'svc__gamma':poly_range_ga,
#                 'svc__kernel':['poly'],
#                 'svc__degree':poly_range}]

#%% PATHS 
# Path to file
cfpath = os.path.dirname(__file__) 
# Path to images to be processed
folderName = os.path.join(cfpath,"images-5HT")
# Path to save bin : saves basic information
saveBin = os.path.join(cfpath,"save_bin")
# Path to training files
trainDatDir = os.path.join(cfpath,'archive-image-bin\\trained-bin-EL-11122021\\')

#%% PARAMS ###
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42

#%% Initialize Image Parsing/Pre-Processing 
#load image folder for training data
im_dir = DataManager.DataMang(folderName)
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [3,4,5,6,10,12,13,14,21,26,27,28,29,35] #[i for i in range(start,im_dir.dir_len)] 
# im_list NOTES: removed 3 (temporary), 
im_list_test = [1]
#define variables for loops
hog_features = [np.array([])]
im_segs = [np.array([],dtype='float32')]
bool_segs = [np.array([],dtype='float32')]
domains = [np.array([],dtype='float32')]
paded_im_seg = [np.array([],dtype='float32')]
X = []
y = []

#%% LOOP: Image Parsing/Pre-Processing 
print('Starting PreProcessing Pipeline...')
## change what channel is being imported on the main image.
im_dir = DataManager.DataMang(folderName)
for gen in im_dir.open_dir(im_list,'test'):
    #load image and its information
    t_start = time.time()
    image,nW,nH,chan,name,count = gen
    print('   '+'{}.) Procesing Image : {}'.format(count,name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    image = image[:,:,channel]
    #Import train data (if training your model)
    train_bool = TrainGUI.import_train_data(name,(nH,nW),trainDatDir)
    plt.figure('image')
    plt.imshow(image)
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, domains, paded_im_seg, paded_bool_seg, hog_features = ProcessPipe.feature_extract(image, ff_width, wiener_size, med_size,True,train_bool)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    tmp_X,tmp_y = ProcessPipe.create_data(hog_features,True,bool_segs)
    X.append(tmp_X)
    y.append(tmp_y)
    t_end = time.time()
    print('     '+'Number of Segments : %i'%(len(im_segs)))
    print('     '+"Processing Time for %s : %0.2f"%(name,(t_end-t_start)))
print('done')
#%% Save Data
tmpDat = [X,y]
tmpSaveDir = os.path.join(trainDatDir, 'joined_data.pkl')
DataManager.save_obj(tmpSaveDir,tmpDat)