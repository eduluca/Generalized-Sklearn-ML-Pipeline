# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:38:44 2020

@author: jsalm
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
# from sklearn import svm, datasets
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import convolve,distance_transform_edt,label, find_objects
from sklearn.metrics import auc
from skimage.feature import hog
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import os
import time
import csv

from ..datmgmt import DataManager
from ..disp import LabelMaker
from ..preproc import Filters

# import xlwings as xw

from IPython import get_ipython
# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['figure.figsize'] = (10,10)
# get_ipython().run_line_magic('matplotlib','qt5')

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save-bin")

global data,labels

def generate_train_sert_ID(boolim,image):
    if type(boolim[0,0]) != np.bool_:
        raise TypeError("args need to be type bool and tuple respectively")
    'end if'
    count = 0
    data = np.zeros((2,boolim.shape[0]*boolim.shape[1]))
    point_data = np.zeros((2,boolim.shape[0]*boolim.shape[1]))
    #generate list of points
    for i,row in enumerate(boolim):
        for j,col in enumerate(row):
            if col == True:
                data[0,count] = image[i,j]
                data[1,count] = 1
                point_data[0,count] = i
                point_data[1,count] = j
                count+=1
            else:
                data[0,count] = image[i,j]
                data[1,count] = 0
                point_data[0,count] = i
                point_data[1,count] = j
                count+=1
            'end if'
        'end for'
    'end for'
    return data,point_data
'end def'


def generate_test_sert_ID(boolim,image):
    if type(boolim[0,0]) != np.bool_:
        raise TypeError("args need to be type bool and tuple respectively")
    'end if'
    count =  0
    t_data = np.sum(boolim)
    data = np.zeros((2,t_data))
    point_data = np.zeros((2,t_data))
    for i,row in enumerate(boolim):
        for j,col in enumerate(row):
            if col == True:
                data[0,count] = image[i,j]
                data[1,count] = 0
                point_data[0,count] = i
                point_data[1,count] = j
                count+=1
    return data,point_data
'end def'

def get_coef(generator):
    weights = []
    for clf in generator:
        weights.append(clf.coef_)
    'end for'
    return weights
'end def'

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def gen_point_vector(image):
    point_data = np.zeros((image.shape[0]*image.shape[1],2))
    count = 0
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            point_data[count,:] = [i,j]
            count += 1
        'end for'
    'end for'
    return point_data
'end def'



def img_to_data(image,mask,keep_all = True,*kwargs):
    """

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    **params : image data type float32[:,:]
        DESCRIPTION.

    Returns
    ------
    array of data of shape [image.shape[0]*image.shape[1],number_of_parameters + image_data] represents
    all the parameters to be enetered into SVM image analysis

    """
    #initialize with original image data
    img_d = image.ravel()
    img_d = img_d.reshape(img_d.shape[0],1)
    
    con_data = img_d
    param_c = 0
    for data in kwargs:
        new_d = data.ravel()
        new_d = new_d.reshape(new_d.shape[0],1)
        con_data = np.concatenate((con_data,new_d),axis = 1)
        param_c += 1
    'end for'
    nonzero = np.sum(mask)
    
    mask_r = mask.ravel()
    mask_r = mask_r.reshape(mask_r.shape[0],1)
    point_data = gen_point_vector(image)
    
    if keep_all:
        data = con_data
        bool_set = mask_r.astype(int)
        
    else:
        masked = np.multiply(con_data,mask_r)
        masked_new = np.zeros((nonzero,con_data.shape[1]))
        point_new = np.zeros((nonzero,2))
        bool_set = np.zeros((nonzero,con_data.shape[1]))
        count = 0
        for i,x in enumerate(masked):
            if x.any() != 0:
                masked_new[count,:] = x
                bool_set[count,:] = mask_r[i,:]
                point_new[count,:] = point_data[i,:]
                count += 1
            'end if'
        'end for'
        data = masked_new
        bool_set = bool_set.astype(int)
        point_data = point_new
    
    return data,bool_set,point_data
'end def'

def data_to_img(mask,predicitons,positions):
    newim = np.zeros((mask.shape[0],mask.shape[1]))
    count = 0
    for i,row in enumerate(mask):
        for j,col in enumerate(row):
            if col == True:
                newim[i,j] = predictions[count]
                count += 1
            'end if'
        'end for'
    'end for'
    return newim

def get_nonzeros(image,val_vector,mask,tru_type = True):
    mask = mask.ravel()
    mask = mask.reshape(mask.shape[0],1)
    
    masklen = np.sum(mask.astype(int))
    
    mask_new = np.zeros((masklen,mask.shape[1]))
    points_new = np.zeros((masklen,2))
    
    points = gen_point_vector(image)
    vals_new = np.zeros((masklen,val_vector.shape[1]))
    
    count = 0        
    for i,x in enumerate(mask.astype(int)):
        if x != 0:
            vals_new[count,:] = val_vector[i,:]
            points_new[count,:] = points[i,:]
            if tru_type:
                # vals_new[count,-1] = 1
                mask_new[count,0] = 1
            else:
                # vals_new[count,-1] = 0
                mask_new[count,0] = 0
            count += 1
    return vals_new,mask_new.astype(int),points_new

# @optunity.cross_validated(x=data,y=labels,num_folds=5,regenerate_folds=True)
# def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, logC, logGamma):
#     model = SVC(C=10**logC,gamma=10**logGamma).fit(x_train,y_train)
#     decision_values = model.decision_function(x_test)
#     auc = optunity.metrics.roc_auc(y_test, decision_values)
#     return auc
# 'end def'

def filter_pipeline(image,ff_width,wiener_size,med_size,multiplier_a=1,multiplier_d=1):
    """
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    ff_width : TYPE
        DESCRIPTION.
    wiener_size : TYPE
        DESCRIPTION.
    med_size : TYPE
        DESCRIPTION.
    direction_features : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    ffimhi_new : TYPE
        DESCRIPTION.
    direction_features : TYPE
        DESCRIPTION.

    """
    direction_features = np.array([])
    #Normalize image
    norm_im = Filters.normalize_img(image)
    
    #Fourier Filter for removing low frequency components
    ffimhi_new = Filters.Hi_pass_filter(norm_im,ff_width)
    
    #denoising
    denoised_im = Filters.wiener(ffimhi_new,wiener_size,None)
    
    #Median Filter
    #Add running window median filter (10x10 to 30x30) and rectify the signal using max(0,val)
    median_im = Filters.median_filt(denoised_im,med_size)
    # median_im = ((denoised_im-median_im)>0)*median_im
    
    #adaptive thresholding
    #lowering the 3rd variable "multiplier_d" tightens border of predictions.
    threshed = convolve(Filters.adaptive_threshold(denoised_im,200,256,False),Filters._d3gaussian(5,multiplier_a,multiplier_d))
    
    
    #various convolution filters to pull out relative directions
    diagnols = np.array([[1,0,0,0,1],
                         [0,1,0,1,0],
                         [0,0,1,0,0],
                         [0,1,0,1,0],
                         [1,0,0,0,1]])
    t_cross = np.array([[0,0,1,0,0],
                        [0,0,1,0,0],
                        [1,1,1,1,1],
                        [0,0,1,0,0],
                        [0,0,1,0,0]])
    di_im = convolve(denoised_im,diagnols)
    t_im = convolve(denoised_im,t_cross)
    #Gaussian Image
    gauim = convolve(denoised_im,Filters._d3gaussian(5,1,1))
    #Differential image
    direction_features = (Filters.diffmat(denoised_im,np.arange(0,2*np.pi,2*np.pi/8),dim=(5,2)))
    return median_im, [threshed,di_im,t_im,gauim,direction_features]

def im_watershed(image,train = True, boolim = np.array([]),a=3,d=2):
    """
    image : np.array(float32)
        DESCRIPTION : 
    train : boolean
        DESCRIPTION : if train == True set boolim = np.array()
    segments image using a watersheding method with distance_transform_edt as the 
    descriminator. Returns list of segments
    """
    MIN_DISTANCE = 20
    im_list = []
    bool_list = []

    gau_im = convolve(Filters.normalize_img(image),Filters._d3gaussian(16,a,d))
    mn = np.mean(gau_im)
    segments = gau_im > mn
    D = distance_transform_edt(segments)
    tmplocalMax = peak_local_max(D, min_distance=MIN_DISTANCE,
                              labels=segments)
    localMax = np.zeros_like(image, dtype=bool)
    localMax[tuple(tmplocalMax.T)] = True
    markers = label(localMax,structure=np.ones((3,3)))[0]
    water_im = watershed(-D,markers,mask=segments)
    f = find_objects(water_im)  
    for seg in f:
        im_list.append(image[seg])
        if train:
            bool_list.append(boolim[seg])
    return im_list,bool_list,f

def cut_segs(im_list,bool_list,standardDim = 40,train = True):
    imOutList = []
    boOutList = []
    #check dimensions
    for k in range(0,len(im_list)):
        nH,nW = im_list[k].shape
        if nH > standardDim or nW > standardDim:
            imOutList.append(im_list[k])
            if train:
                boOutList.append(bool_list[k])
            #endif
            continue
        #endif
        leftoH = np.arange(0,int(nH/standardDim)*standardDim+1,standardDim) #create cuts for horizontal
        leftoW = np.arange(0,int(nW/standardDim)*standardDim+1,standardDim) #create cuts for widths
        if nH%standardDim != 0:
            leftoH = list(np.append(leftoH,nH-leftoH[-1])) #append leftover
        #endif
        if nW%standardDim != 0:
            leftoW = list(np.append(leftoW,nW-leftoW[-1])) #append leftover
        #endif
        for i in range(0,len(leftoH)):
            h1 = leftoH[i]
            h2 = leftoH[i+1]
            for j in range(0,len(leftoW)):
                w1 = leftoW[j]
                w2 = leftoW[j+1]
                imOutList.append(im_list[k][h1:h2,w1:w2]) #generate to image list
                if train:
                    boOutList.append(bool_list[k][h1:h2,w1:w2]) #generate new bool list
            #endfor
        #endfor
        return imOutList, boOutList
#enddef

def pad_segs(im_list,bool_list,f,train = True,fill_val = 0):
    """
    im_list
    bool_list
    f
    reducueFator : int (default = 2)
    train : boolean
        DESCRIPTION : if train == True set bool_list = np.array()
    fill_val = 0 : TYPE, integer or function (e.g. np.nan)
        DESCRIPTION. 
    """
    # get dimensions of larges segment to determine how much padding needs to be added to other images. Then perform reduction
    # based on the value of reduceN
    ySet = 300
    xSet = 300
    yval = []
    xval = []
    count = 0
    for seg in f:
        yval.append(abs(seg[0].stop-seg[0].start))
        xval.append(abs(seg[1].stop-seg[1].start))
    maxy = np.max(yval)
    maxx = np.max(xval)
    
    if maxy < ySet or maxx < xSet:
        maxy = ySet
        maxx = xSet
    else:
        raise ValueError('maxy or maxx too big for current padding setting...')
    #endif

    if maxy != maxx:
        maxMax = max(maxy,maxx) #find maximum index to make image square
        maxMax = maxMax + maxMax%2 #Make even
        maxx = maxx + (maxMax - maxx)
        maxy = maxy + (maxMax - maxy)
        #endif
    #endif
    
    for seg in f:
        dify = maxy - abs(seg[0].stop-seg[0].start) 
        difx = maxx - abs(seg[1].stop-seg[1].start)

        if dify != 0 or difx != 0:
            # Pad image segment on the right edge and bottom edge.
            im_list[count] = np.pad(im_list[count],((0,dify),(0,difx)),'constant',constant_values=fill_val)
            if train:
                bool_list[count] = np.pad(bool_list[count],((0,dify),(0,difx)),'constant',constant_values=fill_val)
        count += 1
    return im_list, bool_list, f

def downSampleStd(im_list, bool_list, train):
    for i in range(0,len(im_list)):
        # reduce image segment using downsampling: figure out how to reduce the image to a specific resolution
        # get shape of image
        # nH,nW = im_list[i].shape
        # determine factor for countless reduction
        redH = 2 #nH/reduceFactor
        redW = 2 #nW/reduceFactor
        # countless algorithm
        outIm = Filters.countless(im_list[i],redH,redW)
        # reassign
        im_list[i] = outIm
        if train:
            # reduce image segment using downsampling: figure out how to reduce the image to a specific resolution
            # get shape of image
            # nH,nW = bool_list[i].shape
            # determine factor for countless reduction
            redH = 2 #nH/reduceFactor
            redW = 2 #nW/reduceFactor
            # countless algorithm
            outIm = Filters.countless(bool_list[i],redH,redW)
            # reassign
            bool_list[i] = outIm
    return im_list, bool_list

def rotateNappend(im_list, bool_list):
    pass
#enddef

def feature_extract(image, ff_width, wiener_size, med_size,reduceFactor = 2, train = True,boolim = np.array([])):
    """
    SUMMARY : 
    image : TYPE
        DESCRIPTION. 
    ff_width : TYPE
        DESCRIPTION. 
    wiender_size : TYPE
        DESCRIPTION. 
    med_size : TYPE
        DESCRIPTION. 
    reduceFactor = 2 : TYPE, int 
        DESCRIPTION. 
    train = True : TYPE, boolean
        DESCRIPTION. 
    boolim = np.array([]) : TYPE, np.array([])
        DESCRIPTION. 
    """
    hog_features = []
    hog_dim = (4,4)
    median_im,feature_list = filter_pipeline(image,ff_width,wiener_size,med_size)
    #segment image using watershed and pad images for resizing
    im_list, bool_list, f = im_watershed(median_im,train,boolim)
    #pad segments
    paded_im_seg,paded_bool_seg,_ = pad_segs(im_list,bool_list,f,train,0)
    #generate hog features
    for seg in im_list:
        normalized = Filters.normalize_img(seg)
        # print(seg.shape) #debug
        testDim = tuple([2*x for x in hog_dim])
        if seg.shape < testDim:
            hogi = hog(normalized, visualize = True, block_norm='L2-Hys', pixels_per_cell=(2,2))
        else:
            hogi = hog(normalized, visualize = True, block_norm='L2-Hys', pixels_per_cell=hog_dim)
        #endif
        hog_features.append(hogi[1]) #grab the array from hog() output
    #endfor
    #downsample feature sets
    tmpInIm = paded_im_seg.copy()
    tmpInBool = paded_bool_seg.copy()
    tmpInHog = hog_features.copy()
    dsIm_segs, dsBool_segs = downSampleStd(tmpInIm,tmpInBool,train)
    dsHog_segs, _ = downSampleStd(tmpInHog,[],False)
    return im_list, bool_list, f, dsIm_segs, dsBool_segs, dsHog_segs

def get_hogs(hog_features):
    hog = []
    for i,val in enumerate(hog_features):
        hog.append(val[0])
    'end for'
    return hog

def create_data(X,y,imNum,train=True):
    y_train = []
    X_in = []
    for i in range(0,len(X)):
        try:
            X_in.append(X[i].ravel())
        except AttributeError:
            X_in = get_hogs(X)
    X_train = np.vstack(X_in)
    if train:
        for i in range(0,len(y)):
            tmpi = y[i] > 0
            tmp = (True in tmpi)
            y_train.append(tmp)
        y_train = np.vstack(y_train).astype(int)
        imArr = np.tile(imNum,y_train.shape)
        y_train = np.hstack((y_train,imArr))
        return [X_train, y_train]
    
    return X_train

def gen_mask(image):
    mask = image > 0
    return np.ma.masked_where(~mask, mask)

def overlay_predictions(image,boolim,preds,y_test,ind_test,f,train=True,**kwargs):
    """
    Parameters
    ----------
    image : np.array(float64)
        image being anlazyed
    boolim : np.array(bool)
        label data that was used to train algorithm
    preds : np.
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    ind_test : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    nH= image.shape[0]
    nW= image.shape[1]
    pred_im = np.zeros((nH,nW)).astype(np.float32)
    # true_im = np.zeros((nH,nW)).astype(np.float32)
    plt.figure("Overlayed Predictions for Test Domain",figsize = (nH/100,nW/100))  
    plt.imshow(image, **kwargs)
    legend_ele = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,label = "label: (actual,predict)"),
                  Patch(facecolor = "red",label = "segmented"),
                  Patch(facecolor = "orange",label = "training data")]
    # plt.set_size_inches(nH/100,nW/100)
    for ind in range(0,len(ind_test)):
        i = ind_test[ind]
        y1 = f[i][0].start
        y2 = f[i][0].stop
        x1 = f[i][1].start
        x2 = f[i][1].stop
        pred_im[y1:y2,x1:x2] = np.ones((y2-y1,x2-x1))
        s = "({0},{1})".format(y_test[ind],preds[ind])
        plt.text(x1, y1-5, s, fontsize = 10, bbox=dict(fill=False, edgecolor='none', linewidth=2))
    plt.legend(handles = legend_ele, loc = 'lower right')

    
    plt.imshow(gen_mask(pred_im), alpha=0.3, cmap=ListedColormap(['red']))
    plt.imshow(gen_mask(boolim), alpha=0.5, cmap=ListedColormap(['orange']))
    plt.savefig(os.path.join(save_bin,'overlayed_predictions.tif'),dpi=200,bbox_inches='tight')
    return 0
#enddef

def overlayValidate(image,predictions,domains,**kwargs):
    """
    Parameters
    ----------
    image : np.array(float64)
        DESCRIPTION.
    predictions : np.array(float64)
        image being anlazyed
    domains : np.array(bool)
        domains on which image was segmented
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    nH= image.shape[0]
    nW= image.shape[1]
    pred_im = np.zeros((nH,nW)).astype(np.float32)
    # true_im = np.zeros((nH,nW)).astype(np.float32)
    plt.figure("Overlayed Predictions for Test Domain",figsize = (nH/100,nW/100))  
    plt.imshow(image, **kwargs)
    legend_ele = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,label = "label: (actual,predict)"),
                  Patch(facecolor = "red",label = "segmented"),
                  Patch(facecolor = "orange",label = "training data")]
    # plt.set_size_inches(nH/100,nW/100)
    for i in range(0,len(domains)):
        y1 = domains[i][0].start
        y2 = domains[i][0].stop
        x1 = domains[i][1].start
        x2 = domains[i][1].stop
        pred_im[y1:y2,x1:x2] = np.ones((y2-y1,x2-x1))
        s = "({0},{1})".format('?',predictions[i])
        plt.text(x1, y1-5, s, fontsize = 10, bbox=dict(fill=False, edgecolor='none', linewidth=2))
    plt.legend(handles = legend_ele, loc = 'lower right')
    plt.imshow(gen_mask(pred_im), alpha=0.3, cmap=ListedColormap(['red']))
    plt.savefig(os.path.join(save_bin,'overlayed_predictions.tif'),dpi=200,bbox_inches='tight')
    return 0
#enddef

def write_auc(fpr,tpr):
    with open(os.path.join(dirname,'save-bin\\svm_auc_roc.csv'),'w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=' ',
                                quotechar='|',quoting=csv.QUOTE_MINIMAL)
        for i in range(len(fpr)):
            spamwriter.writerow([fpr[i],tpr[i]])
    return 0
    
def read_auc():
    fpr = []
    tpr = []
    with open(os.path.join(dirname,'save-bin\\svm_auc_roc.csv'),'r',newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=' ',
                                quotechar='|')
        for row in spamreader:
            fpr.append(float(row[0]))
            tpr.append(float(row[1]))
        
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,roc_auc

def padPreProcessed(X):
    """
    im_list
    train : boolean
        DESCRIPTION : if train == True set bool_list = np.array()
    f
    fill_val = 0 : TYPE, integer or function (e.g. np.nan)
        DESCRIPTION. 
    """
    lenX = []
    padedX = []
    for im in X:
        for seg in im:
            lenX.append(len(seg))
        'end for'
    'end for'
    uVals = np.unique(lenX)
    uMax = np.max(uVals)
    for i in range(len(X)):
        for j in range(len(X[i])):
            nPad = uMax-len(X[i][j])
            padedX.append(np.append(X[i][j],np.zeros(nPad)))
        'end for'
    'end for'
    return padedX
'end def'

import random

def random_ind(a,b,N):
    ints = []      
    for i in range(0,N):
      ints.append(random.randint(0,64))
    return ints

### Testing ###

if __name__ == '__main__':

    ### PARAMS ###
    channel = 2
    ff_width = 121
    wiener_size = (5,5)
    med_size = 5
    ###
    
    dirname = os.path.dirname(__file__)
    foldername = os.path.join(dirname,"images-5HT")
    dirn = dirname
    foldern = os.path.join(dirn,foldername)
    im_dir = DataManager.DataMang(foldern)
    im_list = [3,4,5,6,9,10,11,12,13,14,16,17,26,35] #[i for i in range(0,im_dir.dir_len)]

    #define variables for loops
    hog_features = [np.array([])]
    im_segs = [np.array([])]
    bool_segs = [np.array([])]
    domains = [np.array([])]
    paded_im_seg = [np.array([])]
    X = []
    y = []
    print('Starting PreProcessing Pipeline...')

    ## change what channel is being imported on the main image.

    for gen in im_dir.open_dir(im_list,'test'):
        #load image and its information
        t_start = time.time()
        image,nW,nH,chan,name = gen
        print('   '+'Procesing Image : {}'.format(name))
        #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
        #the notation RGB (0,1,2 respectively))=
        image = image[:,:,channel]
        #Import train data (if training your model)
        train_bool = LabelMaker.import_train_data(name,(nH,nW),'archive-image-bin\\trained-bin-EL-11122021\\')
        plt.figure('image')
        plt.imshow(image)
        #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
        #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
        im_segs, bool_segs, domains, paded_im_seg, paded_bool_seg, hog_features = feature_extract(image, ff_width, wiener_size, med_size,True,train_bool)

        #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
        tmp_X,tmp_y = create_data(hog_features,True,bool_segs)
        X.append(tmp_X)
        y.append(tmp_y)
        t_end = time.time()
        print('     '+'Number of Segments : %i'%(len(im_segs)))
        print('     '+"Processing Time for %s : %0.2f"%(name,(t_end-t_start)))
        break
    print('done')

    plt.figure('Image');plt.imshow(image)
    plt.figure('Bool Image');plt.imshow(train_bool)
    # plt.figure('normalized image');plt.imshow(normalized)
    # plt.figure('thresheld image');plt.imshow(threshed)
    plt.show()

