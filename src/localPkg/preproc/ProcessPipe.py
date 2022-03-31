# -*- coding: utf-8 -*-
"""
Created on Tues Jan 25 19:06:00 2022

@author: Jacob Salminen
@version: 1.0
"""
print(__doc__)
from itertools import count
from socket import IP_MULTICAST_LOOP
import time
import os

from multiprocessing.sharedctypes import Value
from xml.dom.expatbuilder import TEXT_NODE
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import convolve,distance_transform_edt,label, find_objects
from sklearn.metrics import auc
from skimage.feature import hog
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from cv2 import rotate, ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE

import csv

from ..datmgmt import DataManager
from ..disp import LabelMaker
from ..preproc import Filters


callNum = 0
def dispTS(startFunc = True, dispTitle = ""):
    """
    Summary : displays time stamp for when the function is first called to when its next called.
    Parameters
    ----------
    startFunc : TYPE
        DESCRIPTION
    dispTitle : TYPE
        DESCRIPTION

    Returns
    -------
    print(fString)

    """
    global tStart, tEnd, callNum
    if startFunc:
        tStart = time.time()
        outStr = f"{dispTitle}"
    else:
        tEnd = time.time()
        if dispTitle != "":
            dispTitle = f"{dispTitle}"
        #endif
        tDif = tEnd - tStart
        callNum += 1
        outStr = f"{callNum}) {dispTitle}: runtime {tDif}"
    #endif
    return print(outStr)
#enddef

def gen_point_vector(imageIn):
    """
    Parameters
    ----------
    image : TYPE, np.array(dtype=float64)
        DESCRIPTION. image 
    Returns
    -------
    point_data : TYPE, 
        DESCRIPTION
    """
    dispTS()
    pointData = np.zeros((imageIn.shape[0]*imageIn.shape[1],2))
    count = 0
    for i in range(0,imageIn.shape[0]):
        for j in range(0,imageIn.shape[1]):
            pointData[count,:] = [i,j]
            count += 1
        'end for'
    'end for'
    dispTS(False,"gen_point_vector")
    return pointData
'end def'



def img_to_data(imageIn, maskIn, keepAll = True, *kwargs):
    """
    Summary : creates an array of data of shape [image.shape[0]*image.shape[1],number_of_parameters + image_data] represents
    all the parameters to be enetered into SVM image analysis

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    mask : TYPE
        DESCRIPTION
    keepAll : TYPE
        DESCRIPTION
    *kwargs: image data type float32[:,:]
        DESCRIPTION.

    Returns
    ------
    datOut, TYPE
    boolSet, TYPE
    pointData, TYPE

    """
    dispTS()
    #initialize with original image data
    imgD = imageIn.ravel()
    imgD = imgD.reshape(imgD.shape[0],1)
    newD = np.array([])
    count = 0

    for data in kwargs:
        newD = data.ravel()
        newD = newD.reshape(newD.shape[0],1)
        imgD = np.concatenate((imgD,newD),axis = 1)
        count += 1
    #endfor
    nonzero = np.sum(maskIn)
    maskR = maskIn.ravel()
    maskR = maskR.reshape(maskR.shape[0],1)
    pointData = gen_point_vector(image)    
    if keepAll:
        datOut = imgD
        boolSet = maskR.astype(int)
    #endif    
    else:
        masked = np.multiply(imgD,maskR)
        maskNew = np.zeros((nonzero,imgD.shape[1]))
        pointNew = np.zeros((nonzero,2))
        boolSet = np.zeros((nonzero,imgD.shape[1]))
        count = 0
        for i,x in enumerate(masked):
            if x.any() != 0:
                maskNew[count,:] = x
                boolSet[count,:] = maskR[i,:]
                pointNew[count,:] = pointData[i,:]
                count += 1
            #endif
        #endfor
        datOut = maskNew
        boolSet = boolSet.astype(int)
        pointData = pointNew
    #endif
    dispTS(False,"img_to_data")
    return datOut, boolSet, pointData
#enddef

def data_to_img(mask,predictions):
    """
    Parameters
    ----------
    mask : TYPE
        DESCRIPTION
    predictions : TYPE
        DESCRIPTION

    Returns
    ------
    newIm : TYPE

    """
    dispTS()
    newIm = np.zeros((mask.shape[0],mask.shape[1]))
    count = 0
    for i,row in enumerate(mask):
        for j,col in enumerate(row):
            if col == True:
                newIm[i,j] = predictions[count]
                count += 1
            'end if'
        'end for'
    'end for'
    dispTS(False,"data_to_img")
    return newIm

def get_nonzeros(image,valVector,mask,trueType = True):
    """
    Parameters
    ----------
    image : TYPE
        DESCRIPTION
    valVector : TYPE
        DESCRIPTION
    mask : TYPE
        DESCRIPTION
    trueType : TYPE
        DESCRIPTION

    Returns
    ------
    valsNew : TYPE
        DESCRIPTION
    maskNew : TYPE
        DESCRIPTION
    pointsNew : TYPE
        DESCRIPTION
    """
    dispTS()
    mask = mask.ravel()
    mask = mask.reshape(mask.shape[0],1)
    
    masklen = np.sum(mask.astype(int))
    
    maskNew = np.zeros((masklen,mask.shape[1]))
    pointsNew = np.zeros((masklen,2))
    
    points = gen_point_vector(image)
    valsNew = np.zeros((masklen,valVector.shape[1]))
    
    count = 0        
    for i,x in enumerate(mask.astype(int)):
        if x != 0:
            valsNew[count,:] = valVector[i,:]
            pointsNew[count,:] = points[i,:]
            if trueType:
                # vals_new[count,-1] = 1
                maskNew[count,0] = 1
            else:
                # vals_new[count,-1] = 0
                maskNew[count,0] = 0
            count += 1
            #endif
        #endif
    #endfor
    dispTS(False,"get_nonzeros")
    return valsNew,maskNew.astype(int),pointsNew

def filter_pipeline(image,fftWidth,wienerWindowSize,medianWindowSize,multiA=1,multiD=1):
    """
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    fftWidth : TYPE
        DESCRIPTION.
    wienerWindowSize : TYPE
        DESCRIPTION.
    medianWindowSize : TYPE
        DESCRIPTION.
    

    Returns
    -------
    ffimhi_new : TYPE
        DESCRIPTION.
    direction_features : TYPE
        DESCRIPTION.

    """
    dispTS()
    directionFeats = np.array([])

    #Normalize image
    normIm = Filters.normalize_img(image)
    
    #Fourier Filter for removing low frequency components
    ffimhiNew = Filters.Hi_pass_filter(normIm,fftWidth)
    
    #denoising
    denoisedIm = Filters.wiener(ffimhiNew,wienerWindowSize,None)
    
    #Median Filter
    #Add running window median filter (10x10 to 30x30) and rectify the signal using max(0,val)
    medianIm = Filters.median_filt(denoisedIm,medianWindowSize)
    # median_im = ((denoised_im-median_im)>0)*median_im
    
    #adaptive thresholding
    #lowering the 3rd variable "multiplier_d" tightens border of predictions.
    threshed = convolve(Filters.adaptive_threshold(denoisedIm,200,256,False),Filters._d3gaussian(5,multiA,multiD))
    
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
    diagIm = convolve(denoisedIm,diagnols)
    ttIm = convolve(denoisedIm,t_cross)
    #Gaussian Image
    gausIm = convolve(denoisedIm,Filters._d3gaussian(5,1,1))

    #Differential image
    directionFeats = (Filters.diffmat(denoisedIm,np.arange(0,2*np.pi,2*np.pi/8),dim=(5,2)))
    dispTS(False,"filter_pipeline")
    return medianIm, [threshed,diagIm,ttIm,gausIm,directionFeats]

def im_watershed(imageIn,train = True, boolim = np.array([]),multiA=3,multiD=2):
    """
    SUMMARY : segments image using a watersheding method with distance_transform_edt as the 
    descriminator. Returns list of segments

    Parameters
    ----------
    imageIn : TYPE, np.array(float32)
        DESCRIPTION : 
    train : TYPE, boolean (default = True)
        DESCRIPTION : if train == True set boolim = np.array()
    boolim : TYPE, np.array(dtype = int)
        DESCRIPTION :
    multiA : TYPE,
        DESCRIPTION
    multiD : TYPE,
        DESCRIPTION

    Returns
    -------
    imList : TYPE, list
        DESCRIPTION
    boolList : TYPE, list
        DESCRIPTION

    """
    dispTS("")
    MIN_DISTANCE = 20
    imList = []
    boolList = []

    # normalize image than convolve a gaussian kernel to expand feature discovery.
    gausIm = convolve(imageIn,Filters._d3gaussian(16,multiA,multiD))
    # perform a rough thresholding
    segments = gausIm > np.median(gausIm) #(np.mean(gausIm)-np.std(gausIm)/(np.pi)) #adding np.std(gausIm)/2 gives some more specificity. 
    # perform distance calculations to determine segment proximities.
    D = distance_transform_edt(segments)
    # get local maxima
    tmplocalMax = peak_local_max(D, min_distance=MIN_DISTANCE,
                              labels=segments)
    localMax = np.zeros_like(imageIn, dtype=bool)
    localMax[tuple(tmplocalMax.T)] = True
    # label potential segments using distance measures and local maxima
    markers = label(localMax,structure=np.ones((3,3)))[0]
    # watershed segmentation using previous information
    water_im = watershed(-D,markers,mask=segments)
    f = find_objects(water_im)
    # break up segments into rectangles 
    for seg in f:
        imList.append(imageIn[seg])
        if train:
            boolList.append(boolim[seg])
        #endif
    #endfor
    dispTS(False,"im_watershed")
    return imList,boolList,f
#enddef

def _getSmallSquares(seg, nSet):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION
    nSet : TYPE
        DESCRIPTION

    Returns
    -------
    subSegs : TYPE
        DESCRIPTION

    """
    # dispTS()
    # pady = 0 
    # padx = 0
    leftOver = []
    subSegs = []
    orig = []
    yval = abs(seg[0].stop-seg[0].start)
    xval = abs(seg[1].stop-seg[1].start)
    tmpy = [slice(0,yval,None)]
    tmpx = [slice(0,xval,None)]
    tmpyy = slice(0,yval,None)
    tmpxx = slice(0,xval,None)
    dify = yval - nSet
    difx = xval - nSet
    # test if the segment needs to be cut or just taken as is
    if dify < 0 or difx < 0:
        # pady  = abs(dify)
        # padx  = abs(difx)
        leftOver = [tmpyy,tmpxx]
    #endif
    # overwrite axis that is longer than NSET
    if dify > 0 or difx > 0:
        remy  = yval//nSet
        remx  = xval//nSet
        difyy = yval - remy*nSet
        # pady  = (nSet - difyy)*(difyy!=0)
        difxx = xval - remx*nSet
        # padx  = (nSet - difxx)*(difxx!=0)
        if remy > 0:
            cuts = np.arange(0,yval-difyy+1,nSet)
            for i in range(0,remy):
                if i == 0:
                    tmpy[i] = slice(cuts[i],cuts[i+1],None)
                else:
                    tmpy.append(slice(cuts[i],cuts[i+1],None))
            #endif
            if difyy > 0:
                tmpy.append(slice(cuts[i+1],yval,None))
            #endif
        #endif
        if remx > 0:
            cuts = np.arange(0,xval-difxx+1,nSet)
            for i in range(0,remx):
                if i == 0:
                    tmpx[i] = slice(cuts[i],cuts[i+1],None)
                else:
                    tmpx.append(slice(cuts[i],cuts[i+1],None))
            #endif
            if difxx > 0:
                tmpx.append(slice(cuts[i+1],xval,None))
            #endif
        #endif
        for ydim in tmpy:
            for xdim in tmpx:
                tmpO1 = slice(ydim.start + seg[0].start,ydim.stop + seg[0].start,None)
                tmpO2 = slice(xdim.start + seg[1].start,xdim.stop + seg[1].start,None)
                tmpOrig = (tmpO1,tmpO2)
                orig.append(tmpOrig)
                tmp = (ydim,xdim)
                subSegs.append(tmp)
            #endfor
        #endfor
    #endif
    # dispTS(False,"_getSmallSquares")
    return subSegs, orig

def pad_segs(imList,boolList,f,train = True,fill_val = 0):
    """
    Parameters
    ----------
    imList : TYPE
        DESCRIPTION
    boolList : TYPE
        DESCRIPTION
    f : TYPE
        DESCRIPTION
    reducueFator : TYPE, int (default = 2)
        DESCRIPTION
    train : TYPE, boolean (default = True)
        DESCRIPTION : if train == True set bool_list = np.array()
    fill_val = 0 : TYPE, integer or function (e.g. np.nan)
        DESCRIPTION. 

    Results
    -------
    newImList
    newBoolList
    """
    dispTS()
    # Cuts each segment from watershedding process into smaller squares for more efficient processing.
    NSET = 70 # NSETxNSET cuts for each segment
    count = 0
    newImList = []
    newBoolList = []
    newDoms = []

    for seg in f:
        subSegs, origSegs = _getSmallSquares(seg,NSET)
        if len(subSegs) > 0:
            i = 0
            for ss in subSegs:
                yval = abs(ss[0].stop-ss[0].start)
                xval = abs(ss[1].stop-ss[1].start)
                if yval < NSET or xval < NSET:
                    newImList.append(np.pad(imList[count][ss],((0,NSET-yval),(0,NSET-xval)),'constant',constant_values=fill_val))
                    if newImList[-1].shape != (NSET,NSET):
                        ValueError('padded image shape is not square!')
                    #endif  
                    if train:
                        newBoolList.append(np.pad(boolList[count][ss],((0,NSET-yval),(0,NSET-xval)),'constant',constant_values=fill_val))
                    #endif
                    newDoms.append(origSegs[i])
                else:
                    newImList.append(imList[count][ss])
                    newBoolList.append(boolList[count][ss])
                    newDoms.append(origSegs[i])
                #endif
                i += 1
            #endfor
        #endif
        count += 1
    #endfor
    dispTS(False,"pad_segs")
    return newImList, newBoolList, newDoms

def downSampleStd(imList, boolList, train=True):
    """
    Parameters
    ----------
    imList : TYPE
        DESCRIPTION
    boolList : TYPE
        DESCRIPTION
    train : TYPE, boolean (default = True)
        DESCRIPTION

    Returns
    -------
    imList : TYPE
        DESCRIPTION
    boolList : TYPE
        DESCRIPTION

    """
    dispTS()
    for i in range(0,len(imList)):
        # reduce image segment using downsampling: figure out how to reduce the image to a specific resolution
        # get shape of image
        # nH,nW = im_list[i].shape
        # determine factor for countless reduction
        redH = 2 #nH/reduceFactor
        redW = 2 #nW/reduceFactor
        # countless algorithm
        outIm = Filters.countless(imList[i],redH,redW)
        # reassign
        imList[i] = outIm
        if train:
            # reduce image segment using downsampling: figure out how to reduce the image to a specific resolution
            # get shape of image
            # nH,nW = bool_list[i].shape
            # determine factor for countless reduction
            # countless algorithm
            outIm = Filters.countless(boolList[i],redH,redW)
            # reassign
            boolList[i] = outIm
        #endif
    #endfor
    dispTS(False,"downSampleStd")
    return imList, boolList

def rotateNappend(imList, boolList, domains, train = True):
    """
    Summary :

    Parameters
    ----------
    
    Results
    -------

    """
    dispTS()
    imsOut = []
    boolOut = []
    domsOut = []
    for i in range(len(imList)):
        r1 = rotate(imList[i], ROTATE_180)
        r2 = rotate(imList[i], ROTATE_90_CLOCKWISE)
        r3 = rotate(imList[i], ROTATE_90_COUNTERCLOCKWISE)
        r4 = imList[i]
        allR = [r1,r2,r3,r4]
        if train:
            tmpB = boolList[i]
            for j in range(0,len(allR)):
                boolOut.append(tmpB)
            #endfor
        #endif
        for j in range(0,len(allR)):
            imsOut.append(allR[j])
            domsOut.append(domains[i])
        #endfor
    #endfor
    dispTS(False,"rotateNappend")
    return imsOut, boolOut, domsOut
#enddef

def feature_extract(imageIn, fftWidth, wieneerWindowSize, medWindowSize, **kwargs):
    """
    SUMMARY : 

    Parameters
    ----------
    image : TYPE
        DESCRIPTION. 
    fftWidth : TYPE
        DESCRIPTION. 
    wieneerWindowSize : TYPE
        DESCRIPTION. 
    medWindowSize : TYPE
        DESCRIPTION.
    train : TYPE, boolean (default = True)
        DESCRIPTION. 
    boolim : TYPE, np.array([]) (default = np.array([]))
        DESCRIPTION. 

    Returns
    -------

    """
    train = True
    boolIm = np.array([])
    for key, value in kwargs.items():
        if key == "train":
            train = value
        #endif
        if key == "boolIm":
            boolIm = value
        #endif
    #endfor

    dispTS()
    hogFeats = []
    dsFeatSets = []
    HOGDIM = (4,4)

    # Normalize image, Hi-pass filter image (using dfft), wiener filter, and median filter (in that order).
    medIm, _ = filter_pipeline(imageIn,fftWidth,wieneerWindowSize,medWindowSize)

    #segment image using watershed and pad images for resizing
    imList, boolList, origDoms = im_watershed(medIm,train,boolIm)

    #pad segments
    padedImSeg, padedBoolSeg, newDoms = pad_segs(imList,boolList,origDoms,train,0)

    #roate segments and append
    rotatedIms, rotatedBools, rotateDoms = rotateNappend(padedImSeg, padedBoolSeg, newDoms)

    #generate hog features
    dispTS('appending hogs...')
    for seg in rotatedIms:
        # hogIn = Filters.normalize_img(seg)
        hogIn = seg
        # print(seg.shape) #debug
        testDim = tuple([2*x for x in HOGDIM])
        if seg.shape < testDim:
            hogi = hog(hogIn, visualize = True, block_norm='L2-Hys', pixels_per_cell=(2,2))
        else:
            hogi = hog(hogIn, visualize = True, block_norm='L2-Hys', pixels_per_cell=HOGDIM)
        #endif
        hogFeats.append(hogi[1]) #grab the array from hog() output
    #endfor
    dispTS(False, 'hogs appended.')

    #downsample feature sets (optional)
    # tmpInIm = padedImSeg.copy()
    # tmpInBool = padedBoolSeg.copy()
    # tmpInHog = hogFeats.copy()
    # dsIm_segs, dsBool_segs = downSampleStd(tmpInIm,tmpInBool,train)
    # dsHog_segs, _ = downSampleStd(tmpInHog,[],False)  
    # dsFeatSets.append(dsImSegs, dsBoolSegs)  
    
    dispTS(False, "feature_extract")
    return rotatedIms, rotatedBools, hogFeats, newDoms, dsFeatSets

def get_hogs(hogFeats):
    """
    ANTIQUATED
    Summary : Unpacks the hog feature set generated during feature_extract()

    Parameters
    ----------
    hogFeats : TYPE, list

    Returns
    -------
    hogi : TYPE, list

    """
    dispTS()
    hogi = []
    for _,val in enumerate(hogFeats):
        hogi.append(val[0])
    #endfor
    dispTS(False,"get_hogs")
    return hogi

def create_data(datX,imNum,**kwargs):
    """
    Summary :

    Parameters
    ----------
    datX : TYPE, list of np.array()
        DESCRIPTION
    datY : TYPE, list of np.array()
        DESCRIPTION
    imNum : TYPE, list of int
        DESCRIPTION
    train : TYPE, boolean (default = True)
        DESCRIPTION

    Returns
    -------
    outX : TYPE, list of np.array()
        DESCRIPTION
    yTrain (IF 'train' == True) : TYPE, list of int
        DESCRIPTION
    """
    dispTS()
    train = True
    datY = []
    domains = []
    for key, value in kwargs.items():
        if key == "train":
            train = value
        #endif
        if key == "datY":
            datY = value
        #endif
        if key == "domains":
            domains = value
        #endif
    #endfor
        
    
    yTrain = []
    tmpX = []
    for i in range(0,len(datX)):
        try:
            tmpX.append(datX[i].ravel())
        except AttributeError:
            tmpX = get_hogs(datX)
        #endtry
    #endfor

    outX = np.vstack(tmpX)
    if train:
        for i in range(0,len(datY)):
            tmpi = datY[i] > 0
            tmp = (True in tmpi)
            yTrain.append(tmp)
        #endfor
        yTrain = np.vstack(yTrain).astype(int)
        imArr = np.tile(imNum,yTrain.shape)
        yTrain = np.hstack((yTrain,imArr))
        if domains:
            return [outX, yTrain, domains]
        #endif
        return [outX, yTrain]
    #endif
    dispTS(False,"create_data")
    return outX

def gen_mask(imageIn):
    """
    Summary
    Parameters
    ----------
    imageIn : TYPE, np.array()
        DESCRIPTION

    Returns
    -------
    np.ma.masked_where(~mask, mask), where mask = image > 0 .
    """
    mask = imageIn > 0
    return np.ma.masked_where(~mask, mask)

def overlay_predictions(imageIn, boolIm, predIn, yTest, idxTest, f, saveBin, train=True,**kwargs):
    """
    Parameters
    ----------
    imageIn : np.array(float64)
        image being anlazyed
    boolIm : np.array(bool)
        label data that was used to train algorithm
    predIn : np.
        DESCRIPTION.
    yTest : TYPE
        DESCRIPTION.
    idxTest : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    saveBin : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    dispTS()
    nH = imageIn.shape[0]
    nW = imageIn.shape[1]
    predIm = np.zeros((nH,nW)).astype(np.float32)
    # true_im = np.zeros((nH,nW)).astype(np.float32)
    plt.figure("Overlayed Predictions for Test Domain",figsize = (nH/100,nW/100))  
    plt.imshow(imageIn, **kwargs)
    legend_ele = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,label = "label: (actual,predict)"),
                  Patch(facecolor = "red",label = "segmented"),
                  Patch(facecolor = "orange",label = "training data")]
    # plt.set_size_inches(nH/100,nW/100)
    for ind in range(0,len(idxTest)):
        i = idxTest[ind]
        y1 = f[i][0].start
        y2 = f[i][0].stop
        x1 = f[i][1].start
        x2 = f[i][1].stop
        predIm[y1:y2,x1:x2] = np.ones((y2-y1,x2-x1))
        s = "({0},{1})".format(yTest[ind],predIn[ind])
        plt.text(x1, y1-5, s, fontsize = 10, bbox=dict(fill=False, edgecolor='none', linewidth=2))
    #endfor
    plt.legend(handles = legend_ele, loc = 'lower right')    
    plt.imshow(gen_mask(predIm), alpha=0.3, cmap=ListedColormap(['red']))
    plt.imshow(gen_mask(boolIm), alpha=0.5, cmap=ListedColormap(['orange']))
    plt.savefig(os.path.join(saveBin,'overlayed_predictions.tif'),dpi=200,bbox_inches='tight')
    dispTS(False, "overlay_predictions")
    return 0
#enddef

cnt = 0
def overlayValidate(imageIn,predictions,domains,saveDir,**kwargs):
    """
    Parameters
    ----------
    imageIn : np.array(float64)
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
    for key,val in kwargs.items():
        if key == 'boolIm':
            boolIm = val
        #endif
    #endfor
    global cnt
    nH= imageIn.shape[0]
    nW= imageIn.shape[1]
    predIm = np.zeros((nH,nW)).astype(np.float32)
    falseIm = np.zeros((nH,nW)).astype(np.float32)
    # true_im = np.zeros((nH,nW)).astype(np.float32)
    plt.figure("Overlayed Predictions for Test Domain",figsize = (nH/100,nW/100))  
    plt.imshow(imageIn[:,:,2]*5)
    # plt.imshow(imageIn[:,:,1]v, alpha = 0.5)
    legend_ele = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,label = "label: (actual,predict)"),
                  Patch(facecolor = "red",label = "True"),
                  Patch(facecolor = "orange",label = "False")]
    # plt.set_size_inches(nH/100,nW/100)
    for i in range(0,len(domains)):
        y1 = domains[i][0].start
        y2 = domains[i][0].stop
        x1 = domains[i][1].start
        x2 = domains[i][1].stop
        if predictions[i]:
            predIm[y1:y2,x1:x2] = np.ones((y2-y1,x2-x1))
        else:
            falseIm[y1:y2,x1:x2] = np.ones((y2-y1,x2-x1))
        #endif
        # s = "({0})".format(predictions[i])
        # plt.text(x1, y1-5, s, fontsize = 10, bbox=dict(fill=False, edgecolor='none', linewidth=2))
    #endfor
    plt.legend(handles = legend_ele, loc = 'lower right')
    plt.imshow(gen_mask(boolIm), alpha=1, cmap=ListedColormap(['yellow']))
    plt.imshow(gen_mask(predIm), alpha=0.3, cmap=ListedColormap(['red']))
    plt.imshow(gen_mask(falseIm), alpha=0.4, cmap=ListedColormap(['orange']))
    # plt.show()
    plt.savefig(os.path.join(saveDir,f'overlayed_predictions_{cnt}.tif'),dpi=300,bbox_inches='tight')
    cnt += 1
    return 0
#enddef

def write_auc(fpR,tpR,saveDir):
    """
    Summary :

    Parameters
    ----------
    fpR : TYPE, 
        DESCRIPTION
    tpR : TYPE, 
        DESCRIPTION
    saveDir : TYPE,
        DESCRIPTION

    Returns
    -------
    0

    """
    with open(saveDir,'w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=' ',
                                quotechar='|',quoting=csv.QUOTE_MINIMAL)
        for i in range(len(fpR)):
            spamwriter.writerow([fpR[i],tpR[i]])
        #endfor
    #endwith
    return 0
#enddef
    
def read_auc(saveDir):
    """
    Summary :

    Parameters
    ----------
    saveDir : TYPE,
        DESCRIPTION

    Returns
    -------
    fpR : TYPE,
        DESCRIPTION
    tpR : TYPE, 
        DESCRIPTION
    rocAuc : TYPE,
        DESCRIPTION
    
    """
    fpR = []
    tpR = []
    with open(saveDir,'r',newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=' ',
                                quotechar='|')
        for row in spamreader:
            fpR.append(float(row[0]))
            tpR.append(float(row[1]))
        
    fpR = np.array(fpR)
    tpR = np.array(tpR)
    rocAuc = auc(fpR, tpR)
    return fpR,tpR,rocAuc

def padPreProcessed(xIn):
    """
    Summary : 
    Parameters
    ----------
    xIn : TYPE, 
        DESCRIPTION
    
    Returns
    -------
    padedX : TYPE,
        DESCRIPTION
    
    """
    lenX = []
    padedX = []
    for subIm in xIn:
        for subSeg in subIm:
            lenX.append(len(subSeg))
        #endfor
    #endfor
    uVals = np.unique(lenX)
    uMax = np.max(uVals)
    for i in range(len(xIn)):
        for j in range(len(xIn[i])):
            nPad = uMax-len(xIn[i][j])
            padedX.append(np.append(xIn[i][j],np.zeros(nPad)))
        #endfor
    #endfor
    return padedX
#enddef

import random

def random_ind(N, begVal = 0,endVal = 64):
    """
    Summary :
    Parameters
    ----------
    N : TYPE,
        DESCRIPTION
    begVal : TYPE,
        DESCRIPTION
    endVal : TYPE,
        DESCRIPTION

    Returns
    -------
    rndInts : TYPE,
        DESCRIPTION

    """
    rndInts = []      
    for i in range(0,N):
      rndInts.append(random.randint(begVal,endVal))
    return rndInts

def mainLoop(fileNum,rawDatDir,trainDatDir,savePath):
    from datetime import date
    from os.path import join, exists
    #%% Globals
    global dTime
    dTime = date.today().strftime('%d%m%Y')

    try: os.mkdir(savePath)
    except FileExistsError:
        print(f'Path Exists: {savePath}')
    #endtry
    #endif

    #%% Initialize Image Parsing/Pre-Processing 
    #load image folder for training data
    imDir = DataManager.DataMang(rawDatDir)

    #%% PARAMS
    channel = 2
    fftWidth = 121
    wienerWindowSize = (5,5)
    medWindowSize = 10
    # seedN = 42

    #%% MAIN PROCESS
    dispTS()
    #opend filfe
    imageOut,nW,nH,_,imName,imNum = imDir.openFileI(fileNum,'train')
    #load image and its information
    print('   '+'{}.) Processing Image : {}'.format(imNum,imName))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))=
    imageIn = imageOut[:,:,channel]
    #Import train data (if training your model)
    trainBool = LabelMaker.import_train_data(imName,(nH,nW),trainDatDir)
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    _ , padedBoolSeg, hogFeats, doms, _ = feature_extract(imageIn, fftWidth, wienerWindowSize, medWindowSize, train = True, boolIm = trainBool)
    chosenFeats = hogFeats
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    result = create_data(chosenFeats,fileNum,datY = padedBoolSeg,Train = True,domains=doms)
    
    #%% WRAP-UP MAIN
    dispTS(False)
    print('     '+'Number of Segments : %i'%(len(chosenFeats)))
    tmpSaveDir = join(savePath, (f'trained_data_{dTime}_{fileNum}.pkl'))
    DataManager.save_obj(tmpSaveDir,result)
    return result
    #endfor
#enddef

def mainLoopTest(fileNum,aDatGenDir,bDatAggDir,rawDatDir,trainDatDir,aggDatDir,savePath):
    from datetime import date
    from os.path import exists
    #%% Globals
    global dTime
    dTime = date.today().strftime('%d%m%Y')
    xOut = []

    if not exists(savePath):
        print(f'{savePath} does not exist.\n Creating new folder...')
        os.mkdir(savePath)
    #endif

    #%% Initialize Image Parsing/Pre-Processing 
    #load image folder for training data
    imDir = DataManager.DataMang(rawDatDir)

    #%% PARAMS
    channel = 2
    fftWidth = 121
    wienerWindowSize = (5,5)
    medWindowSize = 10

    #%% Main Process
    dispTS()
    #opend file
    imageOut, _, _, _, imName, imNum = imDir.openFileI(fileNum,'test')
    #load image and its information
    print('   '+'{}.) Procesing Image : {}'.format(imNum,imName))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))
    imageIn = imageOut[:,:,channel]
    #extract features from image using method(ProcessPipe.feature_extract) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    _, padedBoolSeg, hogFeats, doms, _ = feature_extract(imageIn, fftWidth, wienerWindowSize, medWindowSize,False)
    chosenFeats = hogFeats
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    tmpX = create_data(chosenFeats,fileNum,train = False, domains = doms)
    xOut.append(tmpX)

    dispTS(False, "mainLoop")
    print('     '+'Number of Segments : %i'%(len(chosenFeats)))
    #stack X
    xOut = np.vstack(xOut)
    #endfor
    return xOut, imageIn, doms
#enddef

### Testing ###
if __name__ == '__main__':
    import multiprocessing as mp
    from os.path import join, dirname, abspath
    from datetime import date

    # change the 'start' in PARAMS to choose which file you want to start with.
    imList = [3,4,5] #[3,4,5,6,10,12,13,14,21,26,27,28,29,35] #[i for i in range(start,im_dir.dir_len)]

    from joblib import Parallel, delayed
    threadN = mp.cpu_count()
    results = Parallel(n_jobs=threadN)(delayed(mainLoop)(i) for i in imList) # only one that works? (03/04/2022)
#endif

