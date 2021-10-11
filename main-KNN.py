# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:19:38 2021

@author: jsalm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""

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
from sklearn.neighbors import KNeighborsClassifier

import SVM
import Filters
import ML_interface_SVM_V3
import DataManager

dirname = os.path.dirname(__file__)
save_bin = os.path.join(dirname,"save_bin")


def robust_save(fname):
    plt.savefig(os.path.join(save_bin,'overlayed_predictions.png',dpi=200,bbox_inches='tight'))
####PARAMS#### 
param_grid = {'kneighborsclassifier__n_neighbors':[5,7,10,13,15,18,20]}

### DATA PROCESSING IMAGE 1 ###

#initialize test data set
dirname = os.path.dirname(__file__)
foldername = os.path.join(dirname,"images_5HT")
im_dir = DataManager.DataMang(foldername)

### PARAMS ###
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42
###

#load image folder for training data
dirname = os.path.dirname(__file__)
foldername = os.path.join(dirname,"images_5HT")
im_dir = DataManager.DataMang(foldername)
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [i for i in range(start,im_dir.dir_len)]
hog_features = []
for gen in im_dir.open_dir(im_list):
    #load image and its information
    image,nW,nH,chan,name = gen
    print('procesing image : {}'.format(name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))
    image = image[:,:,channel]
    #Import train data (if training your model)
    train_bool = ML_interface_SVM_V3.import_train_data(name,(nW,nH),'train_71420')
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, domains, paded_im_seg, paded_bool_seg, hog_features = SVM.feature_extract(image, ff_width, wiener_size, med_size,True,train_bool)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    X,y = SVM.create_data(hog_features,True,bool_segs)
    break

print('done')
#adding in some refence numbers for later
y = np.vstack([y,np.arange(0,len(y),1)]).T
#split dataset

print('Splitting dataset...')
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=count)
ind_train = y_train[:,1]
ind_test = y_test[:,1]

y_train = y_train[:,0]
y_test = y_test[:,0]



print("y_train: " + str(np.unique(y_train)))
print("y_test: " + str(np.unique(y_test)))



#create SVM pipline
#try using a GBC
pipe_knn = make_pipeline(RobustScaler(),KNeighborsClassifier())

#SVM MODEL FITTING
#we create an instance of SVM and fit out data.
print("starting modeling career...")

# gs = GridSearchCV(estimator = pipe_knn,
#                   param_grid = param_grid,
#                   scoring = 'roc_auc',
#                   cv = 5,
#                   n_jobs = -1,
#                   verbose = 10)


# print("Fitting...")
# gs = gs.fit(X_train,y_train)
# print('best score: ' + str(gs.best_score_))
# print(gs.best_params_)
# pipe_knn = gs.best_estimator_
### END Gridsearch ####

### Setting Parameters ###
#{'kneighborsclassifier__n_neighbors': 7}
print('fitting...')

pipe_knn.set_params(kneighborsclassifier__n_neighbors = 7)

### Cross Validate ###
scores = cross_val_score(estimator = pipe_knn,
                          X = X_train,
                          y = y_train,
                          cv = 10,
                          scoring = 'roc_auc',
                          verbose = True,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

### Fitting Model ###
fitted = pipe_knn.fit(X_train,y_train)
y_score = pipe_knn.fit(X_train,y_train).predict_proba(X_test)
print(pipe_knn.score(X_test,y_test))

### DATA PROCESSING IMAGE 2 ###
#pick a test image
# os.chdir(r'C:\Users\jsalm\Documents\Python Scripts\SVM_7232020')
Test_im = np.array(cv2.imread("images_5HT/injured 60s_sectioned_CH2.tif")[:,:,2]/255).astype(np.float32)

#extract features
# im_segs_test, _, domains_test, paded_im_seg_test, _, hog_features_test = SVM.feature_extract(Test_im, ff_width, wiener_size, med_size,False)
# X_test = SVM.create_data(im_segs_test,False)

### SVM MODEL PREDICTION ###
predictions = fitted.predict(X_test)   

# predict_im = data_to_img(boolim2_2,predictfions)
SVM.overlay_predictions(image, train_bool, predictions, y_test, ind_test,domains)

### Confusion Matrix: Save fig if interesting ###
confmat = confusion_matrix(y_true = y_test, y_pred=predictions)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.savefig(os.path.join(save_bin,'confussion_matrix.png'),dpi=200,bbox_inches='tight')
plt.show()

### ROC Curve ###
fpr, tpr,_ = roc_curve(y_test, y_score[:,1])
roc_auc = auc(fpr, tpr)
SVM.write_auc(fpr,tpr)
#fpr,tpr,roc_auc = SVM.read_auc()


plt.figure()
lw = 2

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig(os.path.join(save_bin,'roc_auc_curve.png'),dpi=200,bbox_inches='tight')
# #PLOT IMAGES
# # Filters.imshow_overlay(Test_im,predict_im,'predictions2',True)

# name_list = ["image","denoised_im","median_im","thresh_im","dir_im","gau_im","di_im","t_im"]
# for i in range(0,len(image_tuple)):
#     plt.figure(name_list[i]);plt.imshow(image_tuple[i])