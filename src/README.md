# Generalized-Sklearn-ML-Pipeline
Summary: Image processing pipeline for serotonin detection using SVM and Gradient Bossting Algorithms.

# Installation
Create a new virtual enviornment (or however you'd prefer to setup your python workspace). Go into command prompt (windows) or terminal (mac) and follow this command prompt example:

'''
C:\Windows\system32> cd [your repo path here]
[repo path]> [activate virtual enviornment]
[repo path]> python .\src\setup.py install
'''

The setup.py should add the package to your virtual enviornment so you can use the files and scripts. 

# STEP1: a_dataGeneration
Folders: \_\_pycache\_\_, rawData
Files: \_\_init\_\_.py, TrainGUI.py

### TrainGUI.py

# STEP2: b_dataAggregation
Folders: \_\_pycache\_\_, aggregateData, processedData
Files: \_\_init\_\_.py, dataPreProcessing.py
## Folders
1. \_\_pychache\_\_:: codegen folder for cython compiles.

2. aggregateData:: folder for storing the pickled and concatenated training data.

3. processData:: folder for storing RAW training 

## Files
### dataPreProcessing.py

# STEP3: c_dataValidation
Folders: saveBin
Files: \_\_init\_\_.py, dataCrossValid.py
## Folders
1. saveBin:: folder for storing the train-test split data (k-fold). 

## Files
### dataCrossValid.py

# STEP4: d_modelTrianing
Folders: saveBin
Files: \_\_init\_\_.py, main-KNN.py, main-SVM.py, main-XGB.py
## Folders
1. saveBin:: folder contains folders for saving each model generated from data.

## Files
### main-SVM.py 


### main-KNN.py


### main-SVM.py


### main-XGB.py


# STEP5: e_modelTesting
Folders: saveBin
Files: init.py, modelTesting.py
## Folders
1. saveBin:: folder contains figures and images made availble by using modelTesting.py

## Files
### modelTesting.py


# localModules
Folders: \_\_pycache\_\_
Files: DataManager.py, Filters.py, ProcessPipe.py, ThreeD_Recon_V3.py
## Folders
1. \_\_pycache\_\_: codegen folder for cython compiles.

## Files
### DataManager.py
Handles directory management and image selection. includes funcitonallity to create & load pickled objects.
### Filters.py 
Various filters for producing features and segmentations. (included: wiener filter, median filter, generalized convolver, image normalization, gabor filter, contour generation based on thresholding, adaptive thresholding, Hi-pass filter using dFFT (discrete fourier transform), mean filter, gaussian blurring, edge detection using differential convolution, downsampling using countless & zero-corrected countless.  
### ProcessPipe.py
 
### ThreeD_Recon_V3.py
