#Generalized-Sklearn-ML-Pipeline
Image processing pipeline for serotonin detection using SVM and Gradient Bossting Algorithms.

PYTHON FILES
1) DataManager.py :: Handles directory management and image selection. includes funcitonallity to create & load pickled objects.
2) Filters.py :: Various filters for producing features and segmentations. (included: wiener filter, median filter, generalized convolver, image normalization, gabor filter, contour generation based on thresholding, adaptive thresholding, Hi-pass filter using dFFT (discrete fourier transform), mean filter, gaussian blurring, edge detection using differential convolution, downsampling using countless & zero-corrected countless.  
3) SVM.py :: training data handling (typically as an output from ML_interface_SVM_V3.py), filter pipeline for feature generation to Machine Learning platform, image segmentation for generation of image sections & image padding, statistical diagram outputs.
4) ML_interface_SVM_V3.py :: GUI for tracing images and generating training data.
5) main.py :: implementation of the above modules
