
class ImageClassificate:
    def __init__(self, imagePath):
        ##% PRIMARY INFORMATION
        self.imagePath = imagePath # np.array() of image input
        imageIn = __loadImage(imagePath)
        self.shape = imageIn._getShape() # tuple of the shape of the image (heightxwidth)
        self.muNstd = imageIn._getMeanStd() # mean and standard deviation of the image data
        #%% SECONDARY INFORMATION
        self.imageSegments = []
        self.imageLabels = []
        self.filterOrder = []
    #enddef

    def __loadImage(imagePath):
        
    #enddef

    def _getShape():
        pass
    #enddef 

    def _getMeanStd():
        pass
    #enddef

    def 
#endclass