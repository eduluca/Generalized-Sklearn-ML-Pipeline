
class ImageClassificate:
    def __init__(self, imageIn):
        ##% PRIMARY INFORMATION
        self.image = imageIn # np.array() of image input
        self.shape = imageIn.__getShape() # 
        self.muNstd = imageIn.__getMeanStd()
        #%% SECONDARY INFORMATION
        self.imageSegments = []
        self.imageLabels = []
        self.filterOrder = []
    #enddef
    def __loadImage():
        pass
    #enddef
    
    def __getShape():
        pass
    #enddef 

    def __getMeanStd():
        pass
    #enddef

    def 
#endclass