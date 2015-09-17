import skimage.transform
import numpy as np
#from joblib import Parallel, delayed
from PIL import Image

WIDTH = 224
HEIGHT = 224

class Fetcher :
    def __init__(self,crop_type,mean,n_jobs=16) :
        self.mean = mean
        if crop_type == 'center' :
            self.crop = self._center_crop
        else :
            print 'not implented'
            raise

        #self.thread_pool = Parallel(n_jobs=n_jobs)

    def _preprocess(self,im_loc,ret,i) :
        #read files
        im = np.array(Image.open(im_loc),np.uint8)
        
        #crop
        im = self.crop(im)

        # Shuffle Axes to c*w*h
        im = np.swapaxes(np.swapaxes(im,1,2),0,1)
        # convert to BGR order
        im = im[::-1,:,:]

        im = np.float32(im) - self.mean
        ret[i,...] = im

    def _center_crop(self,im) :
        h, w, _ = im.shape
        return im[h//2-HEIGHT//2:h//2+HEIGHT//2,w//2-WIDTH//2:w//2+WIDTH//2]
    
    def fetch(self,ims) : 
        ret = np.zeros((len(ims),3,WIDTH,HEIGHT),np.float32)
        #self.thread_pool(delayed(_preprocess)(loc[1],self.crop,i,ret,self.mean) for i,loc in enumerate(ims))

        for i,(_,loc) in enumerate(ims) :
            self._preprocess(loc,ret,i)
        return ret

        

     
