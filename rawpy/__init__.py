#a wrapper of rawpy due to custom format camera data
import numpy as np
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)


class Dark_Mu():
    def __init__(self):
        self.param=np.array([ (0.009691417272691908,0.010812937527008336,0.010811112397754551,0.02334004333521536),
                             (-1.7050508439414571,-1.103284108995478,-1.1111501072715306,-3.7673406704110377),
                             (-0.01824922709711304,-0.01754167968112866,-0.017594814414383875,-0.02741881090936765),
                             (+301.27823973628824,+296.7026876897195,+296.64210088307055,+447.75929330966085)])
    def __call__(self,iso,expo):
        assert expo<=0.4 and iso>=0.02
        assert iso<=3100 and iso>=1500
        return (self.param[0]*iso+self.param[1])*expo+(self.param[2]*iso+self.param[3])


class CustomImage:
    def __init__(self,raw_image,h,w,bits=12):
        self.raw_image=raw_image
        self.dark_mu=Dark_Mu()
        self.h=h
        self.w=w
        self.quant_num=(1<<bits)-1

    def __str__(self):
        return f"CustomImage(h={self.h},w={self.w},raw_image=numpy_array)"

    def __repr__(self):
        return f"CustomImage(h={self.h},w={self.w},raw_image=numpy_array)"
    def postprocess(self,iso,expo,use_camera_wb=False):
        raw_image=self.raw_image.astype("float32").copy()
        black_level=self.dark_mu(iso,expo)
        h = raw_image.shape[0]
        w = raw_image.shape[1]

        raw_image[0:h:2, 0:w:2] = raw_image[0:h:2, 0:w:2]-black_level[0]
        raw_image[0:h:2, 1:w:2] = raw_image[0:h:2, 1:w:2]-black_level[1]
        raw_image[1:h:2, 0:w:2] = raw_image[1:h:2, 0:w:2]-black_level[2]
        raw_image[1:h:2, 1:w:2] = raw_image[1:h:2, 1:w:2]-black_level[3]
        show_img=raw_image/self.quant_num
        if use_camera_wb:
            show_img=(show_img-show_img.min())/(show_img.max()-show_img.min())

        #/4095
        show_img=np.clip(demosaicing_CFA_Bayer_bilinear(show_img**(1.0/2.2),pattern='RGGB')*255*1,0,255).astype("uint8")
        return show_img
    
def imread(in_path,h=1536,w=1920,start_offset=64,bits=12):
    frame_sz=w*h
    frame_offset=w*(h+2) 
    raw_vec=np.fromfile(in_path,dtype=np.uint16)
    raw_length=raw_vec.shape[0]
    list_start=list(range(start_offset,raw_length,frame_offset))
    real_raw_vec=[raw_vec[x:(x+frame_sz)] for x in list_start]

    all=[]
    for j,raw_img in enumerate(real_raw_vec):
        if raw_img.shape[0]!=h*w:
            continue        
        im= raw_img.reshape((h,w)).astype("float32")
        all.append(im)
    data=np.stack(all,axis=0)
    if data.shape[0]>1:
        raise NotImplementedError
    if data[0].max()>((1<<bits)-1):
        raise NotImplementedError

    return CustomImage(data[0],h,w,bits=bits)


