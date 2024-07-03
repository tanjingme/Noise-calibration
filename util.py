import rawpy
import numpy as np
import os
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from skimage.color import rgb2gray

def ImgRead(imgpath_0, imgpath_1):
    """ Read image & Data process """
    # Read & Decode raw img
    raw_0 = rawpy.imread(imgpath_0)
    raw_1 = rawpy.imread(imgpath_1)
    raw0_data = raw_0.raw_image.copy().astype(np.float64)
    raw1_data = raw_1.raw_image.copy().astype(np.float64)
    
    # Mean gray value
    h = raw0_data.shape[0]
    w = raw0_data.shape[1]
    mu_y0 = np.sum(raw0_data) / (h * w)
    mu_y1 = np.sum(raw1_data) / (h * w)
    mu_y = 0.5 * (mu_y0 + mu_y1)
    
    # Temporal variance of gray value
    var_y = np.sum((raw0_data - raw1_data) ** 2) / (2 * h * w) - 0.5 * (mu_y0 - mu_y1) ** 2
    
    return var_y, mu_y

def pipeline(n1, n2,root_path=None):
    """ 
    Process pipeline 
    n1: index for the beginning of pipelibe
    n2: index for the end of pipeline
    """
    if root_path is None:
        root_path="images"
    n = int((n2 - n1 + 1) / 2)
    mu_list = []
    sigma_y = []  # Note that sigma_y here denotes sigma_y square for simplicity
    
    for i in range(n):
        imgpath_0 = os.path.join(root_path, '_MG_0{}.CR3'.format(i * 2 + n1))
        imgpath_1 = os.path.join(root_path, '_MG_0{}.CR3'.format(i * 2 + n1 + 1))
        var_y, mu_y = ImgRead(imgpath_0, imgpath_1)
        mu_list.append(mu_y)
        sigma_y.append(var_y)
        
    mu_y = np.array(mu_list)
    sigma_y = np.array(sigma_y)
        
    return mu_y, sigma_y

def color_bias_eval(n1, n2):
    """Evaluate each channel color bias

    Args:
        n1 (_int_): index for the beginning of bias frame
        n2 (_int_): index for the end of bias frame
    
    Return: each channel's color bias
    """
    # Read & Decode raw image
    n = int((n2 - n1 + 1) / 2)
    R_l, G1_l, G2_l, B_l = [], [], [], []
    
    for i in range(n):
        imgpath = os.path.join('./images', '_MG_0{}.CR3'.format(i * 2 + n1))
        raw_img = rawpy.imread(imgpath)
        raw_data = raw_img.raw_image.copy().astype(np.float64)
        h = raw_data.shape[0]
        w = raw_data.shape[1]
        
        # Bayer Pattern: RG/GB
        R = raw_data[0:h:2, 0:w:2]
        G1 = raw_data[0:h:2, 1:w:2]
        G2 = raw_data[1:h:2, 0:w:2]
        B = raw_data[1:h:2, 1:w:2]
        R_mean = np.mean(R)
        G1_mean = np.mean(G1)
        G2_mean = np.mean(G2)
        B_mean = np.mean(B)
        
        # Black level for each channel: 2048 2048 2048 2048
        R_bias = R_mean - 2048
        G1_bias = G1_mean - 2048
        G2_bias = G2_mean - 2048
        B_bias = B_mean - 2048
        R_l.append(R_bias)
        G1_l.append(G1_bias)
        G2_l.append(G2_bias)
        B_l.append(B_bias)
    
    return np.array(R_l), np.array(G1_l), np.array(G2_l), np.array(B_l)

def linearize(img, darkness, saturation):
    """ Scale and Linearize """
    img = img.astype(np.float64)
    ret = (img - darkness) / (saturation - darkness)
    ret[ret<0] = 0
    ret[ret>1] = 1
    return ret

def camerascaleWB(input_img, r_scale, g1_scale, g2_scale, b_scale):
    """RGGB"""
    output_img = input_img.copy()
    output_img[0::2, 0::2] *= r_scale
    output_img[0::2, 1::2] *= g1_scale
    output_img[1::2, 0::2] *= g2_scale
    output_img[1::2, 1::2] *= b_scale
    
    return output_img

def applyColorMatrix(input_img, color_matrix):
    h,w = input_img.shape[0:2]
    output_img = color_matrix.dot(input_img.reshape((h*w, 3), order='C').T)
    output_img = output_img.T.reshape((h, w, 3), order='C')
    return output_img

def scaleImg(imgIn, meanPos=0.25):
    """
    Scale the image

    Here we get the mean value of gray image converted by imgIn,
    and use this value to get the scale value

    Parameters
    ----------
    imgIn: original image after demosaicing.
    meanPos: mean value of the gray image

    Returns
    -------
    Scaled image
    """
    imgInGray = rgb2gray(imgIn)
    scale = meanPos / imgInGray.mean()
    outImg = imgIn.copy()
    outImg *= scale
    outImg[outImg>1] = 1
    outImg[outImg<0] = 0
    return outImg

def gammaCorr(imgIn):
    """
    Gamma Correction
    """
    imgOut = imgIn.copy()
    isDarkPixel = imgIn <= 0.0031308
    isBrightPixel = np.logical_not(isDarkPixel)
    imgOut[isDarkPixel] *= 12.92
    imgOut[isBrightPixel] = (1+0.055)*(imgIn[isBrightPixel])**(1/2.4) - 0.055

    return imgOut

def isp(img, darkness, saturation):
    """ ISP Pipeline

    Args:
        img (_ndarray_): image
        darkness (_int_): black level
        saturation (_int_): white level

    Returns:
        _ndarray_: result
    """
    #----- Linearize -----
    imgLinear = linearize(img, darkness, saturation)
    print(imgLinear.min(), imgLinear.max(), imgLinear.shape, imgLinear.dtype)

    #----- White Balance -----
    # From exiftool we can get:
    # _MG_0771.CR3:
    # Red Balance:1.530273; Blue Balance:1.268555;
    r_scale, b_scale, g_scale = 1.530273, 1.268555, 1.0
    imgWB = camerascaleWB(imgLinear, r_scale, g_scale, g_scale, b_scale)
    
    #----- Demosaicing -----
    imgDemo = demosaicing_CFA_Bayer_Menon2007(imgWB)
    
    #----- Colour Correction -----
    M_sRGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], \
                       [0.2126729, 0.7151522, 0.0721750], \
                        [0.0193339, 0.1191920, 0.9503041]])

    M_XYZ2Cam = np.array([[0.8532, -0.0701, -0.1167], \
                      [-0.4095, 1.1879, 0.2508], \
                        [-0.0797, 0.2424, 0.701]])

    M_sRGB2Cam = M_sRGB2XYZ.dot(M_XYZ2Cam)

    # Normalization
    denomi = np.sum(M_sRGB2Cam, axis=1, keepdims=True)
    M_sRGB2Cam = M_sRGB2Cam / denomi

    # Get inverse
    M_Cam2sRGB = np.linalg.inv(M_sRGB2Cam)
    
    # C_cam onvert to C_sRGB
    imgTrans = applyColorMatrix(imgDemo, M_Cam2sRGB)

    #----- Gamma Correction -----
    # Light Correction
    imgScaled = scaleImg(imgTrans, 0.25)
    
    # Gamma Correction
    # Apply Gamma Correction Function
    imgGamma = gammaCorr(imgScaled)
    imgGamma[imgGamma>1] = 1
    imgGamma[imgGamma<0] = 0
    imgResult = (imgGamma * 255).astype(np.uint8)
    
    return imgResult









