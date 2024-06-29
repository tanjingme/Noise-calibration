""" 重建流程 """

import numpy as np
import rawpy
import scipy.stats as sc
from scipy.stats import tukeylambda
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import util
import imageio

#---------------- Step 1: Extract Parameters ----------------
# This noise image is under the specific conditions
# like specific exposure time, etc.

# Setup: IMX623
# Configuration: ISO-3100  f/#: f/5.6  f:15mm 
# Temperature: room temperature
# Exposure time: 1/50s


import pickle
import os
import pickle as pkl
def get_darkshading(iso, exp=20, naive=False, num=None, remake=True,ds_dir="./darkshading",darkshading={}):
    branch = '_highISO' if iso>1400 else '_lowISO'
    if iso not in darkshading or remake is True:
        ds_path = os.path.join(ds_dir, f'darkshading-iso-{iso}.npy')
        ds_k = np.load(os.path.join(ds_dir, f'darkshading{branch}_k.npy'))
        ds_b = np.load(os.path.join(ds_dir, f'darkshading{branch}_b.npy'))
        if naive: # naive linear darkshading - BLE(ISO)
            with open(os.path.join(ds_dir, f'darkshading_BLE.pkl'), 'rb') as f:
                blc_mean = pkl.load(f)
            BLE = blc_mean[iso]
        else: # new linear darkshading - BLE(ISO, t)
            with open(os.path.join(ds_dir, f'BLE_t.pkl'),'rb') as f:
                blc_mean = pickle.load(f)
            # BLE bias only here
            # kt = np.poly1d(blc_mean[f'kt{branch}'])
            #print(blc_mean)
            BLE =blc_mean[f'{iso}']['b'] # + kt(iso) * exp

        # D_{ds} = D_{FPNk} + D_{FPNb} + BLE(ISO, t)
        darkshading[iso] = ds_k * iso + ds_b + BLE

    if naive:
        return darkshading[iso]
    else:

        kt = np.poly1d(blc_mean[f'kt{branch}'])
        BLE = kt(iso) * exp
        return darkshading[iso] + BLE
    



def raw2rggb(raw_data):
    h = raw_data.shape[0]
    w = raw_data.shape[1]
    R = raw_data[0:h:2, 0:w:2]
    G1 = raw_data[0:h:2, 1:w:2]
    G2 = raw_data[1:h:2, 0:w:2]
    B = raw_data[1:h:2, 1:w:2]
    return R,G1,G2,B
# Read & Decode raw image
raw_img = rawpy.imread('./images/_MG_00.CR3')
raw_data = raw_img.raw_image.copy().astype(np.float64)
h,w=raw_data.shape
DS=get_darkshading(3100,20)
raw_data=raw_data-DS
plt.imshow(raw_data)
plt.show()

def show(raw_data,flag):
    h = raw_data.shape[0]
    w = raw_data.shape[1]
    # extract mean values from each row
    mu_row = np.sum(raw_data, axis=1) / w
    mu_row2d = mu_row.reshape(h, 1)
    mu_row_t = np.tile(mu_row2d, w)

    # maximizing the log-likelihood
    sigma2 = np.sum((raw_data - mu_row_t)**2, axis=1) / w

    # Row noise image
    N_r = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        N_r[i][:] = np.random.normal(loc=0, scale=np.sqrt(sigma2[i]), size=w)
        
    # Subtract row noise image from bias frame
    bias_frame = raw_data - N_r

    # Convert to 1D
    raw_1d = bias_frame.flatten()

    # Tukey Lambda shape parameter
    
    shape_param_max = sc.ppcc_max(raw_1d)
    shape_param_max = 0.165
    print("Shape parameter is ", shape_param_max)

    mean, var, skew, kurt = tukeylambda.stats(shape_param_max, moments='mvsk')

    # Create noise image
    noise_img = np.zeros((h, w), dtype=np.float64)

    #---------------- Step 2: Create Noise Image ----------------
    # Black Level
    noise_img += 256

    # Row Noise
    noise_img += N_r

    # Read Noise
    N_read = sc.tukeylambda.rvs(lam=shape_param_max, loc=mean, scale=np.sqrt(var), size=h*w)
    N_read = N_read.reshape(h, w)
    noise_img += N_read

    # Quantization Noise
    N_q = sc.uniform.rvs(loc= -1 / 2**13, scale= 1 / 2**12, size=h*w)
    N_q = N_q.reshape(h, w)
    noise_img += N_q

    #----------------- Step 3: Evaluation ----------------
    # Histogram
    noise =noise_img.reshape(h*w,)

    plt.subplot(2,1,1)
    plt.hist(noise, bins='auto', histtype='stepfilled', alpha=0.2, log=True)
    # print([noise.min(), noise.max()])
    # plt.set_xlim([noise.min(), noise.max()])
    plt.title("Noise Image Histogram(TL)", fontsize=15)
    plt.tight_layout()
    # plt.show()


    raw_data_ = raw_data.reshape(h*w,)
    plt.subplot(2,1,2)
    plt.hist(raw_data_, bins='auto', histtype='stepfilled', alpha=0.9, log=True)
    print([raw_data_.min(), raw_data_.max()])
    # plt.set_xlim([raw_data_.min(), raw_data_.max()])
    plt.title(f"Raw Data({flag}) Histogram", fontsize=15)
    plt.tight_layout()
    plt.show()


    # R-square
    def linearize(img, darkness, saturation):
        img = img.astype(np.float64)
        ret = (img - darkness) / (saturation - darkness)
        ret[ret<0] = 0
        ret[ret>1] = 1
        return ret

    darkness = 256
    saturation = 4095
    niLinear = linearize(noise_img, darkness, saturation)
    gtLinear = linearize(raw_data, darkness, saturation)
    niLinear_sort = np.sort(niLinear.reshape(h*w,))
    gtLinear_sort = np.sort(gtLinear.reshape(h*w,))
    SSE = np.sum((gtLinear_sort - niLinear_sort)**2)
    SST = np.sum((gtLinear - np.mean(gtLinear))**2)
    R2 = 1 - SSE / SST
    print(f"R2 for estimated model: {R2}")

    # KL-divergence
    print(raw_data_.mean(),noise.mean(),'==============')
    KL = scipy.stats.entropy(raw_data_, noise)
    print("KL-divergence: {}".format(KL))

    #----------------- Step 4: Save Noise Image ----------------
    imgResult = util.isp(noise_img, darkness, saturation)
    plt.grid(visible=False)
    # plt.xticks([])
    # plt.yticks([])
    plt.title("Calibrated Noise Image")
    print(imgResult.shape)
    plt.subplot(2,1,1)
    plt.imshow(raw_data)
    plt.subplot(2,1,2)
    plt.imshow(noise_img)
    imageio.imsave('Calibrated Noise Image.jpg', imgResult)

show(raw_data,'RAW')
plt.show()

