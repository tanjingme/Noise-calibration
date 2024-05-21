""" Evaluation of row noise """

import rawpy
import matplotlib.pyplot as plt
import numpy as np

#--------------- Step 1: Data Process ---------------
# Setup: Canon EOS M50
# Configuration: ISO-3200  f/#: f/5.6  f:15mm 
# Temperature: room temperature

# Expo: 1/4000s
# Read & Decode raw image
raw_img = rawpy.imread('./images/_MG_0771.CR3')
rgb_img = raw_img.postprocess(use_camera_wb=True)
raw_data = raw_img.raw_image.copy().astype(np.float64)

h = raw_data.shape[0]
w = raw_data.shape[1]

# extract mean values from each row
mu_row = np.sum(raw_data, axis=1) / w
mu_row2d = mu_row.reshape(h, 1)
mu_row_t = np.tile(mu_row2d, w)

# maximizing the log-likelihood
sigma2 = np.sum((raw_data - mu_row_t)**2, axis=1) / w

#--------------- Step 2: DFT ---------------
plt.imshow(rgb_img)
plt.title("Original")
plt.xticks([])
plt.yticks([])
plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
img = rgb2gray(rgb_img)

plt.imshow(img, cmap='gray')
plt.title("Gray")
plt.xticks([])
plt.yticks([])
plt.show()

img_fft = np.fft.fft2(img)
img_fftshift = np.fft.fftshift(img_fft)

plt.imshow(np.log(1+np.abs(img_fft)))
plt.title("Spectrum")
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()

plt.imshow(np.log(1+np.abs(img_fftshift)))
plt.title("Centered Spectrum")
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()
