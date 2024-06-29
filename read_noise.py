import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
import statsmodels.graphics.gofplots as sm
import rawpy
import pylab
from scipy.stats import tukeylambda

#---------------- Step 1: Data Process ----------------
# Setup: Canon EOS M50
# Configuration: ISO-3200  f/#: f/5.6  f:15mm 
# Temperature: room temperature

# Expo: 1/4000s
# Read & Decode raw image
raw_img = rawpy.imread('./images/_MG_00.CR3')
raw_data = raw_img.raw_image.copy().astype(np.float64)

h = raw_data.shape[0]
w = raw_data.shape[1]

### Row noise parameters ###
# extract mean values from each row
mu_row = np.sum(raw_data, axis=1) / w
mu_row2d = mu_row.reshape(h, 1)
mu_row_t = np.tile(mu_row2d, w)

# maximizing the log-likelihood
sigma2 = np.sum((raw_data - mu_row_t)**2, axis=1) / w

#---------------- Step 2: Row Noise Image ----------------
raw_row = np.zeros((h, w), dtype=np.float64)
for i in range(h):
    raw_row[i][:] = np.random.normal(loc=mu_row[i], scale=np.sqrt(sigma2[i]), size=w)

# Subtract row noise image from bias frame
raw_read = raw_data - raw_row

# Crop for small array
raw_crop = np.zeros((200, 200))
for i in range(200):
    raw_crop[i][:] = raw_read[raw_read.shape[0]-200+i][raw_read.shape[1]-200+i]


raw_crop = raw_crop.flatten()
# raw_crop = raw_read.flatten()

#---------------- Step 3: Plots for Statistics ----------------

########### Using statsmodel.graphics.gofplots ##########
# # Plots for Normal Probability Plot 
# sns.set_theme()
# fig = sm.ProbPlot(raw_crop).qqplot(line='r')
# plt.ylabel('Ordered Response Values')
# plt.title('Gaussian Probability Plot')
# plt.show()

# # Plot for histogram
# sns.set_theme()
# fig, ax = plt.subplots()
# sns.histplot(raw_crop, kde=True, color='blue')
# plt.title('Histogram')
# plt.show()

################### Using scipy.stats ###################

## Gaussian Probability Plot
sns.set_theme()
sc.probplot(raw_crop, dist='norm', plot=pylab, rvalue=True)
plt.title("Gaussian Probability Plot")
plt.show()

## Tukey Lambda PPCC Plot
lb = -2
ub = 2

fig, ax = plt.subplots()
sc.ppcc_plot(raw_crop, lb, ub, plot=ax)
shape_param_max = sc.ppcc_max(raw_crop)
ax.axvline(shape_param_max, color='r')
plt.title("Tukey Lambda PPCC Plot")
plt.text(0.75, 0.3, s=r"$\lambda$ = "+f"{shape_param_max:.3f}")
plt.show()
print("Shape parameter is ", shape_param_max)

## Tukey Lambda Probability Plot
mean, var, skew, kurt = tukeylambda.stats(shape_param_max, moments='mvsk')
sc.probplot(raw_crop, sparams=(shape_param_max, mean, np.sqrt(var)), dist='tukeylambda', plot=pylab, rvalue=True)
plt.title("Tukey Lambda Probability Plot")
plt.show()
