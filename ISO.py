import rawpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from util import pipeline

"""
Setup: IMX623
Configuration: Exposure time: 1/50s f/#: f/5.6  f:15mm 
Temperature: room temperature    
"""


n1 = 22
n2 = 81
mu_y, sigma_y = pipeline(n1, n2)

#--------------- OLS Model ---------------
# Linear Regression Model by sklearn 
ISO = [200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100]
ISO = np.array(ISO)

ISO = ISO[:, np.newaxis] # sklearn requires 2-D array for processing
mu_y = mu_y[:, np.newaxis] # sklearn requires 2-D array for processing


#iso 1300~3100
# ISO = ISO[5+7-1:, np.newaxis] # sklearn requires 2-D array for processing
# mu_y = mu_y[5+7-1:, np.newaxis] # sklearn requires 2-D array for processing
# sigma_y=sigma_y[5+7-1:]
# Dark current from mean
model1 = LinearRegression()
model1.fit(ISO, mu_y)
mu_y_pred = model1.predict(ISO)
R2_mean = model1.score(ISO, mu_y)
print('Dark current from mean: R2 = %.2f' % R2_mean)

# Dark current from variance
sigma_y = sigma_y[:, np.newaxis] # sklearn requires 2-D array for processing

model2 = LinearRegression()
model2.fit(ISO, sigma_y)
sigma_y_pred = model2.predict(ISO)
R2_var = model2.score(ISO, sigma_y)
print('Dark current from variance: R2 = %.2f' % R2_var)


#---------------- Step 3 Plot ----------------

## Dark current from mean
mu_y_pred = mu_y_pred.reshape(ISO.shape[0],)
mu_y = mu_y.reshape(ISO.shape[0],)

plt.title("Dark current from mean", fontsize=15)
plt.xlabel("ISO", fontsize=13)
plt.ylabel("mean dark signal", fontsize=13)
plt.grid(ls='--', linewidth=0.6, alpha=0.5)
plt.scatter(ISO, mu_y, marker='+', color='#7f7f7f', label='mono data')
plt.plot(ISO, mu_y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
plt.text(2000, 290, s=r"$R^2=$" + f"{R2_mean:.2f}")
plt.legend()
plt.tight_layout()
plt.show()


## Dark current from variance
sigma_y_pred = sigma_y_pred.reshape(ISO.shape[0],)
sigma_y = sigma_y.reshape(ISO.shape[0],)

plt.title("Dark current from variance", fontsize=15)
plt.xlabel("ISO", fontsize=13)
plt.ylabel("variance dark signal", fontsize=13)
plt.grid(ls='--', linewidth=0.6, alpha=0.5)
plt.scatter(ISO, sigma_y, marker='+', color='#7f7f7f', label='mono data')
plt.plot(ISO, sigma_y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
plt.text(2000, 4, s=r"$R^2=$" + f"{R2_var:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
