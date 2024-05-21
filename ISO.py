import rawpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from util import pipeline

"""
Setup: Canon EOS M50
Configuration: Exposure time: 1/400s f/#: f/5.6  f:15mm 
Temperature: room temperature    
"""

n1 = 942
n2 = 957
mu_y, sigma_y = pipeline(n1, n2)

#--------------- OLS Model ---------------
# Linear Regression Model by sklearn 
ISO = [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
ISO = np.array(ISO)

ISO = ISO[:, np.newaxis] # sklearn requires 2-D array for processing
mu_y = mu_y[:, np.newaxis] # sklearn requires 2-D array for processing

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
mu_y_pred = mu_y_pred.reshape(8,)
mu_y = mu_y.reshape(8,)

plt.title("Dark current from mean", fontsize=15)
plt.xlabel("ISO", fontsize=13)
plt.ylabel("mean dark signal", fontsize=13)
plt.grid(ls='--', linewidth=0.6, alpha=0.5)
plt.scatter(ISO, mu_y, marker='+', color='#7f7f7f', label='mono data')
plt.plot(ISO, mu_y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
plt.text(20000, 1000, s=r"$R^2=$" + f"{R2_mean:.2f}")
plt.legend()
plt.tight_layout()
plt.show()


## Dark current from variance
sigma_y_pred = sigma_y_pred.reshape(8,)
sigma_y = sigma_y.reshape(8,)

plt.title("Dark current from variance", fontsize=15)
plt.xlabel("ISO", fontsize=13)
plt.ylabel("variance dark signal", fontsize=13)
plt.grid(ls='--', linewidth=0.6, alpha=0.5)
plt.scatter(ISO, sigma_y, marker='+', color='#7f7f7f', label='mono data')
plt.plot(ISO, sigma_y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
plt.text(20000, 2500, s=r"$R^2=$" + f"{R2_var:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
