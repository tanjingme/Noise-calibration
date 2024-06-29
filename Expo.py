import rawpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from util import pipeline

""" 
Setup: IMX623
Configuration: ISO-3100  f/#: f/5.6  f:15mm 
Temperature: room temperature
"""

n1 = 0
n2 = 21
mu_y, sigma_y = pipeline(n1, n2)

#--------------- OLS Model ---------------
# Linear Regression Model by sklearn
t_exp = [1/50,1/25,1/10,1/8,1/5,1/3,0.4,0.5,0.6,0.8,1.0]
t_exp = np.array(t_exp)

t_expo = t_exp[:, np.newaxis] # sklearn requires 2-D array for processing
mu_y = mu_y[:, np.newaxis] # sklearn requires 2-D array for processing


# Dark current from mean
model1 = LinearRegression()
model1.fit(t_expo, mu_y)
mu_y_pred = model1.predict(t_expo)
R2_mean = model1.score(t_expo, mu_y)
print('Dark current from mean: R2 = %.2f' % R2_mean)
print(model1.coef_,model1.intercept_)

# Dark current from variance
sigma_y = sigma_y[:, np.newaxis] # sklearn requires 2-D array for processing

model2 = LinearRegression()
model2.fit(t_expo, sigma_y)
sigma_y_pred = model2.predict(t_expo)
R2_var = model2.score(t_expo, sigma_y)
print('Dark current from variance: R2 = %.2f' % R2_var)
print(model2.coef_,model2.intercept_)
#dark= 40.39892228*t+272.95957785
#--------------- Plot ---------------
## Dark current from mean
mu_y_pred = mu_y_pred.reshape(t_exp.shape[0],)
mu_y = mu_y.reshape(t_exp.shape[0],)

plt.title("Dark current from mean", fontsize=15)
plt.xlabel("Exposure time(s)", fontsize=13)
plt.ylabel("mean dark signal", fontsize=13)
plt.grid(ls='--', linewidth=0.6, alpha=0.5)
plt.scatter(t_exp, mu_y, marker='+', color='#7f7f7f', label='mono data')
plt.plot(t_exp, mu_y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
plt.text(0.8,280, s=r"$R^2=$" + f"{R2_mean:.2f}")
plt.legend()
plt.tight_layout()
plt.show()


## Dark current from variance
sigma_y_pred = sigma_y_pred.reshape(t_exp.shape[0],)
sigma_y = sigma_y.reshape(t_exp.shape[0],)

plt.title("Dark current from variance", fontsize=15)
plt.xlabel("Exposure time(s)", fontsize=13)
plt.ylabel("variance dark signal", fontsize=13)
plt.grid(ls='--', linewidth=0.6, alpha=0.5)
plt.scatter(t_exp, sigma_y, marker='+', color='#7f7f7f', label='mono data')
plt.plot(t_exp, sigma_y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
plt.text(0.8, 280, s=r"$R^2=$" + f"{R2_var:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
