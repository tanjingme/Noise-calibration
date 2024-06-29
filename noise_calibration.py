import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson
import seaborn as sns
from util import pipeline

#--------------- Linear Regression --------------------
# Please read ./images/README.md for details
# index for the beginning and the end of flat-field frames
n1 = 5122
n2 = 5205
# index for the beginning and the end of bias frames
n3 = 5038
n4 = 5121




mu_y, sigma_y = pipeline(n1, n2)
mu_y_dark, _ = pipeline(n3, n4)
x = mu_y - mu_y_dark
y = sigma_y
x_ = x[:25]
y_ = y[:25]


# # Linear Regression Model by statsmodel
# x_ = sm.add_constant(x_)
# model = sm.OLS(y_, x_, hasconst=1)
# results = model.fit()
# print(results.summary())


# Linear Regression Model by sklearn
x_ = x_[:, np.newaxis]
y_ = y_[:, np.newaxis]
model = LinearRegression()
model.fit(x_, y_)
y_pred = model.predict(x_)
R2 = model.score(x_, y_)
print('R2 = %.2f' % R2)
coef = model.coef_
intercept = model.intercept_
print(f"Coef: {coef}; Intercept: {intercept}")


# # Plot the relationship between sigma_y and (mu_y - mu_{y*dark})
# x_ = x[:25]
# y_ = y[:25]
# y_pred = y_pred.reshape(25,)
# x_fit = np.linspace(0, 10000, 5000)
# y_fit = coef * x_fit + intercept
# y_fit = y_fit.reshape(5000,)
# plt.title(r"$\sigma_y^2 - (\mu_y - \mu_{y\cdot dark})$", fontsize=15)
# plt.xlabel(r"$\mu_y - \mu_{y\cdot dark}$", fontsize=13)
# plt.ylabel(r"$\sigma_y^2$", fontsize=13)
# plt.grid(ls='--', linewidth=0.6, alpha=0.5)
# plt.scatter(x, y, color='#ff7f0e', alpha=0.9, s=30, edgecolors='black', label='Data')
# plt.plot(x_fit, y_fit, linewidth=2, linestyle='--', color='#7f7f7f', label='Fit')
# plt.tight_layout()
# plt.show()


# # Plot for the photon transfer curve
# x_ptc = x[:26]
# y_ptc = y[:26]
# plt.title("Photon Transfer", fontsize=15)
# plt.xlabel(r"$\mu_y - \mu_{y\cdot dark}$", fontsize=13)
# plt.ylabel(r"$\sigma_y^2$", fontsize=13)
# plt.grid(ls='--')
# plt.plot(x_ptc, y_ptc, linewidth=1.0, label='Data')
# x_fit = np.linspace(0, 10000, 5000)
# y_fit = coef * x_fit + intercept
# y_fit = y_fit.reshape(5000,)
# plt.plot(x_fit, y_fit, linestyle='--', linewidth=1.0, color='#ff7f0e', label='Fit')
# plt.scatter([x_ptc[0], x_ptc[24]], [y_ptc[0], y_ptc[24]], color='r', label='Fit range')
# plt.scatter([x_ptc[25]], [y_ptc[25]], color='g', marker='s', label='Saturation')
# plt.legend()
# plt.tight_layout()
# plt.show()


# # Plot SNR     SNR=(mu_y-mu_{y_dark}) / y_std
# y_std = np.sqrt(y)
# SNR = x / y_std
# x_fit = np.linspace(0, 14000, 7000)
# y_fit = coef * x_fit + intercept
# y_fit = y_fit.reshape(7000,)
# y_fit_std = np.sqrt(y_fit)
# SNR_fit = x_fit / y_fit_std
# plt.title("SNR graph", fontsize=15)
# plt.xlabel("irradiation(photon/pixel)", fontsize=13)
# plt.ylabel("SNR", fontsize=13)
# plt.grid(True)
# plt.plot(x[:29], SNR[:29], linewidth=1.0, color='#7f7f7f', label='mono data')
# plt.plot(x_fit, SNR_fit, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.show()


#------------------ ELD calibration -------------------
# ## Photon shot noise
# mu_e = x / coef  # here mu_e denotes I in the ELD paper
# mu_e = mu_e.reshape(36,)
# sample1 = poisson.rvs(mu=mu_e[19], size=100) # choose from 0~35
# sample2 = poisson.rvs(mu=mu_e[20], size=100)
# sample3 = poisson.rvs(mu=mu_e[21], size=100)
# sample4 = poisson.rvs(mu=mu_e[22], size=100)
# # Plot
# sns.set_theme()
# sns.kdeplot(x=sample1, fill=True, label='lambda=371.5')
# sns.kdeplot(x=sample2, fill=True, label='lambda=467.1')
# sns.kdeplot(x=sample3, fill=True, label='lambda=587.3')
# sns.kdeplot(x=sample4, fill=True, label='lambda=739.5')
# plt.title('Photon noise PMF', fontsize=15)
# plt.xlabel('Number of photoelectrons', fontsize=13)
# plt.ylabel('Probability', fontsize=13)
# plt.legend()
# plt.tight_layout()
# plt.show()
