import matplotlib.pyplot as plt
import numpy as np
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson
import seaborn as sns
from util import pipeline
import argparse
import yaml
def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


def str2value(list_):
    return list(map(lambda x: eval(str(x)),list_))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script using argparse')
    # Add arguments
    parser.add_argument('-config',default="config/IMX623.yaml",type=str, help='specify config')
    args = parser.parse_args()
    config=load_config(args.config)
    args.iso=str2value(config['ISO'])
    args.exps=str2value(config['EXPOSURE'])
    stride=len(args.exps)*2
    ble_info=config['DARKSHADING']
    root_path=config['OUTPUT']
    t_exp_ori = np.array(args.exps)
    for branch in ble_info:
        expo_range=ble_info[branch]['EXPO_RANGE']
        mask=(t_exp_ori<=expo_range[1])&(t_exp_ori>=expo_range[0])
        data=[]
        for iso in  ble_info[branch]['ISO']:
            i=iso//100
            print(f"iso:{iso}")
            n1 = 2*stride-2+(i-1)*2*stride
            n2 = 3*stride-3+(i-1)*2*stride
            # index for the beginning and the end of bias frames
            n3 = stride-2+(i-1)*2*stride
            n4 = 2*stride-3+(i-1)*2*stride
            mu_y, sigma_y = pipeline(n1, n2,root_path)
            mu_y_dark, _ = pipeline(n3, n4,root_path)
            x = mu_y - mu_y_dark
            y = sigma_y
            print(mask.shape,x.shape)
            x_ = x[mask]
            y_ = y[mask]


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

            data.append((i*100,coef[0][0],intercept))

            x_ = x[mask]
            y_ = y[mask]
            y_pred = y_pred.reshape(-1,)
            x_fit = np.linspace(0, ble_info[branch]['ISO'][-1],10)
            y_fit = coef * x_fit + intercept
            y_fit = y_fit.reshape(-1,)
        #     plt.title(r"$\sigma_y^2 - (\mu_y - \mu_{y\cdot dark})$", fontsize=15)
        #     plt.xlabel(r"$\mu_y - \mu_{y\cdot dark}$", fontsize=13)
        #     plt.ylabel(r"$\sigma_y^2$", fontsize=13)
        #     plt.grid(ls='--', linewidth=0.6, alpha=0.5)
        #     plt.scatter(x_, y_, alpha=0.9, s=30, edgecolors='black', label='Data')
        #     plt.plot(x_fit, y_fit, linewidth=2, linestyle='--', label='Fit')

        # plt.legend()
        # plt.tight_layout()
        # plt.show()



        data=np.array(data)
        x=data[:,0:1]
        y=data[:,1:2]
        model1 = LinearRegression()
        model1.fit(x, y)
        R2_mean = model1.score(x, y)
        y_pred = model1.predict(x)
        plt.title("K", fontsize=15)
        plt.xlabel("iso", fontsize=13)
        plt.ylabel("coef", fontsize=13)
        plt.grid(ls='--', linewidth=0.6, alpha=0.5)
        plt.scatter(x, y, marker='+', color='#7f7f7f', label='mono data')
        plt.plot(x, y_pred, linewidth=1.0, color='#7f7f7f', linestyle='--', label='mono fit')
        plt.text(0.8,280, s=r"$R^2=$" + f"{R2_mean:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(model1.coef_,model1.intercept_)
