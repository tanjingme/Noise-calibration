import rawpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from util import pipeline
import argparse

import yaml

def str2value(list_):
    return list(map(lambda x: eval(str(x)),list_))


def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script using argparse')
    # Add arguments
    parser.add_argument('-config',default="config/IMX623.yaml",type=str, help='specify config')
    # Parse the command-line arguments
    args = parser.parse_args()
    config=load_config(args.config)
    args.iso=str2value(config['ISO'])
    args.exps=str2value(config['EXPOSURE'])
    stride=len(args.exps)*2
    ble_info=config['DARKSHADING']
    # print(stride)
    # print(config['START_INDEX']+stride-1)
    # exit(0)
    data=[]
    for branch in ble_info:
        expo_range=ble_info[branch]['EXPO_RANGE']
        for iso in  ble_info[branch]['ISO']:
            n3 = config['START_INDEX']+(iso//100-1)*stride*2
            n4 = config['START_INDEX']+(iso//100-1)*stride*2+stride-1
            mu_y, sigma_y = pipeline(n3, n4)

            #--------------- OLS Model ---------------
            # Linear Regression Model by sklearn
            t_exp_ori = np.array(args.exps)
            mask=(t_exp_ori<=expo_range[1])&(t_exp_ori>=expo_range[0])
            t_exp=t_exp_ori[mask]

            t_expo = t_exp[:, np.newaxis] # sklearn requires 2-D array for processing
            mu_y = mu_y[:, np.newaxis][mask] # sklearn requires 2-D array for processing

            # Dark current from mean
            model1 = LinearRegression()
            model1.fit(t_expo, mu_y)
            mu_y_pred = model1.predict(t_expo)
            R2_mean = model1.score(t_expo, mu_y)
            # print('Dark current from mean: R2 = %.2f' % R2_mean)
            # print(model1.coef_,model1.intercept_,i*100)
            data.append((iso,model1.coef_[0][0],model1.intercept_))

            # --------------- Plot ---------------
            # Dark current from mean
            mu_y_pred = mu_y_pred.reshape(t_exp.shape[0],)
            mu_y = mu_y.reshape(t_exp.shape[0],)

            plt.title("Dark current from mean", fontsize=15)
            plt.xlabel("Exposure time(s)", fontsize=13)
            plt.ylabel("mean dark signal", fontsize=13)
            plt.grid(ls='--', linewidth=0.6, alpha=0.5)
            plt.scatter(t_exp, mu_y, marker='+',  label=f'iso:{iso} mono data')#color='#7f7f7f',
            plt.plot(t_exp, mu_y_pred, linewidth=1.0,  linestyle='--', label=f'iso:{iso} mono fit')#color='#7f7f7f',
            plt.text(0.8,100, s=r"$R^2=$" + f"{R2_mean:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.show()




        data=np.array(data)

        params=[]

        x=data[:,0:1]
        y=data[:,1:2]
        model1 = LinearRegression()
        model1.fit(x, y)
        R2_mean = model1.score(x, y)
        y_pred = model1.predict(x)
        plt.title("Back level from mean", fontsize=15)
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

    x=data[:,0:1]
    y=data[:,2:3]
    model2 = LinearRegression()
    model2.fit(x, y)
    R2_mean = model2.score(x, y)
    y_pred = model2.predict(x)
    params.append((model1.coef_[0][0],model1.intercept_[0],model2.coef_[0][0],model2.intercept_[0]))
    print(f"({model1.coef_[0][0]}*iso+{model1.intercept_[0]})*expo+{model2.coef_[0][0]}*iso+{model2.intercept_[0]}")
    print(f"modify {args.config} {branch}'s PARAM")

