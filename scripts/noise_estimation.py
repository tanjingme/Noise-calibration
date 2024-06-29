import numpy as np
from tqdm import tqdm
import h5py
import argparse
import time
import scipy


import matplotlib.pyplot as plt
import os
import cv2

import yaml
def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config
def str2float(list_):
    return list(map(lambda x: eval(str(x)),list_))


def log(string, log=None, str=False, end='\n', notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')
    else:
        pass
        # os.makedirs('worklog', exist_ok=True)
        # log = f'worklog/worklog-{time.strftime("%Y-%m-%d")}.txt'
        # with open(log,'a+') as f:
        #     f.write(log_string+'\n')
    if str:
        return string+end

class Dark_Mu():
    def __init__(self,param,bl):
        self.param=param
        self.bl=bl
        
    def __call__(self,iso,expo):
        return (self.param[0]*iso+self.param[1])*expo +(self.param[2]*iso+self.param[3]-self.bl)
    
def bayer2rggb(bayer):
    H, W = bayer.shape
    return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)

def rggb2bayer(rggb):
    H, W, _ = rggb.shape
    return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H*2, W*2)


def get_darkshading(args,iso,ble, save=True):
    log(f'getting darkshading of ISO-{iso}', log='./ds_naive.log')
    darkframe_file = os.path.join(args["BIAS_ROOT"], f"darkframe-iso-{iso}.npz")

    raw_seq=load(darkframe_file)['data'].astype(np.float32)-args['BLACK_LEVEL']-ble
    print(ble,iso,'=========================')

    # plt.imshow(bayer2rggb(raw_seq)[0,:,:,3])
    # plt.show()
    raw=np.mean(raw_seq,axis=0)
    raw = bayer2rggb(raw)   # ([H,W,4])

    mean_raw = raw.mean(axis=(0,1)) # [4]
    sigma_raw = raw.std(axis=(0,1)) # [4]
    pattern = 'RGGB'
    for c in range(4):
        log(f'{pattern[c]}, mean:{mean_raw[c]:.3f}, sigma:{sigma_raw[c]:.3e}', log='./ds_naive.log')
    # ignore bad points (because different camera usually have different bad points)
    noise_map = raw - mean_raw
    denoised = cv2.medianBlur(noise_map,5)
    sigma_dark = (noise_map-denoised).std(axis=(0,1))
    # get bad points
    bad_points_map = np.abs((noise_map-denoised))>6*sigma_dark.reshape(1,1,4) #https://math.stackexchange.com/questions/964163/normal-distribution-problem-6-times-the-standard-deviation
    bad_pixels = np.array(np.where(bad_points_map==True)).transpose()
    bpc_img = bad_points_map.astype(np.uint8) * 255
    bpc_img = rggb2bayer(bpc_img)
    log(f'bad points:{len(bad_pixels)}, refine_std:{sigma_raw} -> {sigma_dark}', log='./ds_naive.log')
    # refine dark shading by throw bad points
    # dark_shading = denoised * bad_points_map + noise_map * (1-bad_points_map)
    dark_shading = noise_map

    dark_shading = rggb2bayer(dark_shading+mean_raw)

    denoised = rggb2bayer(denoised+mean_raw)
    if save:
        os.makedirs(args["DARKSHADING_OUTPUT"], exist_ok=True)
        np.save(f'{args["DARKSHADING_OUTPUT"]}/bpc-iso-{iso}.npy', bad_pixels)
        cv2.imwrite(f'{args["DARKSHADING_OUTPUT"]}/bpc-iso-{iso}.png', bpc_img)
        np.save(f'{args["DARKSHADING_OUTPUT"]}/darkshading-iso-{iso}.npy', dark_shading)
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(dark_shading.clip(mean_raw.mean()-3*sigma_dark.mean(),mean_raw.mean()+3*sigma_dark.mean()))
        plt.colorbar()
        fig.savefig(f'{args["DARKSHADING_OUTPUT"]}/darkshading-iso-{iso}.png', bbox_inches='tight')
        # plt.show()
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(denoised.clip(mean_raw.mean()-3*sigma_dark.mean(),mean_raw.mean()+3*sigma_dark.mean()))
        plt.colorbar()
        fig.savefig(f'{args["DARKSHADING_OUTPUT"]}/darkshading-denoised-iso-{iso}.png', bbox_inches='tight')
        # plt.show()
        plt.close()
    return dark_shading, bad_pixels

# def estimate_color_bias(config,iso,darkshading,ble):
#     darkframe_file =os.path.join(config["BIAS_ROOT"], f"darkframe-iso-{iso}.npz")
#     raw_seq=load(darkframe_file)['data']-darkshading
#     R_l, G1_l, G2_l, B_l = [], [], [], []
#     for i in range(raw_seq.shape[0]):
#         raw = raw_seq[i].astype(np.float32) - darkshading
#         packed_raw = bayer2rggb(raw)
#         R_bias = np.mean(packed_raw[:,:,0])-config['BLACK_LEVEL']
#         G1_bias = np.mean(packed_raw[:,:,1])-config['BLACK_LEVEL']
#         G2_bias = np.mean(packed_raw[:,:,2])-config['BLACK_LEVEL']
#         B_bias = np.mean(packed_raw[:,:,3])-config['BLACK_LEVEL']
#         R_l.append(R_bias)
#         G1_l.append(G1_bias)
#         G2_l.append(G2_bias)
#         B_l.append(B_bias)
#     return np.mean(np.array(R_l)), np.mean(np.array(G1_l)), np.mean(np.array(G2_l)), np.mean(np.array(B_l))

def estimate_noise_param(config,iso,darkshading,ble):
    dst_dir=config['NOISE_PARAM_OUTPUT']
    if os.path.exists(f'{dst_dir}/noiseparam-iso-{iso}.h5'):
        return
    r = 400
    darkframe_file =os.path.join(config["BIAS_ROOT"], f"darkframe-iso-{iso}.npz")
    raw_seq=load(darkframe_file)['data']
    print(f"nums of raw: {len(raw_seq)}")
    # Read
    sigmaRead = np.zeros((4, len(raw_seq)), dtype=np.float32)
    meanRead = np.zeros((4, len(raw_seq)), dtype=np.float32)
    r2Read = np.zeros((4, len(raw_seq)), dtype=np.float32)
    # R
    sigmaR = np.zeros((4, len(raw_seq)), dtype=np.float32)
    meanR = np.zeros((4, len(raw_seq)), dtype=np.float32)
    r2R = np.zeros((4, len(raw_seq)), dtype=np.float32)
    # TL
    sigmaTL = np.zeros((4, len(raw_seq)), dtype=np.float32)
    meanTL = np.zeros((4, len(raw_seq)), dtype=np.float32)
    r2TL = np.zeros((4, len(raw_seq)), dtype=np.float32)
    lamda = np.zeros((4, len(raw_seq)), dtype=np.float32)
    # Gs
    sigmaGs = np.zeros((4, len(raw_seq)), dtype=np.float32)
    meanGs = np.zeros((4, len(raw_seq)), dtype=np.float32)
    r2Gs = np.zeros((4, len(raw_seq)), dtype=np.float32)
    #color bias
    color_bias = np.zeros((4, len(raw_seq)), dtype=np.float32)

    pbar = tqdm(range(len(raw_seq)))
    for i in pbar:
        raw = raw_seq[i].astype(np.float32)-config['BLACK_LEVEL']-darkshading-ble

        packed_raw = bayer2rggb(raw)
        R_bias = np.mean(packed_raw[:,:,0])
        G1_bias = np.mean(packed_raw[:,:,1])
        G2_bias = np.mean(packed_raw[:,:,2])
        B_bias = np.mean(packed_raw[:,:,3])
        color_bias[0,i]=R_bias
        color_bias[1,i]=G1_bias
        color_bias[2,i]=G2_bias
        color_bias[3,i]=B_bias

        # packed_raw=packed_raw-config['BLACK_LEVEL']

        packed_raw = packed_raw[config['H']//2-r:config['H']//2+r,config['W']//2-r:config['W']//2+r]
        for c in range(4):
            img = packed_raw[:,:,c]
            # Compute σR
            _, (sigRead, uRead, rRead) = scipy.stats.probplot(img.reshape(-1), rvalue=True)
            sigmaRead[c][i] = sigRead
            meanRead[c][i] = uRead
            r2Read[c][i] = rRead**2
            # print(f'uRead={uRead:.4f}, sigRead={sigRead:.4f}, read2={rRead**2:.4f}')
            # img = pack_center_crop(pack_raw_bayer(raw), size=1000)[1]
            row_all = np.mean(img, axis=1)
            _, (sigR, uR, rR) = scipy.stats.probplot(row_all, rvalue=True)
            sigmaR[c][i] = sigR
            meanR[c][i] = uR
            r2R[c][i] = rR**2
            # print(f'uR={uR:.4f}, sigR={sigR:.4f}, r2={rR**2:.4f}')

            # Compute σTL
            img = img - row_all.reshape(-1,1)
            X = img.reshape(-1)

            lam = scipy.stats.ppcc_max(X)
            lamda[c][i] = lam
            _, (sigGs, uGs, rGs) = scipy.stats.probplot(X, rvalue=True)
            # print(f'uGs={uGs:.4f}, sigGs={sigGs:.4f}, rGs2={rGs**2:.4f}')
            _, (sigTL, uTL, rTL) = scipy.stats.probplot(X, dist=scipy.stats.tukeylambda(lam), rvalue=True)
            sigmaTL[c][i] = sigTL
            meanTL[c][i] = uTL
            r2TL[c][i] = rTL**2
            # print(f'λ={lam:.3f}, uTL={uTL:.4f}, sigTL={sigTL:.4f}, rTL2={rTL**2:.4f}')

            _, (sigGs, uGs, rGs) = scipy.stats.probplot(X, rvalue=True)
            sigmaGs[c][i] = sigGs
            meanGs[c][i] = uGs
            r2Gs[c][i] = rGs**2
        pbar.set_description(f"Raw {i:03d}")
        #设置进度条右边显示的信息
        pbar.set_postfix_str(
            f'iso={iso} uRead={meanRead[:, :i+1].mean():.4f}, sigRead={sigmaRead[:, :i+1].mean():.4f}, lam={lamda[:, :i+1].mean()**2:.4f}, '+ 
            f'rR2={r2R[:, :i+1].mean():.4f}, rGs2={r2Gs[:, :i+1].mean():.4f}, rTL2={r2TL[:, :i+1].mean():.4f}'
            )
    param = {
        'lam':lamda, 'wp':config['WHITE_LEVEL'], 'bl':config['BLACK_LEVEL'],
        'sigmaRead':sigmaRead, 'meanRead':meanRead, 'r2Gs':r2Read,
        'sigmaR':sigmaR, 'meanR':meanR, 'r2R':r2R,
        'sigmaTL':sigmaTL, 'meanTL':meanTL, 'r2TL':r2TL,
        'sigmaGs':sigmaGs, 'meanGs':meanGs, 'r2Gs':r2Gs,'color_bias':color_bias
    }
    
    with h5py.File(f'{dst_dir}/noiseparam-iso-{iso}.h5', 'w') as f:
        for key in param:
            f.create_dataset(key, data=param[key])



from sklearn.linear_model import LinearRegression
def regr_plot(x, y, log=True, ax=None, c1=None, c2=None, label=False):
    x = np.array(x)
    y = np.array(y)
    if log:
        x = np.log(x)
        y = np.log(y)
    ax.scatter(x, y)

    regr = LinearRegression()
    regr.fit(x.reshape(-1,1), y)
    a, b = float(regr.coef_), float(regr.intercept_)   
    # ax.set_title('log(sigR) | log(K)')
    x_range = np.array([np.min(x), np.max(x)])
    std = np.mean((a*x+b-y)**2) ** 0.5
    
    if c1 is not None:
        if label:
            label = f'k={a:.5f}, b={b:.5f}, std={std:.5f}'
            ax.plot(x, regr.predict(x.reshape(-1,1)), linewidth = 2, c=c1, label=label)
        else:
            ax.plot(x, regr.predict(x.reshape(-1,1)), linewidth = 2, c=c1)
    
    if c2 is not None:
        ax.plot(x_range, a*x_range+b+std, c=c2, linewidth = 1)
        ax.plot(x_range, a*x_range+b-std, c=c2, linewidth = 1)

    data = {'k':a,'b':b,'sig':std}

    return ax, data
 

def save_params(data, save_path='iso_parts_params_SonyA7S2.txt'):
    if save_path is None:
        print("cam_noisy_params['ISO_%d_%d'] = {"%(data['ISO_start'], data['ISO_end']))
        print("    'Kmin':%.5f, 'Kmax':%.5f, 'lam':%.3f, 'q':%.3e, 'wp':%d, 'bl':%d,"%(data['Kmin'], data['Kmax'], data['lam'], data['q'], data['wp'],data['bl']))
        print("    'sigRk':%.5f,  'sigRb':%.5f,  'sigRsig':%.5f,"%(data['sigRk'], data['sigRb'], data['sigRsig']))
        print("    'sigTLk':%.5f, 'sigTLb':%.5f, 'sigTLsig':%.5f,"%(data['sigTLk'], data['sigTLb'], data['sigTLsig']))
        print("    'sigGsk':%.5f, 'sigGsb':%.5f, 'sigGssig':%.5f"%(data['sigGsk'], data['sigGsb'], data['sigGssig']))
        print("    'sigReadk':%.5f, 'sigReadb':%.5f, 'sigReadsig':%.5f,"%(data['sigReadk'], data['sigReadb'], data['sigReadsig']))
        print("    'uReadk':%.5f, 'uReadb':%.5f, 'uReadsig':%.5f}"%(data['uReadk'], data['uReadb'], data['uReadsig']))
    else:
        f=open(save_path, "a+")
        print("cam_noisy_params['ISO_%d_%d'] = {"%(data['ISO_start'], data['ISO_end']), file=f)
        print("    'Kmin':%.5f, 'Kmax':%.5f, 'lam':%.3f, q':%.3e, 'wp':%d, 'bl':%d,"%(data['Kmin'], data['Kmax'], data['lam'], data['q'], data['wp'],data['bl']), file=f)
        print("    'sigRk':%.5f,  'sigRb':%.5f,  'sigRsig':%.5f,"%(data['sigRk'], data['sigRb'], data['sigRsig']), file=f)
        print("    'sigTLk':%.5f, 'sigTLb':%.5f, 'sigTLsig':%.5f,"%(data['sigTLk'], data['sigTLb'], data['sigTLsig']), file=f)
        print("    'sigGsk':%.5f, 'sigGsb':%.5f, 'sigGssig':%.5f,"%(data['sigGsk'], data['sigGsb'], data['sigGssig']), file=f)
        print("    'sigReadk':%.5f, 'sigReadb':%.5f, 'sigReadsig':%.5f,"%(data['sigReadk'], data['sigReadb'], data['sigReadsig']), file=f)
        print("    'uReadk':%.5f, 'uReadb':%.5f, 'uReadsig':%.5f}"%(data['uReadk'], data['uReadb'], data['uReadsig']), file=f)

import rawpy
# def load(path):
#     i=int(int(path.split("-")[-1].replace(".npz",""))/100)
#     n3 = 82+(i-1)*168
#     data1=rawpy.imread(f"images/_MG_0{n3+19*2}.CR3").raw_image.copy().astype(np.float32)
#     data2=rawpy.imread(f"images/_MG_0{n3+19*2+1}.CR3").raw_image.copy().astype(np.float32)
#     h,w=data2.shape


#     data1[0:70,810:960]=np.flip(data1[0:70,(w-960):(w-810)],axis=1)
#     data1[130:310,80:400]=np.flip(data1[130:310,(w-400):(w-80)],axis=1)

#     data2[0:70,810:960]=np.flip(data2[0:70,(w-960):(w-810)],axis=1)
#     data2[130:310,80:400]=np.flip(data2[130:310,(w-400):(w-80)],axis=1)



#     data=np.stack([data1,data2],axis=0)
#     data_dict={}
#     data_dict['data']=data
#     return data_dict

def load(path):
    data=np.load(path)['data']
    # s,h,w=data.shape
    # data[:,0:70,810:960]=np.flip(data[:,0:70,(w-960):(w-810)],axis=2)
    # data[:,130:310,80:400]=np.flip(data[:,130:310,(w-400):(w-80)],axis=2)

    data_dict={}
    data_dict['data']=data
    return data_dict


def analysis_data(config, param_dir='resources', title='Noise Profile',
                K_ISO=(0.76/800,0), isos=[], save_dir=None):
    
    isos=sorted(isos)

    fig = plt.figure(figsize=(20,8))
    fig2 = plt.figure(figsize=(20,8))
    fig.suptitle(title)
    fig2.suptitle(title)

    axR = fig2.add_subplot(1,3,1)
    axTL = fig2.add_subplot(2,3,2)
    axlam = fig2.add_subplot(2,3,3)
    axGs = fig2.add_subplot(1,3,2)
    axMean = fig2.add_subplot(1,3,3)
    
    params = []

    for iso in isos:
        param = {'ISO': iso}
        f = h5py.File(os.path.join(param_dir, f'noiseparam-iso-{iso}.h5'), 'r')
        for key in f:
            param[key] = np.array(f[key])

            if 'color_bias' in key:
                param[key] = param[key].mean(axis=-1)
            else:
                if len(param[key].shape)>1: 
                    param[key] = param[key].mean(axis=0)
        params.append(param)

    axR.set_title('sigmaR | ISO')
    axTL.set_title('sigmaTL | ISO')
    axlam.set_title('lam | ISO')
    axGs.set_title('sigmaGs | ISO')
    axMean.set_title('uRead | ISO')

    # axR.scatter(isos, params[:]['sigmaR'].mean(axis=-1))
    # axGs.scatter(isos, param['sigmaGs'].mean(axis=-1))
    # axMean.scatter(isos, param['meanRead'].mean(axis=-1))

    axsigR = fig.add_subplot(2,3,1)
    axsigTL = fig.add_subplot(2,3,2)
    axlam = fig.add_subplot(2,3,3)
    axsigGs = fig.add_subplot(2,3,4)
    axMean = fig.add_subplot(2,3,5)
    axRead = fig.add_subplot(2,3,6)
    axsigTL.set_title('log(sigTL) | log(K)')
    axlam.set_title('lam | ISO')
    axsigR.set_title('log(sigR) | log(K)')
    axsigGs.set_title('log(sigGs) | log(K)')
    axMean.set_title('log(uRead) | log(K)')
    axRead.set_title('log(sigRead) | log(K)')
    # axsigR.set_title('sigR | K')
    # axsigTL.set_title('sigTL | K')
    # axlam.set_title('lam | K')
    # axsigGs.set_title('sigGs | K')
    # axMean.set_title('uRead | K')
    # axRead.set_title('sigRead | K')

    iso=[]
    iso_points = []
    sigR=[]
    sigTL=[]
    sigRead=[]
    uRead=[]
    lam=[]
    sigGs=[]
    split_point = [0]
    cnt = 0
    color_bias_iso=[]


    for param in params:

        for i in range(len(param['sigmaR'])):
            point_iso = param['ISO']*K_ISO[0]+K_ISO[1]
            iso.append(point_iso)
            sigR.append(param['sigmaR'][i])
            sigTL.append(param['sigmaTL'][i])
            sigGs.append(param['sigmaGs'][i])
            lam.append(param['lam'][i])
            sigRead.append(param['sigmaRead'][i])
            

        iso_points.append(point_iso)
        color_bias_iso.append(param['color_bias'].tolist())
        uRead.append(param['meanRead'].std())
    

    iso = np.array(iso)

    for i, point_iso in enumerate(iso):
        if iso[split_point[cnt]] != iso[i]:
            cnt += 1
            split_point.append(i)
    split_point.append(len(iso))

    axsigR, dataR = regr_plot(iso, sigR, ax=axsigR, c1='red', c2='orange', log=True, label=True)
    axsigTL, dataTL = regr_plot(iso, sigTL, ax=axsigTL, c1='red', c2='orange', log=True, label=True)
    axsigRead, dataRead = regr_plot(iso, sigRead, ax=axRead, c1='red', c2='orange', log=True, label=True)
    axuRead, datauRead = regr_plot(iso_points, uRead, ax=axMean, c1='red', c2='orange', log=False, label=False)
    axsigGs, dataGs = regr_plot(iso, sigGs, ax=axsigGs, c1='red', c2='orange', log=True, label=True)
    axlam, datalam = regr_plot((iso-K_ISO[1])/K_ISO[0], lam, ax=axlam, log=False, c1='red', c2='orange', label=True)
  
    data = {
        'ISO_start':params[0]['ISO'], 'ISO_end':params[-1]['ISO'], 'q':1/config['WHITE_LEVEL'], 'wp':config['WHITE_LEVEL'], 'bl':config['BLACK_LEVEL'],
        'Kmin':np.min(iso), 'Kmax':np.max(iso), 'lam':np.mean(lam),
        'sigTLk':dataTL['k'], 'sigTLb':dataTL['b'], 'sigTLsig':dataTL['sig'],
        'sigGsk':dataGs['k'], 'sigGsb':dataGs['b'], 'sigGssig':dataGs['sig'],
        'sigRk':dataR['k'], 'sigRb':dataR['b'], 'sigRsig':dataR['sig'],
        'sigReadk':dataRead['k'], 'sigReadb':dataRead['b'], 'sigReadsig':dataRead['sig'],
        'uReadk':datauRead['k'], 'uReadb':datauRead['b'], 'uReadsig':datauRead['sig'],'color':color_bias_iso
    }

    save_params(data, save_dir)

    # # 分段拟合

    for i in range(len(split_point)-2):
        s = split_point[i]
        e = split_point[i+2]
        iso_part = np.array(iso[s:e])
        sigR_part = sigR[s:e]
        sigTL_part = sigTL[s:e]
        sigRead_part = sigRead[s:e]
        uRead_part = uRead[i:i+2]
        sigGs_part = sigGs[s:e]
        lam_part = lam[s:e]
        axsigR, dataR = regr_plot(iso_part, sigR_part, ax=axsigR, c1='blue', c2='green',log=True)
        axsigTL, dataTL = regr_plot(iso_part, sigTL_part, ax=axsigTL, c1='blue', c2='green', log=True)
        axsigRead, dataRead =regr_plot(iso_part, sigRead_part, ax=axRead, c1='blue', c2='green', log=True)
        axuRead, datauRead = regr_plot(iso_points[i:i+2], uRead_part, ax=axMean, c1='blue', c2='green', log=False)
        axsigGs, dataGs = regr_plot(iso_part, sigGs_part, ax=axsigGs, c1='blue', c2='green',log=True)
        axlam, datalam = regr_plot((iso_part-K_ISO[1])/K_ISO[0], lam_part, ax=axlam, log=False, c1='blue', c2='green')
        data = {
            'ISO_start':params[i]['ISO'], 'ISO_end':params[i+1]['ISO'], 'q':1/config['WHITE_LEVEL'], 'wp':config['WHITE_LEVEL'], 'bl':config['BLACK_LEVEL'],
            'Kmin':np.min(iso_part), 'Kmax':np.max(iso_part), 'lam':np.mean(lam_part),
            'sigTLk':dataTL['k'], 'sigTLb':dataTL['b'], 'sigTLsig':dataTL['sig'],
            'sigGsk':dataGs['k'], 'sigGsb':dataGs['b'], 'sigGssig':dataGs['sig'],
            'sigRk':dataR['k'], 'sigRb':dataR['b'], 'sigRsig':dataR['sig'],
            'sigReadk':dataRead['k'], 'sigReadb':dataRead['b'], 'sigReadsig':dataRead['sig'],
            'uReadk':datauRead['k'], 'uReadb':datauRead['b'], 'uReadsig':datauRead['sig'],
        }
        save_params(data, save_dir)

    axsigR.legend()
    axsigTL.legend()
    axsigRead.legend()
    axuRead.legend()
    axlam.legend()
    axsigGs.legend()

    # plt.show()
    fig.savefig(f'{title}.png')
    plt.close()

    return params,color_bias_iso


import pickle as pkl
def get_darkshading_templet(args, legal_iso=[], hint='', denoised=False):
    darkshadings = []
    ds_mean = []
    ds_std = []
    hint_dn = '_denoised' if denoised else ''
    for i, iso in enumerate(legal_iso):
        ds_path = os.path.join(args['DARKSHADING_OUTPUT'], f'darkshading-iso-{iso}.npy')
        ds = np.load(ds_path)

        if denoised:
            ds = bayer2rggb(ds)
            ds = cv2.medianBlur(ds,5)
            ds = rggb2bayer(ds)
        darkshadings.append(ds-ds.mean())
        ds_mean.append(ds.mean())
        ds_std.append(ds.std())
        log(f'ISO:{iso}, mean:{ds_mean[i]:.4f}, std:{ds_std[i]:.4f}', log='./ds_templet.log')

    h,w = darkshadings[0].shape
    ds_data = np.array(darkshadings).reshape(len(legal_iso), -1)
    reg = np.polyfit(legal_iso,ds_data,deg=1)
    ds_k = reg[0].reshape(h, w)
    # plt.imshow(ds_k)
    # plt.show()
    ds_b = reg[1].reshape(h, w)
    # ry = np.polyval(reg, legal_iso)
    print(ds_k.std(), ds_b.mean())
    # ds_b = repair_bad_pixels(ds_b, bad_points=np.load(f'{args["dst_dir"]}/bpc.npy'))
    np.save(os.path.join(args['DARKSHADING_OUTPUT'], f'darkshading{hint}{hint_dn}_k.npy'), ds_k)
    np.save(os.path.join(args['DARKSHADING_OUTPUT'], f'darkshading{hint}{hint_dn}_b.npy'), ds_b)
    plt.figure(figsize=(16,10))
    plt.imshow(ds_k.clip(-3*ds_k.std(),3*ds_k.std()))
    plt.colorbar()
    plt.savefig(f'{args["DARKSHADING_OUTPUT"]}/darkshading{hint}{hint_dn}_k.png', bbox_inches='tight')
    plt.figure(figsize=(16,10))
    plt.imshow(ds_b.clip(-3*ds_b.std(),3*ds_b.std()))
    plt.colorbar()
    plt.savefig(f'{args["DARKSHADING_OUTPUT"]}/darkshading{hint}{hint_dn}_b.png', bbox_inches='tight')
    # 保存BLE
    if os.path.exists(os.path.join(args['DARKSHADING_OUTPUT'], f'darkshading{hint_dn}_BLE.pkl')):
        with open(os.path.join(args['DARKSHADING_OUTPUT'], f'darkshading{hint_dn}_BLE.pkl'), 'rb') as f:
            BLE = pkl.load(f)
    else:
        BLE = {}
    for i, iso in enumerate(legal_iso):
        BLE[iso] = ds_mean[i]
    with open(os.path.join(args['DARKSHADING_OUTPUT'], f'darkshading{hint_dn}_BLE.pkl'), 'wb') as f:
        pkl.dump(BLE, f)
    return BLE

import pickle
def read_darkshading(iso, exp=20, naive=False, num=None, remake=True,ds_dir="./darkshading",darkshading={}):
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
        kt = np.poly1d(blc_mean[f'kt{branch}'][[0,1]])
        BLE = kt(iso) * exp
        return darkshading[iso] + BLE


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script using argparse')
    parser.add_argument('-config',default="config/IMX623.yaml",type=str, help='specify config')
    args = parser.parse_args()
    config=load_config(args.config)
    args.iso=str2float(config['ISO'])
    dst_dir=config['DARKSHADING_OUTPUT']
    expo=config['BIAS_EXPO']
    ble_info=config['DARKSHADING']
    blc_mean={}
    for branch in ble_info:
        ble_param=np.array(ble_info[branch]['PARAM'])
        blc_mean[f'kt{branch}']=ble_param[[0,1]]/1000

    os.makedirs(dst_dir,exist_ok=True)
    os.makedirs(config['NOISE_PARAM_OUTPUT'],exist_ok=True)

    for branch in ble_info:
        ble_param=np.array(ble_info[branch]['PARAM'])#.reshape(-1,4)
        ble_fn=Dark_Mu(ble_param,config['BLACK_LEVEL'])

        for iso in  ble_info[branch]['ISO']:
            os.makedirs(dst_dir, exist_ok=True)
            ble=ble_fn(iso,expo).astype(np.float32)
            if not os.path.exists(f'{dst_dir}/darkshading-iso-{iso}.npy'):
                
                darkshading, bad_points = get_darkshading(config,iso,ble,save=True)
            else:
                darkshading = np.load(f'{dst_dir}/darkshading-iso-{iso}.npy')
                bad_points = np.load(f'{dst_dir}/bpc-iso-{iso}.npy')

            estimate_noise_param(config,iso,darkshading,ble)


        BLE=get_darkshading_templet(config, ble_info[branch]['ISO'], branch)

        get_darkshading_templet(config, ble_info[branch]['ISO'], branch, denoised=True)
        for iso in BLE:
            blc_mean[f'{iso}']={'b':BLE[iso]+ble_param[2]*iso+ble_param[3]-config['BLACK_LEVEL']}

        params,color_bias=analysis_data(config,param_dir=config['NOISE_PARAM_OUTPUT'], K_ISO=ble_info[branch]['K_ISO'], isos=ble_info[branch]['ISO'],title=f"Noise Profile{branch}",save_dir="iso_parts_params_Custom.txt")

        print(color_bias)

    with open(os.path.join(dst_dir, f'BLE_t.pkl'),'wb') as f:
        pkl.dump(blc_mean,f)
    