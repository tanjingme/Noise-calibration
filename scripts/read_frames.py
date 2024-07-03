import numpy as np
black_level=256
data=np.load("./valid_data/frame-iso-3100-20ms.npz")['data']
# from scipy.io import savemat
# savemat("exp{}_iso{}_L.mat".format(lq_exp,lq_iso),{'noisy_list':lq_raw,"gt_list":raw})

import pickle
import os
import pickle as pkl
def get_darkshading(iso, exp=20, naive=True, num=None, remake=True,ds_dir="./darkshading/",darkshading={}):
    branch = '_highISO' if iso>200 else '_lowISO'
    if iso not in darkshading or remake is True:
        ds_path = os.path.join(ds_dir, f'darkshading-iso-{iso}.npy')
        ds_k = np.load(os.path.join(ds_dir, f'darkshading{branch}_k.npy'))
        ds_b = np.load(os.path.join(ds_dir, f'darkshading{branch}_b.npy'))
        if naive: # naive linear darkshading - BLE(ISO)
            with open(os.path.join(ds_dir, f'darkshading_BLE.pkl'), 'rb') as f:
                blc_mean = pkl.load(f)
            BLE = blc_mean[iso]
            print(BLE)
        else: # new linear darkshading - BLE(ISO, t)
            with open(os.path.join(ds_dir, f'BLE_t.pkl'),'rb') as f:
                blc_mean = pickle.load(f)
            # BLE bias only here
            # kt = np.poly1d(blc_mean[f'kt{branch}'])
            #print(blc_mean)
            BLE =blc_mean[f'{iso}']['b'] # + kt(iso) * exp
            print(BLE)

        # D_{ds} = D_{FPNk} + D_{FPNb} + BLE(ISO, t)
        darkshading[iso] = ds_k * iso + ds_b + BLE

    if naive:
        return darkshading[iso]
    else:

        kt = np.poly1d(blc_mean[f'kt{branch}'])
        BLE = kt(iso) * exp
        return darkshading[iso] + BLE
    

# name="exp20_iso3100_L.mat"
# items=name.split("_")
# exp=int(items[0].replace("exp",""))
# iso=int(items[1].replace("iso",""))

DS=get_darkshading(3100,20)
raw=np.clip((np.mean(data.astype(np.float32),axis=0)-DS-black_level)/(4095-black_level),0,1)*20#-np.mean(data1,axis=0)
print(raw.max(),raw.min())
# print(img.max(),img.min())
# from scipy.io import savemat

# lq_raw=(data.astype(np.float32))[:50].astype("uint16")


# # print(lq_raw.max(),lq_raw.min())
# # exit(0)

# savemat("exp20_iso3100_L.mat",{'noisy_list':lq_raw,"gt_list":raw})
# print(img.min())
# print(img.max())

# print(lq_raw.min())
# print(lq_raw.max())
# exit(0)




import matplotlib.pyplot as plt
plt.imshow(raw)
plt.show()
# plt.imshow(lq_raw[0])
# plt.show()