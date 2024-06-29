
import os
import numpy as np
from fabric import Connection
import cv2
def get_iso_value(file_name):
    iso_str = file_name.split("_")[1]  # 提取文件名中的 ISO 值部分
    iso = int(iso_str.replace("iso.bin", ""))  # 去除 "iso.bin" 后缀并转换为整数
    return iso
def imx623_canon_seq(in_path):
    w=1920
    h=1536
    frame_sz=w*h
    frame_offset=w*(h+2) 
    start_offset=64

    raw_vec=np.fromfile(in_path,dtype=np.uint16)
    raw_length=raw_vec.shape[0]
    list_start=list(range(start_offset,raw_length,frame_offset))
    real_raw_vec=[raw_vec[x:(x+frame_sz)] for x in list_start]
    bit=12
    max_v=(1<<bit)
    all=[]
    for j,raw_img in enumerate(real_raw_vec):
        if raw_img.shape[0]!=h*w:
            continue        
        im= raw_img.reshape((h,w)).astype("float32")
        all.append(im)
    data0=np.stack(all,axis=0)
    return data0



import time
if __name__=="__main__":
    host = '192.168.198.198'
    user = 'root'
    password = 'your_password'
    connection= Connection(host=host, user=user,)
    result = connection.run('ls -l')
    connection.run('./print_reg')
    time.sleep(1)
    connection.run('fpga_tools reg w 0x204 1')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0x9c 0x28 w8 0x00 0x01 0x00 0x01 0x00 0x01 0x00 0x01')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0xb4 0x88 w2 0x00 0x01')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0xb4 0x5e w16 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0xb4 0x6e w16 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0')
    time.sleep(1)
    exp=0.02
    root_dir="bias_frames"
    seq_num=75
    num_iteration=6
    ISO_RANGE=range(3100,0,-100)
    for iso in ISO_RANGE:
        for i in range(0,num_iteration):
            os.makedirs(f"bias_frames/data{i}/",exist_ok=True)
            save_name="{}ms_{}iso".format(exp*1000,iso)
            result = connection.run('./save_tan {} {} {} {}'.format(save_name,seq_num,exp,iso))
            os.system(f'scp {user}@{host}:/home/{user}/{save_name}.bin {root_dir}/data{i}/')
            result = connection.run('rm {}.bin'.format(save_name))

        black_frames=[]
        file=save_name
        for i in range(0,num_iteration):
            root_=f"{root_dir}/data{i}/"
            # parts = file.split("_")
            # 提取曝光时间和 ISO 值
            # exposure_time = parts[0].replace("ms", "")  # 去除 "ms" 后缀
            # iso = float(parts[1].replace("iso.bin", "")) # 去除 "iso.bin" 后缀
            file_path = os.path.join(root_, file)
            y=imx623_canon_seq(file_path)
            seq_num=y.shape[0]
            indices=np.where(np.max(y.reshape(seq_num,-1),axis=1)<=4095)
            y=y[indices]
            black_frames.append(y)

        blacks=np.concatenate(black_frames,axis=0)
        df_path = os.path.join("valid_data", f'darkframe-iso-{iso}.npz')
        np.savez_compressed(df_path, data=blacks.astype("uint16"))




    # # root_dir=os.path.join(root_dir,'data0')
    # # for root, dirs, files in os.walk(root_dir):
    # #     files = sorted(files, key=get_iso_value)
    # #     os.makedirs(f"valid_data",exist_ok=True)
    #     for file in files:
