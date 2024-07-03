
import os
import numpy as np
from fabric import Connection
import cv2
import argparse
import yaml
import time
def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def str2value(list_):
    return list(map(lambda x: eval(str(x)),list_))

def get_iso_value(file_name):
    iso_str = file_name.split("_")[1]  # 提取文件名中的 ISO 值部分
    iso = int(iso_str.replace("iso.bin", ""))  # 去除 "iso.bin" 后缀并转换为整数
    return iso

def imx623_canon_seq(in_path,h=1536,w=1920,start_offset=64):

    frame_sz=w*h
    frame_offset=w*(h+2) 

    raw_vec=np.fromfile(in_path,dtype=np.uint16)
    raw_length=raw_vec.shape[0]
    list_start=list(range(start_offset,raw_length,frame_offset))
    real_raw_vec=[raw_vec[x:(x+frame_sz)] for x in list_start]

    all=[]
    for j,raw_img in enumerate(real_raw_vec):
        if raw_img.shape[0]!=h*w:
            continue        
        im= raw_img.reshape((h,w)).astype("float32")
        all.append(im)
    data0=np.stack(all,axis=0)
    return data0


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Example script using argparse')
    # Add arguments
    parser.add_argument('-config',default="config/IMX623.yaml",type=str, help='specify config')
    args = parser.parse_args()
    config=load_config(args.config)
    args.iso=str2value(config['ISO'])
    args.exps=str2value(config['EXPOSURE'])

    args.pre_commands=config['PRE_EXE_COMMAND']
    host = config['HOST']
    user = config['USER']
    password = config['PASSWORD']
    root_dir=config['FRAME_ROOT']
    connection= Connection(host=host, user=user,)
    print(config)
    connection= Connection(host=args.host, user=args.user)
    for command in args.pre_commands:
        connection.run(command)
        time.sleep(1)

    exp=config['BIAS_EXPO']
    seq_num=config['BIAS_MAX_SEQ_NUM']
    num_iteration=config['BIAS_NUM_ITERS']
    ISO_RANGE=sorted(args.iso,reverse=True)
    for iso in ISO_RANGE:
        for i in range(0,num_iteration):
            os.makedirs(f"{root_dir}/data{i}/",exist_ok=True)
            save_name="{}ms_{}iso".format(exp*1000,iso)
            result = connection.run('./save_tan {} {} {} {}'.format(save_name,seq_num,exp,iso))
            os.system(f'scp {user}@{host}:/home/{user}/{save_name}.bin {root_dir}/data{i}/')
            result = connection.run('rm {}.bin'.format(save_name))
        save_name="{}ms_{}iso.bin".format(exp*1000,iso)
        black_frames=[]
        file=save_name
        for i in range(0,num_iteration):
            root_=f"{root_dir}/data{i}/"

            file_path = os.path.join(root_, file)
            y=imx623_canon_seq(file_path,h=config['H'],w=config['W'],start_offset=config['OFFSET'])
            seq_num=y.shape[0]
            indices=np.where(np.max(y.reshape(seq_num,-1),axis=1)<=((1<<config['BITS'])-1))
            y=y[indices]
            black_frames.append(y)

        blacks=np.concatenate(black_frames,axis=0)
        df_path = os.path.join(root_dir, f'frame-iso-{iso}.npz')
        np.savez_compressed(df_path, data=blacks.astype("uint16"))
