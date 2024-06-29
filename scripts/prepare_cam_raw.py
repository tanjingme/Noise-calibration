
import argparse
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)
import os
from fabric import Connection
import rawpy
import cv2
import numpy as np
import yaml
def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def str2float(list_):
    return list(map(lambda x: eval(str(x)),list_))


def download(connection,host,user,save_name,folder,vis_folder,exp,iso,seq_num=1,h=1536,w=1920,start_offset=64,bits=12):

    print(save_name)
    try:
        if not os.path.exists(os.path.join(vis_folder,save_name)+".jpg"):

            result = connection.run('./save_tan {} {} {} {}'.format(save_name,seq_num,exp,iso))
            status=os.system(f'scp {user}@{host}:/home/{user}/{save_name}.bin {os.path.join(folder,save_name)+".CR3"}')
            show_img=rawpy.imread(os.path.join(folder,save_name)+".CR3",h=h,w=w,start_offset=start_offset,bits=bits).raw_image.astype("float32")/((1<<bits)-1)
            show_img=np.clip(demosaicing_CFA_Bayer_bilinear(show_img**(1.0/2.2),pattern='RGGB')*255*1,0,255).astype("uint8")
            cv2.imwrite(os.path.join(vis_folder,save_name)+".jpg",show_img[:,:,[2,1,0]])
            result = connection.run('rm {}.bin'.format(save_name))
        return False
    except Exception as e:
        print("invalid data --- download again")
        return True

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

def download_raw_seq(connection,host,user,root_dir,num_iteration,exp,iso,seq_num=1,h=1536,w=1920,start_offset=64,bits=12):
    save_name="{}ms_{}iso".format(int(exp*1000),iso)
    for i in range(0,num_iteration):
        os.makedirs(f"{root_dir}/data{i}/",exist_ok=True)
        result = connection.run('./save_tan {} {} {} {}'.format(save_name,seq_num,exp,iso))
        os.system(f'scp {user}@{host}:/home/{user}/{save_name}.bin {root_dir}/data{i}/')
        result = connection.run('rm {}.bin'.format(save_name))

def download_bias_frames(connection,host,user,root_dir,num_iteration,exp,iso,seq_num=1,h=1536,w=1920,start_offset=64,bits=12):
    df_path = os.path.join(root_dir, f'darkframe-iso-{iso}.npz')
    if os.path.exists(df_path):
        return
    save_name="{}ms_{}iso".format(int(exp*1000),iso)
    for i in range(0,num_iteration):
        os.makedirs(f"{root_dir}/data{i}/",exist_ok=True)
        result = connection.run('./save_tan {} {} {} {}'.format(save_name,seq_num,exp,iso))
        os.system(f'scp {user}@{host}:/home/{user}/{save_name}.bin {root_dir}/data{i}/')
        result = connection.run('rm {}.bin'.format(save_name))

    black_frames=[]

    file=f"{save_name}.bin"
    for i in range(0,num_iteration):
        root_=f"{root_dir}/data{i}/"
        file_path = os.path.join(root_, file)
        y=imx623_canon_seq(file_path,h=h,w=w,start_offset=start_offset)
        seq_num=y.shape[0]
        indices=np.where(np.max(y.reshape(seq_num,-1),axis=1)<=((1<<bits)-1))
        y=y[indices]
        black_frames.append(y)

    blacks=np.concatenate(black_frames,axis=0)

    np.savez_compressed(df_path, data=blacks.astype("uint16"))



# class Connection:
#     def __init__(self,host,user):
#         pass
#     def run(self,cmd):
#         pass

import time
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script using argparse')
    # Add arguments
    parser.add_argument('-config',default="config/IMX623.yaml",type=str, help='specify config')
    # Parse the command-line arguments
    args = parser.parse_args()
    config=load_config(args.config)
    exps=str2float(config['EXPOSURE'])
    args.iso=str2float(config['ISO'])
    args.si=config['START_INDEX']
    
    args.bits=config['BITS']
    bits=args.bits
    stride=len(exps)*2

    print(f"total_image:{len(exps)*len(args.iso)*2}")

    args.host = config['HOST']
    args.user = config['USER']
    args.password =config['PASSWORD']
    args.pre_commands=config['PRE_EXE_COMMAND']
    root_dir=config['BIAS_ROOT']
    print(config)
    connection= Connection(host=args.host, user=args.user)
    for command in args.pre_commands:
        connection.run(command)
        time.sleep(1)

    folder=config['OUTPUT']
    os.makedirs(folder,exist_ok=True)
    is_dark=config['DARK']
    vis_folder=config['VIS_OUTPUT']

    if is_dark:
        start_index=args.si
    else:
        start_index=args.si+stride
    count=0

    from collections import OrderedDict
    info_dict=OrderedDict()
    os.makedirs(vis_folder,exist_ok=True)
    for j,iso in enumerate(args.iso):

        for i,exp in enumerate(exps):

            save_name=f"_MG_0{start_index+count}"
            if i==0:
                info_dict[iso]=[start_index+count]

            unfinished=True
            while unfinished:
                unfinished=download(connection,args.host,args.user,save_name,folder,vis_folder,exp,iso,h=config['H'],w=config['W'],start_offset=config['OFFSET'],bits=config['BITS'])

            save_name=f"_MG_0{start_index+count+1}"
            if i==len(exps)-1:
                info_dict[iso].append(start_index+count+1)

            unfinished=True
            while unfinished:
                unfinished=download(connection,args.host,args.user,save_name,folder,vis_folder,exp,iso,h=config['H'],w=config['W'],start_offset=config['OFFSET'],bits=config['BITS'])
            count=count+2

        start_index=start_index+stride
        if is_dark:
            print(f"BIAS:{iso}")
            download_bias_frames(connection,args.host,args.user,root_dir,config['BIAS_NUM_ITERS'],config['BIAS_EXPO'],iso,seq_num=config['BIAS_MAX_SEQ_NUM'],h=config['H'],w=config['W'],start_offset=config['OFFSET'],bits=config['BITS'])
    if is_dark:
        split_name='dark'
    else:
        split_name='plain'
    with open(f"{split_name}_split_info.txt",'w') as f:
        f.write(f"iso\t\tbegin\t\tend\n")
        for iso in info_dict:
            f.write(f"{iso}\t\t{info_dict[iso][0]}\t\t{info_dict[iso][1]}\n")


        
